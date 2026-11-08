# Faultline

Faultline is a command-line tool for comparing eager and compiled PyTorch execution. It checks
outputs and gradients, reduces reproducible failures, and exports standalone Python reproducers.
It also supports request sequences and comparisons between two compiler environments.

Use it to check a model before changing PyTorch versions or compiler settings, then package a
failing workload for investigation. The `model` commands execute your original module and preserve
weights and nested inputs in binary snapshots. A separate compact graph format supports fuzzing.
Each run produces JSON results, worker logs, and a local HTML report. Workers have configurable
time and host-memory limits.

Faultline complements PyTorch's debugging tools:

| Tool | Main purpose |
|---|---|
| PyTorch accuracy minifier | Reduce a failing graph from a compiler execution |
| `TORCH_COMPILE_DEBUG` and compiler logs | Inspect graphs, transformations, and generated code |
| `tlparse` / `TORCH_TRACE` | Inspect captured regions, graph breaks, and recompilations |
| Faultline | Freeze workloads and request sequences, compare environments, distinguish new and recurring failures, and export baseline-checked reproducers |

The [PyTorch troubleshooting guide](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_troubleshooting.html)
describes the upstream tools. Faultline's reducer is independent and cannot minimize every failure.

## Installation

Use Python 3.11 or newer and a virtual environment. Install the PyTorch build appropriate for your
machine first. For a CPU-only installation:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install "git+https://github.com/tepiduck/ipsum.git"
faultline --version
```

From a local checkout, use `python -m pip install .`. Commands are available through both
`faultline` and `python -m faultline`.

## Example

```bash
faultline ui
```

The terminal menu can run the demo, test the corpus or a JSON case, browse campaigns, and replay,
reduce, or export a finding. Results are saved under `faultline-runs/`. Choose another location with
`faultline ui --workspace PATH`.

To run the same demo from the command line:

```bash
faultline demo --output faultline-runs/demo
faultline report faultline-runs/demo/injected --open
python faultline-runs/demo/reduced/reproducer/repro.py
```

The demo injects a fault by replacing a sum with a mean, detects the difference, and reduces the
case. The demo command returns `0` on success. Its exported reproducer returns `1` when it observes
the injected fault. Use a new output directory for each run.

## Testing a compiler

```bash
faultline doctor
faultline doctor --backend inductor --probe faultline-runs/probe
faultline run --corpus --count 12 --seed 0 --backend inductor --output faultline-runs/compiled
faultline report faultline-runs/compiled --open
```

This runs the built-in corpus and 12 generated cases. Available backends are:

| Backend | Execution |
|---|---|
| `inductor` | TorchInductor compilation and code generation |
| `aot_eager` | AOT tracing without Inductor code generation |
| `aot_eager_decomp_partition` | AOT tracing with decompositions and partitioning |
| `eager` | Eager execution for checking the test pipeline |

CPU Inductor requires a C++ toolchain and Python development headers. It has been tested on
Ubuntu/WSL. `doctor --probe` checks whether the selected backend works in your environment.

For AMD GPUs, use a compatible PyTorch/ROCm installation and `--device cuda:0`; ROCm uses PyTorch's
`cuda` namespace. To check that installation:

```bash
faultline qualify --device cuda:0 --require-rocm --output faultline-runs/rocm-check
```

This runs forward/backward checks for MLP, convolution, and masked attention in float32, float16,
and bfloat16, with a report for each case. Use `--dtype float32` or `--workload attention` to narrow
the matrix. Use `--require-cuda` for NVIDIA, or select `--device cpu` for CPU checks. Missing
devices produce an unavailable result; the command does not substitute CPU execution.

The host-memory limit does not cover GPU memory. Consult AMD's [ROCm compatibility documentation](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html)
for the supported GPU, driver, OS, and PyTorch combinations.

Both runtime requirements reject a mismatched installation.
The optional [GPU workflow](.github/workflows/gpu.yml) runs on a prepared self-hosted Linux runner;
it does not provision hardware. RX 7900 XT / WSL results are recorded in [RESULTS.md](RESULTS.md).
NVIDIA/CUDA execution remains unvalidated. See [the tested ROCm setup](docs/numerics.md#gpu-environment-and-nightly-checks).

## Checking an existing model

Put model construction and representative inputs in a trusted Python factory:

```python
import torch
from torch import nn


def factory():
    model = nn.Sequential(nn.Linear(768, 768), nn.GELU(), nn.Linear(768, 128)).eval()
    # Load a checkpoint here if needed. Capture freezes the resulting state_dict.
    return model, (torch.randn(8, 768),)
```

```bash
faultline model capture my_model.py:factory --output workloads/my-model
faultline model run workloads/my-model --backend inductor --gradients --timeout 600 --max-rss-mb 8192 --output faultline-runs/model
faultline model replay faultline-runs/model --output faultline-runs/recheck
```

Native runs also accept `--autocast float16|bfloat16` and Inductor's `--compile-mode` choices,
including `reduce-overhead`. Autocast applies to forward execution; the loss and backward run
outside that context. These policies are retained by replay, reduction, and standalone export.
Use `--deterministic` to require deterministic algorithms and a fixed BLAS workspace policy.
With `--numerics`, environment comparisons also report fp64-relative accuracy changes when both
versions already fail tolerance. These advisories leave primary results and exit codes unchanged.

New captures save fixed backward vectors for gradient checks, so each execution receives the same
cotangents. Existing bundles retain their original loss-based policy. Add `--fp64-advisory` to
compare both paths with higher-precision eager execution and report per-tensor error ratios and
distributions. It supplies numerical evidence without changing the primary verdict. See
[gradient probes and numerical evidence](docs/numerics.md) for the scope and limitations.

For gradient findings, `--aot --numerics` adds an eager execution control and compares supported
saved activation masks between AOT eager replay and compiled execution. Reports distinguish
observed mask flips and comparable fp64 error from the primary mismatch. These advisory labels
leave the exit code and reduction predicate unchanged.

`faultline sweep prepare` downloads and freezes the original catalog of 20 pretrained vision and language
models with real image/text inputs. `faultline sweep run` compares two environments, records
per-target diagnostic coverage, and supports interruption/resume with evidence verification.
See [pretrained sweeps](docs/sweeps.md) for commands, dependencies, input scope, and interpretation.
Use `--set held-out` for nine additional model families or `--set all` for all 29. Reports include
absolute fp64 errors, noise-floor triage, and a distributional cross-version accuracy advisory.
`faultline bisect` can localize a CPU accuracy shift across isolated nightly installations;
see [the bisection workflow](docs/bisect.md) for its predicate and resume behavior.
The [recorded CPU and AMD sweep](RESULTS.md#version-09-pretrained-environment-sweep) includes
gradient findings, unstable references, and diagnostic memory limits; it is not a model-correctness claim.

Inputs can be a tensor, a positional tuple, a dictionary of keyword arguments, or
`{"args": (...), "kwargs": {...}}`. Nested dictionaries, lists, and tuples are supported.
Integer token IDs, boolean masks, mixed floating dtypes, noncontiguous views, and shared tensor
storage are retained. With `--trace`, the factory returns an ordered list of requests.

The factory is executed again to construct the model at replay time, then saved weights, inputs,
trainability flags, and RNG state are restored. Its dependencies and any files it opens must still
be available. Use `--include helper.py --include weights.pt` to copy explicitly named local source
and assets into the bundle. Relative assets should be opened relative to the factory's `__file__`.
This is not a serialized arbitrary Python process or a checkpoint downloader.

To check an environment change:

```bash
faultline model compare workloads/my-model --baseline-python .baseline/bin/python --candidate-python .candidate/bin/python --gradients --reduce --timeout 600 --max-rss-mb 8192 --output faultline-runs/model-upgrade
```

Both interpreters need PyTorch, NumPy, psutil, and the model's dependencies. They use the same frozen
workload and comparison policy. A regression requires the baseline to agree with eager execution
and the candidate to fail consistently. A discrepancy present in both environments is reported
separately. The HTML report and JSON result retain both outcomes.

For a saved failing run:

```bash
faultline model reduce faultline-runs/model --max-evaluations 32 --output faultline-runs/model-reduced
python faultline-runs/model-reduced/reproducer/repro.py
```

Reduction first removes requests, then tries compiling individual captured graphs inside the
original model. If one graph independently reproduces the failure, Faultline attempts node bypass
and dead-code removal. Every accepted change must retain the stable failure; upgrade reductions
must also keep the baseline passing. A fresh worker confirms the final result. If a graph cannot
be isolated or serialized, the full model reproducer remains available. Shape reduction and
minification across interacting graphs are not implemented for native workloads.

Forward-only reduction can also replace a graph prefix with a frozen internal tensor, preserving
its recorded input layout. Each accepted proposal must retain the failure and pass the baseline.
For a gradient failure, add `--aot` to capture the compiler's forward and backward graphs:

```bash
faultline model compare workloads/my-model --baseline-python .baseline/bin/python --candidate-python .candidate/bin/python --gradients --aot --reduce --timeout 600 --max-rss-mb 8192 --output faultline-runs/backward-upgrade
```

The reducer first checks whether compiling a single backward graph still causes the original
model's failure. It then tries replaying that graph with its captured activations and cotangents,
and reducing it independently. The standalone reproducer runs the saved derivative graph directly;
it does not differentiate it a second time. Forward failures can be isolated the same way.

AOT capture uses version-dependent PyTorch compiler hooks and disables the AOTAutograd cache for
the diagnostic run. Kernel caching remains enabled. If an independent graph loses the failure,
the full-model reproducer is retained. See [architecture](docs/architecture.md) for the boundary
between whole-model evidence and an independently reproduced graph failure.

The native path has no compact-IR tensor-size or operator-count caps. It still needs enough memory
for the model, compiler, frozen inputs, and comparison observations. It supports evaluation-mode
forward execution and optional input/parameter gradient checks, not optimizer steps or general
training loops. Graph breaks can leave portions of the model running eagerly; the report records
captured graphs and does not claim complete compilation coverage.

[examples/native_models.py](examples/native_models.py) includes ResNet-18, a 48.9-million-parameter
transformer with integer tokens and boolean masks, a controlled reduction example, and a historical
indexing regression. These use synthetic inputs and random weights. See [RESULTS.md](RESULTS.md)
for executed checks, including discrepancies and unvalidated configurations.

## Diagnosing a failure

```bash
faultline model diagnose faultline-runs/model --output faultline-runs/diagnosis
```

This rechecks the saved finding and runs the same frozen workload under `eager`, `aot_eager`,
`aot_eager_decomp_partition`, and `inductor`, each in a separate worker. A fresh bundle can also
be supplied, with execution settings such as `--gradients`. Results include every backend outcome
and a possible failure boundary. An unavailable backend, different symptoms, or a failure that
disappears at a later rung prevents simple attribution. The boundary is diagnostic evidence, not
proof that a particular compiler pass is responsible.

## Historical-issue benchmark

```bash
faultline benchmark list
faultline benchmark historical --good-python .current/bin/python --bad-python .historical/bin/python --diagnose --output faultline-runs/history
```

The initial versioned corpus has four independently sized issue patterns across two related
families. Each report records detection, passing controls, reduction size, wall time, interpreter
versions, and standalone verification. Failed reproductions and recurring discrepancies remain
in the denominator. This is a small calibration set, not a general detection-rate claim.

Use `--case 183986 --case 182012 --require-paired` to require the two cases validated with a
2.15.0 passing control and 2.12.0 failing candidate. Without `--require-paired`, exit 0 means the
benchmark completed conclusively; individual issues may be absent or recurring. See
[diagnostics and GPU validation](docs/diagnostics.md) for interpretation and additional commands.

## Compact-graph testing and reduction

Choose a case in `faultline ui`, or replace `CASE_ID` below with an ID from your report:

```bash
faultline replay faultline-runs/compiled/cases/CASE_ID --output faultline-runs/replay
faultline reduce faultline-runs/replay --max-evaluations 64 --output faultline-runs/reduced
faultline export faultline-runs/reduced/reproducer --output faultline-runs/shareable
python faultline-runs/shareable/repro.py
```

Saved findings retain their backend, device, tolerances, thread and repeat counts, gradient and
dynamic-shape settings, and resource limits. Directory inputs reject settings overrides. Replay
writes a new result and records environment changes.

Reduction checks that the saved failure still occurs, searches within the evaluation budget, and
independently confirms the final case. It does not guarantee a globally minimal result. Standalone
scripts require PyTorch but do not require Faultline. When supplied with a saved failure, they
report whether the original failure signature recurs through `reproduces_expected`.

Artifacts include model weights, input values, logs, and environment metadata. Review their contents
before sharing them. Faultline runs locally and has no telemetry.

## Capturing a compact graph

Save a factory in `my_model.py` that returns an evaluation-mode model and example inputs:

```python
import torch
from torch import nn


def factory():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(17, 31), nn.GELU(), nn.Linear(31, 5)).eval()
    return model, (torch.randn(7, 17),)
```

```bash
faultline capture my_model.py:factory --output my-model.json
faultline run --case my-model.json --gradients --backend inductor --output faultline-runs/model
```

The CLI uses `torch.export` by default. Supported operations include Linear, GELU, Conv2d,
AdaptiveAvgPool2d, scaled dot-product attention, last-dimension LayerNorm, and common arithmetic,
activation, and reshape operations. The importer compares its translation's outputs with the
exported model before saving a case.

Cases store input values, layouts, and model parameters in JSON. `--gradients` compares forward
outputs, a deterministic scalar loss, input gradients, and trainable parameter gradients. Missing
gradients are distinguished from zero gradients.

Capture requires `eval()` and flat tensor inputs/outputs. Stochastic training, mutation, optimizer
steps, unsupported operations, and unsupported strides are rejected. Attention accepts causal or
finite floating-point masks; boolean masks, dropout, and grouped-query attention are unsupported.
Graphs must fit the IR's size limits. Capture executes the supplied Python factory, so use trusted
model code. See [examples/models.py](examples/models.py) for MLP, convolution, and attention
factories. The older FX importer is available through `--frontend fx`. The Python `capture()` API
still defaults to FX for compatibility; pass `frontend="export"` to use the newer importer.

## Request sequences

```bash
faultline capture examples/models.py:requests --trace --output requests.json
faultline run --case requests.json --dynamic dynamic --gradients --output faultline-runs/sequence
```

A trace factory returns one model and an ordered list of input tuples. Execution reuses one compiled
callable across the requests, which can expose failures that depend on previous inputs or shapes.
Requests must share graph structure and frozen model state.

The `reduce` command accepts traces as well as individual cases. It tries removing requests before
simplifying the graph and tensors, repeating the full trace to check each proposed reduction.

## Comparing compiler versions

Install the desired PyTorch versions plus NumPy and psutil in two environments, then run:

```bash
faultline compare --baseline-python .baseline/bin/python --candidate-python .candidate/bin/python --case my-model.json --gradients --reduce --output faultline-runs/upgrade
```

On Windows, use each environment's `Scripts/python.exe`. Both interpreters must run on the same
host OS; run the command inside WSL when using Linux environments. Faultline supplies the same
test code to both workers and records their environments separately. Add multiple `--case`
arguments or use `--corpus`. Cases and traces use the same comparison workflow.

A stable failure that occurs only in the candidate environment is reported as a regression.
With `--reduce`, each smaller case must continue to pass in the baseline and reproduce the original
candidate failure. The final case is checked again in both environments, then exported as separate
baseline and candidate reproducers.

The command returns a nonzero exit code for regressions, changed failures, or inconclusive results
such as timeouts and unavailable devices. The HTML report links to both outcomes and any reduced
reproducers. See [comparison classifications and exit codes](docs/cli.md).

## Commands and documentation

| Command | Purpose |
|---|---|
| `ui` | Open the terminal menu |
| `demo` | Run the injected-fault example |
| `doctor` | Inspect runtime and optionally probe compilation |
| `qualify` | Check forward/backward execution across a device's dtype/workload matrix |
| `bisect` | Localize an advisory CPU accuracy shift across isolated nightly builds |
| `benchmark` | Measure historical-issue detection, reduction, and standalone reproduction |
| `model` | Capture, compare, replay, reduce, or export original-model workloads |
| `generate` | Create a seeded, frozen JSON case |
| `capture` | Import a trusted model factory into frozen JSON |
| `trace` | Combine compatible cases into an ordered request sequence |
| `compare` | Compare compiler environments and optionally reduce new failures |
| `run` | Test a corpus, generated cases, or supplied JSON files |
| `resume` | Continue a compatible interrupted campaign |
| `report` | Build a searchable local HTML report; `--open` opens it |
| `replay` | Recheck a saved stable finding with its original settings |
| `reduce` | Minimize a stable finding within an evaluation budget |
| `export` | Write a standalone PyTorch reproducer |

Use `faultline COMMAND --help` for options. Further documentation:

- [CLI options, exit codes, and artifact formats](docs/cli.md)
- [Architecture, comparison policy, and reduction semantics](docs/architecture.md)
- [Validation results and tested configurations](RESULTS.md)

## Development

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m pytest -q -m compiler
python -m ruff check src/faultline tests/faultline scripts/validate_faultline.py scripts/validate_upgrades.py scripts/check_release.py examples
python scripts/validate_faultline.py --compiler --output faultline-runs/validation
python scripts/validate_upgrades.py --compiler --output faultline-runs/upgrades
python scripts/validate_aot.py --compiler --output faultline-runs/aot
python -m build
```

Portable tests run on Windows and Linux. Compiler tests require a working Inductor toolchain.
Validation includes injected faults, actual compiler comparisons, and upgrade checks across two
PyTorch environments.

The compact importer is limited to its supported operators and size guards. The native model path
executes original modules but cannot reduce every graph or preserve arbitrary Python-side state.
Distributed execution, general training loops, and performance-regression detection are unsupported. Numerical
differences require investigation; exceeding a tolerance alone does not establish a compiler bug.
See PyTorch's [numerical accuracy notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
for relevant floating-point behavior.

Faultline's reducer is independent of PyTorch's internal minifier. Exported scripts can request the
upstream debugger with `--native-minifier`; its behavior depends on the PyTorch version and failure.

## License

MIT. See [LICENSE](LICENSE).
