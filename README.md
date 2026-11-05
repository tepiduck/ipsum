# Faultline

Faultline is a command-line tool for comparing eager and compiled PyTorch execution. It checks
outputs and gradients, reduces reproducible failures, and exports standalone Python reproducers.
It also supports request sequences and comparisons between two compiler environments.

It is intended for debugging small models or model components and testing PyTorch upgrades against
saved workloads. Each run produces JSON results, worker logs, and a local HTML report. Workers run
in separate processes with configurable time and host-memory limits.

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
| `eager` | Eager execution for checking the test pipeline |

CPU Inductor requires a C++ toolchain and Python development headers. It has been tested on
Ubuntu/WSL. `doctor --probe` checks whether the selected backend works in your environment.

For AMD GPUs, use a compatible PyTorch/ROCm installation and `--device cuda:0`; ROCm uses PyTorch's
`cuda` namespace. GPU execution has not yet been validated on hardware. Unavailable devices are
reported explicitly. The host-memory limit does not cover GPU memory.

## Reproducing and reducing failures

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

## Capturing a model

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
python -m build
```

Portable tests run on Windows and Linux. Compiler tests require a working Inductor toolchain.
Validation includes injected faults, actual compiler comparisons, and upgrade checks across two
PyTorch environments.

Current coverage is limited to supported operators in stateless evaluation graphs. Distributed
execution, general training loops, and performance-regression detection are unsupported. Numerical
differences require investigation; exceeding a tolerance alone does not establish a compiler bug.
See PyTorch's [numerical accuracy notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
for relevant floating-point behavior.

Faultline's reducer is independent of PyTorch's internal minifier. Exported scripts can request the
upstream debugger with `--native-minifier`; its behavior depends on the PyTorch version and failure.

## License

MIT. See [LICENSE](LICENSE).
