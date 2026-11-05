# Faultline

**Find a tensor compiler discrepancy. Reduce it. Share a runnable reproducer.**

Faultline is a local command-line tool for testing PyTorch compilers. It generates valid tensor
programs or captures supported models, compares eager and compiled execution, and shrinks stable
failures while preserving their original inputs, settings, and failure signature.

- **Guided terminal interface** for running tests and investigating saved findings.
- **Searchable browser reports** with comparison details and links to raw evidence.
- **Isolated workers** with wall-time and host-memory limits.
- **Failure-preserving reduction** over graphs, dimensions, layouts, and input values.
- **Standalone reproducers** that need only PyTorch, with the same repeated checks as the runner.
- **Model import through `torch.export`**, including GELU, convolution, pooling, and attention.
- **Forward and gradient comparisons**, with named input and parameter diagnostics.
- **Request-sequence reduction** for failures that depend on earlier shapes or requests.
- **Compiler upgrade gates** across two Python environments, with automatic regression reduction.

A numerical discrepancy is a finding to investigate, **not an automatically confirmed compiler bug**.

## Install

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

Or clone this repository and run `python -m pip install .`. PyPI publication is not required;
the repository installation provides the `faultline` command. `python -m faultline` works too.

## Try it

```bash
faultline ui
```

The terminal menu can run the demo, test the corpus or a JSON case, browse campaigns, and replay,
reduce, or export a finding. Results are saved under `faultline-runs/`. Choose another location with
`faultline ui --workspace PATH`.

For a complete, scriptable demonstration:

```bash
faultline demo --output faultline-runs/demo
faultline report faultline-runs/demo/injected --open
python faultline-runs/demo/reduced/reproducer/repro.py
```

The demo deliberately replaces a sum with a mean. It verifies an eager control, detects the
injected fault, reduces the case, and writes a standalone reproducer. **It is a harness demonstration,
not a discovered PyTorch bug.** The demo returns `0` on success; its reproducer intentionally returns
`1` when it observes the injected discrepancy. Use a new output directory for each run.

## Test a compiler

```bash
faultline doctor
faultline doctor --backend inductor --probe faultline-runs/probe
faultline run --corpus --count 12 --seed 0 --backend inductor --output faultline-runs/compiled
faultline report faultline-runs/compiled --open
```

Actual CPU TorchInductor code generation needs a working C++ toolchain and Python development
headers. Ubuntu/WSL is a validated compilation path. `--backend eager` checks the execution pipeline;
`--backend aot_eager` exercises AOT tracing without Inductor code generation. Neither replaces an
Inductor compilation test. Use `doctor --probe` to test the selected backend in your environment.

For an AMD GPU, install a compatible PyTorch/ROCm build and select `--device cuda:0`; ROCm uses
PyTorch's `cuda` device namespace. Missing devices are reported as unavailable. CPU execution is
validated; GPU execution is not yet part of the recorded hardware coverage. Host RSS limits do
not bound GPU memory.

## Investigate a finding

Choose a case in `faultline ui`, or replace `CASE_ID` below with an ID from your report:

```bash
faultline replay faultline-runs/compiled/cases/CASE_ID --output faultline-runs/replay
faultline reduce faultline-runs/replay --max-evaluations 64 --output faultline-runs/reduced
faultline export faultline-runs/reduced/reproducer --output faultline-runs/shareable
python faultline-runs/shareable/repro.py
```

Directory inputs restore the original backend, device, tolerances, thread count, repeat count,
and resource limits. Overrides are rejected. Replay writes new evidence and records environment
changes; it never overwrites the source finding. Reduction verifies the saved failure before
searching and independently confirms the final result. It finds a local fixed point under its
implemented transformations, not a guaranteed global minimum.

Standalone scripts report `reproduces_expected` when an original stable finding is supplied.
Another failure does not count as reproducing the original. Exported model weights, input values,
logs, and environment metadata can contain information from your workload: review your own artifacts
before sharing them. Faultline runs locally and has no telemetry or hosted service.

## Test your model

```python
import torch
from torch import nn

# Save as my_model.py. The factory runs once during capture.
def factory():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(17, 31), nn.GELU(), nn.Linear(31, 5)).eval()
    return model, (torch.randn(7, 17),)
```

```bash
faultline capture my_model.py:factory --output my-model.json
faultline run --case my-model.json --gradients --backend inductor --output faultline-runs/model
```

The CLI defaults to a `torch.export` importer with an explicit operator registry. It supports Linear,
GELU, Conv2d, AdaptiveAvgPool2d, scaled dot-product attention, last-dimension LayerNorm, and common
arithmetic/activation/reshape operations. Parameters retain their identities and trainability.
The importer checks its lowered outputs against the exported model before accepting a case.
`--gradients` compares forward outputs, a deterministic scalar loss, input gradients, and trainable
parameter gradients. It detects missing gradients separately from zero gradients.

Capture requires `eval()` and flat tensor inputs/outputs. Stochastic training, mutation, optimizer
steps, unsupported operations, and unsupported strides are rejected. Attention accepts causal or
finite floating-point masks; boolean masks, dropout, and grouped-query attention are unsupported.
Models must fit the bounded IR; this is a debugger for small models and isolated model components.
Capture executes trusted local Python; saved cases are data-only JSON. See the
[ready-to-run model factories](examples/models.py). The older FX importer remains available with
`--frontend fx`; the Python `capture()` API retains FX as its default for compatibility.

## Debug a request sequence

```bash
faultline capture examples/models.py:requests --trace --output requests.json
faultline run --case requests.json --dynamic dynamic --gradients --output faultline-runs/sequence
```

A trace factory returns one model and an ordered list of input tuples. Each trace execution reuses
one compiled callable across requests, allowing shape-history failures to appear. All requests must
share graph structure and frozen model state. Reduce a failing trace with the usual `reduce`
command: it removes requests before simplifying graphs and tensors. A whole trace is repeated and
rechecked for every proposal. The controlled sequence experiment reduces seven requests to
`A → B → A`, even though every request passes alone.

## Gate a compiler upgrade

Install the desired PyTorch versions plus NumPy and psutil in two environments, then run:

```bash
faultline compare --baseline-python .baseline/bin/python --candidate-python .candidate/bin/python --case my-model.json --gradients --reduce --output faultline-runs/upgrade
```

On Windows, use each environment's `Scripts/python.exe`. Both interpreters must run on the same
host OS; run the command inside WSL when using Linux environments. Faultline supplies the same
harness source to both workers and records each environment independently. Add multiple `--case`
arguments or use `--corpus`. Cases and traces use the same comparison workflow.

A new stable candidate failure fails the gate. Automatic reduction accepts a smaller case only
when **the baseline still passes and the candidate still exhibits the original failure**. It
exports separate baseline and candidate reproducers after fresh confirmation. Timeouts, unstable
results, unavailable devices, and broken environments are inconclusive and cannot pass the gate.
The HTML report links to both outcomes and the reduced reproducers. See [exit codes](docs/cli.md).

## Command reference

| Command | Purpose |
|---|---|
| `ui` | Guided terminal interface |
| `demo` | Complete controlled-fault example |
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

Run `faultline COMMAND --help` for options. See [CLI and artifact semantics](docs/cli.md),
[architecture](docs/architecture.md), and [measured validation](RESULTS.md).

## Development

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m pytest -q -m compiler
python -m ruff check src/faultline tests/faultline scripts/validate_faultline.py scripts/check_release.py examples
python scripts/validate_faultline.py --compiler --output faultline-runs/validation
python scripts/validate_upgrades.py --compiler --output faultline-runs/upgrades
python -m build
```

The portable tests run on Windows and Linux. The `compiler` test needs a working Inductor toolchain.
The experiment harness separates controlled mutations from real compiler comparisons. Forward and
backward checks use stateless evaluation graphs and a restricted operator set. Traces supply concrete
requests to static, automatic, or dynamic compilation. Distributed execution, arbitrary training
loops, arbitrary model coverage, and performance-regression detection are outside the current scope.

Faultline uses public PyTorch FX and `torch.compile` APIs. Its IR reducer is independent of
PyTorch's internal minifier. An exported script can optionally request the upstream debugger with
`--native-minifier`. [PyTorch compiler documentation](https://docs.pytorch.org/docs/stable/torch.compiler.html)
and [numerical accuracy notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
provide background on compiler behavior and floating-point differences.

MIT licensed. Source and issue tracking are hosted in this repository.
