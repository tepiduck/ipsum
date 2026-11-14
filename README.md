# Faultline

Faultline checks a PyTorch model before an upgrade or a compiler-setting change.
It freezes representative inputs, compares eager and compiled outputs and gradients,
and summarizes the evidence in a terminal verdict and an HTML report.

```bash
faultline check model.py:factory --baseline /path/to/old/python --candidate /path/to/new/python
```

It can also check compiled execution against eager in one environment. Numerical
findings are candidates for investigation, not confirmed compiler defects. SAFE is
limited to the inputs, settings, and checks in the report.

## Install and run a first check

Use Python 3.11+ on Linux or WSL for the CPU Inductor quickstart. You need a C++
compiler and Python headers; on Ubuntu, install `g++` and `python3-dev` if missing.
Keep the checkout on the Linux filesystem in WSL for faster compilation and I/O.

These commands build and install a wheel from this repository:

```bash
git clone https://github.com/tepiduck/ipsum.git
cd ipsum
python3 -m venv .venv
. .venv/bin/activate
python -m pip install build
python -m build --wheel
python -m pip install torch==2.15.0 torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install 'dist/faultline_ml-0.19.0-py3-none-any.whl[sweep]'
faultline check --torchvision resnet18 --level quick --output faultline-runs/quickstart
```

The first run downloads pretrained weights and two public example photographs.
It tests three deterministic image batches and writes `report.html`, `verdict.json`,
and detailed worker evidence in the output directory. Quick covers forward execution
only. Download and compilation time depend on the machine; see the
[fresh-install transcript](docs/research/quickstart-test.md).

Use a new output directory for a new check. Use `--resume` to continue an interrupted
check with exactly the same source, frozen inputs, settings, and environments.
Resume covers execution after capture finishes; interrupted preparation needs a new
output directory.
Factories and captured bundles contain executable Python: run code you trust.

## Read the verdict

| Verdict | Exit code | Meaning and next action |
|---|---:|---|
| SAFE | 0 | No measured regression attributable to this change; required executions and accuracy decision are complete. Retain the report. |
| REGRESSION | 1 | Candidate-only stable failure or replicated accuracy degradation. Review the evidence before upgrading. |
| NEEDS_REVIEW | 2 | Change-specific evidence is undecided: an unstable candidate-only difference, or incomplete accuracy replication trending worse. |
| INCONCLUSIVE | 3 | Required execution or accuracy evidence is unavailable. Resolve setup, device, timeout, memory, or coverage problems. |

This table describes the default two-environment [v2 policy](docs/verdict-policy-v2.md).
Differences present in both environments remain visible as **pre-existing compile/eager
differences**, with counts and evidence links. They do not alone block SAFE, including
when the failure signature changes; replicated accuracy degradation still does.
SAFE assesses the change, not whether either environment meets strict eager fidelity.
Use `--verdict-policy v1` for the [original policy](docs/verdict-policy.md).
Single-environment checks retain v1: unexplained or contrary differences need review,
and cannot establish an upgrade regression. The [JSON contract](docs/verdict-schema.md) is versioned.

## Check your model

Create a factory that loads the intended checkpoint, returns an eval model, and
supplies a list of representative input batches:

```python
import torch

def factory():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.GELU()).eval()
    # Replace this demonstration model and random data with your checkpoint and inputs.
    batches = [(torch.randn(2, 32),) for _ in range(10)]
    return model, batches
```

```bash
faultline check model.py:factory --output faultline-runs/model
faultline check --hf openai-community/gpt2 --output faultline-runs/gpt2
faultline check captured/bundle --level quick --output faultline-runs/replay
```

The default standard level checks forward execution and fixed-probe gradients on
ten batches. With two environments it also compares compiled error against fp64.
The replication unit is an input batch, not each gradient tensor. One batch cannot
establish a replicated accuracy shift. Seed-based factories and nested inputs are
covered in the [user guide](docs/user-guide.md).

| Level | Default batches | Checks |
|---|---:|---|
| quick | 3 | Forward execution, no AOT/mask diagnostics |
| standard | 10 | Forward, gradients, paired accuracy; explanations run as needed |
| thorough | 10 | Standard coverage plus full control and AOT/mask diagnostics |

The default `--jobs auto` measures the first batch at the full memory budget, then
selects up to four workers with 25% memory headroom and retries memory-limited batches alone.
The budget is 75% of available RAM, capped at 32 GiB, and the chosen budget and worker
count are printed and saved; `--jobs N` and `--max-rss-mb` allow explicit settings.
Each worker has a 600-second timeout. Accuracy diagnostics run in batch order and stop once the replicated
rule's outcome is mathematically fixed; all planned primary checks still run. A positive
accuracy flag always needs at least ten completed batches. `--no-early-stop` requests
every accuracy measurement. Missing required evidence can prevent SAFE; skipped optional
diagnostics do not erase primary results. [0.16 costs](docs/research/usability-016.md)
remain available under their original execution policy.
The [0.19 efficiency measurements](docs/research/efficiency-019.md) compare current CPU
costs and memory use with 0.18 on four frozen pretrained workloads.

## Compare environments or settings

```bash
faultline env create --torch 2.15.0 --device cpu --output environments/baseline
faultline env create --torch nightly --device cpu --output environments/candidate
# Use the interpreter paths printed by these commands:
faultline check model.py:factory --baseline /path/printed/for/baseline/python --candidate /path/printed/for/candidate/python --output faultline-runs/upgrade
```

The installer records resolved versions and installs torchvision/transformers
dependencies. Add `--requirement package==version` for your model. `--minimal` omits
the catalog dependencies. GPU environments require an explicit `--index-url` for
the compatible CUDA/ROCm runtime; use the [PyTorch installer](https://pytorch.org/get-started/locally/).

To compare settings in one interpreter, pass that interpreter as both baseline and
candidate, then select `--baseline-compile-mode` and `--compile-mode`. For AMD/ROCm
execution use `--device cuda:0`; Faultline does not fall back to CPU. NVIDIA execution
has not been validated in this repository.

## CI and reproducer sharing

[examples/ci/upgrade-check.yml](examples/ci/upgrade-check.yml) creates two environments,
runs a small model, uploads reports, and fails the check gate only for REGRESSION.
NEEDS_REVIEW and INCONCLUSIVE remain visible in the artifact and job summary.
Use your frozen checkpoint and real batches for a deployment decision.

```bash
faultline check model.py:factory --baseline /old/python --candidate /new/python --output check-report
# The terminal prints the reproducer path when a failing workload was exported.
python check-report/reproducer/repro.py
```

Reproducers include exact model weights, inputs, and trusted factory code. Inspect
them before sharing. Large checkpoints and private inputs should stay out of Git.
The existing `model reduce`, `model export`, `sweep`, `bisect`, `benchmark`, and `ui`
commands remain available; check exit codes apply only to the new `check` command.

## Documentation

- [User guide](docs/user-guide.md): factories, levels, reports, and sharing.
- [Troubleshooting](docs/troubleshooting.md): setup failures and incomplete checks.
- [CLI reference](docs/cli.md), [architecture](docs/architecture.md), and [numerics](docs/numerics.md).
- [Sweeps](docs/sweeps.md) and [replicated nightly bisection](docs/bisect.md).
- [Research and validation record](docs/research/README.md), including the historical evidence in RESULTS.md.

## License

Faultline is licensed under the [MIT License](LICENSE).
