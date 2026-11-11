# Fresh-user installation test

The README quickstart was run from its public GitHub clone at commit
[`4021008`](https://github.com/tepiduck/ipsum/commit/4021008dcdc32f5bf30fda14af5a736cbabe18ca),
in a new directory and clean venv. Every command below exited successfully. Faultline
was installed from the built wheel; an import-location check confirms `site-packages`,
not an editable source install. No `PYTHONPATH` override was inherited.

Host: Ubuntu under WSL2, AMD Ryzen 7 7700X, Python 3.12, existing `g++` and Python headers.
The venv and Faultline output/cache directory were new. The machine's pip and pretrained
weight caches were already populated; these timings do not measure a cold network.
Other validation work was active on the host.

| Command | Wall seconds | Exit |
|---|---:|---:|
| `git clone https://github.com/tepiduck/ipsum.git` | 2.25 | 0 |
| `cd ipsum` | 0.002 | 0 |
| `python3 -m venv .venv` | 2.04 | 0 |
| `. .venv/bin/activate` | 0.002 | 0 |
| `python -m pip install build` | 0.64 | 0 |
| `python -m build --wheel` | 1.42 | 0 |
| `python -m pip install torch==2.15.0 torchvision --index-url https://download.pytorch.org/whl/cpu` | 22.69 | 0 |
| `python -m pip install 'dist/faultline_ml-0.16.0-py3-none-any.whl[sweep]'` | 12.17 | 0 |
| `faultline check --torchvision resnet18 --level quick --output faultline-runs/quickstart` | 73.62 | 0 |

Total measured commands: 114.84 seconds. ResNet-18 reports SAFE for three
forward batches. The terminal output, with the temporary checkout path shortened, is:

```text
SAFE - Faultline check
Checked: single_environment, quick; 3/3 batches.
No blocking evidence in the requested checks.
Next: Keep this report with the tested model and environment versions.
HTML: faultline-runs/quickstart/report.html
JSON: faultline-runs/quickstart/verdict.json
Scope: tested inputs and settings only; not a general correctness guarantee.
```

## Friction found

A prepublication cross-environment probe exposed dependency shadowing when Faultline was
installed as a wheel. The fix explicitly loads Faultline without adding the coordinator's
entire `site-packages` to the worker path. The rebuilt wheel's three-batch paired check
completed in 64.24 seconds with SAFE, and worker records confirm distinct 2.15.0+cpu and
2.16.0.dev20261105+cpu runtimes. A regression test places conflicting dependencies next
to the harness and verifies that only the candidate dependency is imported.

The installed-wheel standard follow-up completed ten paired input batches in 224.80
seconds with SAFE and 652.6 MiB peak worker-tree RSS. All ten baseline and ten candidate
primary records retain the expected distinct runtime versions; both quick and standard
verdicts validate against the published schema. This small randomly initialized example
tests the workflow and does not substitute for the pretrained catalog calibration.

The README now names the Linux/WSL toolchain prerequisites, uses an explicit CPU PyTorch
install before the wheel's model dependencies, and explains output directories and scoped
verdicts. Absolute machine paths, raw installation logs, weights, and environments stay local.
