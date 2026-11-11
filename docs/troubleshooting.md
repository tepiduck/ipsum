# Troubleshooting

| Faultline says | Next step |
|---|---|
| Missing compiler / `g++` / `cl.exe` | Install a supported C++ toolchain and Python headers. The tested CPU path is Linux/WSL. Run `faultline doctor --backend inductor --probe NEW_DIRECTORY`. |
| Missing interpreter or import | Check the printed Python path and install PyTorch plus your factory dependencies in that environment. `faultline env create` creates isolated environments and records resolved versions. |
| Factory preparation failed | Read `preparation-supervision/stderr.log`. Return an eval model and inputs; include local checkpoint/modules with `--include`. Check paths relative to `__file__`. |
| Input batches share different weights | Keep initialization/checkpoint fixed. A factory seed must change input selection, not model parameters. |
| Repeated sample / leading dimension error | Supply distinct samples in batch-major tensors. A scalar metadata field is allowed; independent tensor sample dimensions must agree. |
| `replication_unavailable` | Supply ten or more distinct batches and resolve missing fp64 observations. Repeating the same batch does not count. |
| `diagnostic_skipped_memory` | The primary evidence is retained. Use smaller real input shapes or a larger explicit budget if available; missing required evidence can still mean NEEDS_REVIEW. |
| Primary `resource_limit` | Execution exceeded its budget. Reduce workload size or raise `--max-rss-mb` only when the machine has room; retain the incomplete run. |
| Primary timeout | Inspect worker logs for compilation progress. Use a new output directory with a larger `--timeout` if compilation is the bottleneck. |
| Missing GPU / unsupported backend | Install the correct runtime and verify hardware support; Faultline never substitutes CPU/eager. Use `faultline qualify` for a focused device check. |
| Environment changed on resume | Use a new output directory. Resume requires the original source, inputs, settings, and environment identity. |
| HTTPS/download failure | Check network access and model availability. Built-ins download weights and public inputs during preparation; captured bundles replay offline. |
| NEEDS_REVIEW on both versions | Recurring numerical discrepancies are not upgrade regressions by themselves. Inspect explanation coverage and contrary evidence. |

An environment may install successfully without a usable GPU: creation verifies the wheel's runtime
type; execution checks device availability. CUDA/ROCm indexes must be chosen explicitly using the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/).

The new exit codes 0/1/2/3 belong to `faultline check` only. Older `model`, `sweep`, `run`, and
reproducer commands retain their existing codes. Use the documented verdict JSON for new CI integrations.
