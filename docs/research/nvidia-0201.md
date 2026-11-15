# NVIDIA/CUDA validation — 0.20.1

This record covers one Google Colab Tesla T4 runtime on November 15, 2026. Runs use
Faultline main commit `3aabc0bcbc97c4fd98d73436d75321985aa63acf` (0.20.0).
The 0.20.1 change exposes a compiler-skip warning in qualification reports; it does
not change numerical tolerances, primary statuses, verdict policy, signatures or exit codes.
No compiler-defect claim follows from these measurements.

[JSON summaries](nvidia-0201.json) and the [HTML summary](nvidia-0201.html) distinguish
eight runs with original qualification artifacts from ten runs recovered from notebook
output only. Both retain commands and measured costs. Only the first eight retain original
summaries and worker-log hashes. Compiler-skip annotations are post hoc readings of
preserved logs. The original runtime files were absent after reconnection, before later archives were downloaded;
the recovery and missing evidence are described below.

## Runtime and installation

| Item | Observed value |
|---|---|
| GPU | Tesla T4, compute capability 7.5, 15,360 MiB reported by `nvidia-smi` |
| Driver | 580.82.07; `nvidia-smi` reports CUDA 13.0 |
| Host | Linux 6.6.159+, x86-64, two logical CPUs |
| Host RAM | 13,605,818,368 bytes (12.67 GiB) |
| Initial `/content` disk | 120,942,624,768 bytes total; 120,839,045,120 bytes free |
| Stock notebook | Python 3.13.16, PyTorch 2.14.1+cu130 |
| Validation workers | Python 3.13.18 after installing the venv prerequisite |
| Stable | PyTorch 2.15.0+cu132, Triton 3.9.0, torchvision 0.29.2+cu132 |
| Nightly | PyTorch 2.16.0.dev20261114+cu132, Triton 3.9.0+gitb7ddcf86, torchvision 0.30.0.dev20261114+cu132 |
| Shared model dependencies | Transformers 5.23.0, tokenizers 0.23.3, safetensors 0.8.0 |

The initial venv creation failed because Colab lacked working `ensurepip` support.
Installing `python3.13-venv` repaired setup; the failed attempt remains in the logs.
The requested 2.15.0 wheel was absent from the official `cu130` and `cu128` indexes.
The official combined wheel listing identified the `cu132` build. The `cu130` nightly
index stopped at November 2; that separately installed environment was not used for
the stable/nightly validation pair.

Both selected environments were created with `faultline env create`, using explicit
`https://download.pytorch.org/whl/cu132` and
`https://download.pytorch.org/whl/nightly/cu132` indexes. NVIDIA documents
[CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/why-cuda-compatibility.html)
within a driver family, with limitations. Here compatibility was checked by execution:
both `faultline doctor --backend inductor --device cuda:0 --probe ...` runs passed,
taking 17.07 and 15.19 seconds respectively. This is not a claim about every CUDA 13.2 feature.

Full `nvidia-smi`, `torch.__config__.show()`, package versions and build commits are
preserved in the JSON evidence. No CPU execution substituted for a GPU run.

## Qualification

All commands explicitly selected `cuda:0` and required CUDA. The dtype matrix used
the default diagnostic AOT capture; the autocast, sequence and `reduce-overhead`
variants used `--no-aot`, as in the diagnostics guide.

| Run | Primary outcome | Seconds | Host RSS MiB | GPU used MiB |
|---|---|---:|---:|---:|
| Stable dtype matrix | 6 pass, 3 `aot_capture_missing`; inconclusive | 132.54 | 1849.7 | 163 |
| Nightly dtype matrix | 6 pass, 3 `aot_capture_missing`; inconclusive | 129.86 | 1971.8 | 163 |
| Stable bfloat16 autocast | 3 pass; all 3 report compilation skips | 40.76 | 1690.2 | 161 |
| Nightly bfloat16 autocast | 3 pass; all 3 report compilation skips | 41.85 | 1716.2 | 161 |
| Stable buffer/shape sequence | 2 pass | 32.26 | 1660.8 | 161 |
| Nightly buffer/shape sequence | 2 pass | 35.44 | 1686.9 | 161 |
| Stable reduce-overhead | 3 pass | 51.69 | 1735.2 | 183 |
| Nightly reduce-overhead | 3 pass | 51.26 | 1746.9 | 185 |

The twelve float32/float16 dtype cases passed. For bfloat16, Inductor emitted
`Tesla T4 does not support bfloat16 compilation natively, skipping` in both builds.
The dtype runs retained their `aot_capture_missing` outcomes. The bfloat16 autocast
comparisons passed, but those passes are **not compiled-bfloat16 validation**.

The original `TORCH_LOGS=perf_hints` runs did not provide affirmative CUDA-graph
capture evidence. Their passing outcomes alone do not establish graph use.
Additional graph-trace and float16 autocast controls are recorded below.

Those four controls completed. Their measured outcomes survive in the downloaded notebook,
but their original worker logs do not:

| Run | Primary outcome | Seconds | Host RSS MiB | GPU used MiB |
|---|---|---:|---:|---:|
| Stable expanded graph trace | 3 pass | 45.42 | 1633.1 | 183 |
| Nightly expanded graph trace | 3 pass | 44.70 | 1651.9 | 185 |
| Stable float16 autocast | 2 gradient mismatches, 1 pass | 54.92 | 1830.7 | 163 |
| Nightly float16 autocast | 2 gradient mismatches, 1 pass | 55.54 | 1855.4 | 163 |

The expanded trace used `TORCH_LOGS=+inductor,perf_hints,cudagraphs`. Its worker stderr
was not printed into the notebook and was lost with the runtime. Consequently **whether
CUDA graphs were captured or skipped remains unverified**, including for the passing
`reduce-overhead` cases. Float16 autocast reported `gradient_mismatch` for MLP and
convolution in both environments; attention passed. These are numerical findings under
the existing tolerances, not established compiler defects or measured upgrade regressions.
Their per-target error metrics were not recovered.

## Pretrained models and controls

Checks used `--device cuda:0 --level standard`, stable 2.15.0 versus the November 14
nightly, default auto scheduling and unchanged verdict policy v2. Every command selected
one job on the two-CPU runtime. Frozen ResNet inputs were reused for its deterministic
retry; it was not recaptured. The following rows are recovered from the notebook's
command records and printed verdict summaries, not from original `verdict.json` files.

| Run | Verdict / result | Conclusive primary pairs | Seconds | Host RSS MiB | GPU used MiB |
|---|---|---:|---:|---:|---:|
| ResNet-50 default | INCONCLUSIVE: unstable eager references in both environments | 0/10 | 460.64 | 2470.9 | 511 |
| ResNet-50 deterministic | SAFE; 10 batches have pre-existing differences | 10/10 | 925.32 | 2951.9 | 917 |
| BERT default | INCONCLUSIVE: unstable compiled executions in both environments | 7/10 | 3428.87 | 6147.8 | 2029 |
| GPT-2 default | SAFE; 10 batches have pre-existing differences | 10/10 | 2910.34 | 6425.8 | 2245 |
| Injected gradient-scale control | REGRESSION; passing baseline and stable injected candidate failure | 1/1 | 78.19 | 1686.2 | 161 |
| Historical CUDA fixtures | 2 paired reproductions; 1 not observed | 3 fixtures | 119.10 | 1629.9 | 123 |

SAFE addresses the change between environments for these inputs and settings; it does
not claim elementwise compile/eager agreement. BERT retained seven conclusive primary
pairs and seven batches with pre-existing differences, with required accuracy evidence
incomplete. Its status was `unstable_candidate`, not `unstable_reference`; the queued
conditional retry therefore did not run a deterministic BERT check. That control remains
outstanding. No failed batch was removed from the denominator or rerun on CPU.

The model commands took 7.68 minutes for default ResNet-50, 15.42 minutes for its
deterministic rerun, 57.15 minutes for BERT and 48.51 minutes for GPT-2. These include
preparation, compilation, repeat checks, accuracy diagnostics and reporting; they are
not model throughput measurements. Total measured command time across all 18 runs was
8,638.71 seconds (2.40 hours), excluding environment setup and artifact handling.

The injected `mutant:gradient-scale` run is deliberately a **controlled fault**, not an
upstream finding. Its factory supplied one batch, so its stable candidate-only failure
exercised the primary regression rule; it did not establish a ten-batch accuracy shift.

The historical environment installation requested PyTorch 2.12.0 from the official
`cu130` index and completed in 172.38 seconds. The resolved build manifest was not
recovered. The benchmark used that interpreter as the deliberately old candidate and
2.15.0 as the passing comparison environment, with `--no-reduce`:

| Existing upstream fixture | CUDA result |
|---|---|
| 183986 | `paired_reproduction` |
| 182012 | `paired_reproduction` |
| 180164 | `not_observed` |

The third fixture was run, not skipped. These reproduce or fail to reproduce existing
reports; they establish no new upstream discovery. Exact historical per-target errors
and the exported reproducers were lost with the runtime.

## Measurements and preservation

Wall time includes command startup, model preparation where applicable, compilation,
repeated execution, diagnostics and reporting. Host memory is the sampled sum of RSS
for the command and descendants; shared pages can be counted repeatedly. GPU memory
is sampled device-wide `nvidia-smi` usage, including contexts, rather than a PyTorch
allocator peak. Sampling is every 0.5 seconds plus query time, so short peaks can be missed.
Probe/setup commands have wall times but no memory sampler.
The distinction is visible in deterministic ResNet-50: the check's own first-batch probe
reported 3,721 MiB, while the external command sampler observed at most 2,951.9 MiB.
Both measurements are retained; the external samples must not be treated as exact peaks.

The original `qualification-checkpoint.zip` was downloaded before the interruption and
is preserved locally with its extracted JSON/HTML summaries and logs. After browser
control failed, Colab completed the remaining queued commands. The notebook recorded
creation of the ResNet, BERT, GPT-2, final-results and coverage-controls archives. On
reconnection the runtime had only `sample_data` under `/content` and reported 0.08 GB
used disk; the old archives and run directories were no longer present. This is consistent
with a replacement runtime, but the cause of the storage loss was not established.

The notebook was downloaded with all 11 cells and their outputs intact. Its SHA-256 and
reported archive sizes are recorded in the portable JSON. Eight notebook measurement
records were checked for exact agreement with the downloaded qualification checkpoint.
Ten additional records were extracted with their printed outcome excerpts. These are
explicitly marked `notebook_stdout_only`; missing original summaries are `null`, and
parsed summaries have a separate `recovered_summary` field. The HTML table identifies
the evidence source for every run. No missing JSON, metrics or hashes were invented.

Both the original checkpoint and recovered notebook are retained in the repository's
ignored `faultline-runs/nvidia-0201` directory. Public evidence contains only summaries
and relevant printed excerpts. Weights, tensors, environments and personal notebook
metadata are not published. Full resume requires the original frozen inputs and compatible
source/settings/environments; it is no longer possible for the lost runtime. The locally
retained summary checkpoint alone cannot reconstruct those frozen inputs.
Saving a ZIP under `/content` does not make it durable: download each completed archive
before leaving a runtime, or use explicitly configured persistent storage.

## Reporting fix

Qualification now surfaces the observed bfloat16 compilation-skip warning in terminal
output, JSON and HTML as an advisory. A numerical PASS and its existing exit code remain
unchanged. `no_skip_reported` means only that the known warning was absent; it does not
certify compiler coverage. Tests cover a passing comparison with a repeated skip warning,
warning deduplication, visible reporting, missing/unrelated logs, and unreadable logs.
An advisory log-read error returns unavailable coverage rather than replacing the primary
result with a capture failure. The reporting change is tested locally; the original GPU
commands ran main 0.20.0, not the later reporting patch.

## Remaining coverage

Actual CUDA-graph capture, deterministic BERT behavior, explanations for float16 autocast
findings and historical fixture 180164, and the lost per-target evidence remain unresolved.
Other NVIDIA architectures, native bfloat16 compilation, multi-GPU execution and arbitrary
user models are outside this T4 validation. There was no upstream filing and no relaxation
of verdict logic, tolerances or thresholds.
