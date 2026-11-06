# Diagnostics and validation

## Backend ladder

```bash
faultline model diagnose workloads/my-model --gradients --output faultline-runs/diagnosis
faultline model diagnose faultline-runs/saved-failure --python .candidate/bin/python --output faultline-runs/saved-diagnosis
```

A saved run supplies its original settings and request selection; overrides are rejected. A fresh
bundle accepts native execution options. Diagnosis requires the original model, before individual
graph isolation. The chosen interpreter must contain the model's dependencies.

Each rung compares its compiled execution with ordinary eager execution:

| Backend | Additional compiler machinery |
|---|---|
| `eager` | Dynamo graph capture and execution |
| `aot_eager` | AOTAutograd tracing and functionalization |
| `aot_eager_decomp_partition` | Operator decompositions and partitioning |
| `inductor` | Inductor lowering and code generation |

This follows PyTorch's documented [backend ablation workflow](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_troubleshooting.html#ablation).
A boundary is reported only when all earlier rungs pass and all subsequent rungs show the same
observed failure family. Backend names are normalized for this comparison; tensor identities and
failure categories are retained. Different failures, missing backends, and non-monotonic patterns
are reported separately. A matching symptom is not proof of a shared root cause.

For saved evidence, a fresh run must first reproduce the original finding. The ladder then uses
normal backend pipelines without diagnostic AOT capture. If that change loses the original symptom,
the report withholds attribution. Inductor mode options apply only to the Inductor rung; other
backends do not implement those modes. Per-rung policies and all outcomes remain in the report.

Results have `schema_version: 1`, `kind: backend_ladder`, a `cases` array containing per-backend
settings/results/report links, and a `boundary` field. Exit 0 means agreement, 1 means stable
discrepancies were observed, and 2 means attribution was inconclusive or the source finding did not
recur. No pass/flag bisection or automatic nightly installation is performed.

## Historical corpus

```bash
faultline benchmark list
faultline benchmark historical --good-python .current/bin/python --bad-python .historical/bin/python --diagnose --output faultline-runs/history
```

The corpus is bundled Python code and requires no network downloads. Its first version includes:

| Issue | Fixture | Originally reported device |
|---|---|---|
| [183986](https://github.com/pytorch/pytorch/issues/183986) | Index addition over expanded storage with repeated indices | CUDA |
| [182012](https://github.com/pytorch/pytorch/issues/182012) | Diagonal-scatter backward | CUDA |
| [180164](https://github.com/pytorch/pytorch/issues/180164) | Slice-scatter backward | CUDA |
| [185337](https://github.com/pytorch/pytorch/issues/185337) | Explicit float32 → float16 → float32 rounding | CPU |

The fixtures use independent shapes and values. CPU execution of a CUDA-reported operation pattern
is an adaptation, not a replication on the originally reported hardware. Three cases share an
expanded-storage family, so four issue IDs do not represent four independent failure mechanisms.
Upstream closed status is recorded with a date; it does not establish that a particular release
fixed the issue.

`candidate_detections` counts the expected failure category at the documented affected tensor.
`paired_reproductions` additionally requires the control environment to pass. The detection
fraction uses all selected fixtures as its denominator, and the summary retains inconclusive and
not-observed cases. These numbers measure this curated fixture set in the supplied environments.
They are not false-negative estimates for arbitrary compiler bugs.

Only paired reproductions are reduced automatically. Every accepted proposal must preserve the
candidate failure signature and keep the baseline passing. Final confirmation runs in fresh workers,
and both exported scripts are checked without Faultline on `PYTHONPATH`, under time/RSS limits.
Reduction size uses the independent graph's `(requests, call nodes, input elements)` ordering;
it is not a claim about source-code lines or the size of the original upstream program. Wall times
include subprocess startup, compilation, comparison, and export checks. They are tool cost, not
model performance measurements.

The summary has `schema_version: 1`, `kind: historical_benchmark`, and `corpus_version: 1`. It
includes the catalog, selected/family counts, environment provenance, per-case classifications,
reduction metrics, standalone verification, and optional backend diagnoses. Full local evidence
includes exact values and machine metadata.

Use repeated `--case ID` to select fixtures, `--no-reduce` for detection alone, and
`--max-evaluations N` to bound reduction. Exit 0 means the benchmark completed conclusively; it
does not mean every issue reproduced. `--require-paired` returns 1 if any selected issue lacks an
expected candidate failure with a passing control. Inconclusive execution or failed verification
returns 2. The default CPU CI calibration pins 2.12.0/2.15.0 and requires the two independently
verified indexing and diagonal-scatter cases; the full corpus retains the other outcomes.

## GPU and mixed-precision checks

Install a compatible GPU driver and PyTorch runtime in the environment that will execute Faultline.
No driver or runtime is installed by these commands:

```bash
# NVIDIA; use --require-rocm instead for AMD
faultline qualify --require-cuda --output faultline-runs/cuda

# Float32 model storage with mixed-precision execution
faultline qualify --require-cuda --dtype float32 --autocast bfloat16 --no-aot --output faultline-runs/cuda-amp

# Persistent buffers and request shapes 2, 5, 2, 7
faultline qualify --require-cuda --dtype float32 --workload buffer_update --workload shape_sequence --dynamic dynamic --no-aot --output faultline-runs/cuda-sequences

# Request CUDA graph optimization under normal backend caching
faultline qualify --require-cuda --dtype float32 --compile-mode reduce-overhead --workload mlp --workload buffer_update --workload shape_sequence --no-aot --output faultline-runs/cuda-mode
```

`--dtype` controls stored model/input dtype; `--autocast` controls execution and leaves storage
unchanged. Autocast wraps forward only. New native captures use saved VJP probes; existing loss-based
captures retain their policy. With the existing per-output dtype tolerances, low-precision accumulation differences can
therefore produce findings in float32 parameter gradients. Such findings need numerical assessment.
Replay and reduction preserve both autocast and compile mode. Already differentiated AOT graphs
contain their casts and are not autocast or differentiated again.

`--no-aot` keeps the ordinary backend cache behavior. Diagnostic AOT capture remains useful for
backward reduction, but explicitly disables the AOTAutograd cache. `reduce-overhead` requests CUDA
graph optimization; unsupported operations or mutations can make Inductor skip it. A passing case
does not prove CUDA graphs were used. Set `TORCH_LOGS=perf_hints` and inspect each worker's stderr
for skip information. The environment records PyTorch CUDA/HIP versions, devices, and float32
matmul precision.

The manual [GPU workflow](../.github/workflows/gpu.yml) requires a self-hosted Linux runner labeled
`cuda` or `rocm`, with its matching PyTorch environment already prepared. It runs dtype, autocast,
state/shape, and compile-mode checks and uploads evidence. It does not run on public pull requests
or provision a GPU. AMD RX 7900 XT / WSL measurements are recorded in [RESULTS.md](../RESULTS.md).
NVIDIA/CUDA remains unvalidated.

## Remaining work

Config and nightly-version bisection, automatic input/weight sanitization, native tensor-dimension
reduction, task-level quality metrics, RNG-specific classification, and a broader model/bug corpus
are not implemented. Safetensors cannot directly replace the current snapshots without reconstructing
shared storage and views; preserving those layouts is part of reproducing the bugs above.

Native models can opt into `--fp64-advisory`; see [numerical evidence](numerics.md) for its reference
policy and limitations. Explicit rounding, overflow, and dtype conversions are program
semantics, so a compiler result closer to fp64 is not automatically correct. Historical calibration
and controlled fault injection serve different purposes and are reported separately. Compiler
fuzzing research such as [NNSmith](https://github.com/ise-uiuc/nnsmith) targets test generation and coverage; Faultline's current focus is
repeatable workload comparison, regression classification, and failure reduction.
