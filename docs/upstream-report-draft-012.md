# Draft: CPU GPT-2 gradient accuracy shift after the exp_u20 change

Status: local draft; not submitted upstream. This reports a measured accuracy shift and asks
whether it is an expected numerical trade-off. It does not claim a compiler defect or a change
in model quality.

## Observation and source boundary

On a frozen pretrained GPT-2 workload, compiled fp32 gradient RMSE against eager fp64 grows by
a median 1.710× after
[`2450619a4c8a11b6f652154d12173794ebe181d5`](https://github.com/pytorch/pytorch/commit/2450619a4c8a11b6f652154d12173794ebe181d5)
(CPU `exp_u20`, PR #197785). Matched source builds of that commit and its direct parent
`eb48c7e2d9fd6b16c0a7291703c94b16fb33cc42` reproduce the two sides exactly.

The predicate was fixed before this investigation: median log2 compiled/fp64 RMSE growth over
147 gradients with strictly positive finite errors in the stable reference, threshold 1.4×,
three fresh workers per environment, and agreement of all three classifications. Each worker
uses two internal repeats and one CPU thread. The reference vector is unchanged from the
earlier investigation; it is the per-target median of three PyTorch 2.15.0 workers.

| Environment | Fresh workers | Median log2 error growth in each | Paired growth |
|---|---:|---:|---:|
| November 4 CPU nightly | 3 | 0 | 1.000× |
| Parent source `eb48c7e2` | 3 | 0 | 1.000× |
| Child source `2450619a` | 3 | 0.7739668316531798 | 1.710× |
| November 5 CPU nightly | 3 | 0.7739668316531798 | 1.710× |

All 149 complete fp64 evidence rows, comprising 148 gradients and the output, match the
respective nightly exactly in every source worker. The sole exact gradient is excluded from
the positive-error growth predicate but remains in the evidence.

Environment: AMD Ryzen 7 7700X, AVX512 dispatch, WSL Ubuntu 24.04, Python 3.12.3,
transformers 5.22.1, GCC 13.3, MKL 2024.2, oneDNN 3.14.1. Source builds use matching flags:
`USE_CUDA=0 USE_ROCM=0 USE_XPU=0 USE_DISTRIBUTED=0 BUILD_TEST=0 USE_KINETO=0`,
`MKL_THREADING=OMP`, and two build jobs. Both AVX2 and AVX512 attention objects were rebuilt;
normalized runtime build configurations and compilation commands match. Source wheels are
independently installed. Raw logs and large artifacts are retained locally.

## Reproducer

[`examples/gpt2_accuracy.py`](../examples/gpt2_accuracy.py) is a standalone script requiring
only PyTorch, transformers and their dependencies, plus a working CPU Inductor C++ toolchain.
It does not import Faultline. Copy that file to run it independently.

It loads `openai-community/gpt2` at revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`, uses two fixed 64-token inputs and SDPA attention,
and computes a fixed ±1 vector-Jacobian product of the last hidden state (shape 2×64×768).
The model is in evaluation mode with gradients enabled. Every gradient is compared with
eager fp64 using the same checkpoint and probe. Token IDs are embedded in the file; no
tokenizer or input generation is needed. The first run downloads the pinned weights.

Install the chosen PyTorch wheel and `transformers==5.22.1` in separate environments. Run
these commands from a directory containing the script, substituting the two Python paths:

```bash
OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_CACHE_DIR="$PWD/cache-before-0" \
  /path/to/before/bin/python gpt2_accuracy.py --output before-0.json
OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_CACHE_DIR="$PWD/cache-after-0" \
  /path/to/after/bin/python gpt2_accuracy.py --reference before-0.json --output after-0.json
```

Repeat with new output files, cache directories, and processes for observations 1 and 2.
`--offline` can reuse already cached weights. The script records every eager/compiled RMSE,
reference RMS, PyTorch version and commit, and the paired median log2 growth. It makes no
pass/fail correctness verdict. Each invocation is one observation; it does not itself implement
the supervised three-worker predicate or construct a median-of-three reference vector.
Using a single before JSON is equivalent for these recorded runs because all three before
error vectors are identical. Allow roughly 6 GiB for a worker on the tested machine; the
investigation supervised each worker with a 900-second and 12 GiB host-memory budget.

The standalone script was executed separately on stable, November 4, and November 5 wheels.
All 148 eager and compiled gradient RMSEs match the harness exactly in each environment.
An additional November 5 run using normal pinned-checkpoint loading also matches the frozen
assets exactly. These four script checks use one fresh process each; the three-worker source
comparison above uses the frozen investigation harness.

## What changed and what did not

The normalized Inductor forward/backward modules, pre/post-fusion IR, and transformed FX
syntax are identical across the boundary. Both emit 48 forward `addmm`, 96 backward `mm`,
and 12 CPU flash-attention calls per stage. No changed fusion, reduction split, accumulation
dtype, or matrix-call choice was found in those artifacts.

`exp_u20` runs inside CPU float32 flash-attention forward, outside this generated-code diff.
The commit changes AVX2/AVX512 range reduction, scaling, and exceptional-value handling.
Backward recomputes probabilities using a different exponential path. The source pair
attributes the model's shift to the commit; it does not identify which individual arithmetic
change dominates, nor establish that subnormal attention scores occurred.

Two November 5 interventions were each repeated in three fresh workers:

| Intervention | Paired growth | Interpretation |
|---|---:|---|
| Set broadcast reuse threshold to `10**12` | 1.710× | Null result; generated code also unchanged |
| Disable flash SDP | 0.978× | Shift absent; this changes eager, compiled and fp64 backends together |

The flash-disabled code has no CPU flash-attention calls and adds forward/backward `bmm`
calls. It is a broad path control, not independent proof about one exponential instruction.
The default boundary-wheel eager/fp64 gradient error also changes: median paired growth is
1.093× for GPT-2. This observation is not exclusive to compiled execution.

## Magnitudes and scope

Let `u = 2^-24`, fp32 unit roundoff. Per-target errors are reported relative to
`u * RMS(fp64 gradient)`, with absolute errors retained in the
[492-gradient table](gradient-scales-012.csv).

| Workload | Median paired compiled-error growth | Median RMSE / (u × reference RMS), before → after |
|---|---:|---:|
| GPT-2 | 1.710× | 31.889 → 58.763 |
| BERT | 1.309× | 58.298 → 76.802 |
| Pythia-160M | 1.292× | 18,636.454 → 24,044.269 |

For GPT-2, median reference gradient RMS is 10.0353, with range 0.4015–407.0186.
Median absolute compiled RMSE changes from 1.8655e-5 to 3.4101e-5. These medians are separate
statistics: dividing them does not give the median paired growth. Relative scales are not
elementwise ULP counts or theoretical error bounds, and fp64 eager is itself rounded.
Near-zero BERT reference gradients (minimum RMS approximately 3.14e-16) inflate relative
scales; the CSV retains the underlying magnitudes.

BERT and Pythia show their smaller shifts at the same adjacent-nightly boundary in all three
workers, but neither crosses the unchanged 1.4× predicate. They have not been run on the paired
source builds, so attribution to this commit is established only for GPT-2. Earlier AMD
comparisons did not show this shift; no new AMD source-build result is claimed here.

Ten- and eleven-layer variants retain the original embeddings, prefix blocks and final
LayerNorm weights without retraining. Their growth is 1.006× and 0.990×. A twelve-block prefix
starting from frozen first-block activations reproduces 1.712×, showing that executing the
embeddings is unnecessary. Its fp64 reference starts from already-rounded activations, so it
is a different computation. These bounded checks do not establish a general size threshold,
a minimal reproducer, or the mechanism of amplification through the twelfth block.

The upstream change improves exceptional-value handling and reports better local exponential
accuracy. A larger end-to-end gradient error can coexist with that improvement. This may be an
intended numerical trade-off; the evidence alone does not establish an incorrect kernel,
an unacceptable tolerance, or application-level harm.
