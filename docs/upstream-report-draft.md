# Draft: CPU accuracy shift — multi-input evidence

Status: **Not recommended for filing. Repository draft only; not submitted upstream. No compiler-defect claim.**

The original full GPT-2 compiled shift did not meet the predeclared systematic-across-inputs rule.
The single-input shift was not an input-robust regression in this experiment. The upstream report
draft is not recommended for filing. The original numerical measurements and three-model
parent/child attribution remain valid for their frozen inputs; they do not establish a systematic
degradation across passages.

## Twenty-batch replication (0.15)

| Model | Path | Worse / better / ties | Median growth | Batch sign p | Decision |
|---|---|---:|---:|---:|---|
| GPT-2 (12 layers) | compiled | 11 / 9 / 0 | 1.007994× | 0.41190147 | not systematic |
| GPT-2 (12 layers) | eager | 12 / 8 / 0 | 1.027375× | 0.25172234 | not systematic |
| BERT | compiled | 8 / 12 / 0 | 0.994764× | 0.86841202 | not systematic |
| BERT | eager | 6 / 14 / 0 | 0.971831× | 0.97930527 | not systematic |
| Pythia-160M | compiled | 9 / 11 / 0 | 0.990773× | 0.74827766 | not systematic |
| Pythia-160M | eager | 11 / 9 / 0 | 1.008742× | 0.41190147 | not systematic |
| GPT-2 (11 layers) | compiled | 9 / 11 / 0 | 0.997325× | 0.74827766 | not systematic |
| GPT-2 (11 layers) | eager | 9 / 11 / 0 | 0.995580× | 0.74827766 | not systematic |

All 160 fresh workers completed, covering 80 paired batches. Seven original-input controls
reproduced every retained eager and compiled gradient RMSE exactly before the new runs.
The present-gradient counts are 148 for GPT-2, 197 for BERT, 147 for Pythia and 136 for GPT-2-11.
The original GPT-2's 149 value targets included one forward output. No target is treated as
an independent input replicate. All fp64 gradient hashes match across builds in 80/80 pairs.
Each build is compared with its own fp64 reference as predeclared.

No passages were replaced, no thresholds were fitted after observing results, and no previous
labels changed. Exact ties are excluded only from the binomial trial count. The p-values are
descriptive: the batches were sampled without replacement from one book and share model weights
and a fixed VJP. This is neither a false-positive rate nor evidence about task accuracy.

The [predeclared policy](v015-policy.json), [sampling and method](input-accuracy.md),
[per-batch CSV](input-accuracy-015.csv) and [machine-readable evidence](input-accuracy-015.json)
document the final bounded investigation. Existing source builds of parent `eb48c7e2` and
child `2450619a` were reused. The original three-model source attribution is retained below;
it must be read together with the input-level results above. Numerical changes may reflect
an intended trade-off. The full-model propagation mechanism remains unresolved.

The original GPT-2 script below reproduces its frozen-input observation. It is not evidence
that every input gets the same change. The new [batch measurement example](../examples/input_batch_accuracy.py)
and [summary script](../scripts/summarize_input_accuracy.py) provide the separate input-level analysis.

The [0.14 draft](upstream-report-draft-014.md), [0.13 draft](upstream-report-draft-013.md) and
[0.12 draft](upstream-report-draft-012.md) remain archived unchanged. No upstream issue was filed.

## Observation and source boundary

The [0.14 GELU investigation](gelu-investigation.md) searches every retained full-model
forward/backward module from both source builds and resolves the linked tanh calls. Generated
GELU forward/backward and eager `gelu_new` use SLEEF tanh, not exp_u20. Consequently the proposed
generated-GELU exp replacement is not applicable and no recovery is claimed. All twelve blocks
were captured with exact 149-target output/gradient controls. Block 12 is not unusual under
the fixed input/argument range criteria. Its counterfactual exp_u20 disagreement fraction is
below the pooled earlier blocks; those hypothetical evaluations are not actual GELU exposure.
The full-model propagation mechanism remains unresolved.

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

BERT and Pythia show their smaller shifts at the same adjacent-nightly boundary, and 0.13 now
reproduces both on the paired source builds. All three fresh workers per model/build exactly
match every complete fp64 evidence row from the corresponding nightly: 198/198 for BERT and
148/148 for Pythia, including the output. Attribution to this commit therefore covers all three
full models. Neither BERT nor Pythia crosses the unchanged 1.4× predicate. Earlier AMD
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

## Kernel isolation follow-up

Actual q/k/v, upstream gradients, output/logsumexp and scalar settings were captured for blocks
11 and 12 of the frozen twelve-block prefix, on both source builds and in eager/compiled paths.
All four captures exactly match their uninstrumented output/gradient controls in two repeats.
Q/k/v and grad_out layouts agree between paths, but their values differ. Compiled backward uses
a contiguous saved-logsumexp layout; swapping this layout while keeping values fixed changes
none of the five operator results in the additional three-worker controls.

Each of the eight captured inputs was replayed unchanged on both builds, with three fresh workers
per build and explicit fp64 math as reference. Block-12 median dq/dk/dv error growth is:

| Inputs captured from | Parent → child kernel error growth |
|---|---:|
| Parent eager | 0.994052× |
| Parent compiled | 1.004449× |
| Child eager | 0.995670× |
| Child compiled | 0.999003× |

All three observations agree exactly. Out/logsumexp measurements and block-11 controls are also
recorded in the [complete kernel table](attention-kernel-013.csv). Backward outputs are bit-identical
across builds when the same saved out/logsumexp are supplied. The 1.4× kernel predicate is not met.
The [PyTorch-only measurement script](../examples/sdpa_accuracy.py) is not a reproducer of the
1.710× shift; it records this null result on fixed tensors. The compiled/eager calling difference
remains an open question at the full-model level, including propagation of changed activations
and upstream gradients.

Per-head score histograms contain zero finite entries below -87.3 or -103.97. The smallest
score-minus-rowmax values are about -17.8954 in block 11 and -13.2607 in block 12. There are no
exposed heads or rows with which to associate the largest gradient-error ratios. These calls
do not exercise the subnormal/underflow range fixed by the commit.

A sampled AVX512 microbenchmark confirms improved local exp_u20 accuracy: on 1,048,577 points
in [-104,0], the child's maximum observed normal-output error is 1.957 ULP and subnormal error
1.403 ULP versus fp64 exp. The parent's flush-to-zero interval produces much larger errors,
including some normal outputs. Regular exp, used in backward, has identical sampled accuracy
in both builds. These are sampled measurements, not exhaustive error bounds.

A source-matched reconstruction reproduces actual forward logsumexp exactly on all 12,288
captured rows per build. Recomputed backward probability sums differ from one by at most
approximately 3.94e-7–4.39e-7 across cases, with the same case-wise maxima in both builds;
mean deviations and probability differences change slightly in both directions. This does not
show a comparable 1.7× deterioration in local probability consistency. The
[investigation](attention-investigation.md) includes all controls, timings, limitations, and
per-case data. The full-model amplification mechanism remains unresolved.
