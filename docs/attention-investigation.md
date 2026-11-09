# CPU attention investigation

Version 0.13 extends the source-commit attribution to BERT and Pythia-160M, then measures
GPT-2's final attention calls directly. The isolated kernel does not reproduce the full-model
1.710× accuracy shift. This is a recorded null result under the fixed predicate, not a compiler
correctness determination. No upstream report has been submitted.

## Protocol and attribution

The [experimental policy](v013-policy.json) was recorded before the experiments. Additional
[kernel controls](v013-kernel-controls.json) were fixed before kernel execution. These
measurements do not change Faultline's primary statuses, tolerances, signatures, reducer
predicates, or existing exit codes. The numerical harness remains the frozen 0.11 source
used in 0.12. Earlier runs and installations are retained.

The existing source builds are the direct parent `eb48c7e2d9fd6b16c0a7291703c94b16fb33cc42`
and child `2450619a4c8a11b6f652154d12173794ebe181d5`. No PyTorch build was repeated. BERT and
Pythia each receive three fresh workers per build, two internal repeats, one CPU thread,
and the unchanged 1.4× median-gradient-error-growth predicate against their locked 0.12
stable reference vectors.

| Full frozen workload | Parent growth | Child growth | Exact nightly evidence rows, every worker | Fixed predicate |
|---|---:|---:|---:|---|
| BERT | 1.000× | 1.309× | 198/198 | Not shifted on both builds |
| Pythia-160M | 1.000× | 1.292× | 148/148 | Not shifted on both builds |

The table counts gradients and the output, including exact gradients. Every complete fp64
evidence row and every growth score matches the corresponding November 4/5 nightly result
exactly in all twelve workers. Combined with 0.12's GPT-2 source comparison, commit attribution
now covers all three models. BERT and Pythia remain below the fixed 1.4× predicate; the separate
distributional advisory is not substituted for it. [Per-worker attribution data](attribution-013.csv)
retains the unrounded values and primary statuses.

## Capturing actual attention calls

The original frozen twelve-block prefix is executed in eager and compiled modes on both
builds. The compiled forward and backward are warmed first. A dispatch observer then wraps
the already-compiled backend callable, avoiding a Dynamo recapture. It records the actual CPU
flash forward/backward operator arguments for calls 11 and 12. Backward call order is checked
against the sequential prefix, including equality of saved q/k/v and forward outputs.

Each of the four capture workers runs twice. Every observed execution exactly matches all
148 output/gradient tensors from its uninstrumented control. Each observes twelve forward and
twelve backward flash calls. Tensor storage is saved immediately, preserving strides, offsets,
and shared storage. Large saved tensors remain local; the [capture inventory](attention-capture-013.json)
contains shapes, layouts, logical-value hashes, and numerical comparisons.

Both blocks use fp32 q/k/v and upstream gradients of shape `[2,12,64,64]`, scale `0.125`,
`is_causal=True`, dropout zero, and no additive mask. The relevant layouts are:

| Argument | Eager strides | Compiled strides | Storage offset |
|---|---|---|---|
| q/k/v | `[147456,64,2304,1]` | Same | 0 / 768 / 1536 |
| grad_out | `[49152,64,768,1]` | Same | 0 |
| backward saved output | `[49152,64,768,1]` | Same | 0 |
| backward logsumexp, shape `[2,12,64]` | `[768,1,12]` | `[768,64,1]` | 0 |

Q/k/v and grad_out are noncontiguous. The compiled backward receives a contiguous logsumexp
copy, while the eager backward receives the native noncontiguous layout. Their values also
differ between eager and compiled execution. In block 12, the largest q/k/v difference is
5.25e-6 on the parent and 7.63e-6 on the child; the largest grad_out difference is 1.48e-5 and
3.24e-5 respectively. The inventory retains RMSE and magnitudes as well as maxima. These are
observations of different full-model execution paths, not a same-input kernel comparison.

## Same-input kernel comparisons

Each capture becomes a separate frozen workload: two builds × two paths × two blocks gives
eight inputs. Each of three fresh workers per execution build runs all eight, with two internal
repeats per input. Both builds receive identical saved tensors for each comparison. The native
CPU flash forward and backward are called directly; no compiler or transformers dependency is
needed for this stage. The reference is explicit fp64 scaled/masked QK, softmax, PV, and an
autograd VJP using the exact saved upstream gradient.

Forward and backward q/k/v retain their recorded layouts. Newly computed output/logsumexp are
given the recorded backward layouts explicitly where necessary. A matching-build replay exactly
reproduces all five captured operator results. The fp64 reference values and repeated numerical
measurements agree exactly across the workers.

The advisory uses the median log2 child/parent RMSE growth across dq, dk and dv, with the 1.4×
threshold and agreement of all three fresh workers. Out and logsumexp are reported separately.
The [complete target table](attention-kernel-013.csv) also retains absolute errors, reference
magnitudes, errors relative to fp32 unit roundoff, and a noise-floor comparison.

| Captured inputs | Block 11 gradient-error growth | Block 12 gradient-error growth |
|---|---:|---:|
| Parent, eager | 0.997800× | 0.994052× |
| Parent, compiled | 0.994012× | 1.004449× |
| Child, eager | 1.008243× | 0.995670× |
| Child, compiled | 0.984046× | 0.999003× |

All eight cases are `not_shifted`; all three observations agree exactly. Thus the isolated
kernel does not reproduce the 1.710× full-model shift on any preregistered block-12 capture.

Two further controls separate saved values from layouts:

- With captured out/logsumexp held fixed, backward dq/dk/dv are bit-identical across the builds
  for every case. The backward exponential implementation is unchanged.
- Swapping the eager/compiled layouts while retaining identical logical inputs changes only
  backward logsumexp's layout. All five operator results remain exactly equal in all 24
  block-12 control evaluations across six fresh workers, each with two internal repeats.

[`sdpa_accuracy.py`](../examples/sdpa_accuracy.py) measures saved forward/backward records:

```bash
python examples/sdpa_accuracy.py --forward forward-block-12.pt --backward backward-block-12.pt --output parent.json
# Use the other source-build interpreter with the same files:
python examples/sdpa_accuracy.py --forward forward-block-12.pt --backward backward-block-12.pt --reference parent.json --output child.json
```

Each file is a `torch.save` dictionary with `args`, `kwargs`, and `outputs` from the respective
native operator call. Loading uses `weights_only=True` and rejects non-CPU records, changed
forward/backward values, unsupported shapes, and missing required arguments. Each CLI invocation
is one observation; the three-worker supervision belongs to the investigation workflow.
This is a measurement example, **not a reproducer of the full-model shift**. The condition for
promoting a minimal kernel reproducer was not met, so the full GPT-2 script remains the report's
reproducer. The compiled/eager calling difference remains open at the full-model level, including
the changed activations and upstream gradients removed by these same-input comparisons.

## Score distributions and row associations

The [per-head histogram table](attention-heads-013.csv) includes all batch/head combinations.
It uses fp64 scaled/masked QK scores minus each row maximum, with bin edges
`[-104,-103.97,-87.3,-80,-40,-20,-10,-5,-1,0]`; bins include their lower edge and exclude their
upper edge. Masked negative infinities are counted separately from finite permitted scores.

Every capture has 49,920 finite permitted scores and 48,384 masked entries. **None** of the
finite scores is below -87.3 or -103.97. The smallest value across the four block-11 captures
is -17.895388; across block 12 it is -13.260713. Subnormal or zero-underflow exponentials
therefore have no observed exposure in these attention calls.

The [top-row table](attention-rows-013.csv) retains the ten largest finite child/parent RMSE
ratios for every case and dq/dk/dv target, including absolute errors. Dq rows are queries;
dk/dv rows are keys and use key-column exposure. The [association table](attention-association-013.csv)
reports all eligible rows, exposed/unexposed counts and mean growth, and top-ten overlap.
All exposure and overlap counts are zero. There is no exposed group on which to estimate an
association; these data do not support subnormal attention scores as the cause of the shift.

## Exponential accuracy and timing

[`cpu_exp_probe.cpp`](../examples/cpu_exp_probe.cpp) is a small AVX512 measurement adapter built
against each existing installation's headers and libraries. It measures `Vectorized<float>::exp_u20`
and the regular `exp` path used in backward. It does not replace or rebuild PyTorch. The first
adapter could not load because the private CPUBlas wrapper is not exported; the retained retry
uses the exported `sgemm_` called by that wrapper's fp32/highest-precision path. Link-time
undefined-symbol checks and separate synthetic tests cover the adapter before model measurements.

The [sampled accuracy table](exp-accuracy-013.csv) covers 1,048,577 points on [-104,0], plus
262,145 each on [-110,-86] and [0,89], and explicit exceptional values. Samples are generated
in fp64 and rounded to fp32. Real-valued error against fp64 exp is divided by local fp32 spacing;
the subnormal/zero spacing is `2^-149`. Distance to correctly rounded fp32 is reported separately.
This is sampled coverage, not an exhaustive ULP guarantee.

| Implementation, main [-104,0] sample | Normal-output maximum ULP error | Normal p99 | Subnormal maximum ULP error |
|---|---:|---:|---:|
| Parent exp_u20 | 11,862,151.879 | 3.979 | 8,388,070.004 |
| Child exp_u20 | 1.957 | 1.653 | 1.403 |
| Regular exp, both builds | 0.967 | 0.695 | 0.747 |

The parent's enormous maximum includes the previously incorrect flush-to-zero interval,
including some normal outputs near underflow. That interval does not occur in the captured
attention scores. The child fixes this behavior and propagates NaN in the explicit exceptional
check. Regular exp has identical sampled accuracy on both builds. The implementation can improve
local exponential accuracy while changing downstream full-model rounding.

On the fixed main array, median-of-three-worker-median times are 0.149764 ms → 0.129717 ms for
exp_u20 and 0.267188 ms → 0.262903 ms for regular exp. Each worker uses five warmups and eleven
alternating timed calls per implementation, including the ctypes call. No model workers run
concurrently with these timings. [All timings and their spread](exp-timing-013.csv) are retained;
these Zen 4/WSL microbenchmarks are not a full-model speed claim.

To build the small adapter in a matching Linux/AVX512 PyTorch environment, choose a fresh
library/output location and run:

```bash
TORCH_ROOT="$(python -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)')"
c++ -std=c++20 -O3 -shared -fPIC -Wl,--no-undefined \
  -mavx512f -mavx512bw -mavx512vl -mavx512dq -mfma \
  -DCPU_CAPABILITY=AVX512 -DCPU_CAPABILITY_AVX512 \
  examples/cpu_exp_probe.cpp -I"$TORCH_ROOT/include" \
  -I"$TORCH_ROOT/include/torch/csrc/api/include" \
  -L"$TORCH_ROOT/lib" -Wl,-rpath,"$TORCH_ROOT/lib" \
  -ltorch_cpu -lc10 -o cpu_exp_probe.so
python examples/exp_accuracy.py --library ./cpu_exp_probe.so --output exp.json
```

Use a separate adapter and process for each PyTorch installation. The Python wrapper checks
AVX512 dispatch and rejects implicit precision/layout changes. It records the floating-point
control register alongside the sampled accuracy and timings. The SGEMM reconstruction supports
the measured single-key-block, 32-query-block fp32 path; it is not a general attention kernel.

## Forward/backward probability consistency

The source-matched reconstruction uses the actual tensor strides and the native 32-query-block
SGEMM calls, including separate forward scaling and backward GEMM alpha. It reproduces forward
exp_u20 accumulation and scalar tails, then the saved logsumexp. On **all 12,288 rows per build**,
reconstructed forward logsumexp equals the actual kernel output exactly; all three workers agree.
The synthetic checks also match all rows for contiguous and packed-offset tensors on both builds.

Backward probabilities are reconstructed with the regular vector exp and that saved logsumexp.
Their maximum absolute row-sum deviation from one is approximately 3.94e-7–4.39e-7 across cases,
with the same case-wise maximum on both builds. Mean row-sum deviations and forward/backward
probability RMSE change slightly in both directions; there is no comparable 1.7× increase.
The [consistency table](attention-consistency-013.csv) records every case and also includes
fp64-rounded-score and fp32-bmm reconstructions as separately labeled controls. The source-matched
values are reconstructions validated against logsumexp, not direct captures of internal P buffers.

The result narrows the explanation: commit attribution is established for three full models;
subnormal inputs and the logsumexp layout change do not explain these captured calls; the isolated
kernel's accuracy is essentially unchanged. Propagation through the complete model remains
unresolved. The [report draft](upstream-report-draft.md) retains that limitation and has not been filed.
