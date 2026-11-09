# GPT-2 GELU call-path investigation

Version 0.14 checks whether generated GELU kernels explain the previously attributed CPU
accuracy shift. They do not call `exp_u20` in this workload. Both generated forward and backward
use SLEEF tanh, and block 12 is not unusual under the fixed distribution criteria. The full-model
1.710× shift remains attributed to the source commit; its amplification mechanism remains open.
This is advisory evidence, not a compiler-defect claim. Nothing was submitted upstream.

## Scope and fixed predicate

The existing parent `eb48c7e2d9fd6b16c0a7291703c94b16fb33cc42` and child
`2450619a4c8a11b6f652154d12173794ebe181d5` installations are unchanged. The original full frozen
GPT-2 workload supplies the weights, two 64-token inputs, and fixed VJP cotangent. This stage
uses the full model, not the twelve-block prefix used for 0.13 attention capture.

The [policy](v014-policy.json) was fixed before new executions. A conditional intervention
would use the unchanged locked 0.11 reference, median log2 compiled/fp64 error growth across
147 positive finite gradient targets, a 1.4× threshold, and three fresh workers with two
internal repeats each. Missing evidence or mixed classifications would be inconclusive.
Below-threshold growth would not imply exact recovery to the parent.

The intervention gate was not met: there is no generated GELU `exp_u20` call to replace.
Therefore no three-worker replacement or reverse replacement was executed, no disappearance
of the shift is claimed, and no minimal GELU reproducer is promoted. Turning on a different
tanh formula would change arithmetic that does not call `exp_u20`; it would not test the
requested replacement. No PyTorch rebuild was performed.

## Complete generated-code search

The [code inventory](gelu-code-014.json) records exact filenames, SHA-256 hashes, source lines,
kernel boundaries, native call sites, linked symbols, and relevant upstream definitions.
The search covers all three original full-model source workers per build: twelve complete
generated Python modules and 144 cached C++ files. It also covers all 24 retained November
4/5 `output_code.py` debug dumps, including repeated captures.

Each full-model forward defines eleven C++ kernels; backward defines seven. The only workload
math matches for `exp_u20`, `.exp(`, `.tanh(`, `Sleef`, and exp/tanh/sigmoid/erf call expressions
are these two tanh sites, identical across the parent and child:

| Stage | Generated module | Kernel | Exact source line |
|---|---|---|---|
| Forward | `chdaxqkckfvdf676vjbrghyzptw7iybrl7u4tfmocuflik6m6sw5.py` | `cpp_fused_add_mul_pow_tanh_view_3` | 487: `auto tmp13 = tmp12.tanh();` |
| Backward | `c6u6fdvyzjvqhs75isw3axug7xzhg6hfwztanv53ckduuvtw6lk5.py` | `cpp_fused_add_mul_pow_sum_tanh_tanh_backward_view_1` | 171: `auto tmp10 = tmp9.tanh();` |

The forward kernel computes `sqrt(2/pi)*(x + 0.044715*x^3)`, calls tanh, and finishes
`0.5*x*(1+tanh(...))`. The backward kernel recomputes tanh. These definitions are reused
across the twelve blocks. No generated sigmoid/erf exponential decomposition was found.
The list of generated workload C++ kernels reaching `exp_u20` directly or through these
linked tanh paths is empty.

The installed AVX512 header resolves `Vectorized<float>::tanh()` to
`Sleef_tanhf16_u10(values)` at installed line 582, corresponding to upstream source line 581
in `aten/src/ATen/cpu/vec/vec512/vec512_float.h`. All twelve saved GELU shared libraries
(two kernels × three workers × two builds) import that symbol. SLEEF's `xtanhf` calls its own
`expk2f` implementation; it does not call ATen's `exp_u20`. Source and installed header bodies
match exactly apart from the wheel's outer stable-API guard, which shifts line numbers by one.

The cache also contains three ISA capability probes per worker. Their line 11,
`auto tmp1 = tmp0.exp();`, belongs to `__avx_chk_kernel`, not a GPT-2 workload kernel.
The corresponding source is `torch/_inductor/cpu_vec_isa.py:77–80`. Regular vector exp resolves
to `Sleef_expf16_u10`, not `exp_u20`. These matches remain in the inventory rather than being
silently omitted.

`cpp.use_decompose_tanh` is false in both measured environments. In the inspected
`torch/_inductor/codegen/cpp.py:1566–1575`, enabling it emits
`2 / (1 + (-2*a).exp()) - 1`; it already uses regular vector exp. The default is explicitly
disabled in `torch/_inductor/config.py:2128–2132`.

The generated wrappers separately call native CPU flash attention twelve times in forward
and twelve times in backward. Those exact wrapper lines are retained in the inventory.
Native flash forward reaches `exp_u20` through `FlashAttentionKernel.cpp:113`; backward
recomputation uses regular `.exp()` at line 1010. These are native attention calls, not
generated GELU kernels. The [0.13 same-input attention investigation](attention-investigation.md)
still bounds what is known about that path.

## Eager GELU behavior

The frozen configuration selects transformers `gelu_new`, implemented by
`NewGELUActivation.forward` in `transformers/activations.py:65–66`. It computes the Python
arithmetic expression using `torch.pow`, multiplies/adds, and `torch.tanh`. Eager dispatch
capture observes twelve `aten.tanh` calls on `[2,64,3072]` fp32 tensors and the corresponding
backward calls. The native tanh vector path reaches SLEEF. Eager tanh backward computes
`grad*(1-output*output)` (`BinaryOpsKernel.cpp:1020–1029`), without an exponential call.

This actual path is distinct from calling `torch.nn.functional.gelu(approximate="tanh")`.
For completeness, the native fused GELU forward uses `Gelu.h`'s vector tanh helper; native
GELU backward uses `inner_vec.tanh()` (`Activation.cpp:444`). Those also reach SLEEF in
the inspected CPU build. They do not establish that the frozen model used fused eager GELU.

## Captures and distributions

Four fresh capture workers cover parent/child × eager/compiled. Each runs two observed replays
after an uninstrumented forward/backward control; compiled execution is warmed before observation
and the already-compiled backend callable is invoked directly. Every replay exactly matches
all 149 control outputs/gradients. Twelve MLP expansion `addmm` outputs are saved immediately,
preserving the original `[128,3072]` contiguous storage. Each block contains 393,216 fp32 values.
The [capture inventory](gelu-capture-014.json) includes both repeats, layouts, file/value hashes,
operator counts and the exact-control results.

Eager tanh operands are captured directly. For both paths, the comparable mathematical argument
`z = sqrt(2/pi)*(x + 0.044715*x^3)` is also reconstructed in fp64 from the saved fp32 inputs.
The compiled fused intermediate itself is **not** directly captured. This distinction matters
for last-bit comparisons; the tables label reconstructed and observed arguments separately.

The [full distribution table](gelu-ranges-014.csv) has 168 rows covering every block, build,
path and quantity. It retains all 22 histogram bins, extrema, absolute p95/p99, threshold
counts and counterfactual exponential comparisons. Bin edges are fixed in the policy; bins
include their lower edge, with separate outer tails.

Both existing 0.13 adapters are evaluated on identical saved coordinates. First, regenerating
all three fixed grids reproduces every 0.13 sampled-accuracy statistic exactly for each build.
The [grid table](gelu-exp-grid-014.csv) retains parent/child disagreement per occupied bin;
it does not assume that every float in a bin behaves like its sampled points. Directly
evaluating the adapters on the captured coordinates avoids that approximation. Regular exp
outputs are identical across builds throughout these comparisons.

Here are the child-compiled captures; the other three conditions are in the full table:

| Block | Input absolute p99 | Reconstructed z absolute p99 | Hypothetical exp_u20(x) differs | Hypothetical exp_u20(z) differs |
|---|---:|---:|---:|---:|
| 1 | 3.080070 | 3.500038 | 2.013% | 1.851% |
| 2 | 3.174373 | 3.673997 | 1.290% | 1.265% |
| 3 | 3.596097 | 4.528428 | 2.174% | 2.076% |
| 4 | 3.515204 | 4.354417 | 2.356% | 2.278% |
| 5 | 3.313800 | 3.942321 | 1.912% | 1.755% |
| 6 | 3.308754 | 3.932374 | 1.934% | 1.864% |
| 7 | 3.077733 | 3.495801 | 1.674% | 1.535% |
| 8 | 3.008141 | 3.371302 | 1.670% | 1.458% |
| 9 | 3.021039 | 3.394139 | 1.663% | 1.502% |
| 10 | 3.078470 | 3.497138 | 1.764% | 1.615% |
| 11 | 3.143544 | 3.616472 | 1.746% | 1.590% |
| 12 | 3.090838 | 3.519601 | 1.661% | 1.502% |

The exponential percentages are hypothetical evaluations, not actual GELU exposure: neither
`x` nor `z` is passed to `exp_u20` in this model's GELU. The table also evaluates `-2*z`, the
coordinate that the disabled tanh decomposition would use. Its block-12 disagreement is
4.688%, versus 5.252% pooled across blocks 1–11. Both builds produce the same nonfinite counts
on that hypothetical coordinate; all nonfinite counts are reported separately.

Block 12 has a few extreme arguments: reconstructed z ranges from about -144.504 to 34.958,
with three entries below -87.3 and two below -103.97. Block 11 ranges from -124.316 to 12.190;
block 3 is much wider, about -1048.376 to 8912.077. Applying exponential underflow thresholds
to a tanh operand is not evidence of actual exponential underflow inside GELU.

The [block-12 comparison table](gelu-block12-014.csv) applies the fixed criteria independently
to each condition. Block 12's input/z p99 does not exceed twice the median of blocks 1–11,
and no counterfactual disagreement fraction is ten percentage points higher than their pooled
fraction. All fourteen quantity/condition checks are negative. Its absolute p99 rank is seventh
for every quantity and condition. This is descriptive evidence from one frozen workload,
not a population-level outlier test or an explanation of gradient propagation.

## Reusing the range inspection

[`examples/gelu_ranges.py`](../examples/gelu_ranges.py) requires only PyTorch. Given a CPU
`torch.save` dictionary containing `{"value": input_tensor}`, run:

```bash
python examples/gelu_ranges.py --input input-block-12.pt --output ranges.json
```

It records the input checksum and layout, input distribution, reconstructed argument, and
hypothetical `-2*z` distribution. It does not replay GPT-2, determine a compiler verdict,
or claim to reproduce the 1.710× shift. The full-model script remains the report's reproducer.
The [updated draft](upstream-report-draft.md) records this negative GELU-path result; its
[0.13 version](upstream-report-draft-013.md) is preserved verbatim.
