# CPU accuracy investigation

## Case study: from a sweep flag to input replication

The 0.11 sweep flagged GPT-2 on one frozen input. Nightly bisection narrowed the change to
November 4 versus November 5, and source-build tests localized it to commit
`2450619a4c8a11b6f652154d12173794ebe181d5`, following parent
`eb48c7e2d9fd6b16c0a7291703c94b16fb33cc42`. The `exp_u20` mechanism tests then separated
the observed change from proposed explanations; the kernel and generated-GELU null results
are retained in the later investigation notes.

The crucial final check was replication across inputs. On twenty disjoint input batches,
GPT-2's compiled gradients worsened on 11 and improved on 9, with median growth 1.007994x.
None of GPT-2, BERT, Pythia or the eleven-layer GPT-2 variant met the predeclared systematic
effect rule. The [0.15 measurements](input-accuracy.md) retain all batch results and controls.

The original change was real and reproducible for that input, but it did not establish a
systematic accuracy regression. This is how Faultline separates a candidate regression from
an input-sensitive rounding change: preserve the initial flag, localize it, test the mechanism,
then check whether the effect survives new inputs. The original 0.11 flags remain labeled
`accuracy_shift`, annotated **single_input, descriptive only**. No primary result or earlier
measurement has been rewritten, and no upstream defect report was filed.

The account below preserves the original 0.12 investigation; its report-draft discussion is
historical and should be read alongside the later replication outcome above.

The 0.12 investigation follows the GPT-2 accuracy boundary found in 0.11: the November 4
and November 5, 2026 CPU nightly wheels. It compares numerical evidence and generated code.
It does not change Faultline's primary statuses, tolerances, signatures, reducer predicates,
or existing exit codes. Earlier evidence remains in place.

## Fixed experimental policy

The nightly/source predicate remains the 0.11 predicate: median log2 compiled/fp64 RMSE
growth over gradients with finite, strictly positive errors in all three stable-baseline
workers. Each environment gets three fresh workers, two internal repeats, one CPU thread,
a 900-second worker budget, and a 12 GiB host-memory limit. All three scores must reach
1.4× to classify an environment as shifted. Mixed or missing observations are inconclusive.

The BERT and Pythia boundary checks use their full frozen workloads and fresh 2.15.0 baseline
observations. A numerical change below 1.4× remains `not_shifted` under this predicate.
Measured growth and predicate classification are reported separately. The separate 0.11
distributional advisory keeps its existing 1.25× threshold and other conditions.

The investigation executes a frozen copy of the released 0.11 harness so that adding analysis
utilities cannot alter the experiment. Source hashes identify that copy. New runs, compiler
caches, source builds, and logs use separate directories; the 0.11 installations are reused
without upgrading them.

## Inspecting generated code

For each boundary wheel, the full GPT-2 workload is run with `TORCH_LOGS=output_code` and
`TORCH_COMPILE_DEBUG=1`, with a new Inductor cache per worker. Forward and backward artifacts
are retained locally. Debug capture is recorded as an instrumentation intervention and its
accuracy scores are checked against the earlier result.

The read-only comparison utility accepts two debug directories:

```bash
python scripts/compare_inductor.py before/debug after/debug --output code-comparison
```

It writes an inventory, raw and normalized hashes, emitted kernel definitions, external call
sites, reduction/type markers, and a normalized diff. Arithmetic, shapes, kernel names and
types remain in the comparison. Only AOT debug IDs, line endings, and the generated
`cpp_prefix.h` cache path are normalized. Repeated module occurrences remain in the inventory;
content deduplication applies only to the module-set comparison. Unmatched modules are not
assumed to have a one-to-one graph correspondence.
The pre/post-fusion IR and transformed FX graph have separate hashes and completeness counts.
The FX comparison additionally hashes its parsed syntax, keeping comment/location differences
distinct from changed operations, constants, annotations, or shapes. It never executes the graph.

Counts are static definitions and call sites, not measured runtime launches. Type tokens and
reduction markers need inspection before drawing conclusions about accumulation. Equal
generated code does not establish equal implementations in included headers or linked ATen,
BLAS, or oneDNN libraries. That distinction matters for external attention calls.

The [normalized config diff](inductor-config-012.diff) covers all defaults returned by
`torch._inductor.config.get_config_copy()`. Two installation-specific CUTLASS directory values
are replaced by the same `<site-packages>` prefix. The full defaults, source file, build
configuration, and decomposition identities/source hashes remain in local evidence.

## Standalone numerical measurement

[`examples/gpt2_accuracy.py`](../examples/gpt2_accuracy.py) uses PyTorch and transformers
without importing Faultline. It loads a pinned GPT-2 checkpoint and the same frozen token IDs,
uses a fixed Rademacher backward vector, and records each gradient's compiled and eager RMSE
against fp64. It also records `RMSE / (2^-24 * RMS(fp64 gradient))`; this is a relative scale,
not an elementwise ULP bound. The fp64 reference itself has rounding error.

```bash
python examples/gpt2_accuracy.py --output baseline.json
# Run under the other interpreter:
python examples/gpt2_accuracy.py --reference baseline.json --output candidate.json
```

The checkpoint revision is fixed in the script. `--offline` uses an existing Hugging Face
cache. `--assets PATH` instead reads the original frozen `model.json`, `weights.pt`, and
`inputs.pt` source directory and checks the checkpoint identity, file checksums, input values,
dtypes, strides, and storage offsets. Large checkpoint
files are not distributed in this repository. Each invocation is one fresh-process
observation; the script alone does not implement the supervised three-worker bisection.

The experiment's source builds and interventions are reported in [RESULTS](../RESULTS.md).
No upstream issue is filed automatically. A downstream accuracy shift can follow an intended
algorithm or code-generation trade-off; it is not by itself a compiler-defect determination.

## Boundary-wheel code comparison

All three debug-captured observations per wheel reproduce the earlier GPT-2 scores: 1.000×
on November 4 and 1.710× on November 5. Both wheels emit the same two normalized modules,
one forward and one backward. Pre-fusion and post-fusion IR are identical, and the transformed
FX graphs have identical parsed syntax. Their raw comments contain different installation paths.
The [generated-code diff](inductor-code-012.diff) and [FX syntax diff](inductor-fx-012.diff)
are therefore empty; this is a measured equality, not a missing capture.

| Static observation, equal on both wheels | Forward | Backward |
|---|---:|---:|
| Emitted C++ kernel definitions | 11 | 7 |
| Generated-kernel call sites | 38 | 61 |
| Top-level fused scheduler groups | 25 | 61 |
| External matrix multiplication calls | 48 `addmm` | 96 `mm` |
| External CPU flash-attention calls | 12 | 12 |
| IR sum-reduction marker occurrences | 50 | 148 |

The reduction markers specify float32 input and output. Inspection found no changed reduction
splits, accumulation types, fusion groups, matrix-call choices, or arithmetic in these artifacts.
There are no emitted packed-oneDNN, `linear`, `bmm`, or matrix-template call sites in these
two modules. This does not describe the matrix operations inside the external attention kernel.
The [machine-readable inventory](inductor-code-012.json) retains individual kernel names,
type markers, counts, hashes, and all six observation scores.

The default decomposition registry has 1,108 identical identity records. The 599 available
Python source hashes match; 509 entries lack inspectable Python source. Registry identity does
not establish equality of native implementations. Config inspection finds 671 defaults before
and 672 after; the only substantive
addition is `realize_broadcast_reuse_threshold = 64`. Both wheels report MKL 2024.2 and the same
oneDNN 3.14.1 revision. Identical generated Python/C++ does not imply identical linked ATen code.

## Candidate inspection

The inspection starts from the [78-commit main interval](https://github.com/pytorch/pytorch/compare/2e10e9e8998bf3d420dac3530dfff53bf784c880...b7b26ca70e5b0d33b5e4f43ec213b2e6810c14cb)
identified by the two wheel snapshots. The following changes received closer source inspection:

| Change | Relevance to the recorded workload |
|---|---|
| [CPU `exp_u20`, `2450619a`](https://github.com/pytorch/pytorch/commit/2450619a4c8a11b6f652154d12173794ebe181d5) | Changes the AVX2/AVX512 float32 exponential used inside CPU flash-attention forward. It can change execution despite identical Inductor output. |
| [Broadcast materialization, `1933a0a9`](https://github.com/pytorch/pytorch/commit/1933a0a9ee51f2b03ad3cababd727b442d23630e) | Adds the only changed Inductor default. Disabling its trigger in three fresh November 5 workers retains exactly 1.710× growth. |
| [Native matmul row reductions, `01910287`](https://github.com/pytorch/pytorch/commit/01910287629bbab55aa8ffcaaeb1e822016856a2) | Changes the opt-in Triton native-matmul path; `triton.native_matmul` is false and the observed CPU graph uses external matrix calls. |
| [Device-independent padding refactor, `d690e74f`](https://github.com/pytorch/pytorch/commit/d690e74fd1893c752a6875d101ce26c552f6bbac) | The CPU exclusion remains in the eligibility check; emitted matrix shapes and code are unchanged. |
| [MKL update, `87f1120b`](https://github.com/pytorch/pytorch/commit/87f1120b4b1bb3d853770fbb113a0f228fa751ad) | Reverted by the interval's final commit. Both actual boundary wheels report MKL 2024.2. |

The `exp_u20` change modifies range reduction, scaling, and handling of subnormal values and
NaNs. Its upstream description reports improved local exponential accuracy. The CPU attention
forward kernel calls this approximation for float32, while backward recomputes probabilities
using a different exponential path. This is a reason to test the commit, not evidence that the
new approximation is defective. End-to-end gradient accuracy need not improve monotonically
with the accuracy of one elementary operation.

The changed error is also not exclusive to compiled execution: across positive gradient errors,
the boundary-wheel median eager/fp64 error growth is 1.093× for GPT-2, 1.215× for BERT, and 0.929×
for Pythia. These additional measurements describe the same frozen inputs and references.

Disabling flash attention is a broader intervention: it changes the backend for eager,
compiled, and fp64 execution. It cannot isolate `exp_u20` by itself. The source comparison
therefore builds the candidate and its direct parent with matching build settings and applies
the original three-worker predicate to each. The source builds are distinct from the official
nightly wheels; their environment identities and numerical outcomes must be reported separately.

## Source-commit attribution

The paired source builds isolate the GPT-2 shift to
[`2450619a4c8a11b6f652154d12173794ebe181d5`](https://github.com/pytorch/pytorch/commit/2450619a4c8a11b6f652154d12173794ebe181d5),
whose direct parent is `eb48c7e2d9fd6b16c0a7291703c94b16fb33cc42`.
The commit changes the AVX2 and AVX512 `exp_u20` implementations and their tests, with no
Inductor changes. All three parent workers score exactly 0 median log2 growth (1.000×),
and all three child workers score 0.7739668316531798 (1.710×), against the locked 0.11 reference.
The primary status remains `gradient_mismatch` in every worker.

This is stronger than matching the aggregate score: all 149 fp64 evidence rows, covering
148 gradients and the output, match the respective November 4/5 nightly observations exactly
in every source worker. Both source builds use matching compiler flags, GCC 13.3, MKL 2024.2,
oneDNN 3.14.1, and AVX512 dispatch on a Ryzen 7 7700X. Their normalized build configurations
and compilation-command hashes agree; the logs confirm that both AVX2 and AVX512 attention
objects were recompiled. The environments and wheels remain separate. Raw build logs,
wheel hashes, flags, and earlier unsuccessful build attempts are retained locally, with a
portable summary in `validation.json`.

The result attributes this frozen GPT-2 accuracy shift to one source commit. BERT and Pythia
were checked at the nightly boundary, not on the paired source builds, so commit-level
attribution is not established for them. The tests also do not identify which individual
arithmetic change within `exp_u20` dominates or why retaining the twelfth block amplifies
the effect. There is no claim that subnormal values occurred in the attention scores.

The [upstream report draft](upstream-report-draft.md) includes the standalone reproducer,
environment details, and error scales. It has not been submitted. This may be an intended
numerical trade-off in a CPU kernel, rather than a compiler defect.

## Error scale

The [per-gradient table](gradient-scales-012.csv) records all 492 gradient targets across GPT-2,
BERT, and Pythia, including exact gradients. For each wheel it includes compiled and eager RMSE,
fp64 reference RMS, and `compiled_RMSE / (2^-24 * reference_RMS)`. Each row uses the first worker;
the complete gradient evidence is identical across all three workers within each model/date.

| Full frozen workload | Gradients | Median error / (`u` × reference RMS), Nov 4 | Nov 5 |
|---|---:|---:|---:|
| GPT-2 | 148 | 31.889 | 58.763 |
| BERT | 197 | 58.298 | 76.802 |
| Pythia-160M | 147 | 18,636.454 | 24,044.269 |

Here `u = 2^-24`. These are medians of per-target relative errors, not ratios of medians,
and differ from the paired-growth predicate. They are not elementwise ULP counts or theoretical
error bounds. Near-zero reference gradients can make this normalization very large: BERT's
smallest reference RMS is approximately 3.14e-16. The CSV retains absolute errors and magnitudes
so that such targets remain interpretable. The fp64 eager computation is itself a numerical
reference, not an exact-real or application-quality oracle.
