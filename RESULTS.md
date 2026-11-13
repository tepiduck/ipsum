# Validation results

Recorded experiments: **November 5–8, 2026**. The [public experiment summary](docs/validation.json)
contains check outcomes, compiler counts, reduction sizes, and the source fingerprint used for
those experiments. Raw local logs and machine paths are not distributed.

## Controlled experiments

All **14 experiment checks passed** under Python 3.12.3 and PyTorch 2.15.0+cpu on Ubuntu/WSL.

- Six eager-control cases agreed.
- Four injected numerical faults were detected: sum replaced with mean, ReLU replaced with
  negation, softmax on the wrong axis, and NaN output.
- An alternating candidate was classified as unstable. Compiler exceptions were confirmed
  repeatedly, and an abrupt worker exit was recorded separately as a process crash.
- Reduction shrank the controlled sum/mean fixture from **7 operations and 144 input values**
  to **1 operation and 2 values**, using **17 evaluations**, including an independent final check.
- Standalone numerical and exception reproducers preserved the expected failure signature.
- Saved-finding replay retained the original settings and left source evidence unchanged.
- Campaign resumption preserved completed evidence.

These are debugger-validation results using deliberate faults. They are not upstream bug discoveries.
The reducer reached a local fixed point under its implemented transformations; global minimality
is not claimed. Run `faultline demo --output NEW_DIRECTORY` to inspect this workflow yourself.

## Actual compiler execution

An 18-case float32 CPU TorchInductor campaign produced **17 agreements and one numerical
discrepancy**, both before and after shared-predicate hardening. The campaign combined six targeted
cases with twelve generated cases, seeds 0–11, using three repeats, one CPU thread, `rtol=1e-4`,
and `atol=1e-5`.

The discrepant graph combined matrix multiplication, averaging, squaring, sine, and softmax.
Its maximum eager/compiled difference was approximately **6.18e-5**, affecting one of 31 output
values. Both paths were similarly far from a CPU float64 advisory (about **5.08e-4** and **5.05e-4**).
This is consistent with amplified floating-point error and does not identify an incorrect compiler.

A bounded reduction removed three operations, **11 to 8**, within a 28-evaluation budget. The budget
was exhausted, so local irreducibility is not claimed. The reduced candidate reproduced in a fresh
worker and an independent standalone script with matching comparison metrics and failure signature.
A captured MLP using Linear, LayerNorm, and ReLU also passed CPU TorchInductor comparison.

**New upstream bug discoveries: zero.** No upstream issue has been submitted for the numerical candidate.

## Coverage and limits

The portable test suite exercises graph validation, model capture, numerical/nonfinite handling,
repeat stability, worker lifecycle, reduction, saved-finding integrity, standalone export, and
terminal workflows. The separate compiler test executes real TorchInductor code generation.
CI runs the portable suite on Windows and Linux, the compiler suite on Linux, a two-environment
upgrade gate, and historical calibration. GPU qualification is a separate manual workflow.

CPU eager corpus tests cover float16, bfloat16, float32, and float64. The compiled 18-case campaign
used float32. Later sections record additional forward/backward, dynamic-shape, and AMD GPU checks.
Distributed execution and arbitrary-model coverage have not been established.
Diagnostic timing fields are not throughput or speedup evidence.

The recorded experiment predates the terminal interface release. Current behavior is checked by
the automated suite; rerun the experiment harness to record evidence for another revision:

```bash
python scripts/validate_faultline.py --compiler --output faultline-runs/validation
```

## Version 0.3 model and upgrade experiments

All **eight additional experiment checks passed** on CPU under Ubuntu/WSL. The two independent
Python environments used **PyTorch 2.14.1+cpu and 2.15.0+cpu**, both on Python 3.12.3.

- Captured GELU/LayerNorm MLP, convolution/pooling, and causal-attention workloads passed real
  TorchInductor forward and backward comparison. All three also passed in both compiler environments.
- A deliberately doubled derivative produced `gradient_mismatch` while forward outputs and loss
  still agreed. The unit suite separately distinguishes missing gradients from zero gradients.
- A controlled history-dependent failure was absent in all seven isolated requests. Whole-trace
  reduction retained **three requests with batch sizes 2 → 3 → 2**, within a 32-evaluation budget.
  A fresh standalone script reproduced the same failure. This is an injected fixture, not a real
  compiler cache bug; the exhausted budget does not establish local irreducibility.
- A seven-request captured MLP stream passed real dynamic TorchInductor forward/backward checks.
- The environment gate detected a controlled candidate regression and reduced **3 operations / 136
  input values to 1 operation / 4 values** in eight dual-environment evaluations. The final baseline
  passed and the final candidate retained the original numerical failure. Separate baseline and
  candidate reproducers were exported. This bounded reduction does not claim global minimality.

The compiler unit suite additionally covers approximate GELU, grouped/dilated convolution, and
noncausal attention. Standalone exports are exercised without importing Faultline. Tests include
baseline rejection during reduction, saved gradient/trace policy integrity, and CLI factory capture.
These checks establish behavior on small representative components, not correctness for all models.
No AMD/ROCm hardware result or throughput improvement is claimed.

Reproduce the new experiment (the baseline environment needs PyTorch, NumPy, and psutil):

```bash
python scripts/validate_upgrades.py --compiler --baseline-python .baseline/bin/python --output faultline-runs/upgrades
```

Omit `--baseline-python` for a same-environment control. CI pins both PyTorch releases and uploads
the full evidence after success or failure. The public JSON summary retains the measured source
fingerprint and separates these observations from the original 18-case campaign.

## Version 0.4 original-model validation

The native workload harness passed its five experiment groups on CPU, under Ubuntu/WSL and
Python 3.12.3. The regular comparison used PyTorch 2.14.1+cpu and 2.15.0+cpu. The historical check
used a separate PyTorch 2.12.0+cpu environment. These runs used two repeats, one CPU thread, and
the unchanged float32 tolerances `rtol=1e-4`, `atol=1e-5`.

| Workload | Size | Observed result |
|---|---|---|
| torchvision ResNet-18 | 11,689,512 parameters; input 1×3×224×224 | Forward agreement in both releases |
| Six-layer TransformerEncoder, embeddings and output head | 48,917,120 parameters; int64 tokens and boolean padding mask | Forward agreement in both releases |
| Expanded-storage indexing fixture | Float values plus integer indices | Agreement on 2.14.1 and 2.15.0 |
| Controlled ReLU fault in a 768-wide module | 590,592 parameters | Baseline passed; injected candidate failure detected and reduced |
| Historical indexing regression | PyTorch issue #183986 | Reproduced on 2.12.0; absent on 2.15.0 |

The vision and transformer fixtures use random weights and synthetic inputs. They establish that
the tool executes original models of these sizes; they do not establish accuracy, throughput,
pretrained-checkpoint coverage, or correctness for every model.

The controlled reduction isolated the failing graph inside the original module, then reduced
**four operations / 592,128 tensor input elements to one operation / 1,536 elements** in eight
dual-environment evaluations. The baseline remained passing and the candidate retained its
independent graph failure signature. Both were checked in fresh final workers. The exported
candidate script reproduced the injected failure without importing Faultline.

The [documented indexing issue](https://github.com/pytorch/pytorch/issues/183986) was reproduced
with an independently sized CPU fixture. The historical release was deliberately the candidate,
so this is not a claim that a current upgrade introduced the bug. Full-model graph isolation and
independent graph replay both retained the numerical mismatch. The three-operation graph could
not be reduced further by the implemented bypass proposals; all five evaluations, including final
confirmation, retained a passing 2.15.0 baseline whenever the failure was accepted. Separate
standalone scripts reproduced the failing 2.12.0 and passing 2.15.0 outcomes. This validates a
known upstream bug, not a new discovery.

A separate ResNet-18 forward/backward check reported **gradient_mismatch in both 2.14.1 and
2.15.0**, with agreeing forward outputs. The environment gate classified it as `recurring`.
The default thresholds were not relaxed to hide it. This observation needs numerical investigation
and is not evidence of a newly introduced compiler defect. A smaller 768-wide native module passed
actual Inductor forward/backward testing; controlled identical-forward/wrong-gradient tests also
exercise the derivative oracle.

The test suite covers binary alias/stride preservation, nested integer/boolean inputs, exact int64
comparison, input mutations, registered-state reset, request deletion, baseline rejection,
standalone predicate parity, modified artifact rejection, and the CLI. Native-model checks are
also part of the two-environment CI job. AMD/ROCm hardware, distributed workloads, optimizer steps,
and performance-regression detection remain unvalidated or unsupported.

Reproduce the native experiment groups (install torchvision in both environments for `--large`):

```bash
python scripts/validate_native.py --compiler --large --baseline-python .baseline/bin/python --output faultline-runs/native-validation
```

Add `--historical-python .historical/bin/python` with PyTorch 2.12.0, NumPy, and psutil to run the
historical bug check. Omit `--large` for the smaller CI workload. Public summaries omit local paths
and exact model data; full evidence remains in the chosen output directory.

## Version 0.5 backward graphs and device checks

The AOT validation ran on CPU under Ubuntu/WSL and Python 3.12.3, using PyTorch 2.14.1, 2.15.0,
and a separate 2.12.0 historical environment. The comparison retained two repeats, one CPU thread,
and the existing dtype-dependent tolerances. AOTAutograd caching was disabled for graph capture;
Inductor's kernel cache stayed enabled.

- A controlled fault doubled the backward graph's outputs while preserving forward results.
  Whole-model isolation retained the gradient failure. Independent graph reduction went from
  **12 operations / 597,504 tensor input elements to 4 operations / 3,072 elements** in seven
  dual-environment evaluations. Every accepted proposal passed in the baseline, and final fresh
  workers confirmed both outcomes. Separate exported scripts reproduced the passing baseline
  and injected gradient failure.
- The same native workload passed actual Inductor forward/backward checks on 2.14.1 and 2.15.0.
  Its independently captured backward graph also passed on 2.15.0, including standalone replay.
  These replays execute saved derivative operations without taking second derivatives.
- A five-by-five `diagonal_scatter(...).sum()` fixture reproduced the gradient error described in
  [PyTorch issue #182012](https://github.com/pytorch/pytorch/issues/182012). Forward results agreed
  on both versions; gradients failed on 2.12.0 and agreed on 2.15.0. Compiling only the backward
  stage retained the original full-model failure. The independent **four-operation backward graph**
  reproduced it with one scalar cotangent input. Its only bypass proposal removed the failure and
  was rejected. Five dual-environment evaluations, including final confirmation, left the graph
  unchanged. Both exported scripts reproduced their expected outcomes. Preserving Inductor's
  output-visibility metadata was necessary to retain this failure in the independent graph.

The historical release was deliberately selected as the failing candidate. This is a known bug
reproduction, not a new discovery or a claim that a current upgrade regressed. Controlled faults
are labeled separately in artifacts. Reduction does not claim a globally minimal graph.

`faultline qualify --device cpu --repeats 2` passed all **nine** combinations of MLP, convolution,
and boolean-masked attention with float32, float16, and bfloat16 on Inductor 2.15.0. Each performed
forward/backward checks with AOT capture. `qualify --require-rocm` returned `unavailable` on the local
CPU-only PyTorch installation and ran no fallback cases. AMD hardware execution is still unvalidated.
The matrix is a fixed set of small components, not evidence of arbitrary-model or full-training coverage.

The compiler suite now also checks independent AOT backward replay. Portable tests cover backward
stage isolation, baseline-preserving reduction, standalone parity, saved AOT policy, compiler
metadata serialization, static-input roles after argument pruning, and unavailable-device behavior.
The two-environment CI job runs the AOT experiment in addition to the existing model/upgrade checks.

```bash
python scripts/validate_aot.py --compiler --baseline-python .baseline/bin/python --output faultline-runs/aot-validation
```

Add `--historical-python .historical/bin/python` with PyTorch 2.12.0, NumPy, and psutil to check the
known backward bug. The Inductor adapter uses private compiler interfaces tested on the releases
listed above; other PyTorch versions and GPU code paths need their own validation.

## Version 0.6 diagnostics and historical calibration

The versioned historical corpus was run on CPU under Ubuntu/WSL and Python 3.12.3, with
PyTorch 2.15.0 as the control and 2.12.0 deliberately selected as the older candidate. Each
comparison used two repeats, one thread, and unchanged dtype-dependent tolerances. All four
selected fixtures remain in the denominator:

| Issue pattern | 2.12.0 | 2.15.0 | Calibration result | Total wall time |
|---|---|---|---|---|
| [183986](https://github.com/pytorch/pytorch/issues/183986), expanded-storage indexing | Numerical mismatch | Pass | Paired reproduction | 157 s |
| [182012](https://github.com/pytorch/pytorch/issues/182012), diagonal-scatter backward | Gradient mismatch | Pass | Paired reproduction | 143 s |
| [180164](https://github.com/pytorch/pytorch/issues/180164), slice-scatter backward | Pass | Pass | Not observed | 21 s |
| [185337](https://github.com/pytorch/pytorch/issues/185337), explicit cast rounding | Numerical mismatch | Numerical mismatch | Recurring; no passing control | 41 s |

The expected symptom appeared in **3 of 4 fixtures**; **2 of 4** also had a passing control.
The first three reports originated on CUDA; these executions used independently sized CPU
adaptations. They also share an expanded-storage family, so the corpus represents two families,
not four independent mechanisms. This is calibration on a small selected set, not a general
detection rate or a false-negative estimate. Closed upstream issues can still reproduce on a
given release, as the cast-rounding fixture demonstrates.

Both paired cases were isolated and checked in fresh workers and separate exported scripts:

| Fixture | Original independent graph | Retained graph | Evaluations | Reduction wall time |
|---|---|---|---|---|
| Expanded-storage indexing | 3 operations / 15 tensor input elements | 3 / 15 | 5 | 82 s |
| Diagonal-scatter backward | 4 operations / 1 scalar cotangent | 4 / 1 | 5 | 80 s |

The graphs were already small: the implemented bypass proposals did not remove further operations.
Their passing baseline and failing candidate scripts reproduced all four expected outcomes.
The cast case was not automatically reduced because neither environment supplied a passing
control. Timing includes startup, compilation, supervision, and export verification; the total
also includes diagnosis. Caches were enabled and other validation used the same host, so these
times describe this run's cost rather than a reproducible performance benchmark.

For all three detected candidates, `eager`, `aot_eager`, and `aot_eager_decomp_partition` passed;
only `inductor` failed. The ladder therefore reported an observed Inductor lowering/code-generation
boundary. This narrows investigation without establishing which pass or transformation caused it.

On Inductor 2.15.0 CPU, float32 model storage with bfloat16 autocast passed convolution and masked
attention checks. The MLP reported `gradient_mismatch`, with agreeing forward outputs. Seven
input/parameter gradient tensors exceeded the existing float32 thresholds; the largest absolute
difference was **0.000732421875**. Thresholds were not relaxed. Low-precision accumulation can
produce such differences, so this remains a numerical finding rather than a compiler-bug claim.
Both a three-request registered-buffer mutation stream and a batch-size **2 → 5 → 2 → 7** stream
passed forward/backward checks with dynamic compilation and normal backend cache behavior.

The compiler suite separately verifies all four backend rungs and a `reduce-overhead`/autocast
run with AOT capture and standalone replay. Running this mode on CPU does not validate CUDA graphs.
`qualify --require-cuda` correctly returned `unavailable` with zero cases on the CPU-only runtime.
**Neither NVIDIA/CUDA nor AMD/ROCm hardware execution is validated.** The manual GPU workflow
collects dtype, autocast, mutation/shape, and compile-mode evidence on a prepared runner; its
presence is not a hardware result.

Reproduce the calibration with PyTorch, NumPy, and psutil installed in both environments:

```bash
faultline benchmark historical --good-python .current/bin/python --bad-python .historical/bin/python --diagnose --max-evaluations 12 --output faultline-runs/history
faultline qualify --device cpu --dtype float32 --autocast bfloat16 --no-aot --repeats 2 --output faultline-runs/amp
faultline qualify --device cpu --dtype float32 --workload buffer_update --workload shape_sequence --dynamic dynamic --no-aot --repeats 2 --output faultline-runs/sequences
```

The benchmark returns 0 for a conclusive completed measurement, even when a fixture does not
reproduce. CI selects the two paired historical cases and uses `--require-paired` to enforce them.
See [diagnostics](docs/diagnostics.md) for the artifact fields, exit codes, and remaining limitations.

## Version 0.7 fixed backward probes and native numerical evidence

New native captures save fixed Rademacher cotangents for each request/output. Eager and compiled
backward receive the same vectors. The tests verify that an output offset does not change the
gradient seed, that a wrong derivative remains detectable at zero-valued outputs, and that probe
capture leaves initial state unchanged. Existing bundles retain their square-loss policy. The
fp64 advisory compares both executions with promoted eager computation without changing the
primary verdict. [Numerics](docs/numerics.md) describes the policies and their limits.

ResNet-18 was rerun with these VJP probes on CPU under PyTorch 2.14.1 and 2.15.0, with two repeats,
one thread, `rtol=1e-4`, and `atol=1e-5`. It still reported `gradient_mismatch` in both releases,
with agreeing forward outputs. The 2.15.0 input-gradient RMSE against the fp64 reference was:

| Execution | RMSE against the promoted reference |
|---|---|
| Eager float32 | 1.0156e-8 |
| Compiled float32 | 1.8795e-5 |
| Compiled/eager ratio | 1,850.6 |

The large ratio alone does not implicate the compiler. Follow-up saved-mask diagnostics found two
activation-selector flips, while eager execution with oneDNN disabled produced a comparable
gradient discrepancy without compilation. Compiled and eager backward agreed when given identical
saved inputs. This is evidence of derivative sensitivity at activation boundaries; the earlier
wording directing investigation toward the compiled path was misleading. Fixed cotangents remove
differences in the final backward seed, but do not fix internal activation branches. The model used
random weights and synthetic inputs, not a pretrained checkpoint or task-quality evaluation.

The CPU MLP with float32 storage and bfloat16 autocast also retained a gradient finding. Its
input-gradient RMSE was **0.0009200 eager versus 0.0009088 compiled**, a ratio of **0.988**.
Both paths were similarly far from the promoted reference; the measured compiled error was slightly
smaller. This gives substantially different numerical context from the ResNet finding without
automatically dismissing either result. Convolution and masked attention passed this CPU autocast
matrix. The original compact-IR matmul/softmax example's similar eager/compiled fp64 errors remain
recorded in the earlier campaign above.

## Version 0.7 known bug embedded in a transformer

The constructed indexing experiment used the **48,917,120-parameter**, six-layer transformer.
An aggregation after the classifier consumed actual transformer outputs and applied the operation
pattern from [issue #183986](https://github.com/pytorch/pytorch/issues/183986). No manual graph break
or compiler mutant was inserted. This deliberately places a known operation pattern in a larger
random-weight workload; it is not a newly discovered bug in an unmodified production model.

PyTorch **2.15.0 passed** and **2.12.0 failed numerically** on the same frozen workload. One Dynamo
graph was captured. Its whole-model failure survived compilation isolation and independent graph
replay. Freezing an internal tensor then removed the transformer prefix:

| Stage | Call operations | Tensor input elements |
|---|---:|---:|
| Original independent graph | 15 | 48,917,248 |
| Retained indexing graph | 4 | 12 |

The original graph uses PyTorch's fused transformer-layer operations, which explains the small
operation count despite the parameter count. The retained graph constructs its integer indices
as constants; those constants are not counted as tensor inputs. Reduction used **seven evaluations**,
including supervised boundary capture and uncached final confirmation. The baseline remained passing
whenever a proposal was accepted. The four-operation result could not be shortened by its bypass
proposal, which removed the failure. Global minimality is not claimed.

Both exported scripts were run independently under supervision: the baseline passed and the
candidate reproduced the numerical failure. The original workload and rejected proposals remain
in local evidence; exporting a smaller graph does not sanitize its retained values.
The indexing operation is near the output, within reach of the twelve-boundary prefix search.
This result does not validate reduction of a failure between encoder layers; suffix cutting and
that harder placement remain unimplemented.

## Version 0.7 AMD GPU execution

These checks ran on the **AMD Radeon RX 7900 XT (gfx1100)** in Ubuntu 24.04 / WSL, using a separate
environment with **PyTorch 2.14.0+rocm10.1.0**, reported HIP runtime **7.16.26385**, and
Triton **3.8.0+git669b31ac.rocm10.1.0**. The Windows display driver was **32.0.32015.7029** and
was not changed. Runs explicitly required ROCm and used `cuda:0`; they did not fall back to CPU.

| Matrix | Result |
|---|---|
| MLP, convolution, masked attention × float32/float16/bfloat16 storage | All 9 passed forward/backward checks with AOT capture |
| Float32 storage, bf16 autocast, dynamic shapes, normal backend caching | MLP and changing-shape sequence reported gradient discrepancies; buffer mutation passed |
| `reduce-overhead`, float32 MLP/buffer mutation/changing-shape sequence | All 3 passed forward/backward checks |

The autocast findings use the original per-output dtype thresholds and remain in the evidence.
Their same-device promoted references completed. A passing requested compile mode does not prove
that GPU graph capture was used. These are component and sequence checks on one AMD configuration,
not throughput measurements, general training validation, or NVIDIA/CUDA coverage.

The local release checks passed **215 portable Linux tests, 214 Windows tests with one platform
skip, 11 actual CPU compiler tests, and 31 legacy tests**. The public JSON includes the common
executed source fingerprint, numerical metrics, reduction sizes, and GPU outcomes. A separate
nightly CPU workflow now checks the compiler adapters and compares original-model workloads against
stable PyTorch; a failed nightly is evidence to investigate, not an automatic upstream bug report.

## Version 0.8 numerical sensitivity diagnostics

`--numerics` adds advisory classifications, an eager execution control, and optional AOT saved-mask
comparisons. It retains the primary status, tolerances, exit code, and reducer predicate. These
checks ran on the same CPU and RX 7900 XT environments described above, with two repeats and fixed
VJP cotangents. The executed source fingerprint and selected measurements are in
`docs/validation.json` under `numerical_context_validation`.

| Workload | Primary result | Additional evidence |
|---|---|---|
| Random-weight ResNet-18, CPU 2.15.0 | Gradient mismatch | `discontinuity_sensitive`, `eager_control_sensitive` |
| Float32 MLP with bf16 autocast, CPU 2.15.0 | Gradient mismatch | Comparable fp64 error on the failing gradients; input-gradient RMSE ratio 0.988 |
| Float32 MLP with bf16 autocast and AOT, ROCm 2.14.0 | Gradient mismatch | Comparable fp64 error on the failing first-layer bias gradient; ratio 0.978 |
| Linear(4, 8) followed by ReLU, ROCm | Pass | 16 saved selector elements compared, zero flips; standalone verified |
| 768-wide Linear/ReLU/Sigmoid/Tanh, ROCm | Pass | Recomputed ReLU selector explicitly unsupported; same-input backward agreement |

For ResNet, sixteen supported selectors covered **1,505,280 entries** and contained **two flips**.
One additional selector required recomputation and was not compared; lowered max-pool scatter
selectors are outside the current coverage. The flips were in backward nodes `where_1` and
`where_14`, consuming saved `relu_15` and `relu_2` values. These are AOT graph identifiers, not a
claim that every original-module activation was inspected. With the compiled saved inputs held
fixed, eager and compiled backward agreed under the original tolerances.

Disabling oneDNN in eager execution produced input-gradient RMSE **2.1925e-5 relative to original
eager**, versus **1.8795e-5 for compiled relative to original eager**: the eager-only discrepancy
was 1.167 times as large. The diagnostic replay matched the primary candidate and affected targets
on both repeats. Its exported standalone reproduced the complete numerical context. This supports
activation-boundary sensitivity as the explanation of the large 0.7 fp64 ratio; it is not a new
compiler bug. The general diagnostic still allows a separate backward defect to coexist with flips.

The ROCm control perturbed floating input storage by one representable step and retained input
layouts, aliases, integer roles, and captured cotangents. Its MLP has no recognized saved mask
selectors and reports that explicitly. No GPU graph-capture claim is added: the saved 0.7
`perf_hints` logs had no matching skip messages, but absence of a skip is not positive capture evidence.

Local validation passed the portable suite, ten focused diagnostic tests, twelve actual CPU
Inductor tests, and 31 legacy tests. Tests cover wrong derivatives with identical forward results,
missing gradients, changed saved activations, RNG and storage preservation, unstable controls,
advisory labels, and worker/standalone parity. The wrong-backward control remains a stable failure.

## Version 0.9 triage criteria

Numerical context schema 2 replaces the any-flip label with two labels. `explained_by_discontinuity`
requires flips, agreement in every completed same-input backward check, no failed backward replay,
and a mapped path from a flipped selector to every failing model gradient. Gradient mapping uses
actual tensor storage identity and layout; copied or accumulated gradients remain unmapped.
`discontinuity_present` retains suspicion when those conditions are not met.

Whole-finding fp64 comparability and eager-control sensitivity now require every failing target
to qualify. Reports retain matched/total counts, the exact control policy, and partial mask coverage,
including unverified max-pool selectors. These criteria do not change primary comparisons or
reduction predicates. They are evidence labels, not proofs of correctness.

A controlled AOT test combines a real ReLU mask flip with an optional doubled backward result.
The correct backward receives the explanation label; the planted wrong backward remains suspicious.
Mixed-target tests reject whole-finding reassurance when only one of two failing targets qualifies,
including when only one gradient is downstream of a flip. A failed backward replay also blocks
the explanation. Local validation passed 231 Windows portable tests (one platform skip),
12 Linux Inductor tests, and 31 legacy tests.
The built wheel's CLI and publication allowlist checks passed.

## Version 0.9 pretrained environment sweep

The [sweep commands](docs/sweeps.md) prepared 20 pretrained models: five ResNet/ResNeXt variants,
five EfficientNets, three MobileNets, ViT-B/16, Swin-T, ConvNeXt-Tiny, BERT, GPT-2, DistilBERT,
and SmolLM2-135M. Models range from 2,542,856 to 134,515,008 parameters. Each vision workload uses
the same two dog photographs with its resolved weight enum's inference transforms. Each text
workload uses two public-domain prose passages, padded/truncated to 64 tokens, and returns hidden
states. Models run in evaluation mode with float32 storage and fixed captured VJP cotangents.
These are concrete real-input paths, not representative task datasets or optimizer-step tests.

Each device attempted all 20 model pairs, using identical frozen weights and inputs across its
two interpreters. Settings were Inductor with AOT capture, gradient/numerical checks, two repeats,
one thread, `rtol=1e-4`, `atol=1e-5`, 600 seconds and 12 GiB host RSS per worker. Campaigns ran
sequentially on Ubuntu 24.04 / WSL, Python 3.12.3. No models were substituted, no tolerances were
relaxed, and completed inconclusive results were retained.

- CPU compared **2.15.0+cpu** with **2.16.0.dev20261105+cpu**. Matching torchvision versions were
  **0.29.2+cpu** and **0.30.0.dev20261105+cpu**. Their differences are part of the environment change.
- The **RX 7900 XT (gfx1100)** compared **2.15.0+rocm10.3.0a20261105** with
  **2.16.0a0+rocm10.3.0a20261105**, both using torchvision **0.30.0a0+rocm10.3.0a20261105**.
  Both came from AMD's [nightly package index](https://nightly.repo.amd.com/rocm/whl-next/), with the
  same ROCm SDK snapshot. The 2.15 release-version baseline is **not an AMD stable-channel build**.
- Transformers was **5.22.1** in all four environments. Exact package versions, input hashes,
  checkpoint revisions/URLs, per-model results, target counts, and source fingerprints are in
  `docs/validation.json` under `pretrained_sweep_validation`.

| Result, counted per model/device pair | CPU | RX 7900 XT |
|---|---:|---:|
| Planned and attempted | 20 | 20 |
| Full eager/compiled agreement in both versions | 0 | 0 |
| Stable finding pairs | 19 | 13 |
| Inconclusive pairs | 1 | 7 |
| `comparable_to_eager_against_fp64` in either version | 5 | 9 |
| `eager_control_sensitive` in either version | 4 | 2 |
| `explained_by_discontinuity` in either version | 0 | 0 |
| `discontinuity_present` in either version | 2 | 0 |
| Finding pairs still unexplained | 16 | 9 |

Labels overlap. A pair contributes once to a label present in either version; whole-finding labels
still require every failing target **within that version** to qualify. A pair remains unexplained
if either failing version lacks sufficient evidence. Disagreeing same-input backward checks or
`discontinuity_present` retain suspicion even alongside another label. Inconclusive pairs are
counted separately from unexplained stable findings.

CPU had ten recurring and nine changed failure signatures; AMD had six recurring and seven changed
signatures. **No pair had a passing baseline followed by a stable candidate failure.** A changed
signature is not a newly failing model or a confirmed regression: both environments already failed,
but different gradient targets exceeded tolerance. All completed finding pairs reported gradient
discrepancies; their forward outputs stayed within the configured tolerance.

The incomplete results are material:

- SmolLM2 crossed the 12 GiB host RSS budget during diagnostics in **all four environments**.
  Its saved CPU primary observations contain forward and gradient discrepancies; its AMD primary
  observations contain gradient discrepancies. These remain observations inside inconclusive runs.
- ViT, Swin, ConvNeXt, BERT, GPT-2, and DistilBERT had **unstable repeated eager references** in both
  AMD environments. They stopped before compiler comparison. The cause of that instability was
  not isolated by this experiment.
- Some saved AOT backward graphs could not replay eagerly with their original layouts. Larger
  graphs also exceeded saved-mask snapshot limits, and some AMD mask diagnostics were unstable.
  These gaps prevented a discontinuity explanation; zero qualifying labels does not mean no
  activation-boundary sensitivity exists. Max-pool coverage remains explicitly incomplete.

These counts measure discrepancies and diagnostic coverage. They are **not a false-positive rate**
and establish **no new compiler bug**. All 40 planned model/device pairs have recorded results,
but eight pairs lack a complete usable comparison. The sweep identifies reference stability,
diagnostic memory usage, and saved-graph replay coverage as concrete limits to investigate before
treating these labels as an automatic correctness decision.

Both exported CPU ResNet-18 scripts were independently rerun under supervision, from new working
directories with all `faultline` imports blocked. They reproduced the original failure signatures,
labels, and matched/total target coverage. This validates that representative pretrained export
pair; the other exported scripts were not independently rerun in this experiment.

After the sweep, the preparation loader was corrected to serialize Transformers' unused-checkpoint
key sets as sorted JSON lists. A real cached DistilBERT preparation then produced exactly the same
checkpoint revision, weight hash, input hash, and parameter count as the measured bundle. The
numerical comparison and triage code did not change. Public evidence distinguishes the executed
sweep source fingerprint from the final release-validation fingerprint.

## Version 0.10 GPT-2 investigation

Before changing the implementation, the original CPU GPT-2 sweep was examined target by target
and rerun in fresh workers with the same frozen bundle. The [149-target table](docs/gpt2-accuracy.csv)
lists compiled-versus-fp64 RMSE, eager-versus-fp64 RMSE, ratios, model locations, and primary failure
membership in both environments. **Every fp64 measurement and primary comparison reproduced
exactly** in the fresh 2.15.0 and 2.16.0.dev20261105 workers.

The baseline had 132 failing gradient targets, 129 within the [0.5, 2] fp64 ratio band; the nightly
had 141, with 70 in the band and 71 above it. The union contains 47 attention, 48 LayerNorm, 44 MLP,
and two embedding gradients. The final LayerNorm gradients and the forward output agreed.

| Target | Baseline compiled/eager fp64-error ratio | Nightly ratio | Compiled RMSE growth |
|---|---:|---:|---:|
| Block 1, second LayerNorm weight | 0.662 | 2.602 | 1.849× |
| Block 6, first LayerNorm weight | 2.054 | 6.819 | 2.129× |
| Block 9, second LayerNorm weight | 0.974 | 3.045 | 3.030× |
| Block 9, second LayerNorm bias | 1.074 | 3.261 | 2.657× |

Both environments used Transformers 5.22.1, tokenizers 0.23.3, and identical GPT-2 implementation
code, checkpoint revision, config, factory, frozen token tensors, and weight hashes. Replay loads
the saved token IDs; it does not tokenize again. CPU torchvision differs between environments but
is not used by this GPT-2 factory. PyTorch's eager arithmetic also changes between versions, so the
table includes absolute errors as well as ratios against each version's own eager reference.

The nightly backend ladder passed eager and AOT-eager, then reported 135 failing targets with
decomposition/partitioning and 141 with Inductor. The median ratio across the recorded fp64 targets
was approximately 1.000 and 1.991, respectively. The ladder classified this as `different_failures`,
not a single attributed boundary. In a six-evaluation-budget reduction, five evaluations ran:
backward-only compilation changed the full-model failure signature; forward-only compilation
retained it, but that detached graph's eager reference was nonfinite. The original workload was
retained and independently confirmed. Optional numerical diagnostics were disabled during this
bounded reduction; primary settings and failure predicates were retained. It did not produce a
reduced reproducer that preserves the cross-version accuracy change.

This is a **reproduced accuracy-regression candidate with unresolved attribution**, not a confirmed
compiler defect. The advisory policy was declared before inspecting this target table: candidate
ratio greater than 2, at least twice baseline, on at least 20% of the failing-target union. GPT-2
qualifies on **28/141 targets (19.858%)**, so it does **not** receive the automatic advisory. The
cutoff was not changed to catch this case. The observed shift remains recorded separately.

## Version 0.10 AMD eager stability investigation

BERT was rerun on the RX 7900 XT using its original frozen weights, tokens, and VJP probes. Each
mode used five eager observations with restored model/RNG state. Both modes set explicit BLAS
workspaces; only the deterministic mode required deterministic algorithms and enabled the
associated cuDNN policy. No comparison tolerance was changed.

| Environment | Default eager: failing targets in four repeat comparisons | Deterministic eager |
|---|---|---|
| 2.15.0+rocm10.3.0a20261105 | 29, 31, 35, 28 | All five observations exactly equal |
| 2.16.0a0+rocm10.3.0a20261105 | 30, 29, 32, 32 | All five observations exactly equal |

Neither build raised an unsupported deterministic-operation error. The baseline's largest
per-target eager-repeat RMSE was approximately 7.18e-5; it was zero under the deterministic policy.
This supports using the opt-in policy for this workload, but does not isolate embedding, LayerNorm,
attention, or any particular kernel as the source. `--deterministic` now saves and replays this
execution policy, including standalone exports. The default stability rule is unchanged.

## Version 0.10 pretrained sweeps

The same 20 frozen pretrained workloads from 0.9 were compared again, with gradients, AOT capture,
numerical diagnostics, two repeats, one thread, a 600-second worker timeout, and 12 GiB of host RSS.
CPU used default execution; the RX 7900 XT used **explicit deterministic execution in both
environments**. The CPU versions were 2.15.0+cpu and 2.16.0.dev20261105+cpu. AMD used
2.15.0+rocm10.3.0a20261105 and 2.16.0a0+rocm10.3.0a20261105. The AMD baseline is a release-version
PyTorch build from AMD's nightly SDK packaging, not a stable-channel installation. The complete
package versions, workload identities, source fingerprints, target coverage, and per-model rows
are in `accuracy_triage_validation` in [docs/validation.json](docs/validation.json).

| Environment pair | Models | Full agreement | `accuracy_regression` | Explained | Contrary evidence | Missing evidence | Inconclusive |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU, default execution | 20 | 0 | 0 | 3 | 16 | 0 | 1 |
| AMD, deterministic execution | 20 | 0 | 0 | 10 | 7 | 2 | 1 |

Full agreement, explained, contrary evidence, missing evidence, and inconclusive sum to 20 in
each row. The cross-version accuracy advisory is an independent count, not another partition.
These are **advisory classifications, not a false-positive rate or compiler bug confirmations**.
AMD's execution policy changed, so differences from 0.9 cannot be attributed solely to new labels.
The original 0.9 evidence remains unchanged.

All 40 CPU primary comparisons, including SmolLM2's saved primary observations, exactly match
0.9. ResNet-18, ResNeXt-50, and Swin have complete explanations in both CPU environments. In
particular, ResNet-18's failed optional mask replay no longer overrides its 27/27 eager-control
explanation. Every stable pair has paired fp64 measurements for its full failing-target union.
The only targets meeting the cross-version degradation criterion were CPU BERT (1/178), CPU
GPT-2 (28/141), and AMD Swin (1/127); none reaches the fixed 20% fraction. SmolLM2 has no paired
fp64 measurements, with all 243 CPU and 261 AMD union targets retained as unavailable evidence.

ViT, Swin, ConvNeXt, BERT, GPT-2, and DistilBERT all complete stable primary comparisons in both
AMD environments under the deterministic policy. BERT has forward and gradient discrepancies;
the other five have gradient discrepancies. SmolLM2 still exceeds the host-memory budget during
diagnostics in both versions on both devices. Its primary observations remain available inside
inconclusive outcomes.

The remaining diagnostic limits are visible in the classifications:

- The new same-input backward criterion is not always more permissive than the elementwise
  check. For CPU EfficientNet-B0, the elementwise backward check agrees, but one recorded output
  has compiled RMSE 5.29e-6 versus eager RMSE 1.76e-6 against fp64, a ratio of approximately 3.0.
  It therefore supplies contrary evidence under the fixed factor-of-two rule. That rule is a
  reporting convention, not a calibrated bug detector.
- AMD BERT and DistilBERT remain in the missing-evidence category. Their same-input fp64 backward
  replay reports an unsupported operation, and the input-ULP eager control cannot perturb their
  integer-only requests. In the candidate environment, 173/174 BERT and 47/48 DistilBERT targets
  are in the [0.5, 2] comparability band; each remaining gradient is **closer** to fp64, with ratios
  0.283 and 0.328. The existing whole-finding label requires every target to be inside that band.
  Missing-evidence triage here does not mean the candidate is less accurate.
- Saved-mask limits, unsupported graph replay, and incomplete max-pool selector coverage remain.
  An explained label does not erase these omissions or establish correctness.

The original AMD coordinator stopped while exporting the final SmolLM2 pair: memory-limited
outcomes omitted the new deterministic-policy field, causing the strict export-policy check to
reject them. The supervisor now retains that field for every outcome, including timeouts and
memory limits. Dedicated tests cover both cases. The original outcomes and failed export were
preserved; SmolLM2 was rerun in a separate directory, exported successfully, and reproduced both
original primary comparisons exactly. The AMD table combines the 19 originally committed rows
with this separately recorded final pair. The evidence distinguishes both source fingerprints
and verifies that the only runtime change between them is this supervisor metadata field.
It also records the verified CRLF/LF difference between executed working-tree bytes and Git's
source bytes.

Final local checks passed: 254 portable tests on Windows (one skipped), 12 Linux compiler tests,
23 Linux advisory/policy tests, 31 legacy tests, lint, formatting, and all 14 validation-harness
checks. The harness's real compiler campaign recorded 17 passes and one unconfirmed numerical
candidate. CPU ResNet-18 and deterministic AMD BERT standalone exports were rerun with all
`faultline` imports blocked; both retained their primary signatures, labels, triage, target
coverage, and execution policy. The clean wheel passed an isolated install, CLI, and eager-corpus
check. The staged source, wheel, and source archive were checked against the publication allowlist.

## Version 0.11 CPU nightly bisection

The GPT-2 investigation used the original frozen pretrained workload and fixed VJP probes. A
Conv1D/addmm workload with input shape 2×64×768 and a single full GPT-2 block were tried first,
with three fresh baseline/candidate worker pairs each. Median compiled/fp64 gradient-error growth
was 1.000× for Conv1D and 0.995× for the block. Prefixes of two, four, and eight blocks measured
1.009×, 0.990×, and 1.007× in one screening pair each. None reproduced the full-model shift. A
twelve-block prefix attempt lost its worker and is inconclusive. The full frozen GPT-2 model
therefore remained the smallest tested workload that reliably reproduced the shift; no minimal
reproducer is claimed.

The bisection policy was fixed before searching: a threshold of **1.4×** median gradient-error
growth against the per-target median of three fresh PyTorch 2.15.0 CPU observations. Each nightly
ran in three fresh workers, with two internal repeats, one thread, a 900-second timeout, and a
12 GiB host-memory limit. All three scores had to meet the threshold to count as shifted; an
unavailable or mixed result could not count as good. The 147 paired gradient targets had finite,
strictly positive compiled/fp64 errors in every baseline observation. This excludes one exact
gradient and differs from the all-value-target distributional advisory below.

The original [GPT-2 table](docs/gpt2-accuracy.csv) contains two different growth measures.
Its 137 increases among 148 finite ratio-growth entries refer to compiled/eager error ratios
(including the model output). Absolute compiled/fp64 error increases on 144 of 148 gradients,
or 145 of 149 value targets including the output. The 0.11 distributional advisory uses the
latter measure; the bisection uses the 147 gradients with strictly positive baseline errors.

| CPU nightly | Fresh observations | Median compiled/fp64 error growth | Advisory |
|---|---:|---:|---|
| 2026-10-05 | 3 | 1.000× | Not shifted |
| 2026-10-20 | 3 | 1.000× | Not shifted |
| 2026-10-27 | 3 | 1.000× | Not shifted |
| 2026-10-31 | 3 | 1.000× | Not shifted |
| 2026-11-02 | 3 | 1.000× | Not shifted |
| **2026-11-04** | **3 + 3 independent confirmations** | **1.000×** | **Last good under this predicate** |
| **2026-11-05** | **3 + 3 independent confirmations** | **1.710×** | **First bad under this predicate** |

Every nightly observation retained the primary `gradient_mismatch` status. All three scores
within each date were identical. The search used 31 compatible Python 3.12 Linux CPU wheels from
the [official nightly index](https://download.pytorch.org/whl/nightly/cpu/torch/), starting at the
[2.15 branch cut](https://dev-discuss.pytorch.org/t/pytorch-release-2-15-branch-cut-is-complete/3461).
November 3 had no compatible wheel and remains explicitly missing. Binary search assumes a
monotone transition; the sampled dates do not rule out an earlier temporary shift. The
[per-observation table](docs/bisection-011.csv) retains dates, primary statuses, wheel hashes,
source SHAs, and scores.

The last-good and first-bad wheel snapshot SHAs are `7f24a19c0967db04f2d5ba44c1e56eca489932ec`
and `084f8661f5bb5de149860f74d82f5313766f4256`. Their release commit messages identify underlying
main SHAs `2e10e9e8998bf3d420dac3530dfff53bf784c880` and
`b7b26ca70e5b0d33b5e4f43ec213b2e6810c14cb`: a
[78-commit source interval](https://github.com/pytorch/pytorch/compare/2e10e9e8998bf3d420dac3530dfff53bf784c880...b7b26ca70e5b0d33b5e4f43ec213b2e6810c14cb).
Both boundary builds use oneDNN 3.14.1, so a oneDNN version change alone does not account for
the adjacent-nightly boundary.

A post-bisection control tested the interval's
[wide-broadcast materialization change](https://github.com/pytorch/pytorch/commit/1933a0a9ee51f2b03ad3cababd727b442d23630e).
Setting `torch._inductor.config.realize_broadcast_reuse_threshold = 10**12` in the November 5
process disabled its new trigger without editing installed packages. Two fresh runs retained
exactly the same 0.773967 median log2 growth (1.710×) as two default-setting runs. This intervention
does not support attributing the shift to that heuristic. The date boundary is established;
the individual causal commit remains unresolved. An accuracy shift may reflect an intended
algorithm or code-generation trade-off. No compiler defect is claimed and no upstream issue
was filed.

The reusable [`faultline bisect` workflow](docs/bisect.md) installs each nightly in a separate
environment, supervises installation and execution, preserves logs and partial attempts, and
checks source, workload, settings, dependency locks, environment identity, and committed evidence
before resuming. Missing wheels and failed installations remain separate from accuracy results.
The experiment used an earlier frozen 0.11 source snapshot; subsequent changes added installation
retry and interrupted-initialization recovery without changing the numerical predicate.

Three setup attempts were preserved before the completed campaign: a WSL clock correction made
the old process-creation-time watchdog misidentify its live parent; the system lacked `ensurepip`;
and a full environment freeze included an unrelated local package unavailable from PyPI. Linux
worker identity now uses kernel process-start ticks, with PID-reuse and clock-correction tests.
The successful campaign pinned the model dependency closure and installed each CPU wheel
separately. No baseline model environment was upgraded to make the search work.

## Version 0.11 advisory policy and retrospective application

The [fixed policy](docs/v011-policy.json) was motivated by GPT-2's 0.10 result. `accuracy_shift`
requires at least eight paired value targets, 80% fp64 coverage, 75% worsening, median compiled
error growth of at least 1.25×, and a one-sided paired sign-test p-value at most 0.01. It examines
all output and gradient values, including those within the primary tolerance. Ties remain in
the fraction and median and leave only the sign-test trials. Correlated targets make the p-value
descriptive; it is not a calibrated probability that the model or compiler is incorrect.

The unchanged rule was applied retrospectively to all 40 original 0.10 CPU/AMD pairs. The
[complete per-pair table](docs/accuracy-shift-010.csv) includes unavailable pairs, coverage,
worsening/improving/tied counts, median log2 growth, and p-values. **CPU GPT-2 and CPU BERT flag;
no AMD pair flags.** GPT-2 has the largest median growth among the 38 available pairs, but it is
not the only flagged model. This ranking does not establish a statistically calibrated outlier.

| Distribution of per-pair median log2 growth | Planned | Available | Minimum | 25th percentile | Median | 75th percentile | Maximum |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU | 20 | 19 | -0.006229 | -4.13e-13 | 0 | 0 | 0.770341 |
| AMD deterministic | 20 | 19 | -0.065776 | 0 | 0 | 0 | 0.264394 |

Percentiles use inclusive linear interpolation over available pair medians. Unavailable pairs
are reported separately rather than assigned zero growth.

| 0.10 pair | Paired targets | Worsened | Median growth | Sign-test p | `accuracy_shift` |
|---|---:|---:|---:|---:|---|
| CPU GPT-2 | 149/149 | 145/149 | 1.706× | 1.51e-39 | Yes |
| CPU BERT | 198/198 | 179/198 | 1.308× | 7.74e-35 | Yes |
| CPU DistilBERT | 101/101 | 84/101 | 1.149× | 1.30e-12 | No: growth below threshold |
| AMD EfficientNet-B1 | 303/303 | 241/303 | 1.201× | 9.15e-27 | No: growth below threshold |
| AMD GPT-2 | 149/149 | 0/149 | 1.000× | 1 | No |
| AMD BERT | 198/198 | 0/198 | 1.000× | 1 | No |

This application is exploratory, not held-out validation. SmolLM2 lacks paired fp64 evidence on
both devices and remains in the denominator. The old fraction-based `accuracy_regression` rule
is unchanged, including its zero flags on these 40 pairs. In two actual same-version comparisons
between the three fresh 2.15.0 GPT-2 baseline observations, all 149 value errors tied: median log2
growth 0, p=1, and no shift flag. A planted uniform 1.5× degradation flags in the behavior tests.

The numerical floor uses **c=32**, chosen before the new sweeps:
`max(eager_rmse, eager_control_rmse_vs_same_fp64, 32 * 2^-24 * RMS(fp64_reference))`.
Compiled RMSE must exceed twice this floor to supply positive fp64 contrary evidence. This is an
engineering allowance, not a fitted error bound. Unsupported controls are recorded as unavailable;
same-input backward currently has no alternate backward control. The old errors, ratios, and
ratio-band labels remain in the evidence for comparison.

Incomplete comparability or eager-control coverage now means missing evidence, with matched/total
counts. Contrary evidence requires a completed beyond-floor fp64 check, a same-input backward
check beyond that floor, or a mapped failing gradient not downstream of observed flips. Tests
cover all three categories and partial explanations. These changes affect advisory reporting
only. Primary statuses, tolerances, failure signatures, reducer predicates, and existing command
exit codes are unchanged. The 0.10 artifacts and results above are preserved.

The repeated CPU EfficientNet-B0 candidate illustrates both effects of the floor. Its same-input
backward output 9 retains eager RMSE 1.76419e-6, compiled RMSE 5.29149e-6, and the old ratio 2.99938.
Reference RMS 2.97174 gives a floor of 5.66813e-6 and a cutoff of 1.13363e-5, so that target no
longer supplies contrary evidence. Of fourteen backward outputs beyond twice eager error in the
old candidate check, thirteen are now within twice their floor. Output 19 still exceeds its floor:
compiled RMSE 9.49417e-6 versus a cutoff of 7.62523e-6. The whole finding therefore remains
`unexplained_contrary_evidence`. The errors and legacy ratios are unchanged; the declared c=32
is not adjusted to clear the remaining target. These are AOT backward output positions, not
original-model parameter indices. Full values are retained in `noise_floor_examples` in
[validation.json](docs/validation.json).

## Version 0.11 held-out and original-model sweeps

The held-out catalog adds RegNet-Y-400MF, DenseNet-121, ShuffleNet-V2-x1.0, SqueezeNet-1.1,
MNASNet-1.0, T5-small's encoder, RoBERTa-base, OPT-125M, and Pythia-160M. All nine use pretrained
weights and the same real-input policy as the original catalog. The original twenty frozen
workloads are reused. Model identities, weights, inputs, and checkpoint revisions are retained.

The first preparation completed eight models and rejected OPT's missing embedding weight.
The existing safetensors conversion stores that tied weight under `lm_head.weight`; direct
base-model loading would initialize an embedding randomly. Preparation now loads the declared
causal-language-model weight tie and then extracts its base model, with no missing or mismatched
weights permitted. The repaired OPT preparation used the same pinned checkpoint and byte-identical
input sources. Its successful bundle was combined with the eight completed preparations in a
new directory. The original failed attempt remains preserved; no model was substituted.

CPU uses PyTorch 2.15.0+cpu versus 2.16.0.dev20261105+cpu. AMD uses the RX 7900 XT with
2.15.0+rocm10.3.0a20261105 versus 2.16.0a0+rocm10.3.0a20261105 and explicit deterministic
execution. These are the same environments as 0.10. Both sets use gradients, AOT capture,
numerical diagnostics, two repeats, one thread, 600 seconds and 12 GiB of host RSS per worker.
The thresholds above were fixed before these sweeps and are not adjusted for their outcomes.

An initial CPU campaign completed seven pairs before being stopped for a memory fix. RoBERTa
exceeded 12 GiB during diagnostics in both environments. The eager-control handoff retained an
unneeded model-state snapshot alongside the output/gradient values used by fp64 comparison.
It now retains only those value tensors, without copying them or changing their layouts. A
behavior test verifies identical advisory fields with the full and compact observations. All
completed results and the interrupted eighth attempt were preserved. The full campaign was
restarted in new directories with the same thresholds, inputs, and memory budget.

Both RoBERTa CPU environments then completed with full fp64 coverage, peaking at 12,104 and
12,070 MiB, below the unchanged 12,288 MiB limit. The first six completed model pairs reproduced
their primary comparisons, fp64 measurements, and numerical-context records exactly after this
change. The old RoBERTa runs were stopped at the memory limit, so their unrestricted peak memory
is unknown; no percentage memory reduction is inferred from those truncated observations.

The completed rerun keeps every selected model in the denominator:

| Catalog / device | Models | Fully agree | Explained | Contrary evidence | Missing evidence | Inconclusive | Old ratio flag | `accuracy_shift` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Held-out / AMD deterministic | 9 | 1 | 6 | 2 | 0 | 0 | 0 | 0 |
| Held-out / CPU | 9 | 0 | 5 | 4 | 0 | 0 | 1 | 1 |
| Original / AMD deterministic | 20 | 0 | 16 | 3 | 0 | 1 | 0 | 0 |
| Original / CPU | 20 | 0 | 13 | 6 | 0 | 1 | 0 | 2 |

The five outcome columns partition each catalog/device set. The last two advisory columns
are independent flags and are not additional partition categories. Explained means the
recorded diagnostic criteria are met, not independently adjudicated correctness. These
counts are not a false-positive rate.

SmolLM2 exceeds the unchanged 12 GiB host-memory budget during diagnostics in both versions
on both devices. Its two model/device pairs stay inconclusive, with completed primary
observations retained and all 273 value targets unavailable to the distributional advisory.
The other 56 pairs have full paired fp64 coverage. No model was removed from the denominator.

The [complete 58-pair table](docs/accuracy-shift-011.csv) retains every model, primary status,
triage category, coverage, worsening/improving/tied counts, median log2 growth, and sign-test
p-value. The distribution below summarizes per-pair median log2 growth, including unflagged pairs:

| Catalog / device | Planned | Available | Minimum | 25th percentile | Median | 75th percentile | Maximum |
|---|---:|---:|---:|---:|---:|---:|---:|
| Held-out / AMD deterministic | 9 | 9 | 0 | 0 | 0 | 0 | 0.124031 |
| Held-out / CPU | 9 | 9 | -0.00550835 | 0 | 0 | 0 | 0.367669 |
| Original / AMD deterministic | 20 | 19 | -0.0657764 | 0 | 0 | 0 | 0.264394 |
| Original / CPU | 20 | 19 | -0.00622877 | -4.13003e-13 | 0 | 0 | 0.770341 |

The unchanged distributional rule flags these pairs:

| Catalog / device / model | Paired / total | Worsened / paired | Median growth | Sign-test p |
|---|---:|---:|---:|---:|
| Held-out / CPU / Pythia-160M | 148/148 | 132/148 | 1.290× | 7.45e-25 |
| Original / CPU / BERT | 198/198 | 179/198 | 1.308× | 7.74e-35 |
| Original / CPU / GPT-2 | 149/149 | 145/149 | 1.706× | 1.51e-39 |

This is completed held-out application of the predeclared engineering rule: nine new
pretrained models on both devices, with no threshold changes after seeing their results.
It validates application and repeatability on these workloads, not statistical calibration
or broad production-model coverage. The two real input examples per model remain a limited input set.

CPU Pythia is also reproduced independently through both exported scripts with all Faultline
imports blocked. Both exports retain the original primary outcome, every fp64 error, labels,
and coverage. Recomputing `accuracy_shift` from those fresh observations reproduces every
field of the sweep advisory. CPU OPT and deterministic AMD ShuffleNet exports also pass
independent execution with matching evidence and execution policy.

Pythia's candidate also meets the per-version fp64 comparability criterion. That does not
contradict its cross-version shift: error can grow by 1.29× while remaining below twice the
candidate environment's noise floor. The two advisories answer different questions and neither
changes its primary numerical mismatch.

RegNet illustrates why the two accuracy advisories are separate. On CPU, the old ratio rule
flags 185/208 failing targets (88.9%), while all 227 paired compiled/fp64 errors are unchanged.
Its eager denominator changed. The new rule records 227 ties, median log2 growth 0, p=1, and
no accuracy shift. The old flag remains visible; neither rule modifies the primary result.

AMD BERT and DistilBERT's previously missing-evidence pairs now have complete fp64 explanations.
Candidate coverage is 174/174 and 48/48 failing targets, respectively. Results closer to fp64
qualify under the new upper-bound criterion; their old ratios of 0.283 and 0.328 remain recorded.
Unsupported same-input backward replay and unavailable integer-input ULP controls still appear
as diagnostic gaps. These explanation labels do not erase the primary discrepancies.

All 80 original-catalog primary observations (20 models × two devices × two environments)
reproduce the saved 0.10 primary status and comparison exactly. All 76 stored failure
fingerprints match. The remaining four memory-limit checkpoints precede fingerprint
serialization; applying the unchanged fingerprint function with the recorded native Inductor
and gradient settings gives identical fingerprints, with those derivations marked in the evidence.
All 76 completed fp64 observations also retain identical eager/compiled error statistics,
ratios, reference dtypes, exact-result flags, and legacy labels. New advisory labels do not
rewrite the old evidence. Full environments and executed source hashes
are recorded in [validation.json](docs/validation.json); editable package metadata in those
unchanged environments can retain the prior version, so the source hash identifies the harness.

## Version 0.11 executed checks

| Check | Result |
|---|---|
| Windows portable tests, Python 3.12.10 / PyTorch 2.15.0+cpu | 282 passed, 1 skipped; 12 compiler tests deselected |
| Linux CPU Inductor tests, Python 3.12.3 / PyTorch 2.15.0+cpu | 12 passed |
| Compiler validation campaign | All 14 checks passed; real-compiler corpus: 17 passes and 1 unconfirmed numerical discrepancy |
| Legacy tests | 31 passed with asyncio auto mode; legacy source was unchanged by later Faultline-only edits |
| Ruff lint and requested formatting checks | Passed; 80 files formatted |
| Independent standalone exports with Faultline imports blocked | OPT CPU; both Pythia CPU environments; deterministic AMD ShuffleNet |

The behavior tests cover uniform 1.5× degradation, unchanged-version results, mixed targets,
noise-floor terms, incomplete versus contrary evidence, backward/selector disagreement,
environment installation failures, interrupted initialization, resume compatibility, and
independent bisection confirmation. Neither the numerical policy nor the execution budgets
were retuned after the held-out results.

The final executed harness source hash is
`9eb40b2a72a15d214b5cac7bf45594565c781116374f67d62bcbbc9c7d78dc2b`.
The Git-normalized source hash is
`94152d83a320d1c9da32c0e5658f624462d645573fe769eaeeb7a1eb2cebedf0`.
The difference is verified CRLF/LF normalization only, with affected filenames recorded in
the evidence. Published CSV hashes refer to UTF-8/LF bytes. No primary comparison, tolerance,
failure-signature function, reducer predicate, or existing command exit code was changed.

The release allowlist audit checked 106 tracked files, 57 wheel members, and 109 source-archive
members. Private guides, notes, model weights, environments, and local experiment directories
are excluded from those distributions. An isolated import from the installed 0.11.0 wheel
matches the Git source hash above. Its version command, deterministic model options, bisection
retry options, held-out sweep options, and repeated eager corpus all pass. The evidence audit
also verifies the published CSV hashes, every previous validation record, the unchanged original
GPT-2 CSV bytes, and the unchanged primary/signature/reducer files.

## Version 0.12 CPU accuracy investigation

This investigation retains the 0.11 bisection predicate and frozen GPT-2 reference vector.
Every environment receives three fresh workers, each with two internal repeats, one CPU thread,
900 seconds and 12 GiB of host RSS. The advisory threshold remains 1.4× median compiled/fp64
gradient-error growth. The experimental harness is the released 0.11 source snapshot with hash
`9eb40b2a72a15d214b5cac7bf45594565c781116374f67d62bcbbc9c7d78dc2b`.
New analysis utilities do not participate in the numerical predicate. All earlier evidence above
and in `validation.json` is preserved. This investigation runs on CPU; the 0.11 AMD results
remain the AMD evidence.

### Shared nightly boundary

BERT and Pythia use their full frozen workloads and three fresh 2.15.0 reference workers each.
GPT-2 reuses its original locked 147-target baseline. The boundary-wheel observations are:

| Workload | Positive baseline gradients | November 4 growth | November 5 growth | Fixed 1.4× classification |
|---|---:|---:|---:|---|
| GPT-2 | 147 | 1.000× | 1.710× | Not shifted → shifted |
| BERT | 196 | 1.000× | 1.309× | Not shifted → not shifted |
| Pythia-160M | 146 | 1.000× | 1.292× | Not shifted → not shifted |

All three scores within each model/date agree exactly. BERT and Pythia show increased error at
the same adjacent-nightly boundary, but **neither flips the unchanged binary predicate**.
Their separate distributional `accuracy_shift` labels use the existing 1.25× rule; that rule
is not substituted into the bisection. BERT retains primary `gradient_mismatch`; Pythia retains
primary `numerical_mismatch`; GPT-2 retains primary `gradient_mismatch` in both wheels.

### Generated code and interventions

Full GPT-2 forward/backward code was captured in three fresh workers per boundary wheel with
`TORCH_LOGS=output_code`, `TORCH_COMPILE_DEBUG=1`, and separate compiler caches. Instrumented
workers reproduce 0.11's scores exactly. Both normalized emitted modules, both pre/post-fusion
IRs, and the transformed FX syntax are identical. The raw FX comments differ in installation
paths. The [small generated-code and configuration evidence](docs/accuracy-investigation.md#boundary-wheel-code-comparison)
records kernel definitions, call sites, fusion groups, reduction/type markers, and hashes;
full debug dumps remain local. Empty normalized diffs denote observed equality, not absent dumps.

Both versions emit 48 forward `addmm` calls, 96 backward `mm` calls, and 12 external CPU
flash-attention calls in each stage. No changed fusion, reduction split, accumulation type,
or matrix-call choice was found in the emitted code. The 1,108 default decomposition identity
records match, including all 599 available Python source hashes; 509 entries lack inspectable
Python source. Of 671/672 Inductor defaults, the only substantive addition is the broadcast
reuse threshold. Both actual wheels report MKL 2024.2 and the same oneDNN 3.14.1 revision.
Equal generated code does not establish equality of linked ATen kernels or included headers.

| November 5 GPT-2 intervention | Fresh workers | Growth in every worker | Fixed predicate |
|---|---:|---:|---|
| Default, debug capture enabled | 3 | 1.710× | Shifted |
| Broadcast reuse threshold set to `10**12` | 3 | 1.710× | Shifted |
| CPU flash attention disabled | 3 | 0.978× | Not shifted |

The broadcast control's emitted code is identical to the default. The flash-disabled captures
contain no CPU flash-attention calls: attention instead adds 24 forward and 48 backward `bmm`
call sites. That intervention changes eager, compiled, and fp64 backend execution together;
it implicates the attention path but does not isolate one elementary operation or commit.
All six controlled runs retain primary `gradient_mismatch`. Their standalone exit code of 1
denotes that retained mismatch, not a failed experiment or a changed primary policy.

Source inspection of the 78-commit interval identifies the CPU `exp_u20` change in
[`2450619a`](https://github.com/pytorch/pytorch/commit/2450619a4c8a11b6f652154d12173794ebe181d5)
as a candidate outside the generated-code diff. CPU float32 flash-attention forward uses this
AVX2/AVX512 exponential; backward recomputes probabilities using a different exponential path.
The upstream change improves local approximation and exceptional-value handling according to
its own tests. That does not guarantee monotonic improvement of a complete model's gradients.
The [candidate inspection table](docs/accuracy-investigation.md#candidate-inspection) records
the other reviewed changes and the negative evidence for them.

The [492-target gradient table](docs/gradient-scales-012.csv) retains absolute eager/compiled
errors and fp64 gradient magnitudes for all three models, including exact gradients. With
`u = 2^-24`, median compiled `RMSE / (u * RMS(reference))` changes from 31.889 to 58.763 for
GPT-2, 58.298 to 76.802 for BERT, and 18,636.454 to 24,044.269 for Pythia. These relative scales
are not elementwise ULP bounds. Near-zero reference gradients can inflate them, so the absolute
values remain available. No model-quality conclusion or compiler-defect claim follows from them.

### Bounded depth checks

The earlier twelve-block prefix was retried without changing its frozen inputs or weights.
Two further variants set the full GPT-2 configuration to ten or eleven layers, retaining the
original embeddings, first N blocks, and final LayerNorm weights without retraining. Each
variant has three fresh 2.15.0 baseline workers and three November 5 workers, with the same
1.4× predicate and execution settings. Its reference vector is constructed from its own
baseline gradients; gradients from different variants are not pooled.

| Variant | Positive baseline gradients | Growth in every candidate worker | Advisory |
|---|---:|---:|---|
| Full model configured for 10 layers | 123 | 1.006× | Not shifted |
| Full model configured for 11 layers | 135 | 0.990× | Not shifted |
| 12-block prefix with frozen first-block inputs and final LayerNorm | 146 | 1.712× | Shifted |
| Original full 12-layer GPT-2 | 147 | 1.710× | Shifted |

The prefix includes an input gradient and omits the embedding parameters; its fp64 reference
starts from the saved, already-rounded activations. It is not the same reference computation
as the full model. The prefix nevertheless reproduces the shift, so embedding execution is
not required to observe it. On these inputs, the effect appears when the final pretrained
block is retained. These bounded variants do not prove a universal model-size threshold or
distinguish conditioning from cancellation of numerical errors. They are manually constructed
investigations, not accepted reducer outputs or a claim of minimality. The previously lost
0.11 worker remains recorded as inconclusive; its evidence was not overwritten.

### Source-commit isolation

Two matching local PyTorch source builds bracket the shift at the single commit
[`2450619a4c8a11b6f652154d12173794ebe181d5`](https://github.com/pytorch/pytorch/commit/2450619a4c8a11b6f652154d12173794ebe181d5),
which changes CPU AVX2/AVX512 `exp_u20` and its tests. Its direct parent is
`eb48c7e2d9fd6b16c0a7291703c94b16fb33cc42`. The original full frozen GPT-2 workload and locked
0.11 reference were used without changing the three-worker predicate.

| Source build | Fresh workers | Median log2 growth in every worker | Growth | Advisory |
|---|---:|---:|---:|---|
| Direct parent `eb48c7e2` | 3 | 0 | 1.000× | Not shifted |
| Child `2450619a` | 3 | 0.7739668316531798 | 1.710× | Shifted |

Every worker retains primary `gradient_mismatch`. In addition to matching the aggregate score,
all 149 complete fp64 evidence rows (148 gradients and one output) match the respective
November 4/5 nightly evidence exactly in all six workers. The source pair therefore attributes
this workload's shift to the single commit; further binary subdivision of the 78-commit interval
was unnecessary. BERT/Pythia were not run on these source builds, so their shared nightly
boundary does not establish commit-level attribution for them.

Builds ran under WSL Ubuntu 24.04 on a Ryzen 7 7700X with Python 3.12.3, GCC 13.3, MKL 2024.2,
oneDNN 3.14.1 and AVX512. Both used CPU-only settings, disabled distributed/test/Kineto builds,
OpenMP MKL threading, and two build jobs. The common incremental build directory was reused
between the child and parent, but the wheels and installed environments are separate. Logs
verify recompilation of both AVX2 and AVX512 attention objects; normalized build configurations
and the complete compilation-command hashes agree. The successful child build took 3,829 s
with 5,499 MiB peak process-tree RSS; the incremental parent build took 818 s with 2,995 MiB.
These are build observations, not model or kernel performance benchmarks.

All unsuccessful build attempts remain recorded: two invalid/cached MKL-threading configurations,
one intentional interruption to change build parallelism, and one attempt ending in a WSL
`psutil` pidfd `EINVAL` cleanup error without a serialized completion result. The successful
retry used a local build-helper workaround that verified the exited parent had no live owned
descendants before accepting that specific cleanup condition. Faultline's production watchdog
was not changed. The successful source predicates used the unmodified 0.11 worker supervisor.

The commit fixes exceptional-value handling and changes the normal exponential approximation;
this investigation does not show that subnormal inputs triggered the model result. Its local
accuracy improvements can coexist with a larger end-to-end gradient error. The exact mechanism
of the twelve-block amplification remains unresolved. This is an attributed accuracy shift,
possibly an intended numerical trade-off, not a compiler-bug claim.

The [upstream report draft](docs/upstream-report-draft.md) has not been filed. The
[54-observation table](docs/accuracy-investigation-012.csv) includes every baseline, boundary,
intervention, depth, and source worker from this investigation; the additional standalone
script checks are recorded separately. `accuracy_investigation_validation` in
`docs/validation.json` retains settings, source/environment identities, build attempts,
artifact hashes, and per-worker scores. Large raw dumps, checkpoints, wheels, and environments
remain local. All 19 prior validation entries and the earlier result text are preserved.

### Executed release checks

| Check | Result |
|---|---|
| Windows portable suite, Python 3.12.10 / PyTorch 2.15.0+cpu | 290 passed, 1 skipped; 12 compiler tests deselected |
| Linux CPU Inductor suite, Python 3.12.3 / PyTorch 2.15.0+cpu | 12 passed |
| Compiler validation campaign | All 14 checks passed; 17 corpus/generated passes and 1 unconfirmed numerical discrepancy |
| Legacy tests | 31 passed with asyncio auto mode; legacy code unchanged by subsequent additions |
| Ruff lint and requested formatting | Passed; 84 files in the requested formatting scope |
| Independent PyTorch/transformers-only GPT-2 script, frozen assets | All 148 eager and compiled gradient RMSE values exactly match the harness in stable, November 4, and November 5 environments |
| Same script, normal pinned-checkpoint loading on November 5 | All gradient evidence exactly matches the frozen-assets execution |
| Publication allowlist and content audit | 119 public files; no internal instructions, personal paths, notes, raw runs, or environments included |
| Built distribution audit | 58 wheel members and 122 source-archive members checked |
| Independent installed-wheel smoke checks | Version, imported-package identity, eager corpus, and comparison of the actual boundary debug dumps passed |

The standalone script uses one fresh process per environment, not the three-worker bisection
protocol. Its 147-positive-gradient comparison nevertheless reproduces the exact 0 and
0.7739668316531798 log2-growth scores. These are independently executed measurements; the script
does not read Faultline outcome files or import Faultline. Each execution uses a new compiler
cache. It verifies frozen asset checksums, token values, dtypes, strides, and storage offsets
before measurement. Behavior tests reject corrupted assets and equal-valued inputs with a
different dtype or layout.
The normal loading path also reproduces the same comparison and gradient evidence without
reading the local frozen-assets directory. All four independent script runs completed within
the recorded supervision budgets.

Generated-code tests cover meaningful arithmetic/type changes, repeated captures, incomplete
artifacts, changed IR with identical emitted code, and FX comment differences versus arithmetic
differences. The release adds inspection utilities and a standalone measurement example; all
50 existing implementation files apart from the separately updated package-version file are
unchanged. Primary statuses, tolerances, failure signatures, reducer predicates, and existing
command exit codes are preserved.

The distribution checks build from the audited Git index and install the wheel plus its
dependencies into an independent virtual environment. The imported package location and
Git-normalized source hash are verified; the installed code comparator reproduces the same
two equal boundary modules. The packaged source hash is
`463b89eab2bfebd0106fe98ca72c1828e2506fbe48e09a15bb66851060ecc907`.
This normalizes repository line endings through Git; the raw Windows/compiler-test snapshot
hashes are retained separately. Build and installation logs remain local.

## Version 0.13 attention investigation

The existing parent `eb48c7e2d9fd6b16c0a7291703c94b16fb33cc42` and child
`2450619a4c8a11b6f652154d12173794ebe181d5` source installations are reused without rebuilding
PyTorch or changing their dependencies. The [policy](docs/v013-policy.json) and
[additional kernel controls](docs/v013-kernel-controls.json) fix the comparisons before the
corresponding experiments. All outputs are advisory. Primary statuses, tolerances, signatures,
reducer predicates, and existing exit codes remain unchanged; no upstream report was filed.

### Attribution now covers three full models

BERT and Pythia-160M use their original frozen workloads and locked 0.12 reference vectors.
Each source build receives three fresh workers per model, two internal repeats, one CPU thread,
900 seconds and a 12 GiB host-memory limit. The original 1.4× predicate is unchanged.

| Model | Parent growth | Child growth | Exact corresponding nightly evidence rows, every worker | Advisory |
|---|---:|---:|---:|---|
| BERT | 1.000× | 1.309× | 198/198 | Not shifted on both builds |
| Pythia-160M | 1.000× | 1.292× | 148/148 | Not shifted on both builds |

All twelve workers reproduce the complete per-target fp64 evidence and growth scores exactly
from the respective November 4/5 nightlies. The counts include gradients and the output.
BERT retains `gradient_mismatch`; Pythia retains `numerical_mismatch`. Together with the
0.12 GPT-2 source comparison, this attributes all three full-model accuracy shifts to the
same commit. BERT/Pythia still do not cross the fixed 1.4× predicate. The
[per-worker table](docs/attribution-013.csv) records unrounded values and artifact hashes.

### Exact runtime capture and standalone kernel null result

Actual CPU flash-attention forward/backward arguments were captured at blocks 11 and 12 of
the unchanged twelve-block prefix, in eager and compiled execution on both builds. Compiled
forward/backward were warmed before observing the backend callable. All four capture workers
match all 148 uninstrumented output/gradient control tensors exactly in both internal repeats;
each observes twelve forward and twelve backward attention calls. Tensor strides, storage
offsets, and shared storage are preserved in local snapshots.

All q/k/v and grad_out tensors have shape `[2,12,64,64]`, dtype fp32, scale 0.125, causal mode,
zero dropout and no additive mask. Eager/compiled q/k/v strides agree at `[147456,64,2304,1]`,
with offsets 0/768/1536. Grad_out strides agree at `[49152,64,768,1]`. Values differ between
the full-model paths: block-12 q/k/v maximum differences reach 5.25e-6 on the parent and
7.63e-6 on the child; grad_out reaches 1.48e-5 and 3.24e-5. Backward logsumexp has shape
`[2,12,64]` and differs in layout: eager `[768,1,12]`, compiled `[768,64,1]`. The
[capture inventory](docs/attention-capture-013.json) retains all metadata and comparisons.

Each of the eight frozen captures is run unchanged on both builds. Three fresh workers per
build each execute all eight cases, with two internal repeats. Reference out/logsumexp/dq/dk/dv
come from explicit fp64 scaled/masked QK, softmax, PV and VJP. The advisory threshold is 1.4×
median log2-gradient-error growth over dq/dk/dv, with all three workers required to agree.

| Captured inputs | Block 11 growth | Block 12 growth |
|---|---:|---:|
| Parent eager | 0.997800× | 0.994052× |
| Parent compiled | 0.994012× | 1.004449× |
| Child eager | 1.008243× | 0.995670× |
| Child compiled | 0.984046× | 0.999003× |

All eight cases are `not_shifted`, with identical measurements in all three workers. The
matching-build standalone replay reproduces every captured operator result exactly. The
fp64 reference values agree exactly across builds. Holding the same captured out/logsumexp
fixed makes backward dq/dk/dv bit-identical across builds. Swapping eager/compiled layouts
while keeping the logical inputs fixed changes only logsumexp's layout and leaves all five
operator results exactly equal in all 24 additional block-12 evaluations across six workers.

Thus the isolated kernel does **not** reproduce the 1.710× full-model shift. The
[PyTorch-only measurement example](examples/sdpa_accuracy.py) is not presented as a reproducer
of that shift. The condition for promoting a minimal kernel reproducer was not met; the full
GPT-2 workload remains the report's reproducer. The full-model compiled/eager calling difference,
including changed activations and upstream gradients, remains open. The
[40-target kernel table](docs/attention-kernel-013.csv) retains absolute errors and noise-floor context.

### No subnormal-score exposure

Each capture has 49,920 finite permitted score-minus-rowmax entries and 48,384 masked entries.
None of the finite entries is below -87.3 or -103.97. The minimum across block-11 captures is
-17.895388; for block 12 it is -13.260713. The
[192-row batch/head histogram table](docs/attention-heads-013.csv) retains every bin and threshold
count. Masked negative infinities are separate from finite subnormal-range candidates.

The [top-row data](docs/attention-rows-013.csv) rank the ten largest finite child/parent error
ratios for every capture and dq/dk/dv target, retaining absolute errors. Dq uses query-row exposure;
dk/dv use key-column exposure. All [exposure/association counts](docs/attention-association-013.csv)
are zero, including top-ten overlaps. There is no exposed group from which to estimate an
association. These captures do not support subnormal attention scores as the explanation.

### Exponential and probability-consistency measurements

Small AVX512 adapters were compiled against the existing installations; PyTorch was not rebuilt.
The first adapter linked but could not load a private CPUBlas symbol. That attempt is retained;
the replacement calls the exported SGEMM entry point used by the inspected source path and
rejects unresolved symbols at link time. Synthetic contiguous and packed-offset checks reproduce
all 3,072 logsumexp rows per build before the captured-model measurements.

The fixed [-104,0] sample contains 1,048,577 fp32 points. Additional grids cover [-110,-86] and
[0,89], plus explicit boundary/NaN/infinity inputs. On the main sample, the child's maximum
observed exp_u20 error is 1.957 ULP for normal outputs and 1.403 ULP for subnormals. The parent's
flush-to-zero interval gives much larger errors, including some normal outputs; its normal-output
p99 is 3.979 ULP versus the child's 1.653. Regular vector exp, used in backward, has identical
sampled accuracy across builds (normal maximum 0.967 ULP). The child also propagates NaN where
the parent returns infinity. These are sampled measurements, not exhaustive error bounds.

Median-of-three-worker-median times on the fixed array are 0.149764 → 0.129717 ms for exp_u20
and 0.267188 → 0.262903 ms for regular exp. Each fresh worker uses five warmups and eleven
alternating timed calls; model workers are not running concurrently. The
[accuracy](docs/exp-accuracy-013.csv) and [timing](docs/exp-timing-013.csv) tables retain full
range/category counts and timing spread. This is a Zen 4/WSL microbenchmark, not model throughput.

The source-matched reconstruction preserves native SGEMM shapes/strides, forward scaling,
backward GEMM alpha, vector exp accumulation and scalar tails. Reconstructed forward logsumexp
matches all 12,288 actual kernel rows per build exactly, and all three workers agree. Backward
probability sums have maximum absolute deviations from one of about 3.94e-7–4.39e-7 across cases,
with the same case-wise maxima on both builds. Mean deviations and forward/backward probability
RMSE change slightly in both directions, without a comparable 1.7× increase. The
[consistency table](docs/attention-consistency-013.csv) also retains the separately labeled
fp64-rounded-score and fp32-bmm reconstructions. Internal P buffers were reconstructed, not
directly captured; exact agreement with saved logsumexp is the validation check.

The [investigation guide](docs/attention-investigation.md) contains commands, protocols and limits.
The [updated report draft](docs/upstream-report-draft.md) has not been submitted; its
[0.12 version](docs/upstream-report-draft-012.md) is preserved verbatim. The cause of full-model
amplification remains unresolved despite three-model commit attribution. No compiler-bug claim
follows from this investigation.

### Executed checks

| Check | Result |
|---|---|
| Windows portable suite | 295 passed, 1 skipped; 12 compiler tests deselected |
| Linux CPU Inductor suite | 12 passed; 296 tests deselected |
| Compiler validation campaign | All 14 checks passed; 17 passes and 1 unconfirmed numerical discrepancy |
| Legacy tests | 31 passed with asyncio auto mode |
| Ruff lint and requested formatting | Passed; 88 files in the requested formatting scope |

Five new behavior tests cover preserved packed layouts/storage offsets, the fp64 attention
reference and a planted gradient error, rejection of incomplete/changed saved arguments,
explicit backward-layout materialization, and ULP accounting through zero/subnormal/overflow
ranges. Adapter checks reject implicit precision/layout changes. No existing runtime
implementation changes apart from the package version are required for this investigation.

The publication audit checks all 136 Git-index files and both distributions: 57 wheel members
and 139 source-archive members. The wheel and dependencies install into an independent Linux
virtual environment; package location, version, source hash, eager corpus execution and the
installed code comparator pass. The standalone attention CLI reproduces the recorded parent
and child block-12 measurements exactly, including reference-file comparison. License files
and license metadata remain absent from both archives following the requested removal.

All 51 earlier runtime files other than the version file are unchanged. All 20 prior validation
entries, earlier results text and evidence remain intact, and the original 0.12 report draft is
archived byte-for-byte. The Git-normalized packaged source hash is
`98599a6cbd96b9cc1e23269cfb960ab37146981683bcd31e90766ee4f553691c`.
Raw captures, compiler caches, adapters, environments and supervision logs remain local.

## Version 0.14 GELU investigation

The existing parent `eb48c7e2` and child `2450619a` source installations and original full
frozen GPT-2 workload are unchanged. The [policy](docs/v014-policy.json) fixes the original
1.4× median-log2-growth predicate, locked reference vector, three fresh workers and two internal
repeats before any conditional intervention. All new results are advisory; primary statuses,
tolerances, signatures, reducer predicates and existing exit codes are unchanged.

### Generated GELU does not call exp_u20

The complete search covers twelve source-build generated Python modules, 144 cached C++ files
and all 24 original boundary-nightly debug dumps. Each full-model forward defines eleven
generated C++ kernels and backward defines seven. The only workload math matches are:

| Stage | Generated kernel | Exact module source line |
|---|---|---|
| Forward | `cpp_fused_add_mul_pow_tanh_view_3` | 487: `auto tmp13 = tmp12.tanh();` |
| Backward | `cpp_fused_add_mul_pow_sum_tanh_tanh_backward_view_1` | 171: `auto tmp10 = tmp9.tanh();` |

Both builds emit those same calls. All twelve corresponding saved shared libraries import
`Sleef_tanhf16_u10`. The installed AVX512 header resolves vector tanh to SLEEF, whose own
exponential helper does not call ATen's `exp_u20`. No generated GELU sigmoid/erf/exponential
decomposition was found. The [inventory](docs/gelu-code-014.json) includes every module/kernel,
hash, exact match, native attention call site and source/link resolution.

Three `.exp()` matches per worker belong to ISA capability probes, not model kernels. Native
CPU flash forward separately reaches `exp_u20`; backward recomputation uses regular exp.
`cpp.use_decompose_tanh` is false. Enabling that option would emit regular vector `.exp()`
already, changing the tanh formula rather than replacing an existing `exp_u20` call.

The frozen model's eager `gelu_new` uses Python arithmetic and `aten.tanh`, also reaching
SLEEF. Its tanh backward is `grad*(1-output*output)`, without an exponential. The native fused
GELU-tanh forward/backward implementations were inspected separately and likewise use vector
tanh; the frozen model does not use that fused eager GELU entry point.

Consequently the requested generated-GELU replacement and reverse replacement are
**not applicable**. No three-worker replacement runs were made, no disappearance of the
1.710× full-model shift is claimed, and no minimal GELU reproducer is promoted. This is a
negative call-path result, not a passing numerical intervention. No PyTorch rebuild was needed.

### Full-model GELU distributions

Four fresh workers capture parent/child × eager/compiled execution of the full model.
Each performs two observed replays after an uninstrumented forward/backward control. All
eight replays exactly match all 149 output/gradient control tensors. Twelve actual expansion
addmm outputs per replay retain their original `[128,3072]` fp32 storage. The
[capture inventory](docs/gelu-capture-014.json) records counts, layouts, hashes and controls.

Eager tanh operands are directly captured. For comparable distributions in both paths,
`z = sqrt(2/pi)*(x+0.044715*x^3)` is reconstructed in fp64 from the saved fp32 inputs;
the compiled fused intermediate is not directly observed. The
[168-row table](docs/gelu-ranges-014.csv) includes all twelve blocks, both builds and paths,
all histogram bins, extrema, absolute p95/p99, and observed/reconstructed distinctions.

The existing 0.13 adapters regenerate all three fixed grids with exactly the same sampled
accuracy statistics. The [per-bin comparison](docs/gelu-exp-grid-014.csv) reports 26 occupied
sample/bin combinations; it does not infer exact behavior for untested floats in a bin.
Both adapters are also evaluated directly on identical saved inputs and reconstructed
arguments, plus hypothetical `-2*z` coordinates. Regular exp results are identical throughout.
These are hypothetical exponential evaluations; generated GELU does not call exp_u20.

For child-compiled captures:

| Quantity | Block 11 | Block 12 | Blocks 1–11 comparison |
|---|---:|---:|---:|
| Input absolute p99 | 3.143544 | 3.090838 | Median 3.143544 |
| Reconstructed z absolute p99 | 3.616472 | 3.519601 | Median 3.616472 |
| Hypothetical exp_u20(x) parent/child disagreement | 1.746% | 1.661% | Pooled 1.836% |
| Hypothetical exp_u20(z) parent/child disagreement | 1.590% | 1.502% | Pooled 1.708% |
| Hypothetical exp_u20(-2*z) parent/child disagreement | 4.960% | 4.688% | Pooled 5.252% |

Block 12's absolute p99 rank is seventh in every condition. None of the fourteen
quantity/condition checks meets either fixed criterion: p99 above twice the earlier-block
median, or disagreement at least ten percentage points above the earlier-block pooled fraction.
The [comparison table](docs/gelu-block12-014.csv) retains all scores and ranks, including nulls.

Block-12 reconstructed z reaches approximately -144.504 to 34.958, with three values below
-87.3 and two below -103.97. Earlier blocks also have tails: block 11 reaches -124.316;
block 3 spans roughly -1048.376 to 8912.077. These tanh operands are not attention scores
or actual exp_u20 inputs, so the threshold counts do not show exponential underflow in GELU.
Nonfinite outputs of the hypothetical exponential evaluations are counted separately.

The [investigation guide](docs/gelu-investigation.md) includes the complete per-block summary
and call-path details. The PyTorch-only [range example](examples/gelu_ranges.py) reproduces
the recorded block-12 distributions through its CLI. It is not a reproducer of the accuracy
shift. The full GPT-2 script remains the lead of the [report draft](docs/upstream-report-draft.md),
whose [0.13 version](docs/upstream-report-draft-013.md) is preserved verbatim. Nothing was filed
upstream. The source-commit attribution stands; the full-model amplification remains unresolved.

### Executed checks

| Check | Result |
|---|---|
| Windows portable suite | 298 passed, 1 skipped; 12 compiler tests deselected |
| Linux CPU Inductor suite | 12 passed; 299 tests deselected |
| Compiler validation campaign | All 14 checks passed; 17 passes and 1 unconfirmed numerical discrepancy |
| Legacy tests | 31 passed with asyncio auto mode |
| Ruff lint and formatting | Passed; 90 files in the requested format scope, 95 with all scripts |
| Standalone range CLI | Exactly reproduces the recorded child-compiled block-12 distributions |
| Publication audit | 146 allowed Git-index files; 57 wheel and 149 source-archive members |
| Independent installed-wheel checks | Package location, version, source hash, eager corpus and code comparator passed |

Three new behavior tests cover histogram boundaries/nonfinite inputs, exponential disagreements
with nonfinite outputs, and positive/null block-12 comparisons with a complete denominator.
All 51 earlier runtime implementation files other than the version file are unchanged.
All 21 prior validation entries, earlier results and evidence remain intact; the 0.13 report
is archived byte-for-byte. Large captures, adapters, caches, environments and logs remain local.
License files and license metadata remain absent from the distributions.

The Git-normalized packaged source hash is
`3f3fc9b7f303cbbdb06a042d106897ab881a3a027704f37e3f38823dfd4424cc`.
This release changes the version and adds a separate diagnostic example and evidence; no
numerical tolerance, primary verdict or reducer behavior changes.

## 0.15 — replication across input batches

The original full GPT-2 compiled shift did not meet the predeclared systematic-across-inputs rule.
The single-input shift was not an input-robust regression in this experiment. The upstream report
draft is not recommended for filing. The original numerical measurements and three-model
parent/child attribution remain valid for their frozen inputs; they do not establish a systematic
degradation across passages.

The policy in [v015-policy.json](docs/v015-policy.json) was frozen before execution. Twenty
batches per model were drawn with seed 150015 from disjoint 200-word chunks of the frozen
public-domain Alice text, excluding both original paragraphs. The original pinned tokenizers
produce 2×64-token inputs. Full GPT-2, BERT, Pythia-160M and the existing eleven-layer GPT-2
variant run on the unchanged parent and child source builds; GPT-2 variants share token tensors.
The fixed seed-zero ±1 VJP, weights, layouts and one-thread settings are retained.

For each path, one batch contributes the median log2 child/parent gradient RMSE growth against
each build's fp64 reference. The forward output is excluded. The predeclared rule requires
at least 16/20 worsening batches and median growth ≥1.25× for “systematic across inputs”;
≤13/20 is “not systematic”, otherwise “mixed”. The exact one-sided binomial p-value on batch
signs is descriptive and is not an additional threshold.

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

The [investigation guide](docs/input-accuracy.md), [complete per-batch table](docs/input-accuracy-015.csv),
[input manifest](docs/input-batches-015.json) and [measurement evidence](docs/input-accuracy-015.json)
retain all models and paths. Large raw measurements, weights, source builds and caches remain local.
The target-level `accuracy_shift` documentation now explicitly distinguishes correlated targets
from input replicates. Its thresholds and prior labels remain unchanged. A separate standalone
batch summary implements the new rule; the production sweep extension remains a design note.

This concludes the bounded exp_u20 investigation. The kernel and generated-GELU null results
from 0.13–0.14 remain intact; no new causal mechanism or compiler defect is claimed. The
[report draft](docs/upstream-report-draft.md) leads with the new evidence and preserves the
earlier single-input context. Its [0.14 version](docs/upstream-report-draft-014.md) is archived
byte-for-byte. Nothing was filed upstream.

### 0.15 executed checks

| Check | Result |
|---|---|
| Original-input calibration | 7 controls, every retained eager/compiled gradient RMSE exact |
| New-input measurements | 160/160 workers complete; all 80 paired fp64 gradient references exact |
| Independent batch calculation | All scores agree to 1e-12; signs, decisions and exact p-values agree |
| Summary CLI | Matches the recorded summaries for all four models |
| Windows portable suite | 309 passed, 1 skipped; 12 compiler tests deselected |
| Linux CPU Inductor suite | 12 passed; 310 tests deselected |
| Compiler validation campaign | All 14 checks passed; 17 passes and 1 unconfirmed numerical discrepancy |
| Legacy tests | 31 passed with asyncio auto mode |
| Ruff lint and formatting | Passed; 92 files in the requested format scope, 98 with all scripts |
| Publication audit | 155 allowed Git-index files; 57 wheel and 158 source-archive members |
| Independent installed-wheel checks | Package location, version, source hash, eager corpus and code comparator passed |

Eleven new tests cover the fixed decision boundaries, a planted uniform 1.5× shift,
same-error ties, correlated targets counted as one batch, missing/duplicate batches,
zero-error extended values, mismatched identities, changed gradient presence and reference
hashes, and exact sign-test handling of ties. No primary oracle tests or tolerances were altered.

All 51 earlier runtime implementation files other than the version file are unchanged.
All 22 prior validation entries and the earlier RESULTS prefix remain intact. Earlier numerical
evidence is unchanged; the 0.14 report is archived byte-for-byte. The public input manifest
retains exact token IDs and sampling metadata. Raw per-gradient measurements, environments,
weights, logs and caches remain local. License files and license metadata remain absent.

The Git-normalized packaged source hash is
`9324cff1ca078196fe7c03075996963fdf90d11273d9140cae37a33789cd694f`.

## 0.16 usability workflow

`faultline check` adds a terminal verdict, HTML report and versioned JSON contract over
the recorded primary evidence. It accepts trusted factories, captured bundles, torchvision
models and Hugging Face model IDs. Existing primary statuses, tolerances, signatures,
reducer predicates and command exit codes remain unchanged. Only the new command uses
SAFE/0, REGRESSION/1, NEEDS_REVIEW/2 and INCONCLUSIVE/3.

The fixed policy requires candidate-only stable failure or input-replicated accuracy
degradation for REGRESSION. Input batches, not correlated gradient tensors, are the unit
of replication: at least ten completed scores, at least 80% worsening and median growth
of at least 1.25x. Old single-input labels remain descriptive and unchanged. Replicated
nightly bisection uses the same rule. Environment creation reuses supervised installers;
the GitHub Actions example gates only on REGRESSION and retains all other verdicts.

Quick, standard and thorough levels separate primary execution from diagnostics.
Memory-limited diagnostics preserve completed primary observations. All ten cost cells
completed their primary batches within the 12-GiB worker budget. Standard took 15.00
minutes for ResNet-50, 33.67 for BERT, 36.44 for GPT-2 and 17.86 for SmolLM2-135M.
BERT and GPT-2 miss the approximate 30-minute target; thorough took 78.43 and 85.86
minutes respectively. SmolLM2 peaked at 8,015.6 MiB and returned NEEDS_REVIEW. These
are CPU operational costs for the documented workload sizes, not universal bounds.

The complete fixed-policy calibration follows. All 29 models and ten batches per model
remain in both catalog denominators. The research record contains every pair, accuracy
score, miss and cohort limitation.

| Cohort | Planned | SAFE | REGRESSION | NEEDS_REVIEW | INCONCLUSIVE |
|---|---:|---:|---:|---:|---:|
| same_version | 29 | 4 | 0 | 25 | 0 |
| upgrade | 29 | 4 | 0 | 25 | 0 |
| historical | 4 | 0 | 2 | 2 | 0 |
| controlled_injection | 3 | 0 | 3 | 0 | 0 |
| exp_u20 | 4 | 0 | 0 | 0 | 4 |

Same-version SAFE target misses: 25/29. Upgrade replicated accuracy flags: none.
The policy still requires manual review for most tested models in an unchanged
environment; the automatic SAFE target was not met for this catalog.
All three controlled injected faults were detected. Two of four historical fixtures
meet the REGRESSION target; the other two remain NEEDS_REVIEW. No exp_u20 variant
triggers the replicated rule; its accuracy-only historical data yield INCONCLUSIVE
for the full verdict because primary observations are missing. None of these counts
is a false-positive rate. The policy was not retuned after seeing calibration results.
Incomplete accuracy coverage is shown in the per-model table. A false replicated flag
with fewer than ten usable scores cannot establish that an accuracy shift is absent.

The full-diagnostic upgrade campaign retains its original frozen source, two repeats,
one thread, 600-second deadline and 12,288-MiB RSS budget. Twenty-six same-version
controls use fresh independent workers against the matching preserved upgrade baselines;
three use independent paired standard-cost workers. Shared baselines and source material
make cohort counts correlated. No new GPU validation is claimed.

A wheel-install check found and fixed cross-environment dependency shadowing. Workers
now import only the harness package from the coordinator and resolve model dependencies
in the selected interpreter. Actual installed-wheel workers record 2.15.0+cpu and
2.16.0.dev20261105+cpu. Quick and standard small-model verdicts are SAFE and conform
to the published schema. A fresh public clone and clean-venv wheel install completed
the README's nine commands in 107.78 seconds, including a 67.30-second pretrained
ResNet-18 quick check. Global download caches were warm; these are not cold-network
timings. The earlier 114.84-second walkthrough is retained.

The finite exact-match comparator shortcut and omission of redundant diagnostic tensor
dumps preserve all fields and exceptions in 2,238 comparator cases. Fresh BERT workers
and 82 before/after observations across six cost cells retain identical selected primary
comparisons and numerical diagnostics. Earlier costs and incomplete attempts remain
recorded; different host contention prevents an isolated optimization speedup claim.

### Executed checks and preservation

| Check | Result |
|---|---|
| Linux portable suite | 366 passed; 12 compiler tests deselected |
| Windows portable suite | 365 passed, one skipped; 12 compiler tests deselected |
| Actual CPU Inductor suite | 12 passed |
| Compiler validation campaign | All 14 checks passed; 17 passes and one unconfirmed numerical discrepancy |
| Legacy suite | 31 passed |
| Ruff lint and requested formatting | Passed |
| Installed-wheel quick and standard | SAFE; independent worker versions and JSON schema verified |
| GitHub Actions at `2f7874e` | All six jobs passed; small-model ten-batch upgrade check SAFE |
| Preservation audit | 23 prior validation entries and 32 historical evidence files unchanged |

Earlier frozen and integrated snapshots also passed their recorded checks: 326 and
355 portable tests respectively, all 12 Inductor tests, all 14 compiler-validation
checks and all 31 legacy tests. These are separate from the final runtime checks above.

The initial overlapping runs encountered a host OOM and later WSL interruptions.
All 130 completed upgrade-pair commits and nine completed partial-cost commits passed
hash verification. Interrupted artifacts were retained; the final ten-cell timing
series restarted in new directories and the upgrade campaign resumed unchanged.
An exclusive queue then completed the remaining upgrade pairs and same-version controls.
No interrupted host process was relabeled as a compiler failure.
Windows also suspended the host for nearly eight hours during catalog calibration;
the workers resumed with their saved evidence. The timing study had already completed,
so no reported preset cost spans that pause.
A later WSL restart interrupted DistilBERT after 180 upgrade pairs had committed.
All 180 commit records passed verification before another unchanged resume. The
supervisor then used a persistent background client and durable logs; monitoring checks
the boot ID and process start ticks so a reused PID cannot imply that a worker survived.

Large completed calibration tensor copies are retained in verified lossless archives
with per-file SHA256 values and restore tools. Frozen input bundles, weights and result
JSON remain in place. Twelve archive lifecycle checks and a 52-file, 21.2-GiB real-data
round trip passed; all result-JSON hashes remain unchanged. This local research storage
step does not change normal tool output. Restore archived copies before local graph
export or reduction.

The [usability validation record](docs/research/usability-016.md),
[calibration JSON](docs/research/calibration-016.json),
[calibration CSV](docs/research/calibration-016.csv),
[cost evidence](docs/research/cost-016.json) and
[fresh-user transcript](docs/research/quickstart-test.md) provide the detailed evidence.
The tested Git-normalized runtime hash is
`9d07dfc043138e51979a3fba32fc47e964fb8f3821cc693c97283e980f64e31d`.
All pre-0.16 RESULTS content, historical evidence and the license situation are preserved.
Nothing was filed upstream and no compiler defect is claimed.

## 0.17 — upgrade change attribution

Two-environment `check` defaults to the predeclared [v2 policy](docs/verdict-policy-v2.md).
Shared compile/eager discrepancies, including changed signatures, are reported separately
as pre-existing differences. Candidate-only stable failures and replicated accuracy shifts
remain REGRESSION. Missing required execution/accuracy evidence still prevents SAFE.
Single-environment checks and `--verdict-policy v1` retain the original policy. Primary
statuses, tolerances, signatures, reducer predicates and exit codes are unchanged.

Re-scoring all 69 retained 0.16 cohort entries changed only the advisory summary. Every
old v1 verdict, reason count, primary count, accuracy distribution and evidence hash matched.
The v2 results below are **post hoc**, with no model reruns or simulated early stopping:

| Cohort | Planned | v1: SAFE / REGRESSION / NEEDS_REVIEW / INCONCLUSIVE | v2 (post hoc) |
|---|---:|---|---|
| same_version | 29 | 4 / 0 / 25 / 0 | 28 / 0 / 0 / 1 |
| upgrade | 29 | 4 / 0 / 25 / 0 | 27 / 0 / 0 / 2 |
| historical | 4 | 0 / 2 / 2 / 0 | 1 / 2 / 0 / 1 |
| controlled_injection | 3 | 0 / 3 / 0 / 0 | 0 / 3 / 0 / 0 |
| exp_u20 | 4 | 0 / 0 / 0 / 4 | 0 / 0 / 0 / 4 |

SmolLM2 remains INCONCLUSIVE in both catalog cohorts because paired accuracy evidence
is unavailable. Upgrade GPT-2 remains INCONCLUSIVE with nine of ten scores. Both
baseline-confirmed historical regressions and all three injections retain REGRESSION.
The other historical fixtures were not regressions under v1. The exp_u20 accuracy-only
cohort lacks primary observations and remains INCONCLUSIVE without a replicated flag.
The complete 0.16 evidence and verdict counts remain unchanged; these are not false-positive rates.

Fresh fixed-policy validation covered six new checkpoint variants, three vision and
three text, on ten real-input batches each. Each used one CPU job, two primary repeats
and the retained 2.15.0+cpu / 2.16.0.dev20261105+cpu environments. The variants were absent
from the 0.16 catalog but share architecture families and input sources with earlier work.

| Cohort | Planned | SAFE / REGRESSION / NEEDS_REVIEW / INCONCLUSIVE |
|---|---:|---|
| same_version | 6 | 6 / 0 / 0 / 0 |
| upgrade | 6 | 6 / 0 / 0 / 0 |

Fresh replicated accuracy flags: none.
Three fresh, explicitly injected control runs also returned REGRESSION.

Primary batch workers now run in parallel with exclusive reusable per-slot caches and
a divided total RSS budget. Accuracy diagnostics consume frozen batch order and stop
only with a verified bound on every possible remaining completed-score count. Positive
flags still require ten completed scores. All planned primary comparisons run, including
after negative accuracy stopping; a test verifies detection of a last-batch failure.
Primary and diagnostic commits are separate so interruption does not rerun completed primaries.

The requested standard-level timing comparison used four jobs and a total 12-GiB budget
(3072 MiB per worker), against 0.16's sequential 12-GiB workers:

| Model | 0.16 minutes | 0.17 minutes | v2 verdict | Conclusive primary batches |
|---|---:|---:|---|---:|
| resnet50 | 14.97 | 4.63 | SAFE | 10/10 |
| bert | 33.61 | 1.54 | INCONCLUSIVE | 0/10 |
| gpt2 | 36.36 | 1.66 | INCONCLUSIVE | 0/10 |

Elapsed-time comparisons include changed diagnostic coverage and concurrency; they do not
isolate a parallelism speedup. Any resource-limited timing cell is incomplete, not a speedup.
Use `--jobs 1` when a model needs the full budget. Per-batch coverage, memory observations,
fixed stopping certificates and the preserved old costs are linked in the research record.

Executed checks: Linux portable 381 passed; Windows 380 passed and one skipped; actual
CPU Inductor 13 passed; all 14 compiler-validation checks passed; legacy 31 passed;
Ruff lint and requested formatting passed. Actual eager and Inductor workers confirm
serial/parallel primary-result parity. The installed 0.17 wheel passed paired standard v2
and explicit quick v1 checks; both JSON schemas and independent worker versions were verified.

Preservation checks verified all 24 prior validation entries, 37 prior data files and
the complete preceding RESULTS content. The raw tested source hash is
`43fc70267291fee131123c50b2135f0cdd1426c1ef66f64524b509918c7956d8`;
the installed Git-normalized hash is
`72e84c2e13648d1bad721c3a89c7091995012a6d9a0c5b829566c910dbd914df`.
Five existing files differ only in line endings between those snapshots.

See the [0.17 research record](docs/research/usability-017.md),
[post hoc re-score JSON](docs/research/rescore-017.json),
[re-score CSV](docs/research/rescore-017.csv) and
[fresh evidence](docs/research/validation-017.json).
The initial cold-per-batch-cache attempt remains retained; the final measurements use
new directories and the same frozen inputs. No compiler defect or new GPU validation
is claimed, and nothing was filed upstream.
