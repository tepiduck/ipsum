# Validation results

Recorded experiments: **November 5–6, 2026**. The [public experiment summary](docs/validation.json)
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
