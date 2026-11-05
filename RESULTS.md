# Validation results

Recorded experiments: **November 5, 2026**. The [public experiment summary](docs/validation.json)
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
CI runs the portable suite on Windows and Linux, the compiler suite on Linux, and a separate
two-environment upgrade gate.

CPU eager corpus tests cover float16, bfloat16, float32, and float64. The compiled 18-case campaign
used float32. GPU execution, distributed execution, and arbitrary-model coverage have not been
established. Forward/backward and dynamic-shape coverage for version 0.3 is described below.
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
