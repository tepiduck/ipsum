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

**Confirmed upstream bugs: zero.** No upstream issue has been submitted for the numerical candidate.

## Coverage and limits

The portable test suite exercises graph validation, model capture, numerical/nonfinite handling,
repeat stability, worker lifecycle, reduction, saved-finding integrity, standalone export, and
terminal workflows. The separate compiler test executes real TorchInductor code generation.
CI runs the portable suite on Windows and Linux and the compiler suite on Linux.

CPU eager corpus tests cover float16, bfloat16, float32, and float64. The compiled 18-case campaign
used float32. GPU execution, gradients, distributed execution, dynamic shapes, and arbitrary-model
coverage have not been established. Diagnostic timing fields are not throughput or speedup evidence.

The recorded experiment predates the terminal interface release. Current behavior is checked by
the automated suite; rerun the experiment harness to record evidence for another revision:

```bash
python scripts/validate_faultline.py --compiler --output faultline-runs/validation
```
