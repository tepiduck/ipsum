# Gradient probes and numerical evidence

## Captured backward inputs

New native captures use a fixed vector-Jacobian product (VJP) probe. Capture executes the eager
request stream once and saves a Rademacher vector (values -1 and +1) for every floating output of
each request. A private CPU generator creates the vectors; they do not depend on output values.
The original state, inputs, and RNG snapshot are saved before probe execution, so any mutations
while collecting output shapes do not become the starting state of the replay.

```bash
faultline model capture model.py:factory --backward-probe vjp --output workloads/model
faultline model run workloads/model --gradients --fp64-advisory --output faultline-runs/numerics
```

Gradient evaluation calls `torch.autograd.grad` with the same saved `grad_outputs` for eager and
compiled execution. The vectors are exactly representable in supported floating dtypes. Detached
outputs remain observable but do not enter autograd. Changing the output contract produces an
error rather than silently drawing new vectors. Request deletion retains each request's original
vector, including when its position in the reduced sequence changes.

This removes the output-dependent backward seed from the previous mean-square loss. It also
provides a nonzero probe at zero-valued outputs. It does **not** remove the effects of different
forward intermediates, activation branches, or saved tensors inside backward. One vector probes
one projection of the Jacobian; it is not a full Jacobian comparison or a proof of correctness.
The API follows PyTorch's [vector-Jacobian product interface](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad).

VJP workload manifests use schema version 3, which older readers reject. Existing version-1
captures retain the square-loss policy. Use `--backward-probe loss` to make another legacy-policy
capture. The historical corpus explicitly retains that policy to keep its fixtures comparable.
Capture now requires the forward stream to execute successfully; legacy loss capture can still
freeze a workload whose failure must be investigated later.

Use `--aot --gradients` when reducing backward failures. Independent AOT backward graphs already
contain the actual saved cotangents and are never differentiated again. Independent Dynamo graph
replays still use their own square-loss derivative predicate: their output boundary differs from
the model's. Such localization establishes a separate predicate and must not be described as
retaining the original model's VJP. Whole-model isolation and request reduction retain it.

## Native fp64 advisory

`model run`, `model compare`, and `qualify` accept `--fp64-advisory`. After the repeated primary
comparison, the worker executes an eager reference with floating model storage and inputs promoted
to float64 on the **same device**. Autocast is disabled for this reference. Integer inputs, boolean
masks, strides, offsets, repeated references, and same-dtype storage aliases are retained. Explicit
casts inside the model remain intact; unsupported layouts and mixed-dtype storage aliases fail the
advisory rather than being silently rewritten.

For each floating output, loss (legacy captures), and gradient, the report records:

- Eager and compiled RMSE and maximum absolute error against the reference.
- The compiled/eager RMSE ratio, with separate exact-result flags when the denominator is zero.
- Median, 95th, and 99th percentile absolute errors from at most 32,768 regularly spaced elements.
  Each record gives the sampled count and stride. RMSE and maximum error use every element.
- The reference output dtype, since an explicit cast can make it lower than float64.

Large ratios must be considered alongside absolute errors. Promoting arithmetic changes its
rounding and can change control flow or activation derivatives near discontinuities. A result
closer to fp64 is not automatically the correct implementation of the original dtype semantics.
Stochastic operators can also draw different values across dtypes despite restoring RNG state.
The advisory never relaxes tolerances, changes failure signatures, or turns a discrepancy into a
pass. It is evidence for investigating a candidate.

The reference is repeated to check stability. Errors, nonfinites, and instability receive separate
advisory statuses. Unsupported reference execution does not overwrite the primary result. The
worker's total time/RSS limits include the advisory; `primary-result.json` checkpoints the completed
primary comparison first. If the advisory exceeds a budget, the run remains inconclusive and the
coordinator includes that primary observation. Standalone scripts use the same advisory code, but
need an external supervisor to enforce resource limits.

## Freezing internal graph values

Forward-only native reduction can now cut a graph at a tensor computed during eager execution.
A supervised worker records candidate boundaries, then replaces a prefix with a new frozen input.
The joint input snapshot retains storage sharing and views. Proposals must decrease the existing
complexity ordering, retain the stable candidate failure, and pass the baseline when one was supplied.
Final confirmation reruns both environments without the reducer's evaluation cache.

This extends node bypassing to prefixes whose input and output shapes differ. The preparation
worker and each tested proposal count toward the evaluation budget. At most twelve internal
boundaries near the output are captured in one preparation pass to bound snapshot cost. Boundary
capture is currently unavailable for gradient checks and saved AOT graphs. It does not perform
tensor-dimension shrinking or sanitize data; retained boundary tensors can contain sensitive values.

The constructed transformer example places the historical indexing operation after all six encoder
layers and the classifier. Its values depend on the transformer output. No manual graph break or
compiler mutant is added:

```bash
faultline model capture examples/native_models.py:transformer_indexing --timeout 600 --max-rss-mb 12288 --output workloads/transformer-indexing
faultline model compare workloads/transformer-indexing --baseline-python .current/bin/python --candidate-python .historical/bin/python --reduce --max-evaluations 24 --repeats 2 --timeout 600 --max-rss-mb 12288 --output faultline-runs/transformer-indexing
```

The recorded experiment uses PyTorch 2.15.0 as the passing control and 2.12.0 as the failing
candidate. This tests localization of a known operation pattern in a larger constructed workload;
it is not a new bug discovery or an unmodified pretrained-model benchmark.

## GPU environment and nightly checks

The AMD validation uses a separate WSL environment, the RX 7900 XT, and AMD's PyTorch/ROCm
packages. The Windows display driver was not changed. With an appropriate driver and WSL setup,
the environment can be created using AMD's [package instructions](https://github.com/ROCm/TheRock/blob/main/RELEASES.md):

```bash
python -m venv .rocm
.rocm/bin/python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1100]==2.14.0+rocm10.1.0" "rocm[libraries,devel,device-gfx1100]==10.1.0"
.rocm/bin/python -m pip install numpy psutil
.rocm/bin/python -m pip install --no-deps -e .
.rocm/bin/python -m faultline qualify --require-rocm --repeats 2 --output faultline-runs/rocm
```

The versions above identify the tested environment, not a promise that every driver/OS combination
works. See [RESULTS.md](../RESULTS.md) for measured coverage. NVIDIA/CUDA remains unvalidated.
Requesting `reduce-overhead` does not prove graph capture was used; inspect compiler skip logs.

The [nightly workflow](../.github/workflows/nightly.yml) runs daily and on manual dispatch. It tests
the current CPU nightly's compiler adapters and compares native workloads against stable PyTorch
2.15.0. Artifacts retain actual versions and source fingerprints. A failed job is a signal to
investigate; it does not automatically file an upstream issue or establish a new compiler bug.
