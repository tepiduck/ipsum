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
- A descriptive accuracy label: `comparable_to_eager_against_fp64` for a compiled/eager RMSE
  ratio in [0.5, 2], `candidate_closer_to_fp64` below that band, and
  `candidate_farther_from_fp64` above it. Zero errors and unavailable comparisons have separate
  labels. The factor-of-two band is a reporting convention, not a correctness threshold.

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

## Mask and eager-control diagnostics

```bash
faultline model run workloads/model --gradients --aot --numerics --timeout 600 --max-rss-mb 12288 --output faultline-runs/model-numerics
python examples/relu_sensitivity.py --output faultline-runs/resnet-sensitivity
```

The example uses random-weight torchvision ResNet-18, seed 21, synthetic inputs, and fixed VJP
cotangents. It requires torchvision and a CPU Inductor toolchain. Exit 1 records a stable numerical
mismatch; the advisory does not turn it into a pass.

`--numerics` includes fp64 evidence and adds two checks after the stable primary comparison:

1. **Eager control.** On CPU with oneDNN enabled, repeat eager with oneDNN disabled, restoring the
   backend setting afterward. Otherwise, repeat eager after moving each finite floating input
   storage value one representable step toward positive infinity. This input perturbation uses
   freshly loaded requests, retains aliasing/layouts, leaves integers and whole-model VJP cotangents unchanged,
   and avoids overflow to infinity. It is explicitly a different-input sensitivity control.
   No perturbable input, unsupported execution, nonfinite results, and unstable repeats have
   separate statuses. `eager_control_sensitive` requires every failing target to have a control that also
   exceeds the primary tolerance and its RMSE from eager is at least half the compiled/eager RMSE.
   Similar disagreement establishes sensitivity, not proof that a compiler defect is absent.
   For an independently saved derivative graph, floating graph inputs include lifted cotangents;
   the ULP control perturbs those too. This is separate from the original-model VJP comparison.
2. **AOT saved masks.** With a whole-model `--aot --gradients` workload, run a separate diagnostic
   replay. At each forward compiler boundary, execute the same AOT graph eagerly on jointly cloned
   inputs with restored PyTorch RNG. Match its outputs to the compiled outputs by position, then
   to actual backward inputs by storage identity, offset, shape, stride, and dtype. This avoids
   guessing correspondences from equal shapes or node names. Ambiguous aliases are unsupported.
   Compare threshold/ReLU selectors, boolean `where` selectors (including supported clamp
   comparisons), and explicit max-pool backward indices. Record exact flip counts, up to sixteen
   changed flat indices, saved-input origins, and reachable backward output positions. Also compare
   eager and compiled backward outputs with **identical compiled saved inputs**.

The mask diagnostic must reproduce the primary candidate observation and affected-target signature
on every repeat. The returned model gradients are mapped to AOT backward outputs by exact storage
identity and layout. Graph-local output positions are never treated as model parameter positions.
Copied, accumulated, ambiguous, or otherwise unmapped gradients cannot satisfy downstream coverage.

- `explained_by_discontinuity` requires observed flips, agreement in **every** same-input backward
  check, no failed backward replay, and a mapped path from a flipped selector to **every** failing gradient.
- `discontinuity_present` records flips when either the backward check disagrees or downstream
  coverage is incomplete. It remains suspicious, including when another numerical label qualifies.
- `comparable_to_eager_against_fp64` and `eager_control_sensitive` apply to the whole finding only
  when all failing targets qualify. Per-target evidence remains available for mixed findings.

Numerical context schema 2 records matched/total target counts under `label_coverage`, including
the exact eager control policy (`onednn_disabled` or `input_one_ulp_up`). The two controls are
different experiments and are not pooled as equivalent evidence. Changed diagnostic execution or
unstable masks cannot qualify for either discontinuity label. The explanation label expresses this
evidence criterion, not a proof of compiler correctness or an adjudicated false-positive label.
Schema 1's `discontinuity_sensitive` and any-target aggregate labels are historical 0.8 behavior.

Mask coverage is deliberately bounded. Recomputed intermediates, unknown expression forms,
lowered max-pool scatter patterns, unavailable saved values, and graphs outside the supported AOT
adapters remain unsupported. Observed max-pool operations without established selector coverage
produce an explicit coverage gap and `partial` status, even when the compared masks have zero flips.
A graph with no recognized selectors reports `no_supported_masks`,
not proof of smoothness. Joint shadow-input storage is limited to 512 MiB per graph call; retained
paired outputs are limited to 256 MiB. The outer worker time and RSS budgets still apply.
`partial` reports include supported comparisons and explicit omissions. These comparisons use
**AOT eager graph replay**, which can differ from the original module's eager execution. Graph
breaks and recomputation prevent claiming complete original-model mask coverage or precise
layer-by-layer causal attribution.

The ResNet finding illustrates why the fp64 ratio alone is insufficient: the follow-up check
observed two selector flips and an eager-only convolution-library change produced a comparable
gradient discrepancy. The compiled backward agreed with eager backward on identical saved inputs.
The 0.7 RMSE ratio is evidence of derivative sensitivity around activation boundaries; it should
not be used as evidence of a compiler defect. See the follow-up measurements in [RESULTS](../RESULTS.md).

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
The location near the output favors this bounded prefix-cut search. A pattern between encoder
layers would be a stronger reduction test. Suffix cutting and that middle-of-network experiment
are not implemented or validated by the 0.7 result.

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
No skip message in a log is also insufficient to establish capture; positive runtime capture or
replay evidence is needed. The saved 0.7 `perf_hints` logs contained no matching skip messages,
so those results remain execution passes with GPU graph capture unverified.

The [nightly workflow](../.github/workflows/nightly.yml) runs daily and on manual dispatch. It tests
the current CPU nightly's compiler adapters and compares native workloads against stable PyTorch
2.15.0. Artifacts retain actual versions and source fingerprints. A failed job is a signal to
investigate; it does not automatically file an upstream issue or establish a new compiler bug.
