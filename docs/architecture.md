# Faultline architecture and validity

## Native workloads

`model capture` runs a trusted factory in a supervised worker and saves its state dictionary,
trainability flags, nested requests, and Python/NumPy/PyTorch RNG states. Tensor data uses
`torch.save`, retaining strides, storage offsets, repeated references, and shared storage. Loading
uses `weights_only=True`; the copied factory is still executable trusted Python. Artifact hashes
detect accidental edits, not maliciously coordinated replacements. Included local source and asset
files must be named explicitly. Factory dependencies and external files are not automatically vendored.

`model run` compares the original module with `torch.compile(module)`. It restores saved state at
the start of every whole request stream, while retaining one compiled callable across repeats and
requests. It compares nested outputs, post-call tensor inputs, final registered buffers and
parameters, and optionally input/parameter derivatives. New captures use saved VJP cotangents;
legacy captures use the original scalar-loss policy. Missing derivatives
are distinct from zeros. Integer and boolean comparisons are exact, including int64 values beyond
float64 precision. Comparison storage retains two anchor observations and the current repeat;
memory still grows with model size and request-stream length. Timings include reset, copying,
graph capture, and observation overhead and must not be treated as performance measurements.

The backend callback records Dynamo graphs and actual arguments before the first execution.
It delegates to the explicitly selected compiler backend. Native graph snapshots retain PyTorch
operations instead of translating them into the compact IR. Unsupported graph targets, symbolic
objects, or opaque submodules prevent independent graph export, but do not prevent the original
model from running. Compilation failures before real arguments reach the callback can also leave
only the full-workload reproducer. No captured graph under a real compiler produces the
inconclusive `no_compiled_graph` status rather than an apparent compiler pass.

Native reduction checks the original failure again, removes requests, and tries compiling one
graph at a time in the full model while executing the others eagerly. The full-model failure
signature must survive that isolation. It then establishes a separate independent graph predicate:
output names and lifted parameter names differ at this boundary. The category must match, but this
does not prove root-cause identity. Both stages require a passing baseline when comparing
environments. Backend flat output lists are adapted to the captured FX tuple ABI only at the
independent graph boundary; original-model output structure remains part of the comparison.

For a portable independent graph, reduction bypasses shape/dtype-compatible nodes in chunks,
prunes dead code and unused arguments, and accepts only strict decreases in the lexicographic
ordering `(requests, call nodes, first-request tensor elements)`. It preserves the independent
failure signature. A final uncached evaluation is reserved; dual-environment checks rerun both
interpreters. Graph shape reduction and multi-graph interaction minification are not implemented.
The full original snapshot is retained, including when no graph can be isolated. Attempt records
and proposals consume disk space; large models and long request streams can make reduction costly.

Forward-only Dynamo graphs also support frozen tensor boundaries. A supervised worker executes
the eager graph, jointly snapshots retained inputs and internal tensors, and emits bounded cut
proposals. Each is checked against the same candidate and baseline predicates. Gradient and AOT
graphs do not use these cuts. Details of the probe policies, fp64 advisory, and boundary reductions
are in [numerical evidence](numerics.md).

Optional `--numerics` runs fp64, eager-control, and supported AOT saved-mask checks after the
primary comparison. Mask shadow execution runs in a separate diagnostic replay and must reproduce
the original candidate and failure targets across repeats. Advisory labels are not part of the
reducer's acceptance signature. They never relax tolerances, change an exit code, or erase the
underlying finding. Unsupported/partial evidence is retained explicitly.

Standalone native bundles copy the same comparison, execution, and graph reconstruction modules
used by workers. They verify file hashes and need no Faultline installation. Full-model bundles
also need factory dependencies; graph bundles use standard PyTorch/operator/math targets. The
standalone process does not apply the coordinator's timeout/RSS watchdog.

The compiler adapter also travels with each standalone bundle. It resolves installed backend
implementations and Inductor mode options without falling back to another compiler. Native runs
can scope autocast to forward execution; the loss and backward remain outside autocast. Saved
policies participate in export validation and failure matching. Independently saved AOT graphs
already contain their casts and are executed without a second autocast context. The backend
ladder and historical benchmark use these same workers; see [diagnostics](diagnostics.md).

This path requires `eval()`. It does not restore arbitrary Python attributes, optimizer state,
external services, or distributed process state. Such state can cause unstable or misleading
results and must be removed or explicitly reset in a workload. Stochastic operators can use
different eager and compiled RNG algorithms even with the same seed; this is a numerical
comparison tool, not a statistical equivalence test. Complex tensors are unsupported. CPU
validation does not establish GPU correctness.

### AOT forward/backward capture

`model run/compare --gradients --aot` observes the graphs produced by the selected backend's
AOTAutograd pipeline. Inductor retains its decompositions, partitioner, and original compiler
options. The adapter wraps the `CompiledFxGraph.current_callable` execution boundary, preserving
the OutputCode object and its runtime hooks. AOT-eager uses boxed forward/backward compiler
callbacks. Both record actual arguments before execution, including backward activations and
cotangents; fake compile-time tensors are not used as numerical evidence.

Diagnostic capture disables `enable_autograd_cache` so cache hits cannot bypass these callbacks.
Inductor's generated-kernel cache stays enabled. This is an explicitly modified diagnostic
execution mode, not an observation of unmodified cache behavior. Unsupported adapter controls
produce `unsupported_aot`, and a passing run with no backward graph produces `aot_capture_missing`.
Both are inconclusive. The Inductor adapter depends on private PyTorch APIs; tests and environment
metadata identify the versions actually exercised.

Each AOT graph is associated with its parent Dynamo graph, stage, and compilation ordinal. Stage
isolation compiles only the selected stage, using eager execution for the others, and requires
the original whole-model failure signature to recur. An independent backward graph consumes the
captured inputs and returns gradients as ordinary outputs. It is never differentiated again.
The manifest records `aot_stage`, static-input roles by placeholder name, and inference mode.
Graph metadata retains output visibility and static-input flags, while binary storage preserves
input aliases and strides. Naming static inputs avoids index drift when reduction prunes arguments.

AOT graph manifests use schema version 2 so older readers reject their changed execution semantics.
Graph JSON also uses version 2 when compiler metadata is present; existing version-1 workloads and
job identities remain supported. Replay reports `aot_backward_outputs` or `aot_forward_outputs`.
The backward output comparison maps numerical discrepancies to `gradient_mismatch`.

Compiler metadata is deliberately limited to the fields needed by the tested adapter. A graph may
depend on surrounding state or layout context that cannot be independently reconstructed. The
reducer must observe a stable independent failure and a passing baseline before adopting that
graph. If it cannot, the original workload is exported. This is particularly relevant to failures
involving interacting compiled regions or compiler-generated layout decisions.

`qualify` runs fixed MLP, convolution, and masked-attention workloads over selected floating dtypes
on an explicit device, with optional registered-buffer mutation and changing-shape request streams.
Each case uses the native worker, gradient oracle, and export path; AOT capture is enabled unless
`--no-aot` is specified. Device discovery alone is not a successful execution check. No CPU fallback
is permitted; missing required CUDA/ROCm support returns an unavailable report before the matrix. Results are
component checks, not a hardware certification or a benchmark.

## Execution chain

`generator or trusted model capture → validated data-only IR → frozen inputs → FX graph`

`eager reference + candidate → repeated comparisons → durable outcome → reduction → reproducer`

Inputs contain exact logical values plus a layout recipe. Transposed, strided, and zero-stride
expanded tensors are reconstructed intentionally. A graph contains only an allowlisted set of
operators, earlier-node references, and validated parameters. All inputs share one floating dtype.
The IR caps ranks, dimensions, individual and aggregate tensor elements, input count, and nodes.
These are allocation guards; they are not a complete estimate of compiler working memory.

## Comparison policy

Both paths use the same frozen inputs, dtype, device, and graph semantics. TF32 is disabled
and float32 matmul precision is set to highest. Default `(rtol, atol)` pairs are float64
`(1e-8, 1e-10)`, float32 `(1e-4, 1e-5)`, float16 `(1e-2, 1e-3)`, and bfloat16 `(5e-2, 5e-3)`.
These are configurable engineering thresholds, not universal scientific guarantees.

For finite values, a discrepancy satisfies `abs(candidate-reference) > atol + rtol*abs(reference)`.
Shape and dtype disagreements are structural failures. Reference nonfinites are inconclusive.
Candidate nonfinites are explicit discrepancies. Matching NaNs between candidate repeats can
establish repeat stability, but cannot establish correctness against a finite reference.

At least two evaluations are required; the default is three. Material variation between repeats
is unstable and not reducible. Compiler exceptions are retried and must retain their exception
class and normalized leading message. Every repeated comparison must retain the same agreement
decision and affected-output/nonfinite classification, even when within-path differences are
small. Crossing a tolerance boundary between invocations is inconclusive. The first compiled
invocation is checked too. It is timed
separately because it includes compilation. These timings are diagnostics, not speedup evidence.

CPU float64 eager results are included for compact-IR numerical candidates. This can expose cases where
both float32 paths are similarly far from a higher-precision result. It does not decide which
implementation is correct: algorithms, floating-point semantics, and conditioning still matter.

## Reduction

The ordering is lexicographic: operator count, input elements, noncontiguous input count, and
nonzero stored input values. Proposals bypass nodes with same-shape earlier values, prune dead
code, shrink equal dimensions together or individual axes, simplify layouts, and zero whole inputs. Cropping uses
stored logical values; it does not generate a fresh unrelated random input.

The original failure category, affected outputs/nonfinite category, backend, or exception
signature must survive. A shape error, timeout, or unrelated exception cannot replace a numeric
candidate. Signatures are symptom filters, not proof of identical root cause. Graph validation
and repeated evaluation reduce false progress, but cannot eliminate that ambiguity.

Evaluation results are cached by full case hash. One evaluation is reserved for an independent
final confirmation. Exhausting a budget does not imply local irreducibility. Even reaching a
local fixed point does not establish global minimality or novelty.

When starting from a saved finding directory, the first evaluation must match the original
saved signature before any proposals are tried. Reproducing some new failure does not establish
that the old finding survived an environment change.

## Execution and persistence

The coordinator starts one worker at a time to avoid resource interference. Worker subprocesses
and compiler descendants have a wall deadline and sampled aggregate RSS budget. Shared pages
can be counted more than once, so the RSS budget is conservative. Sampling cannot prevent a
brief overshoot and does not bound GPU memory. The worker monitors coordinator liveness; normal
cancellation kills the owned process tree. This is process isolation, not a malicious-code sandbox.

Workers emit JSON atomically. The coordinator commits a settings fingerprint to each completed
outcome and rebuilds summaries from those files. Campaign locks allow a single writer. Uncommitted
worker outcomes are rerun on resume. Completed committed outcomes are retained byte-for-byte.
Resume compares package versions, runtime/device metadata, Python/platform/build, and the actual
Faultline source hash. Hashes identify inputs and source; they do not make arbitrary hardware
or compiler versions numerically interchangeable.

## Saved-finding replay

`finding.py` loads campaign case directories, worker attempts, or standalone bundles. It checks
case hashes, frozen values, settings/outcome agreement, worker jobs, campaign membership and
commit hashes when available. New exports and replays also carry content hashes for case,
settings, outcome, and environment JSON. These checks detect accidental corruption; they do
not authenticate artifacts against deliberate coordinated edits.

Replay runs in a fresh isolated worker using the saved settings and writes into a new directory.
It records the source evidence and environment differences without modifying the original. A
matching stable failure family is `reproduced`; an agreeing execution is `not_observed`; another
stable failure is `changed_failure`; timeouts, unavailable devices, instability, and other
nonreducible results are `inconclusive`. A fresh observation does not prove root-cause identity
or a compiler fix. Missing historical environment metadata is explicitly recorded. Re-exporting
a saved finding retains its original execution environment, including unknown provenance; it
does not substitute the machine performing the export.

Replay deliberately permits environment changes. Campaign resume still rejects them, since
mixing completed observations inside one campaign would hide the change in experimental conditions.

## Standalone reproducers and upstream minification

Exports contain both generated FX forward functions and exact values/layout construction in
one Python file, using only PyTorch. They can run without the Faultline package. An environment
manifest and data-only case accompany the script. The export preserves the original tolerances.
`portable.py` implements the shared predicate and is embedded verbatim in generated scripts,
including nonfinite handling, repeat classification, cold/warm agreement, and exception retries.
There is no second independently maintained copy of those semantics. CPU float64 advisories
remain a runner feature; standalone scripts report the same primary comparison and fingerprint.

A saved stable outcome becomes the expected failure in the script. Another failure or a passing
execution returns exit code 2 with `reproduces_expected=false`. The optional `--device` switch
is an explicit exploratory change and therefore cannot match an expectation for another device.
Standalone scripts do not include the process-tree watchdog; use `faultline replay` for isolated
execution with saved wall/RSS limits.
The historical benchmark separately supervises both exported scripts with those limits and checks
their reported expected outcome and exit status.

`--native-minifier` sets PyTorch's upstream AOT repro environment options before importing torch.
Whether upstream minification triggers depends on the PyTorch version and failure. Faultline's
own reducer does not import or modify private PyTorch minifier internals.

## Compact-IR limits

Stateless evaluation graphs with forward/backward checking; no optimizer state, stochastic training,
arbitrary control flow, in-place mutation, sparse tensors, distributed execution, or
performance-regression classification. Traces use concrete requests with one shared graph; they
cannot express arbitrary shape-dependent Python branches. The operator registry is deliberately
bounded. The small corpus is engineering coverage, not representative evidence for all models.
Mutant detection measures harness behavior, not compiler bug prevalence.

## Model import, backward checking, and request sequences

The `torch.export` importer uses graph-signature metadata to distinguish user inputs, parameters,
buffers, and constants. Schema 2 retains those roles and parameter trainability and adds explicit
contracts for GELU, convolution, pooling, permutation, and attention. Schema 1 serialization and
hashes remain unchanged. Translation checks compare the normalized IR's eager outputs to the
exported model. Imported operations are normalized into this IR, so capture is not byte-for-byte
preservation of the original exported graph. Unknown operators and mutations are rejected.

Backward checks differentiate a deterministic mean-square loss through eager and compiled forward
functions. AOTAutograd can therefore compile the candidate's backward path. Observations include
forward outputs, loss, gradient-presence flags, and gradient tensors. Repeats use fresh leaves and
autograd graphs, preserving strides. Gradient signatures name tensor identities. This is an
eager-versus-compiled derivative check for one loss, not finite-difference validation or a full
Jacobian proof. The controlled wrong-gradient fixture preserves forward values but doubles the
derivative; it is clearly labeled as injected.

A trace contains ordered frozen cases sharing graph structure and model state. Each full trace
creates a fresh callable and invokes it sequentially; repetition replays the whole trace. Compiler
process-global caches are not forcibly cleared between repetitions. Isolated reduction attempts
run in new worker processes. Evidence records failed request indices. Failure signatures retain
the affected tensor identity and category but omit the request index, allowing earlier requests
to be deleted. Sequence length precedes graph/input size in the reduction ordering. Accepted
proposals must retain the repeated whole-trace failure, and the final trace is independently
confirmed. Signatures do not prove which earlier request caused an error.

## Environment regression gates

Protocol-versioned jobs run through explicitly selected Python interpreters, with the same harness
source, frozen cases, and comparison policy. Each worker records its own package/runtime metadata.
Baseline failures, unavailable environments, and unstable comparisons cannot be presented as new
candidate regressions. Automatic reduction invokes the baseline first and rejects any proposal
that stops passing there; the candidate must then retain the original stable failure. A fresh
dual-environment confirmation precedes export. Source-controlled CI pins both compiler releases
and retains comparison artifacts even when the gate fails.

## Pretrained sweep coordinator

`sweep_models.py` resolves the pretrained catalog, prepares real inputs, and writes offline factories
with exact model weights and inputs. `sweep.py` captures these through the native workload interface
and executes the existing two-environment comparison. The coordinator does not implement another
numerical oracle. Preparation and execution failures remain in the planned model denominator.

Campaigns lock their output directory and record source, input, settings, and environment identity.
Each attempt commits hashes of its result documents only after the comparison is complete. Resume
verifies completed documents and starts a new attempt for interrupted work, retaining earlier
artifacts. Numerical labels are advisory; counts distinguish pair agreement, stable findings,
inconclusive execution, and findings without sufficient diagnostic evidence. See [sweeps](sweeps.md).

## Accuracy bisection

`bisect_env.py` discovers compatible CPU nightly wheels and supervises isolated venv installation,
package probes, and wheel verification. `bisect.py` freezes a reference error vector, searches
nightly dates using repeated observations, and independently confirms neighboring builds.
It uses the existing native worker and preserves primary results. The new predicate and its
reports are advisory and never feed the reducer or the primary oracle. See [bisection](bisect.md).

`accuracy.py` owns the cross-version fraction and distributional advisories. `native_precision.py`
retains absolute errors, legacy ratios, and the declared noise floor. `native_numerics.py` combines
this evidence with mask reachability and same-input backward checks for advisory triage.

## Generated-code inspection

`code_diff.py` inventories Inductor debug dumps without importing or executing generated code.
It records normalized source hashes, static call sites, and separate IR/FX comparisons, retaining
duplicate capture occurrences and missing companion artifacts. The comparison writes to a new
directory and has no connection to primary verdicts or reduction. Equal emitted code does not
imply equal implementations in linked libraries or included headers. See the
[CPU accuracy investigation](accuracy-investigation.md) for its use and limits.
