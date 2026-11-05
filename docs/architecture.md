# Faultline architecture and validity

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

CPU float64 eager results are included for numerical candidates. This can expose cases where
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

`--native-minifier` sets PyTorch's upstream AOT repro environment options before importing torch.
Whether upstream minification triggers depends on the PyTorch version and failure. Faultline's
own reducer does not import or modify private PyTorch minifier internals.

## Current limits

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
