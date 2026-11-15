# Compilation structure

Native checks and sweeps record advisory telemetry from their existing compiled primary
invocations. The report compares baseline and candidate frame attempts, captured Dynamo
graphs, graph-break events/reasons, recompilation events/guard-failure reasons and compile
wall time. Reasons are deduplicated with available source filenames and line numbers;
event counts remain separate. Per-batch records, adapter coverage and errors are in JSON.

The [fixed v1 policy](compilation-policy-v1.json) was written before the 0.20 measurements:

| Note | Condition |
|---|---|
| `new_graph_breaks` | A normalized reason/location appears only in the candidate of a complete paired batch. |
| `recompile_increase` | At least one more candidate recompilation event in a complete paired batch. |
| `compile_time_increase` | Candidate mean compile wall time is at least 1.5 times baseline and at least 10 seconds higher per batch, with complete timing coverage for all planned pairs. |

These are summary notes. They cannot change v2 verdicts, primary statuses, tolerances,
signatures, reducer acceptance or exit codes. Fewer graphs or fewer compilations do not
establish correctness, and an increase does not establish a compiler defect.

## Collection and limits

`compile_tracking.py` contains the version-dependent adapter. It takes deltas of
`torch._dynamo.utils.counters`, reads `graph_break_reasons`, and consumes existing
`get_compilation_metrics()` records. These are internal PyTorch interfaces, isolated
behind an adapter tested against actual 2.15.0 and nightly CPU Inductor. There is no stable
public API providing all requested counters. The collector does not call `explain`, add
model executions, reset counters, enable TORCH_LOGS, or change compiler configuration.
PyTorch also documents [graph-break and recompile logging](https://docs.pytorch.org/docs/2.15/torch.compiler_troubleshooting.html)
for interactive investigation outside the saved checks.

Only the compiled cold call, existing warm repeats, and any existing exception confirmations
are in scope. Eager execution and optional fp64/mask/control replays are excluded. Lazy
backward compiler events are included. Frame counts are attempts, not distinct Python source
functions. A failed guard may supply several reasons for one recompile; the count uses
compilation events with a recompile reason, not the number of reason strings.

Compile wall time is the union of reported forward/backward compilation intervals, so nested
events are not counted twice. It includes tracing and cache lookup. It excludes worker startup,
model loading, comparisons and ordinary tensor execution. These operational observations
depend on machine load, compiler cache state and how many repeats/frames were executed;
they are not an isolated compilation benchmark.

Each input batch still gets a fresh worker. A dynamic-shape request sequence inside one
native workload can show recompilation, but separate batch workers cannot reveal recompilation
across batches. Totals sum independent workers; they do not describe one persistent serving
process. The graph-break and recompile notes compare paired batch evidence, and the time note
uses the same fully covered batch set in both environments.
The summary uses each batch's effective committed primary result. Earlier memory-limited
attempts remain in resource evidence; an aborted worker may not have produced complete
telemetry and cannot be treated as a zero-cost compilation attempt.

Unavailable fields and metric-buffer overflow produce partial/unavailable coverage. An absent
adapter is not evidence of zero graph breaks. Primary execution proceeds unchanged, and old
artifacts without telemetry remain readable. Timing values and absolute event timestamps are
observational metadata, not failure signatures or resume predicates.
