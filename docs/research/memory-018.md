# 0.18 memory scheduling

This release changes execution scheduling, not verdict v2, primary comparisons, tolerance,
failure signatures, reducer predicates, or exit codes. All 0.17 evidence is retained.

Before measurements, the default total budget is fixed at 75% of available RAM, capped at
32 GiB. Availability comes from the operating system and, where mounted, the Linux cgroup
memory limit. The first frozen batch runs alone in both environments at the full budget.
The largest measured worker-tree RSS determines concurrency with 25% headroom, capped at
four workers, half the available CPUs, and the planned batch count. Unknown peaks or a
memory-limited first batch select one job. Auto accuracy diagnostics run sequentially and
use the full budget. Neither a probe nor a retry changes numerical policy.

Explicit job counts retain their initial equal budget allocation. Any recorded parallel
memory limit receives a full-budget serial retry after the pool drains; both attempts are
preserved. Generic crashes and timeouts are not inferred to be OOM. A serial failure stays
visible. Completed schedules and budget choices are retained across resume, with source,
settings, environments and evidence hashes checked as before.

The fixed measurement queue uses the original frozen 0.16 collections and stable 2.15.0+cpu
versus 2.16.0.dev20261105+cpu: ResNet-50, BERT, GPT-2 and SmolLM2 with auto scheduling, then
BERT and GPT-2 with one job. Standard level keeps ten primary batches, two repeats, one
thread, the 600-second worker timeout and the existing accuracy stopping rule. Each cell
starts in a new directory with empty slot caches. Results include chosen budgets, measured
peaks, retries, coverage and wall time; incomplete executions are not speedups. The target
is a conclusive transformer check in roughly 10–15 minutes, not a changed acceptance rule.

The missing GPT-2 score from 0.16 is batch index 9: the baseline's combined numerics worker
was terminated at its RSS limit (14,707 MiB sampled peak), after saving a primary result.
The candidate completed. The missing score and original 9/10 count are preserved.
The current separated diagnostic path is checked
on that exact frozen batch in a new directory.

## Measured results

| Model | Requested → chosen jobs | Total MiB | Probe MiB | Minutes | Primary pairs | Accuracy scores | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| resnet50 | auto → 4 | 10981 | 1751 | 6.44 | 10/10 | 3/10 | SAFE |
| bert | auto → 1 | 10957 | 6094 | 21.56 | 10/10 | 8/10 | SAFE |
| gpt2 | auto → 1 | 11069 | 7405 | 20.90 | 10/10 | 5/10 | SAFE |
| smollm2 | auto → 1 | 11100 | 7318 | 20.19 | 10/10 | 4/10 | SAFE |
| bert | 1 → 1 | 11104 | — | 20.96 | 10/10 | 8/10 | SAFE |
| gpt2 | 1 → 1 | 11112 | — | 20.93 | 10/10 | 5/10 | SAFE |

All six cells remain in the denominator. Unmeasured accuracy batches remain marked as skipped,
with a stopping certificate explaining why they cannot change the decision. A completed decision can
therefore use fewer than ten scores while all ten primary pairs still run.

The fresh BERT and GPT-2 explicit-one-job runs matched their auto runs exactly in primary
comparisons, completed accuracy distributions and verdicts. Auto selected one job for both
models on this host.
Recorded full-budget serial retries across these six cells: 0. The memory-retry
behavior is also covered by controlled resource-limit and interrupted-retry tests.

All five transformer cells missed the 15-minute-and-decisive target.
Budgets were selected independently from available RAM at each invocation and retained in
the evidence. All cells used one CPU thread and `TORCHINDUCTOR_COMPILE_THREADS=1`.
The host was a Ryzen 7 7700X (16 logical CPUs); the WSL guest reported 15.2 GiB total RAM.
The developer suites finished before the timing queue; other host applications were uncontrolled.
These are frozen-workload operational costs, excluding download/capture, not isolated benchmarks.
The 0.17 timing series used same-version pairs, so it is not a direct upgrade-speedup baseline.
Workers load the frozen 0.18 harness through its bootstrap; older installed Faultline package
metadata in an environment does not identify the executed harness. The recorded source hash does.

The exact former batch-9 workload produced 1/1 complete paired accuracy scores in the current
separated diagnostic path. Its single-batch verdict is INCONCLUSIVE: one batch cannot satisfy
the unchanged ten-batch replication requirement. The frozen bundle hash matches the old
AOT/numerics job. This targeted coverage check leaves the 0.16 results unchanged.

The [machine-readable record](memory-018.json) retains every cell, selected schedule,
budget, stopping certificate, primary/diagnostic status count, aggregate worker duration,
environment identity and evidence hash. Worker-duration sums include overlapping workers;
they are not elapsed times. `scripts/validate_check_memory.py` runs the fixed queue and
`scripts/summarize_check_memory.py` verifies commits before compacting the evidence.

SmolLM2's actual Inductor graph contains `torch._C._set_grad_enabled`, which the independent
graph exporter does not support. Its full frozen workload remains available; this graph-export
limitation does not prevent the whole-model primary comparisons or accuracy checks.

Executed checks: Linux portable 390 passed; Windows 389 passed and one skipped;
actual CPU Inductor 13 passed; legacy 31 passed; all 14 compiler validation checks passed;
Ruff lint and formatting passed. Tests cover measured allocation, memory retries after
the pool drains, full-budget failure, diagnostic retry, interruption/resume, unchanged
serial/parallel decisions and actual eager/Inductor result parity.
The installed wheel also completed a three-batch CPU Inductor upgrade check with default
auto scheduling and resumed it without changing its budget or committed evidence.

All 43 protected evidence/policy files and 25 prior validation entries
are preserved. Tested raw runtime: `0aead688339c0e8cbb4bc325d489639096020717d7ba208853873e2440435355`; Git-normalized runtime:
`abd41cb00a5e3195d4de8a79374bfdaa8afaa2eac8536a7c0d95798056a39aa2`. Normalized package bytes match the frozen source used by workers.
Nothing was filed upstream and no compiler defect or new GPU validation is claimed.
