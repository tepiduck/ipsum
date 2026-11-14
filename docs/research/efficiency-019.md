# 0.19 bounded efficiency pass

The baseline is 0.18 (`23243a8`). A standard GPT-2 check uses the retained ten frozen
input batches, PyTorch 2.15.0+cpu and 2.16.0.dev20261105+cpu, one CPU thread, two repeats,
auto scheduling and a new cache/output directory. Numerical rules and earlier evidence
remain unchanged.

Worker timers record exclusive wall time for startup, model setup, snapshot loading,
eager/compiled forward and backward, compiler calls, observation/reset, comparisons,
serialization and fp64 statistics. Repeat time is a tagged subset of those categories,
not another amount to add. Compiler timing includes backend preparation and cache loading;
PyTorch's nested compilation metrics are retained separately.
Compiled forward time includes Dynamo tracing outside the measured Inductor backend calls.
Tensor serialization is timed directly; JSON/report construction and artifact hashing remain
in worker or coordinator overhead rather than being presented as pure model execution.

RSS is sampled every 50 ms. Weak references inventory identifiable CPU tensor storages,
deduplicating aliases, without retaining model observations. Saved activations are also
measured at forward boundaries. fp64 storage is an exclusive category. The residual
includes allocator arenas, libraries, comparison temporaries and opaque native allocations;
logical tensor bytes are not a complete accounting of resident pages. Independent category
maxima must not be added together as if they occurred at the same instant.

Instrumentation smoke tests preceded the full measurement. Earlier attempts are retained
locally: an empty closure cell caused a profiling error; a compiler callback default bypassed
one timer; and frame-local inspection extended tensor lifetimes. The final instrumentation
uses weak references and checks that compilation is actually timed.

## Baseline GPT-2 profile

The complete instrumented command took 1,388.50 seconds (23.14 minutes), versus 1,253.80
seconds in the preserved uninstrumented 0.18 run. This is a profiling run with overhead,
not an isolated timing comparison. It completed all ten primary pairs and five fp64 pairs
before the unchanged early-stopping rule fixed the accuracy decision. All numerical
evidence, advisory distributions, verdict reasons and stopping decisions match 0.18 exactly.

| Exclusive category | Seconds |
|---|---:|
| Worker startup, 30 workers | 94.76 |
| Model construction | 94.85 |
| Loading snapshots | 48.80 |
| Input reset and observation copies | 139.36 |
| Eager forward / backward | 15.31 / 36.70 |
| Inductor compilation and cache loading | 66.66 |
| Compiled forward / backward, excluding compiler | 56.94 / 53.79 |
| Pointwise comparisons and reference health | 557.56 |
| fp64 forward / backward / error statistics | 8.51 / 23.02 / 31.44 |
| Graph capture, excluding serialization | 6.36 |
| Tensor serialization | 20.83 |
| Other worker overhead | 52.54 |
| Outside timed workers: coordinator, supervision, reports, process teardown | 81.05 |

Repeat executions account for 173.67 seconds **within** these categories. This subset
includes eager, compiled and fp64 repeat execution/reset; comparisons performed after
those executions are in the comparison row, not in the subset. Primary workers contribute
829.40 seconds and diagnostic workers 478.05 seconds. The JSON retains both phase breakdowns.

The largest sampled worker-tree RSS was 7,469.83 MiB, during comparison:

| Category at that instant | MiB |
|---|---:|
| Known model/checkpoint weight storage | 949.40 |
| Retained observation and captured-input storage | 3,324.03 |
| Raw gradients, saved activations, fp64 storage | 0 identified live |
| Compiler subprocesses | 0 observed live |
| Unattributed worker RSS | 3,196.40 |

At other recorded boundaries, identified raw gradients reached 474.70 MiB, activations
126.85 MiB and fp64 storage 3,799.10 MiB. These are separate maxima, not extra allocations
to add to the peak. Compiler subprocesses were absent at the reported peak and recorded
forward/backward boundaries; this does not establish a zero transient compiler-process peak.
Gradient copies held in observations belong to the retained-observation category; the raw
gradient category covers the returned autograd tensors before observation ownership.

Comparisons account for 40% of command wall time; cached compilation accounts for much less.
The first warm GPT-2 worker spent 17.49 seconds in comparisons and 0.47 seconds in compiler
calls. These measurements selected bounded comparison scratch space and shorter lifetimes/
shared immutable state copies. Fresh-worker isolation remains in place: a persistent-worker
default was not justified by the measured warm-cache compiler cost in this pass.

All four fresh checks exactly match the retained 0.18 primary statuses, signatures, per-target
comparison fields, fp64/control evidence, replicated accuracy distributions, verdict reasons
and stopping decisions. Source, resource, timing and scheduling metadata differ. No tolerance
or decision threshold was changed.

## Repeating the profile

With Faultline installed, use an existing frozen collection and separate compiler environments:

```bash
python scripts/profile_check_workers.py captured/bundle \
  --baseline /path/to/stable/python --candidate /path/to/nightly/python \
  --output faultline-runs/profile-new
```

The script copies the selected harness into a new directory and instruments that copy.
It leaves the installed package and captured inputs unchanged. Per-worker `profile.json`
files retain exclusive durations, repeat subsets, tensor inventories and sampled RSS.
`--package /path/to/faultline` selects another source version. This research profiler uses
private PyTorch compiler hooks, tested with the retained 2.15/2.16 environments; it is not
part of the check decision path. The published full-check profile uses the initial category
split; a supplemental before/after primary run separates retained gradient copies and tracks
child-process RSS throughout compilation. Neither replaces the uninstrumented timing series.


## Fresh uninstrumented checks

| Model | 0.18 min | 0.19 min | Jobs before → after | Probe MiB before → after | Accuracy pairs | Verdict |
|---|---:|---:|---|---|---:|---|
| resnet50 | 6.44 | 6.58 | 4 → 4 | 1751 → 1331 | 3/10 | SAFE |
| bert | 21.56 | 16.11 | 1 → 2 | 6094 → 3091 | 8/10 | SAFE |
| gpt2 | 20.90 | 15.02 | 1 → 2 | 7405 → 3431 | 5/10 | SAFE |
| smollm2 | 20.19 | 15.63 | 1 → 2 | 7318 → 3839 | 4/10 | SAFE |

All four checks completed all ten primary pairs. Fewer than ten accuracy pairs means the
unchanged negative stopping certificate was decisive; no skipped batch receives an invented score.
These are single-run operational costs on the same Ryzen 7 7700X/WSL host, using CPU
2.15.0+cpu and 2.16.0.dev20261105+cpu. Downloads and initial model capture are excluded.
The memory-derived budget is selected independently at invocation time; the JSON records
each budget, peak, retry and schedule. The probe column is primary worker-tree RSS, not
the maximum across fp64 diagnostics or combined concurrent workers.

See [the portable record](efficiency-019.json) for the full-check profile, supplemental
primary profiles, per-model schedules, exact-evidence checks and original artifact hashes.
Fresh-worker isolation remains the default and only batch execution mode.

Executed checks: Linux 407 passed; Windows 406 passed and one skipped; actual CPU Inductor
13 passed; legacy 31 passed; all 14 compiler campaign checks passed; Ruff lint and formatting
passed. A standalone installed-wheel check and its resume also passed. The first Windows
run exposed two mocked scheduling tests depending on live host RAM; a declared-memory
fixture fixed those tests, and both the focused Linux checks and full Windows rerun passed.
No production resource limit or scheduling policy changed. All 26 prior validation entries
and 64 protected documentation/evidence files are preserved. Nothing was filed upstream.

Cells without a lower measured wall time: resnet50.
This single-run comparison does not establish the statistical significance of small differences.


## Peak RSS including diagnostics

| Model | 0.18 peak worker MiB | 0.19 peak worker MiB |
|---|---:|---:|
| resnet50 | 1844 | 1678 |
| bert | 6316 | 5375 |
| gpt2 | 7405 | 5982 |
| smollm2 | 7531 | 6189 |

These maxima include fp64 workers, which retain reference gradients in addition to the
fp32 eager/compiled observations. The 2–3-GiB aspiration is not met throughout transformer
checks. Smaller primary peaks can allow more auto jobs while sequential fp64 diagnostics
still require a larger full-budget worker. The JSON retains both kinds of measurements.


## Supplemental primary profiles

| Supplemental stable GPT-2 primary | 0.18 | 0.19 |
|---|---:|---:|
| Worker time, s | 59.90 | 48.21 |
| Comparison time, s | 17.67 | 5.48 |
| Reset and observation time, s | 4.17 | 3.00 |
| Compiler time, s | 22.24 | 22.75 |
| Peak worker-tree RSS, MiB | 7234.62 | 3413.77 |
| Weights at peak, MiB | 949.40 | 474.70 |
| Gradient observation copies at peak, MiB | 1424.10 | 949.40 |
| Other retained observations at peak, MiB | 1899.93 | 950.62 |
| Raw gradients at peak, MiB | 0.00 | 0.00 |
| Activations at peak, MiB | 0.00 | 0.00 |
| fp64 copies at peak, MiB | 0.00 | 0.00 |
| Compiler subprocesses at peak, MiB | 0.00 | 0.00 |
| Unattributed worker RSS at peak, MiB | 2961.19 | 1039.05 |
| Largest sampled compiler-subprocess RSS, MiB | 270.02 | 268.09 |

These are separate, fresh-cache, single-threaded stable-PyTorch batch-0 workers using the
same frozen GPT-2 input. Their primary evidence agrees exactly. The refined inventory
separates gradient observation copies and samples child-process RSS even when total RSS
is below its prior maximum. They are primary-only runs; the complete profile above includes
the fp64 workload. Logical storage inventories and RSS remain distinct measurements.

The peak moved from comparison in 0.18 to snapshot loading in 0.19. Category values
at those different instants are not measurements of identical execution phases. The
complete boundary samples remain in the JSON. Comparison time fell from 17.67 to 5.48
seconds; reset/observation time fell from 4.17 to 3.00 seconds in this supplemental pair.
