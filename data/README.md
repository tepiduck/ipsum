# Data — RTPTorrent project selection

Profiled from RTPTorrent Table 2 (Mattis et al., MSR 2020). For measuring a
**slope**, two things matter: (1) **many build jobs** = many CI cycles in
chronological order = resolution to see `dQ/dt`; (2) **enough failing test
methods** = positive labels to learn from (CI is heavily green-imbalanced, so
low-failure projects starve the signal).

The decisive metric is **total failing-test-method observations ≈ Builds × (TM Failing/build)** —
how many positive examples the whole replay yields.

## Per-project profile (sorted by total positives)

| Project | Builds (CI cycles) | TM failing / build | ≈ Total positives | Lines | Notes |
|---|---:|---:|---:|---:|---|
| sling | 8,552 | 7.38 | **~63,100** | 673k | high raw signal, but poor v1 job→commit→file coverage; drop for compounding v1 |
| **sonarqube** | 53,307 | 0.68 | ~36,200 | 661k | most CI cycles by far → best slope resolution; heavy |
| **okhttp** | 9,772 | 1.62 | ~15,800 | 69k | clean, well-known, mid-size; great starter |
| DSpace | 3,338 | 2.37 | ~7,900 | 384k | balanced mid-size |
| jade4j | 932 | 7.55 | ~7,000 | 10k | tiny + dense → fast iteration |
| titan | 1,075 | 5.97 | ~6,400 | 60k | dense, few cycles |
| Achilles | 997 | 5.46 | ~5,400 | 54k | dense, few cycles |
| dynjs | 1,020 | 5.23 | ~5,300 | 57k | dense, few cycles |
| buck | 1,148 | 3.48 | ~4,000 | 563k | large codebase, few cycles |
| cloudify | 5,206 | 0.29 | ~1,500 | 133k | many cycles, weak signal |
| jOOQ | 3,245 | 0.34 | ~1,100 | 351k | weak signal |
| graylog2-server | 10,622 | 0.10 | ~1,060 | 127k | AVOID: many cycles, almost no failures |

(Other projects in the set: HikariCP, LittleProxy, deeplearning4j, jcabi-github,
jetty.project, jsprit, optiq — generally lower on one axis. Full set is 20 projects.)

## Recommended picks

**Tier 1 — start here (2 projects):**
- **okhttp** — clean, well-known, mid-size, and the best v1 changed-file coverage among the current real-data runs.
- **sonarqube** — 53k builds gives the finest-grained slope curve; use it as the heavier second project.

**Tier 2 — add for breadth:**
- **DSpace** (balanced) and **Achilles** or **jade4j** (small + dense → fast debug loops).

**Avoid** for v1: sling, graylog2-server, cloudify, jOOQ. Sling has many raw
positive labels, but the RTPTorrent v1 job→commit→patch join covers too few jobs
for a change-aware abstraction test; the others have too few failures per build,
so the positive class is too thin to learn or to measure improvement reliably.

## Caveats (carry into analysis)
- Travis logs span 2007–2016; logging/config changes can make pre/post test runs
  non-comparable within a project (paper §3.3 "Reliability"). Segment around config changes.
- Multiple build types/platforms interleave (one build → several jobs). De-duplicate
  to a canonical job per commit before replay.
- Durations < Java clock resolution are logged as 0.0s (9.28% of test cases); 17 negative
  durations exist. Clean before using duration as a feature/cost.
- IDs join to TravisTorrent (`travistorrent_8_2_2017.csv`) for branch/timestamp, and
  SHA1s join to GHTorrent for PR/issue/author links.

## How to fetch + verify
RTPTorrent is on Zenodo (record 3712290). Download there, then run the profiler to
confirm these numbers on the actual CSVs:

```bash
cd ~/issum/ipsum
python data/profile_rtptorrent.py /path/to/rtptorrent
```

Use `data/rtptorrent.py` to load one project CSV into chronological CI cycles
with per-test outcomes. Pass `changes_csv=` when you have job- or commit-keyed
changed-file metadata; the loader supports common `job_id` / `commit_id` and
`file_path` / `changed_files` aliases. The RTPTorrent per-test CSV alone does not
include changed-file sets, so change-aware baseline features and ipsum
abstractions require that metadata join.

Once the project CSV, `tr_all_built_commits.csv`, and project patches CSV are
available, run the real-data compounding harness with:

```bash
python experiments/compounding_rtptorrent.py \
  --dataset okhttp \
  --project-csv /path/to/rtptorrent/square@okhttp/square@okhttp.csv \
  --built-commits-csv /path/to/rtptorrent/tr_all_built_commits.csv \
  --patches-csv /path/to/rtptorrent/square@okhttp/square@okhttp-patches.csv \
  --change-granularity directory \
  --change-depth 3 \
  --min-support 2 \
  --cochange-threshold 0.02
```

The command writes `experiments/runs/<run_id>/` artifacts for weekly-retrain,
data-matched abstraction-off control, and ipsum. Changed files are joined from
RTPTorrent v1 metadata as
`travisJobId -> tr_all_built_commits.git_commit_id -> patches.name`; jobs whose
changed-file union exceeds 30 files are dropped as infra/merge noise. The harness
coarsens changed files before candidate proposal, for example
`a/b/c/Foo.java -> a/b/c` at directory depth 3, and logs the admission funnel so
zero-admit runs distinguish proposal starvation from strict admission.
