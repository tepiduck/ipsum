# Data notes

The real-data work used RTPTorrent v1, a dataset of Java projects with Travis CI
test results. The useful part for ipsum is the per-build stream:

```text
build job -> changed files -> per-test outcomes
```

The dataset does not ship that stream as one table. It has to be joined from three
CSV files.

## Files used

For a project such as okhttp:

- `square@okhttp/square@okhttp.csv`
  - per-test results
  - key column: `travisJobId`
  - test column: `testName`
- `tr_all_built_commits.csv`
  - job to commit mapping
  - columns: `tr_job_id`, `git_commit_id`
- `square@okhttp/square@okhttp-patches.csv`
  - commit to changed file mapping
  - columns: `sha`, `name`

For each job, the loader takes the union of all patch files for all commits attached
to that job. Jobs touching more than 30 files are dropped as merge or infra noise.

No git diff reconstruction is used for this v1 path.

## Project notes

These numbers came from the original RTPTorrent project profile and the later v1 join
checks.

| Project | Builds | Failure density | Notes |
|---|---:|---:|---|
| okhttp | 9,772 | medium | best v1 changed-file coverage; use first |
| sonarqube | 53,307 | low-medium | much larger; useful second project, heavier to run |
| sling | 8,552 | high | many failures, but poor job-to-file coverage in this v1 join |
| DSpace | 3,338 | medium | possible breadth project |
| jade4j | 932 | high | small, dense, useful for quick checks |
| graylog2-server | 10,622 | very low | not useful for v1; too few failures |

The important lesson was that failure density alone is not enough. sling looked good
on raw failing tests, but the job-to-commit-to-patch join covered too little of the
stream, so its zero-admit result was not meaningful.

For v1, the useful projects were:

- **okhttp** as the main real-data smoke test,
- **sonarqube** as the larger second project.

## Running the harness

Example for okhttp:

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

Example for sonarqube, using a coarser large-project cadence:

```bash
python experiments/compounding_rtptorrent.py \
  --dataset sonarqube \
  --project-csv /path/to/rtptorrent/SonarSource@sonarqube/SonarSource@sonarqube.csv \
  --built-commits-csv /path/to/rtptorrent/tr_all_built_commits.csv \
  --patches-csv /path/to/rtptorrent/SonarSource@sonarqube/SonarSource@sonarqube-patches.csv \
  --change-granularity directory \
  --change-depth 3 \
  --min-support 2 \
  --cochange-threshold 0.02 \
  --max-candidates 64 \
  --admission-interval 1000 \
  --admission-warmup 1000 \
  --eval-interval 1000 \
  --eval-window 1000
```

Each run writes artifacts under:

```text
experiments/runs/<run_id>/
```

The files are shaped by [INTERFACE.md](../INTERFACE.md).

## Change granularity

Full file paths were too sparse on real projects. For okhttp, full paths produced
hundreds of mostly singleton identifiers. Coarsening to directory tokens restored
candidate proposal support:

```text
a/b/c/Foo.java -> a/b/c    # directory depth 3
```

The harness supports:

- `--change-granularity path`
- `--change-granularity directory`
- `--change-granularity java_package`

and:

- `--change-depth N`

## What to check before trusting a run

Do not read a flat slope as a thesis result until these are nonzero and sane:

- changed-file coverage,
- candidates proposed,
- candidates passing the support diagnostic,
- admitted count.

The admission funnel is logged into `events.json` and summarized in `metrics.json`.
It exists because early real-data nulls were not real results. They were coverage and
granularity artifacts.

## Historical caveats

RTPTorrent covers old Travis-era projects, mostly 2007-2016. That brings a few limits:

- build and logging behavior can change over time,
- jobs can represent different build configurations,
- retry information is limited,
- branch and timestamp metadata are not present in the v1 zip used here,
- some projects have weak job-to-commit coverage.

Those caveats do not make the data useless. They just mean every run needs coverage
and funnel diagnostics before any slope is interpreted.
