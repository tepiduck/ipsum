# Results: ipsum v1

v1 ended as a clean negative result.

An online, inspectable abstraction store did **not** produce compounding improvement
on CI predictive test selection. The main reason was not mysterious: the task was
mostly solved by a simple historical test-failure-rate feature. Co-change
abstractions found real file-change signal, but that signal did not add much once the
cheap baseline feature was already in the model.

This means the v1 task was a poor place to demonstrate the larger idea. It does not
prove that compounding through abstractions is impossible. It does say that CI test
selection, at least in this RTPTorrent setup, does not have enough headroom for this
particular abstraction store to matter.

## What was tested

The broad question was whether a system could improve over time by accumulating
explicit, readable abstractions.

The v1 task was predictive test selection:

- Input: a code change.
- Output: a ranked subset of tests to run.
- Budget: fixed selection rate.
- Metric: TestRecall at that fixed selection rate.
- Abstractions: co-change file or directory clusters.

The important comparison was ipsum against a **data-matched, abstraction-off
control**. That control saw the same stream and the same labels, but could not use
the abstraction store.

Weekly retraining was included as a baseline, but it was not the thesis test.
Beating weekly retraining mostly shows that online cumulative data is useful. The
real question was whether abstractions add anything beyond that.

## Synthetic work

The synthetic testbed came first because it lets us know when a mechanism should
work. The model never sees the oracle state; the oracles are only used for scoring.

The synthetic phase gave mixed but useful results:

- **Instrument self-check: pass.** On drifted synthetic data, the harness detected a
  planted abstraction advantage. With the abstraction store disabled, the negative
  control stayed exactly flat. This made the measuring stick trustworthy.
- **Admission: modest pass.** The held-out log-likelihood gate recovered true
  clusters much better than admit-everything across the cluster sweep.
- **Eviction: iterate.** Eviction helped post-drift plateau accuracy on average, but
  the effect was not strong enough, and eviction precision/recall were weak.
- **Coverage guard: not demonstrated.** The mechanism was implemented correctly, but
  the expected benefit did not grow cleanly with coverage skew.

The pattern mattered. Nothing exploded, but nothing became a strong result either.
By the time the project moved to real data, the prior should already have been lower.

## Real-data work

The real-data runs used RTPTorrent v1. The stream was built from three CSVs:

1. per-project test results,
2. `tr_all_built_commits.csv`,
3. per-project patch files.

Jobs were ordered by ascending `travisJobId`. Labels were de-flaked before counting.
Jobs touching more than 30 files were dropped as large infra or merge noise.

Two issues had to be fixed before the real-data result meant anything:

- **Project coverage.** sling looked attractive because it had many failing tests, but
  only about 16% of jobs could be joined to changed files. Its null result was a
  coverage artifact.
- **Change granularity.** Full file paths were too sparse. Coarsening changed files
  into directory tokens restored candidate proposal support.

### okhttp

okhttp was the meaningful v1 real-data run.

- Raw jobs: `9,772`
- Used cycles: `9,703`
- Changed-file coverage: `52%`
- Full paths collapsed to change tokens: `971 -> 64`
- Candidates proposed: `9,691`
- Abstractions admitted: `12`
- Abstractions retained at the end: `0`
- ipsum vs data-matched slope gap: `0.0`
- plateau gap: `0.0`

In other words, the pipeline worked and admissions happened, but the abstractions did
not improve prediction.

### sonarqube

sonarqube was wired as the second project.

- Raw jobs: `53,307`
- Used cycles: `49,179`
- Changed-file coverage: `31%`
- Full paths collapsed to change tokens: `6,605 -> 159`
- Candidates proposed: `2,734`
- Abstractions admitted: `0`

Because it admitted nothing, the sonarqube slope result is not a thesis result. It is
an admission-starved diagnostic.

## The decisive diagnostic

To check whether this was an implementation failure or a task failure, we compared
against RTPTorrent's own baseline strategies on okhttp. The metric was how much of
the suite had to run before the first failing test was found. Lower is better.

| strategy | first failure at | signal type |
|---|---:|---|
| `optimal-failure` | 0.073 | oracle ceiling |
| `recently-failed` | 0.109 | historical failure rate |
| `matrix-naive` | 0.187 | file-change signal |
| `matrix-conditional-prob` | 0.287 | file-change signal |
| `random` / `untreated` | ~0.51 | no useful signal |

This explains the outcome:

- File-change signal exists.
- Historical failure rate is much stronger.
- Historical failure rate is already close to the oracle ceiling.
- Co-change abstractions are therefore mostly redundant on this task.

That matches the admission funnel: held-out LL gains were near zero.

## What the result means

This result supports a narrow claim:

> Co-change abstractions do not compound on this CI test-selection setup.

It does not support the broader claim:

> Explicit abstractions can never create compounding expertise.

A fairer next task would need more headroom: something where cheap historical-rate
features do not already capture most of the predictable signal. Incident prediction,
review routing, flaky-test diagnosis, or regression triage might be better candidates.

## Lessons worth keeping

A few failure modes showed up repeatedly:

- A positive control can show a one-time level shift without showing compounding.
- Recovery-time slopes are dangerous when unrecovered epochs are censored.
- A metric can improve because its denominator changed, not because the mechanism did.
- Real-data nulls are meaningless until coverage and candidate proposal are visible.
- When a variant "wins," diff its predictions against the control. If predictions are
  identical, the win is in the measurement, not the mechanism.

The standing rule for future work is simple:

> Report uncertainty, report worst seeds, and do not trust a pass until the control
> comparison is visibly different in predictions, not just in labels or bookkeeping.

## Status

v1 is complete. No further mechanism tuning is justified on this task.

Reusable pieces remain:

- synthetic testbed,
- compounding harness,
- RTPTorrent loader,
- artifact format,
- research notes,
- and the experimental scars in `RESEARCH_LOG.md`.

Artifacts live under `experiments/runs/`. The detailed experiment record is in
`RESEARCH_LOG.md`.
