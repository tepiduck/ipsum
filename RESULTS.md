# RESULTS — ipsum v1

**Verdict: a clean, well-explained negative.** An online, inspectable abstraction
store (co-change file/directory clusters with statistically-gated admission and active
eviction) **does not produce compounding improvement on CI predictive test selection.**
The cause is identified and independently confirmed: on this task the predictable signal
is dominated by a trivially cheap feature (historical test-failure rate), leaving no
headroom for accumulated abstractions to add value. This is a *task-instantiation*
failure, not an engineering failure and not a refutation of the broader thesis.

---

## 1. What was tested

**Thesis.** A system can acquire domain expertise that *compounds* — its rate of
improvement keeps climbing where a stateless model plateaus — by accumulating explicit,
inspectable abstractions and reusing them.

**Instantiation (v1).** Domain = software CI. Task = predictive test selection (given a
code change, predict which tests can fail). Abstractions = co-change file/directory
clusters. Decision = which tests to run under a fixed budget. **Compounding metric** =
does ipsum's TestRecall@SelectionRate gap over a **data-matched, abstraction-off control**
*widen* with experience. Beating a weekly-retrain-from-scratch baseline is table stakes
(it only shows online > batch); the thesis lives entirely in the gap over the
data-matched control.

---

## 2. Synthetic phase — instrument validated, mechanisms marginal

A controllable synthetic testbed with ground-truth oracles was built first to debug
mechanisms where the right answer is known.

**Instrument self-check (Card I) — PASS.** On a drifting synth stream, the harness shows
a widening ipsum-vs-control slope gap *only* when the abstraction store is enabled
(early/mid/late gap `0.000 / 0.183 / 0.287`), and **exactly zero** when ipsum is
byte-equivalent to the control (negative control parity). The measuring instrument is
trustworthy — it detects planted compounding and does not fabricate it.
(`runs/20260623-011049-instrument-*`)

**Admission (Card A) — PASS, modest.** A one-standard-error lower bound on held-out
log-likelihood gain minus a complexity cost recovers true clusters far better than
admit-everything: mean cluster-F1 **0.783 vs 0.209**, with positive margins at every
granularity (4→16 clusters). (`runs/20260623-002149-A-synth`)

**Eviction (Card B) — ITERATE.** Across 5 seeds and 7 drift epochs, an evicting store
sustains post-drift accuracy where an append-only store degrades (mean plateau advantage
**+0.046**, variance `0.000686` ≈ 1.7σ; better at every epoch after the first). But the
multi-seed effect is smaller than the single-seed result and **eviction quality is poor
(precision 0.568, recall 0.449)**. Directionally real, not strong.
(`runs/20260623-045427-B-synth`)

**Positivity/coverage guard (Card D, the "keystone") — NOT DEMONSTRATED.** The guard
mechanism is correct (it reduces thin-region false admissions; a provisional variant
avoids most of the strict guard's post-drift adaptation freeze). But a coverage-skew
sweep showed the benefit **does not scale with coverage severity** — slope `-0.000138`,
non-monotone, CIs crossing zero at the extreme. The hypothesized keystone curve was not
found. (`runs/20260623-062919-D-skew-sweep-synth`)

**Pattern:** every mechanism passed *weakly or null* on synth. Each null was individually
explainable; the accumulation lowered the prior going into real data.

---

## 3. Real-data phase — RTPTorrent

Dataset: RTPTorrent (20 Java projects, 100k+ TravisCI build logs). Stream built by
joining test results → `tr_all_built_commits.csv` (`tr_job_id`→`git_commit_id`) →
project patches (`sha`→`name`), ordered by ascending `travisJobId`, de-flaked, large
changes (>30 files) dropped.

**Two pipeline issues found and fixed first** (so the verdict is not an artifact):
1. *Project coverage.* sling — picked for failure density — had only **16%**
   job→commit coverage (worst in the dataset). Its initial null was a coverage artifact,
   not a result. (`runs/20260623-074628-compounding-sling`)
2. *Granularity.* Full file paths yield thousands of singleton files with no co-change
   support. Coarsening to directory tokens (okhttp `971 → 64`) restored candidate proposal.

**okhttp — the valid v1 test.** Coverage 52%, 9,703 build cycles.
- Admission funnel: **9,691 candidates proposed → 12 admitted → 0 retained** (all evicted).
- Abstraction held-out LL gain: **max 0.0033, mean −0.00006** across the run.
- Result: ipsum ≡ data-matched control. **`ipsum_vs_data_matched_slope_gap = 0.0`**,
  plateau gap 0.0. (ipsum/control plateau TestRecall ≈ 0.75; both beat weekly-retrain
  0.48 — but that gap is online>batch, i.e. the data-accumulation confound, not the thesis.)
  (`runs/20260623-081638-compounding-okhttp`)
- sonarqube (second project, 31% coverage): **0 admitted** — admission-starved,
  inconclusive. (`runs/20260623-082055-compounding-sonarqube`)

The store could never stay populated, because nothing in it actually improved prediction.

---

## 4. The decisive diagnostic — engineering or thesis?

To distinguish "our implementation is weak" from "the signal isn't there," we measured
RTPTorrent's *own* shipped baseline strategies on okhttp — fraction of the suite run
before the first failure (lower = better), over 749 failing builds:

| strategy | first-failure at | signal type |
|---|---:|---|
| `optimal-failure` (oracle) | 0.073 | ceiling |
| `recently-failed` | **0.109** | historical failure-rate |
| `matrix-naive` | **0.187** | file-change |
| `matrix-conditional-prob` | 0.287 | file-change |
| `random` / `untreated` | ~0.51 | none |

This isolates the cause unambiguously:
- **File-change signal is real** — `matrix-naive` (0.187) ≫ random (0.51). The
  abstractions were chasing genuine signal; this is **not an engineering failure**.
- **But historical failure-rate (0.109) is much stronger and nearly hits the ceiling
  (0.073).** A trivially cheap feature already captures the predictable signal.
- **So co-change abstractions are real but redundant** — they add ~nothing *on top of*
  historical rates, which is exactly what ipsum's funnel found (LL gain ≈ 0).

**Conclusion: test selection is a task with almost no headroom.** The thesis was not
tested fairly here — not because the code is wrong, but because a cheap baseline dominates.

---

## 5. What this does and does not establish

**Establishes:**
- Co-change abstractions do not compound on CI test selection, with a precise,
  independently-confirmed mechanism (redundancy with historical failure rates).
- A trustworthy methodology and instrument for measuring compounding (slope vs a
  data-matched control, multi-seed, uncensored metrics, pre-registered gates).

**Does NOT establish:**
- That compounding-via-abstractions is impossible. Only one weak instantiation was tested,
  on a task without headroom. A fair test needs a task where cheap features do *not*
  already saturate the signal (e.g., regression/incident prediction, review routing) and,
  possibly, a richer abstraction substrate than co-change.
- External validity beyond two old Java/Travis-era (2007–2016) public repos.

---

## 6. Methodological lessons (the part that transfers)

Repeatedly, an early "pass" evaporated under honest measurement, and each was caught:
- **Card I:** a stationary positive control showed a level *shift*, not a widening slope —
  fixed by requiring drift and gating on sustained level-growth, not a censored slope scalar.
- **Card B:** a single-seed recovery-time trend was an artifact of where censored "never
  recovered" values fell — fixed with 5 seeds, uncensored plateau accuracy, and
  recovered-fraction + median-recovery instead of an OLS slope on capped values.
- **Card D:** a "provisional guard" that improved a metric by redefining its denominator
  was cosmetic — fixed by measuring harm, not a gameable count; then the skew sweep showed
  the benefit was within noise.
- **Real data:** the first "does not compound" was a 16%-coverage artifact; the second was
  a granularity artifact; only after both were fixed was the negative result real.

Standing rule that would have caught all of them: **report mean ± SE and the worst seed,
gate on significance not the mean, and when a new variant "passes," diff its predictions
against the control — if they're identical, the win is in the metric, not the mechanism.**

---

## 7. Status

v1 is **complete and banked as a negative result.** No further mechanism work is justified
on this instantiation — the signal isn't there to tune toward. The synthetic testbed,
the measurement harness, and the design/research notes remain reusable for any future
instantiation on a headroom-rich task.

*Artifacts referenced above live under `experiments/runs/`. Full per-experiment record in
`RESEARCH_LOG.md`. Design rationale and verified literature in `research/`.*
