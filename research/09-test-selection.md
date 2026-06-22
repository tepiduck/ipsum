# 09 — Predictive Test Selection (Machalica et al., 2019)

## What this paper actually is

- **Title:** Predictive Test Selection
- **Authors:** Mateusz Machalica, Alex Samylkin, Meredith Porth, Satish Chandra (Facebook, Inc.)
- **Venue/Year:** arXiv:1810.05286v2, submitted October 2018, revised May 2019. Published at ICSE-SEIP '19 (Software Engineering in Practice track of the 41st International Conference on Software Engineering, 2019).
- **Confirmed:** This IS the paper you expected. It is the canonical industrial-baseline paper for ML-based predictive test selection.

## Core idea

**Problem:** Given a code change submitted to CI, select a minimal subset of tests to run such that you catch failures without running everything.

Facebook's monorepo (tens of thousands of changes/week) triggers ~10,000 tests per change via build-dependency traversal. Running all is infeasible at peak. The solution is to learn a binary classifier over `(change, test)` pairs that predicts P(test fails | change). Only tests above a score threshold (plus top-N by rank as a safety net) are run.

**The core safety/efficiency tradeoff:** You cannot guarantee catching all failures (that would require running all tests). You can tune the score cutoff to hit a target recall at acceptable selection rate. The stabilization stage (full suite, every few hours) provides a backstop — so diff/land-time selection only needs to be "good enough."

## Mechanism

### Features used (gradient boosted decision trees — XGBoost)

**Change-level features:**
- File change history: number of changes to modified files in last 3, 14, 56 days (active areas break more)
- File cardinality: number of files touched
- Target cardinality: number of test targets triggered
- File extensions: bit vector of programming languages
- Distinct authors of modified files

**Test-target-level features:**
- Historical failure rates: vector over last 7, 14, 28, 56 days
- Project name (categorical)
- Number of test cases in target

**Cross features (most impactful):**
- Minimal graph distance between modified files and test target in build dependency DAG
- Number of common tokens in file paths (lexical proximity)

**Feature importance findings (Table I):** Top performers are `historical failure rates` (1.37x classification gain), `minimal distance` (1.23x), `project name` (1.15x), `number of tests` (1.07x), `file extensions` (1.04x), `change history` (1.03x). Common tokens and distinct authors actually hurt (regression below 1.0). For the ranking metric, `number of tests` (2.89x), `file extensions` (1.62x), `failure rates` (1.62x), and `change history` (1.59x) dominate.

### Model architecture
- XGBoost binary classifier; no feature normalization needed; handles class imbalance naturally (passing tests vastly outnumber failing)
- Training window: 3 months of historical test outcomes
- Train/test split: most recent week held out as evaluation set
- Flakiness handling: each failed test retried up to 10 times; `FailedTests(d)` = all attempts failed; `FlakedTests(d)` = mixed outcomes. Training only on de-flaked labels is essential — training on raw outcomes trains the model to predict flakes, not real failures (Experiment B shows this destroys real recall)

### Selection strategy
`SelectedTests(s*, d) = LikelyFailing(s*, d) ∪ HighlyRanked(s*, d)`
- `LikelyFailing`: all tests with score ≥ ScoreCutoff (drives TestRecall)
- `HighlyRanked`: top-N by score regardless of cutoff (drives ChangeRecall — catches cases where all scores are low but one test will fail)
- Two parameters calibrated independently: ScoreCutoff → TestRecall ≥ 0.95; CountCutoff → ChangeRecall ≥ 0.999

### Evaluation metrics (formal definitions in §III)
1. **TestRecall(s, D):** fraction of individual failing tests caught across all changes
2. **ChangeRecall(s, D):** fraction of faulty changes where at least one failing test is caught
3. **SelectionRate(s, D):** fraction of build-dependency-selected tests actually run (relative to the dependency baseline, not all tests)

**Key results achieved in production:**
- TestRecall > 0.95 (catch >95% of individual failures)
- ChangeRecall > 0.999 (catch >99.9% of faulty changes)
- SelectionRate < 0.33 (run less than 1/3 of build-dependency-selected tests)
- Infrastructure cost reduction: 2x (total machines); test execution count reduction: 3x

### Deployment cadence
Weekly automated pipeline: retrain → assert performance thresholds → auto-deploy. No manual tuning. The model adapts to codebase drift weekly.

## Why it matters for this project

**This is the problem definition and existing bar for your v1 wedge.** Machalica et al. establish:
- The canonical task formulation: learn P(fail | change, test) from history
- The right metrics: TestRecall / ChangeRecall / SelectionRate
- The existing industrial ceiling: 95%+ recall, 3x cost reduction, deployed at scale

What this means for you: if you build predictive test selection on public GitHub repos, you are building in this problem domain. Your baseline to beat (or match cheaply) is a weekly-retrained XGBoost on build-dependency-graph distances + historical failure rates. The features are mostly available from GitHub data: commit history, changed files, test outcome logs from CI (GitHub Actions), dependency manifests.

**What signal defines success here:** ChangeRecall is the critical safety metric. TestRecall at 95% is the minimum acceptable. SelectionRate < 0.33 is the efficiency target. Any system you build must be evaluated on these three quantities to be comparable.

## What to extract / reuse

### Feature ideas directly applicable to public repos
- **File-level change history** from git log (3/14/56 day windows) — trivially available
- **File extensions** as language indicator — trivially available
- **Build dependency distance** — approximable via package imports, `import` graph analysis, Makefile/CMake/package.json dependency graphs; less clean without a build system but doable
- **Historical test failure rates** — available from GitHub Actions run logs (via API)
- **Common token overlap** between changed paths and test file paths — trivially computable
- **Project/module name** — inferrable from directory structure

### Evaluation methodology to steal
- Hold out most recent week; train on 3 months prior — clean temporal split avoids leakage
- Run "learning test runs" (run all tests on a random sample of changes to measure true recall) — on public repos this translates to measuring recall on CI runs that already ran all tests
- De-flake by requiring test to fail all retry attempts — can approximate with "consistent failure across N runs" from CI history
- Report TestRecall/ChangeRecall/SelectionRate as the three primary metrics

### Data availability on public GitHub repos
- GitHub Actions logs give test outcomes per commit — but are noisy, inconsistent across repos, and often don't run the full suite
- Build dependency graphs are repo-specific (Python: imports; JS: package.json; Java: pom.xml/Gradle) — need language-specific parsers
- The "learning test run" equivalent is any CI run that happened to trigger all tests — grep for them from history
- Flakiness detection requires multiple runs of the same test on the same code — hard on public repos unless there are re-runs in CI logs

## Gaps / where a learning-loop approach could differentiate

### What this paper does NOT do — the opening for you

1. **Static retraining, not online learning.** Their model is retrained weekly from scratch on 3 months of data. There is no incremental update, no online gradient step, no Bayesian posterior update. Each new test outcome is discarded until the next weekly retrain. A system that updates its prior after every CI run — cheaply, without full retrain — would respond to drift faster and compound learning from the first day of use.

2. **No abstraction over what "matters."** Their features are hand-engineered by humans. The model learns weights over these fixed features but cannot discover new relevant structure — e.g., "this test cluster is semantically related to this file cluster via API contracts, not just file paths." Your thesis is that the prior itself should be learned and recursively refined. Autonomous abstraction discovery (open problem #1) is completely absent here.

3. **No credit assignment across changes.** If a test fails on change 47 and was last selected on change 31, there is no mechanism to attribute that miss to a specific feature gap and tighten the prior accordingly. Their calibration is global and periodic, not per-failure. Targeted credit assignment (open problem #3) would close this gap.

4. **Drift detection is reactive, not proactive.** They detect model degradation via weekly threshold assertions. There is no proactive drift/staleness modeling — no estimate of how much the prior has decayed since last retrain. A learning system with an explicit drift model (open problem #4) could flag its own uncertainty and trigger retraining selectively.

5. **No compounding curve.** Because they retrain weekly from scratch, their improvement over time is bounded by data volume and feature engineering. The slope of improvement flattens once data is sufficient. A recursive update system that compounds — each new run improves the prior for subsequent runs — would show a steeper slope over time, which is your moat.

6. **No test correlation modeling.** They explicitly acknowledge (§VIII) that their model treats each test independently and cannot capture that subsets of tests have correlated coverage. This is a stated open problem — credit assignment across correlated tests would reduce redundant selection.

## Limitations & risks relevant to us

- **Monorepo assumption:** Their whole approach depends on a build dependency DAG that is explicitly maintained. Public GitHub repos have no such uniform graph — you have to reconstruct it from imports/manifests, which is noisy and language-specific. Distance in the DAG is their strongest cross-feature (1.23x); approximating it on public repos degrades signal quality.
- **Scale assumption:** The paper implicitly requires large history (3 months, tens of thousands of changes) to train well. Small public repos with sparse CI histories will have insufficient signal; the model will underfit or overfit on rare failure events.
- **Flakiness is worse on public repos:** Facebook can enforce retry infrastructure. Public repo CI runs rarely have retry logs. Conflating flaky failures with real ones (as Experiment B shows) can silently corrupt the model.
- **No test oracle:** Their gold labels come from "learning test runs" where all tests are run. On public repos, you only observe the tests that CI actually ran (usually a subset). This creates survivorship bias in your training labels.
- **Their closed system can't transfer directly:** Features like "target cardinality" and "project name" are build-system concepts. On public repos, you need proxies (directory depth, test-file naming conventions, etc.).

## One-line takeaway

Machalica et al. prove that a weekly-retrained XGBoost over build-graph distance, file change history, and historical failure rates can cut CI costs 3x with 95%+ recall — establishing the industrial baseline and metric framework for the v1 wedge, but leaving the learning loop (online updates, autonomous abstraction, credit assignment, drift modeling) entirely untouched.
