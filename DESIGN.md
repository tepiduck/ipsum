# DESIGN

This document is the working architecture and research roadmap. It is deliberately honest about what is borrowed (solved) and what is novel (open).

## 1. Thesis, stated precisely

Let a domain produce a stream of experiences `e_1, e_2, …`. Maintain a prior `θ_t` and an abstraction set `A_t`. On each new experience:

```
(θ_t, A_t) ──▶ decision d_t ──▶ outcome o_t ──▶ (θ_{t+1}, A_{t+1})
```

**Claim:** with the right update rule, decision quality `Q_t` improves with a *slope* that a stateless predictor `f(e_t)` — including a strong LLM + retrieval — cannot match, because the stateless predictor has no `(θ_t, A_t)` to accumulate.

The empirical object we report is `dQ/dt`, measured against a frozen baseline, on held-out future experiences.

## 2. What we borrow (do not innovate here)

| Component | Borrowed from | Role |
|---|---|---|
| Prior as a network | MAML | the prior is parameters; adaptation is cheap |
| Bayesian grounding | Grant et al. 2018 | the init *is* an empirical-Bayes prior mean; inner optimizer is its covariance |
| Amortized O(1) conditioning | Conditional Neural Processes | encode→mean-pool→decode; new experience updates the latent in one pass, with uncertainty |
| Consolidation w/o forgetting | EWC | Fisher-weighted anchor; "posterior → next prior" |
| Inspectable abstraction store | Voyager | explicit, readable abstractions; updated without weight training |
| Admission criterion (exact case) | DreamCoder | add an abstraction iff it compresses solved tasks more than it costs |
| Compounding loop shape | AlphaZero | learned prior + search + clean signal; the slope is the proof |

The prior + consolidation stack (CNP + EWC + FOMAML) is a complete, buildable recipe. Assemble and move on.

## 3. What we must build (the open mechanisms)

### 3.1 Admission under uncertainty
DreamCoder's MDL admission assumes outcomes are binary/exact. Replace with **Bayesian model selection**: maintain, for each candidate abstraction `a`, an estimate of held-out predictive log-likelihood gain `ΔLL(a)` minus a complexity penalty `c(a)`. Admit iff `ΔLL(a) − c(a) > 0` with sufficient posterior confidence. This doubles as the project's core ablation metric.

### 3.2 Delayed, noisy credit assignment
CI outcomes are delayed (minutes–hours) and label-noisy (flaky tests). Need:
- an **eligibility structure** mapping a landed outcome back to the abstractions that informed the decision;
- **de-flaking** before an outcome counts as a label (retry / consistency check — Predictive Test Selection retries up to 10x);
- outcome-weighted reinforcement of abstractions, not embedding-similarity reuse (Voyager's gap).

### 3.3 Eviction / anti-staleness
DreamCoder and Voyager only add. Code drifts. Maintain per-abstraction:
- a **usefulness trace** (recent `ΔLL` contribution);
- a **decay**; evict when the trace falls below `c(a)`.
- EWC Fisher importance is a candidate signal for what to protect vs. release.

## 4. Reference architecture

```
                 ┌─────────────────────────────────────────┐
   experience ──▶│ encoder  h_θ(change, test, context) → r_i│
                 └─────────────────────────────────────────┘
                                  │  mean-pool (O(1) streaming)
                                  ▼
                       latent prior  z_t  ──────────────┐
                                  │                     │
            ┌─────────────────────┼─────────────┐       │ consolidation
            ▼                     ▼              ▼       │  (EWC anchor)
   abstraction store A_t   decoder g_θ(test,z)   credit  │
   (admit / evict)         → P(fail), σ          assigner│
            │                     │              ▲       │
            └─────── outcomes o_t ◀──────────────┴───────┘
```

## 5. v1 experiment: compounding vs. weekly-retrain

**Task:** predictive test selection on public GitHub repos with CI history.

**Baseline:** reimplement Predictive Test Selection — XGBoost over (change, test) pairs, features = build-dependency-graph distance, historical per-test failure rate, file-change history windows, etc. Retrain weekly from scratch. (This is the plateau.)

**ipsum:** the architecture above, updated online per CI run.

**Metric:** TestRecall at fixed SelectionRate (≤ 0.33), on a rolling held-out future window. Plot both systems' metric vs. wall-clock/repo-age.

**Hypothesis to falsify:** `slope(ipsum) > slope(baseline)`, and the gap widens month 3 → month 6. If it doesn't, the thesis is wrong on this domain — and we've learned that cheaply.

**Confounders to control:**
- *more data, not better abstractions* → **the primary control is "same model, same cumulative data, abstraction store OFF."** This is non-negotiable: the ICSME-2023 study (arXiv 2311.13413) found existing ML techniques' improvement over CI cycles comes *mainly from the growing amount of training data, not code evolution* — so naive "improves over time" is illusory. ipsum's slope is only meaningful if it exceeds the data-matched, abstraction-off control.
- *bigger model* → keep the predictor small; any gain must come from `A_t`, not capacity.
- *flaky labels* → de-flake before scoring.

**Public-repo caveats:** build-dependency graph (the baseline's strongest feature) must be reconstructed from imports/manifests; small repos lack history volume; flakiness retry logs are usually unavailable. Pick repos accordingly.

## 6. Milestones

1. Data: use **RTPTorrent** (20 Java projects, 100k+ Travis builds, per-test pass/fail, 9-yr history) rather than scraping CI from scratch. Pick 3–5 long-history, high-failure-density projects. See `research/10-datasets.md`.
2. Baseline: stand up the **ICSME-2023 replication package** (arXiv 2311.13413, Zenodo 7036507) as the baseline harness; reproduce sane TestRecall/SelectionRate numbers. Define the **data-matched, abstraction-off control** as the primary comparison.
3. Prior: CNP-style amortized predictor with uncertainty; online conditioning.
4. Abstraction store + admission (3.1); ablation harness.
5. Credit assignment (3.2) + consolidation/eviction (3.3).
6. The slope plot. This is the deliverable.
