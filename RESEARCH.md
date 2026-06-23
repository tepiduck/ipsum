# RESEARCH.md — how we actually do the open mechanisms

This file is the research playbook. `DESIGN.md` says *what* the three open
mechanisms are; this says *how to find them* without a paper to copy. Read
`research/00-synthesis.md` first for why these three are the novel surface.

The three open mechanisms (from DESIGN.md §3):
1. **Admission under uncertainty** — keep an abstraction iff it earns its complexity in held-out predictive likelihood.
2. **Delayed, noisy credit assignment** — attribute a CI outcome (after de-flaking) back to the abstractions that informed the decision.
3. **Eviction / anti-staleness** — decay and prune abstractions as the domain drifts.

---

## 0. The meta-method (read this before touching a mechanism)

A research mechanism is **not a feature to build — it is a claim to test.** The
work is a loop, not an implementation:

```
hypothesis ─▶ simplest version ─▶ measure (isolating metric vs control) ─▶ keep / kill ─▶ repeat
```

Four rules make the loop actually work:

1. **Instrument the thing you are testing before judging it.** For synthetic
   mechanism work, that means the synthetic generator, oracle metrics, and
   per-card controls. For RTPTorrent, that means the loader, slope plot, and the
   *data-matched, abstraction-off control*. Do not declare a mechanism successful
   without the measuring instrument appropriate to that stage.

2. **Debug on the synthetic testbed, not on RTPTorrent.** Build a generator where
   *you* control ground truth (which files truly co-vary, when drift happens, how
   noisy outcomes are, how delayed feedback is). On real data a null result is
   ambiguous — bad mechanism or just noise? On synthetic data you know the right
   answer, so you can check the mechanism *recovers* it. Get it working on synth,
   then move to sling/okhttp. This is the highest-leverage rule here; skipping it
   is how solo research loses months. Spec is in §1.

3. **Reduce to the degenerate case first, then add the hard part.** Don't start
   with the full version. Zero-delay before delayed. Immediate/exact outcomes
   before noisy. Get the easy instance correct and measured, then turn one knob.

4. **Write the hypothesis and metric before the code, every time.** Literally:
   "I expect M to improve [metric] over [control] because [reason]; if it doesn't,
   I've learned [X]." Log it (see §4). Solo research dies from forgetting what was
   already tried and why.

**Sequence:** synthetic testbed + oracle metrics → **admission** (easiest to
isolate) → **instrument self-check (Card I, both controls)** → **eviction** (Card B,
needs drift) → **positivity/coverage guard (Card D — the keystone)** → **credit
assignment** (Card C, hardest) → RTPTorrent instrument/data-matched control. The
rigorous admission gate (e-LOND + held-out-LR e-values, design §3) and the v2
uncertainty layer are upgrades layered on *after* the first clean compounding result.

---

## 1. The synthetic testbed (`src/ipsum/synth.py`)

A controllable mock of "a codebase under CI" whose only purpose is to give every
mechanism a ground-truth oracle to be validated against. The model never sees the
oracle; evaluation does.

**Entities**
- Files `F = {f_1 … f_N}` and tests `T = {t_1 … t_M}`.
- **True dependency** `D: t → subset(F)` — the files whose change can make `t` fail.
- **True clusters** `C` — groups of files that co-change *and* co-affect tests.
  These are the latent abstractions the system is supposed to discover. Membership
  is the ground truth for admission.

**Dynamics (one CI cycle)**
- A **commit** = a sampled set of changed files. Sampling is cluster-correlated:
  changes tend to hit whole clusters (this is the structure to be recovered).
- **Outcome model:** test `t` fails with prob `p_hit` if the commit touches any
  file in `D(t)`, and with prob `p_flaky ≪ p_hit` otherwise. `p_flaky` is the
  **noise knob**.
- **Delay knob:** the outcome of a decision at cycle `k` is revealed at cycle
  `k + delay` (delay may be stochastic). Default 0.
- **Drift knob:** at scheduled cycles, mutate `D`/`C` (rewire some `t→f` deps,
  split/merge clusters) so previously-true abstractions go stale.

**Oracle accessors (evaluation only — never fed to the model)**
- `true_clusters()` → ground-truth abstraction membership (for admission scoring).
- `true_deps(t)` → ground-truth dependency set.
- `cause(outcome)` → which file/cluster actually caused a given failure (for
  credit-assignment scoring).
- `drift_schedule()` → when drifts happen (for eviction scoring).

**Knobs summary:** `n_files, n_tests, n_clusters, p_hit, p_flaky (noise),
delay (mean/var), drift_schedule, seed`. Everything seeded and reproducible.

With this, each mechanism has a clean, isolating check (see the cards in §2/§3).

---

## 2–3. Experiment cards

Each card is a self-contained, falsifiable unit. Work them in order. Each one
ends with concrete sub-tasks suitable to hand to a coding agent.

### Card A — Admission under uncertainty
- **Hypothesis:** an abstraction admitted iff it raises *held-out predictive
  log-likelihood* by more than its complexity cost will recover the testbed's
  true clusters (high cluster precision/recall) and improve prediction over a
  no-abstraction predictor.
- **Mechanism / module:** `abstractions.AbstractionStore.admit` (+ `candidates`).
- **Isolating metric (unit-level, not end-to-end):**
  (1) cluster precision/recall of admitted abstractions vs `synth.true_clusters()`;
  (2) ΔLL on a held-out set from admitting each candidate.
- **Control:** no abstractions (raw predictor); and a naive "admit everything"
  store (to show admission ≠ accumulation).
- **Testbed config:** noise off→low, no delay, no drift. Isolate admission only.
- **Borrow from:** Bayesian model selection; minimum description length under
  noisy likelihoods; online/streaming feature selection. ("How many abstractions"
  has a principled prior in Bayesian nonparametrics — Indian Buffet Process — if
  needed later; don't start there.)
- **Keep / kill:** keep iff cluster-recovery F1 ≫ admit-everything control AND
  held-out LL improves. If admitting true clusters doesn't help prediction, the
  abstraction *representation* is wrong — fix that before anything else.
- **Agent sub-tasks:** implement candidate proposal (co-change clustering over
  recent commits); implement held-out ΔLL estimator; implement the admit rule;
  write a synth experiment that sweeps `n_clusters` and reports cluster-F1 + ΔLL.
- **Status / upgrade:** v1 passed with a one-SE held-out-LL-gain threshold. The
  *rigorous* gate (design §3) is **e-values + e-LOND** for FDR under the dependent,
  overlapping candidate hypotheses (a held-out likelihood-ratio test-martingale is the
  e-value), plus the **Card D coverage guard**. Layer these after the first compounding
  result; the simple threshold is an honest v1 stand-in, not the final rule.

### Card I — Instrument self-check (positive + negative control) — DO BEFORE Card B
The compounding harness must be trusted before Cards B/C are judged by it. This is
a positive control for the *instrument*, not a new mechanism. The synth slope is
currently ≈ 0; we don't yet know if that's "no signal" or "blind instrument."

- **Hypothesis:** if `ipsum` is given a structural predictive edge the controls
  lack (the Card-A admitted abstraction store) and that edge accumulates as more
  abstractions are admitted over cycles, the harness will show ipsum's
  TestRecall@SelectionRate slope exceeding the data-matched control's, with the gap
  *widening*. If it doesn't show this on synth where we planted the advantage, the
  instrument is broken and must be fixed before any mechanism is trusted.
- **Setup:** synth (noise low, no delay, no drift) → chronological `(commit,
  per-test outcomes)` stream. Task: rank tests by predicted failure prob, select
  top subset under the SelectionRate cap, score TestRecall = fraction of
  actually-failing tests selected. Three systems consume the **same** stream
  (data-matched): `weekly_retrain` (refit from scratch every K cycles, no carried
  state); `data_matched_control` (same model, same cumulative data online,
  abstraction store OFF); `ipsum` (identical to control PLUS the admitted store —
  when a commit touches an admitted cluster's files, raise failure prob for tests
  with observed co-failure history; **no oracle**).
- **Run BOTH controls (the non-negotiable rigor):**
  - *Positive (signal present):* ipsum MUST show a clearly widening gap over
    `data_matched_control`.
  - *Negative (signal absent):* disable ipsum's store (or feed useless/shuffled
    abstractions) so it is equivalent to the control. The harness MUST then show
    **no** meaningful gap. This catches an instrument that fabricates a gap from
    implementation asymmetry (extra data, RNG drift, leakage). A stick that always
    shows ipsum winning is as useless as one that never does — most people build
    only the positive control.
- **Isolating metric:** per-window TestRecall@SelectionRate series per system; fit
  a slope each; report `slope_ipsum − slope_control` and a widening test (final-window
  gap > early-window gap). Pos-control gap must clearly exceed neg-control gap.
- **Keep / kill (for the INSTRUMENT):** PASS iff positive control widens AND
  negative control is flat. Positive flat → harness can't see compounding that
  exists; fix windowing/metric/online-state plumbing before Card B. Negative shows
  a gap → harness fabricates differences; find and remove the asymmetry.
- **Artifacts:** two runs, `card="instrument-poscontrol"` and
  `card="instrument-negcontrol"`, `dataset="synth"`; `slope.json`
  `metric_name="test_recall_at_selrate"`; `metrics.json` with per-system slopes,
  gap-widening boolean, pos-vs-neg summary; update `index.json`.
- **Agent sub-tasks:** TestRecall@SelectionRate scorer; the three systems sharing
  one stream with strict data-matched parity (only the store differs); wire admitted
  abstractions into ipsum's predictor via observed co-failure stats; run both modes;
  `RESEARCH_LOG.md` entry with both results + PASS/FAIL verdict; a **parity unit
  test** asserting the negative control yields a near-zero gap at a fixed seed.
- **Guardrails:** keep the predictor small (edge from abstractions, not capacity);
  oracles are evaluation-only; data-matched parity is sacrosanct; verify on Python
  3.10 (the declared minimum), not just the local interpreter.

### Card B — Eviction / anti-staleness
Build on the validated Card I harness (drift already plumbed). The question is not
"does eviction run" but whether it produces the *compounding-relevant* behavior: a
store that stays useful across MANY drifts while a never-evicting store rots.

- **Hypothesis (two parts, both falsifiable):**
  1. *Eviction helps recovery:* after a drift, an evicting store recovers prior
     accuracy faster than an append-only (never-evict) store.
  2. *The benefit compounds across drifts:* over many successive drifts, the
     evicting store **holds or improves** post-drift recovery while the append-only
     store **degrades** as stale abstractions accumulate and clutter prediction.
     This trend — recovery quality vs. drift-epoch number — is the headline, because
     it separates true anti-staleness from "the bag just kept filling."
- **Mechanism / module:** `abstractions.AbstractionStore.decay_and_evict`,
  `.reinforce`. Usefulness trace updated by immediate held-out-LL contribution;
  decay each cycle; evict when usefulness < complexity cost; optional drift-suspicion
  trigger to evict faster post-change. EWC Fisher importance (`consolidation.py`) is
  an optional "what to protect" signal — start with decay/threshold, don't over-engineer.
- **Systems / controls:** `ipsum_evict` (the mechanism); `append_only` (identical
  but never evicts — isolates eviction's effect); `no_store` (raw predictor floor).
- **Isolating metrics:**
  - recovery time per drift epoch (cycles to return within ε of pre-drift accuracy);
  - **headline — recovery quality vs. drift-epoch #** for evict vs append-only
    (evict flat/improving, append-only worsening);
  - eviction latency vs `synth.drift_schedule()` (and check evicted abstractions were
    actually stale, via `true_clusters()` pre/post — evaluation-only);
  - stable-period cost: no-drift accuracy must not be worse than append-only.
- **Testbed config:** noise low, no delay (degenerate first — immediate reinforcement;
  delayed credit is Card C). **≥6–8 drift epochs** so the cross-drift trend is visible;
  a single drift can't distinguish eviction-helps from noise.
- **Borrow from:** concept-drift detection (ADWIN, DDM) for *when* to suspect
  staleness; EWC Fisher importance as the *what to protect* signal.
- **Keep / kill:** keep iff (1) `ipsum_evict` recovers faster than `append_only`
  after drifts, (2) the cross-drift trend favors eviction (append-only degrades,
  evict doesn't), and (3) no stable-period accuracy loss. If eviction hurts stable
  periods, decay is too aggressive — tune and re-run, don't ship. Do NOT mark passed
  on "eviction runs" or "recovers faster once"; it passes only on the sustained trend.
- **Artifacts:** `card="B"`; `slope.json` `metric_name="post_drift_recovery"` for
  `ipsum_evict` / `append_only`; `events.json` admit/evict/drift timeline;
  `metrics.json` per-epoch recovery times, cross-drift trend slope, eviction latency,
  stable-period delta. Update `index.json`.
- **Agent sub-tasks:** recovery-time estimator; append-only as a store config flag;
  drift-suspicion trigger; cross-drift trend computation; `RESEARCH_LOG.md` entry with
  the per-epoch table and verdict; tests — eviction latency on a hand-built drift, and
  a stable-period parity test (evict ≈ append-only with no drift).

### Card C — Delayed, noisy credit assignment
- **Hypothesis:** an eligibility structure that holds in-flight decisions until
  de-flaked outcomes land will attribute credit that concentrates on the
  testbed's true cause (`synth.cause()`), and online reinforcement using that
  credit beats embedding-similarity reuse (Voyager's gap).
- **Mechanism / module:** `credit.CreditAssigner` (`record`/`deflake`/`settle`)
  feeding `abstractions.reinforce`.
- **Isolating metric:** **attribution accuracy** — fraction of credit landing on
  the true causal cluster vs `synth.cause()`; then end-to-end lift over similarity-reuse.
- **Control:** (1) immediate-credit (delay=0) upper bound; (2) similarity-only reuse.
- **Testbed config:** start delay=0, noise off (validate attribution recovers the
  oracle), then turn up `delay` and `p_flaky` one at a time.
- **Borrow from:** **delayed-feedback learning** from ad-click prediction
  (Chapelle, KDD 2014) — survival-weighted EM over a conversion model + a delay
  model; weight a not-yet-failed test by `w = p(x)·exp(−λ(x)·elapsed)` = prob the
  failure is merely pending. RL eligibility traces / TD as secondary intuition.
- **Delay model (build it right):** exponential delay = constant hazard, wrong for
  CI (queue wait + ~fixed runtime). Use a **flexible hazard (Weibull/Gamma)**; but a
  single Weibull is **unimodal**, so if the resolved-delay histogram is genuinely
  **bimodal**, use a **mixture** (mixture of exponentials/Weibulls) — do *not* call a
  single Weibull "bimodal". A **max-wait window** `T` closes the loop: after `T`,
  absence-of-failure = true pass. See `research/ipsum-design.md` §5.
- **De-flake is a separate axis from delay.** Flakiness is label *noise*, not late
  arrival — require retry/consistency before an outcome counts; never reinforce on a
  flaky or unresolved signal.
- **Keep / kill:** keep iff attribution accuracy stays high as delay/noise rise,
  and reinforcement beats similarity-reuse end-to-end. This is the hardest card;
  expect it to fail first and teach the most.
- **Agent sub-tasks:** implement the eligibility buffer; a de-flaking resolver
  (retry/consistency model — synth can emit repeated trials); the survival-weighted
  settle/credit map with the delay model above; a synth experiment sweeping `delay`
  and `p_flaky` and reporting attribution accuracy.

### Card D — Positivity / coverage guard (the keystone) — v1 coarse-coverage only
The design doc's single most novel contribution (`research/ipsum-design.md` §3), and
the build slot it was missing. ipsum learns on a **non-uniformly covered** Boolean
cube; abstractions whose support is barely observed are **unidentifiable** (positivity/
overlap failure — D'Amour et al. 2021). Admitting from thin coverage yields confident-
but-wrong structure. This also targets Card B's measured weakness (eviction precision).

- **Hypothesis (two parts):**
  1. A coverage-gated admission rule reduces false admissions and improves held-out
     accuracy vs a no-guard control — *specifically in thin-coverage regions* — when
     the cube is covered non-uniformly.
  2. *Guard ⟂ drift resolution:* post-drift, **provisional** admission under a wide band
     recovers adaptation speed a **strict** guard would freeze — provisional ≈ no-guard
     on post-drift recovery latency, while still beating no-guard on false-admission rate.
- **Mechanism / module:** admission path in `abstractions.AbstractionStore` (a coverage
  gate before `admit`).
- **Coverage statistic — v1 COARSE ONLY:** for a candidate file-subset, count observed
  commits whose changed-files cover its support (or a sufficient-statistic proxy) and
  gate on a finite-sample standard-error / count threshold for identifiability to ±ε.
  **Do NOT build the per-coefficient Bayesian posterior — that is the v2 uncertainty
  layer, explicitly parked.** One coverage rule.
- **Systems / controls:** `no_guard` (current Card A — LL-gain alone); `strict_guard`
  (refuse below coverage threshold); `provisional_guard` (refuse *or* provisionally admit
  under a wide band + wider eviction tolerance, tightening as coverage accrues).
- **Testbed config:** add a **coverage-skew knob** to synth (commits hit some true
  clusters far more than others, so some abstractions are well-covered and others thin —
  you control which). Run with and without drift.
- **Isolating metrics:** false-admission rate in thin-coverage regions vs `no_guard`
  (scored against synth oracles, eval-only); held-out accuracy **split by well- vs
  thin-covered regions**; post-drift admission latency (`provisional` must not inflate it
  vs `no_guard` — the tension test); bonus: effect on Card B eviction precision/recall.
- **Borrow from:** positivity/overlap (Rosenbaum–Rubin 1983; Hernán & Robins 2020);
  high-dimensional overlap failure (D'Amour et al. 2021, arXiv:1711.02582).
- **Keep / kill:** keep iff (1) the guard reduces false admissions / improves thin-region
  accuracy vs `no_guard`, and (2) the provisional variant does NOT freeze post-drift
  adaptation. If strict improves precision but kills adaptation, that's the documented
  tension — provisional is the resolution and the data should show it.
- **Guardrails:** v1 coarse coverage only (no Bayesian posterior, no active probing —
  both v2); small predictor; oracles eval-only; verify on Python 3.10.
- **Agent sub-tasks:** coverage-skew knob on synth; coverage statistic + guard (strict +
  provisional); the non-uniform-coverage experiment with the three controls; metrics incl.
  thin/thick split + post-drift latency; tests (guard refuses a hand-built thin candidate;
  provisional admits post-drift without latency blowup); `RESEARCH_LOG.md` entry.

---

## 4. Research log convention

Keep `RESEARCH_LOG.md` append-only. One dated entry per experiment:

```
## YYYY-MM-DD — <card> — <one-line claim>
- Hypothesis: I expect ___ to improve ___ over ___ because ___.
- Setup: testbed knobs / dataset / control.
- Result: the number(s). Plot path if any.
- Decision: keep / kill / iterate, and why.
- Next: the single next question.
```

If an entry has no metric and no control, it is not an experiment — it's a vibe.
Don't log vibes.

---

## 5. What success looks like

The headline is one plot: TestRecall @ fixed SelectionRate vs CI-cycle, for
**(a)** weekly-retrain baseline, **(b)** data-matched abstraction-off control,
**(c)** ipsum. The thesis holds iff (c)'s slope exceeds (b)'s and the gap widens.
Beating (a) is table stakes; beating (b) is the whole point — it proves the gain
comes from accumulated abstractions, not accumulated data.
