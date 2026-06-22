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
isolate) → RTPTorrent instrument/data-matched control → **eviction** (needs drift;
builds on the testbed) → **credit assignment** (hardest; do last, on the
foundation of the other two).

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

### Card B — Eviction / anti-staleness
- **Hypothesis:** with a usefulness trace + decay + eviction rule, after an
  injected drift the system prunes the now-stale abstraction and recovers accuracy
  *faster* than a no-eviction (append-only, à la DreamCoder/Voyager) control.
- **Mechanism / module:** `abstractions.AbstractionStore.decay_and_evict`,
  `.reinforce`.
- **Isolating metric:** post-drift **recovery time** (cycles to return to within
  ε of pre-drift accuracy); and stale-abstraction **eviction latency** vs
  `synth.drift_schedule()`.
- **Control:** append-only store (never evicts).
- **Testbed config:** noise low, no delay, **drift on** (a few scheduled rewires).
- **Borrow from:** concept-drift detection (ADWIN, DDM) for *when* to suspect
  staleness; EWC Fisher importance (`consolidation.py`) as the *what to protect* signal.
- **Keep / kill:** keep iff recovery is faster than append-only with no loss of
  pre-drift accuracy. If eviction also hurts stable periods, the decay is too aggressive.
- **Agent sub-tasks:** implement usefulness trace + decay; a drift-suspicion
  trigger; the eviction rule; a synth experiment plotting accuracy vs cycle for
  evict vs append-only across a drift.

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
  (Chapelle's delayed-feedback model) — the closest match: label arrives hours
  later, attribute it back. RL eligibility traces / TD as secondary intuition.
- **Keep / kill:** keep iff attribution accuracy stays high as delay/noise rise,
  and reinforcement beats similarity-reuse end-to-end. This is the hardest card;
  expect it to fail first and teach the most.
- **Agent sub-tasks:** implement the eligibility buffer; a de-flaking resolver
  (retry/consistency model — synth can emit repeated trials); the settle/credit
  map; a synth experiment sweeping `delay` and `p_flaky` and reporting attribution accuracy.

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
