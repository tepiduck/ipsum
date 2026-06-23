# ipsum design — principles distilled from the literature

This turns `research/ipsum-related-work.md` (the reading) into design decisions. It is
opinionated on purpose: each principle states the choice and what we deliberately do *not*
do. This layer feeds the build spec (`DESIGN.md`) and the experiment cards (`RESEARCH.md`);
it does not replace them.

> **Accuracy pass (2026-06-23).** Every load-bearing citation below was re-verified against
> the primary source; corrections from that pass are folded in and the changed claims are
> marked **[verified]** / **[corrected]**. Confirmed references with arXiv ids are listed at
> the end. Two corrections matter most: (1) the admission gate controls **mFDR, not FDR**,
> under ipsum's dependent hypotheses — the honest fix is **e-values + e-LOND**; (2) the
> identifiability guard is the **positivity/overlap** condition from causal inference, and its
> high-dimensional failure is a *known* result (D'Amour et al. 2021) — so it must be cited, not
> claimed as invented.

## The spine (one commitment everything serves)
ipsum is **a small, sparse, inspectable interaction model over the file-change Boolean
cube, maintained online by bounded sufficient statistics, with statistically-gated
admission and actively-triggered eviction.** Every principle below serves that sentence.
Anything that doesn't is out of scope.

**The honest core (read this before claiming novelty).** Recovering a sparse, low-degree
function over the **Boolean cube** from random samples is provably hard — "at least as hard
as noisy parity," with an `n^{Ω(d)}` barrier even for a single interaction **[verified:
Andoni et al., SODA 2014]**. The clean polynomial-time results are for *continuous* product
distributions; ipsum's file-changed indicators are Boolean, so ipsum lives on the hard side.
Even the *positive* Boolean result (s-sparse recovery in **O(s²n)** uniform samples via L1
**[verified: Negahban–Shah, Allerton 2012]**) and the tight uniform-L∞ price of **Θ(ns²)**
per-coefficient **[verified: van Doornmalen et al., arXiv:2606.17319, 2026]** assume **i.i.d.
uniform** sampling. The L∞ lower bound holds *even against a learner that designs its own
adaptive queries* — so designed sampling cannot beat Θ(ns²), and ipsum, which gets no queries
at all and observes a **non-uniform, correlated, drifting** slice of the cube, is strictly
harder still.

That gap — **online sparse-interaction recovery over a non-uniformly-covered Boolean cube
under drift** — is ipsum's contribution. State it precisely: it is open *within the
sparse-Boolean-recovery literature and the SPEX program*, not "open in all of learning
theory" (covariate-shift, agnostic/SQ learning, and streaming compressed sensing all bear on
pieces of it). The load-bearing engineering consequence is the **§3 positivity guard**; if one
thing in this document is the thesis, it is that.

---

## 1. Representation — bounded, sparse, symbolic
- **Low degree by decree.** Abstractions are file-subsets of order ≤ `k_max` (start at 3).
  *Why:* recovering interactions on the Boolean cube is parity-hard and cost explodes in
  degree (`n^{Ω(d)}`; Andoni); low-degree is both cheaper and the realistic regime. **Don't**
  chase high-order interactions to look clever.
- **Symbolic, not dense.** Abstractions stay named file-subsets you can read. *Why:* Mairal
  gives us the online *update pattern*, not the representation — its dictionary atoms are
  dense and uninspectable, which would forfeit ipsum's whole bet. Borrow the math, not the form.
- **Small predictor.** The model over abstraction-features is a sparse logistic — the
  admitted abstractions *are* its support. *Why:* the thesis only holds if gains come from
  accumulated structure, not capacity. **Don't** let the predictor grow.

## 2. Candidate proposal — co-change, done right
The front of the funnel. Keep it cheap and high-precision so admission isn't drowned in noise.
- Group changes by **PR/MR, not raw commit**; drop commits touching **>30 files** and merge
  commits; score candidate subsets by **support AND confidence** (never frequency alone);
  add a **recency** signal; **skip sparse-history regions**. (evolutionary-coupling lit;
  evidence co-change→defects is real but *modest* — odds ratio ≈1.08 per coupling, Kirbas 2017
  — so treat candidates as cheap proposals, not strong priors.)
- **Re-propose on a ~60-day cadence** — coupling decays; this dovetails with drift handling (§6).
- **Don't** propose from thin history and trust it equally.

## 3. Admission — a controlled statistical decision (Card A)
The biggest upgrade over the current ad-hoc threshold — and the section the accuracy pass
changed most.
- **Each candidate is a hypothesis** (null = "adds no held-out predictive value"). The test
  statistic must be **valid under model selection**: the asymptotic χ²(2·LL-gain) on data the
  model was tuned on is **anti-conservative** **[verified]**. Use a held-out construction.
- **Use e-values, not p-values, and gate with e-LOND. [corrected — this is the important fix.]**
  The original plan (SAFFRON/ADDIS over p-values "controlling the false-admission rate over the
  whole stream") is wrong as stated: those procedures control **mFDR** under conditional
  super-uniformity, but **full FDR control requires independence (or only local positive
  dependence)** **[verified: Ramdas et al.; Zrnic/Fisher arXiv:2110.08161]**. ipsum's candidates
  are **nested, overlapping file-subsets → strongly dependent p-values**, so p-value procedures
  retain only mFDR, not FDR. The dependence-robust route is **e-LOND** **[verified: Xu & Ramdas,
  AISTATS 2024, arXiv:2311.06412]**, which consumes **e-values** and controls **FDR under
  arbitrary, unknown dependence** (the online lift of e-BH, Wang–Ramdas arXiv:2009.02824), at a
  modest power cost. The natural e-value for "this abstraction adds predictive value" is a
  **held-out likelihood-ratio test-martingale**: bet each held-out point's with-abstraction
  predictive density against the without-abstraction one; the running product is a nonnegative
  martingale starting at 1 = a valid e-value at any stopping time. This is both more honest under
  dependence *and* a cleaner fit to ipsum than permutation p-values.
  - *Acceptable fallback:* if you start with p-values, claim **mFDR only**, or apply a
    Benjamini–Yekutieli / Javanmard–Montanari ξ-discount for conservative FDR under local
    dependence — never claim plain FDR with SAFFRON/ADDIS on overlapping hypotheses.
- **Positivity guard — the ipsum-specific constraint (name it correctly). [corrected.]** Admit an
  abstraction *only* where the commit stream has covered its support region enough to estimate
  its coefficient to ±ε. This is exactly the **positivity / overlap** assumption from causal
  inference **[verified: Rosenbaum & Rubin 1983; Hernán & Robins 2020]** — "you cannot identify
  an effect where you have no support" — applied per-coefficient. Do **not** claim to have
  invented it. Its high-dimensional failure is itself a known result: **strict overlap becomes
  essentially impossible as dimension grows** **[verified: D'Amour et al., J. Econometrics 2021,
  arXiv:1711.02582]** — propensity mass is forced toward 0/1, the problem concentrates. A Boolean
  cube of many file-indicators is precisely that regime, so *most interaction regions have no
  overlap by the curse of dimensionality, not by accident.* The guard is ipsum operationalizing a
  known impossibility as a per-coefficient admit/refuse rule.
  - **Honest novelty:** the *combination*, not the guard. ipsum's contribution is the four-way
    intersection — **sequential + interaction-structured + non-stationary + observational
    positivity-failure**. Each ingredient (positivity, its high-dim breakdown, per-coefficient
    uncertainty gating, online testing) is leveraged, not introduced.
- **The guard ⟂ drift tension (must be designed for, not ignored). [new.]** Positivity and fast
  adaptation pull opposite ways: right after a drift, coverage of the *new* structure is thin by
  definition, so a strict guard *refuses to admit exactly when Card B demands fast recovery* — it
  would make ipsum slower to adapt, fighting its own thesis. Resolution: make the guard
  **coverage-relative and uncertainty-aware** rather than a hard cutoff. Post-changepoint (§6),
  knowingly accept higher per-coefficient uncertainty and **provisionally admit** new abstractions
  under a wider band, tightening or evicting as coverage accrues. This is where the v2
  per-coefficient-uncertainty layer earns its keep (it tells you *where* you're thin), and where
  v2 active probing is the principled fix (spend budget to cover the new regime). Track this
  explicitly as a metric: post-drift admission latency must not blow up.
- **Don't:** admit everything with positive gain; admit from thin coverage; hand-tune a threshold;
  claim FDR when you only have mFDR.

## 4. Memory & update — bounded sufficient statistics
- Keep **fixed-size running summaries** — co-occurrence counts, per-abstraction usefulness,
  delay-model stats — and **never replay history**. Each commit folds in by simple update.
  (Mairal's `A_t, B_t` sufficient-statistic recursion **[verified]**.)
- A **forgetting factor β** down-weights stale evidence: the standing, always-on drift knob,
  distinct from the discrete eviction trigger in §6. (Mairal's `β_t=(1−1/t)^ρ` rescaling.)
- This is "yesterday's posterior is today's prior" made operational. **Don't** keep a growing log
  you periodically batch over. *Caveat:* Mairal's convergence assumes i.i.d. compact-support data;
  the forgetting variant is heuristic under drift — borrow the update *pattern*, not a proof.

## 5. Credit assignment — resolve outcomes before crediting (Card C)
- **Never treat "not failed yet" as a pass.** Model the **failure-arrival delay** and weight a
  pending outcome by its survival probability `w = p(x)·exp(−λ(x)·elapsed)`; credit flows to
  abstractions only as outcomes resolve. (Chapelle, KDD 2014, EM over a conversion model + a delay
  model **[verified]**.)
- **Delay distribution — pick it correctly. [corrected.]** Exponential delay assumes a *constant
  hazard*, which is wrong for CI (a job waits in a queue, then runs for a roughly fixed time).
  Weibull/Gamma flexibilize the hazard **shape**, but a single Weibull is **unimodal** and cannot
  represent a genuinely **bimodal** delay (queue-wait mode + runtime mode) — for that you need a
  **mixture** (e.g., mixture of exponentials/Weibulls). Start with Weibull; escalate to a small
  mixture only if the resolved-delay histogram is visibly bimodal. Don't write "Weibull because
  bimodal" — those are in tension.
- **De-flake first.** Flakiness is label *noise*, a separate axis from delay — require
  retry/consistency before an outcome counts. A `max-wait` window closes the loop: after `T`,
  absence-of-failure = true pass.
- **Don't** reinforce abstractions on unresolved or flaky signals.

## 6. Eviction — active maintenance, two layers (Card B)
The core thesis claim lives here: **plasticity requires eviction; accumulation alone rots.**
- **Layer 1 — continuous.** A usefulness trace with decay; evict below complexity cost; a
  **maturity grace period** protects freshly-admitted abstractions. (Dohare's utility-based
  prune-and-renew **[verified]**.)
- **Layer 2 — triggered.** Run **BOCPD** on the store's prediction-error stream (Adams–MacKay
  2007 **[verified]**); on a detected changepoint, sweep-evict abstractions whose support predates
  it and re-propose (§2). *Tie its sensitivity to eviction precision:* every false changepoint
  causes a sweep-evict, which is exactly the **over-eviction already measured** in the Card B run
  (`stale_eviction_fraction ≈ 0.665` — a third of evictions removed still-good abstractions). The
  BOCPD hazard rate must be tuned **against** eviction precision/recall vs `synth.true_clusters()`,
  not in isolation, or the detector formalizes the bug.
- **On the append-only prediction. [corrected — softened.]** Dohare shows dense deep nets *lose
  plasticity* and decay toward a shallow baseline under long task streams — this is **suggestive
  analog evidence** that pure accumulation rots, from a *different* setting (continuous weights,
  not a symbolic store), so treat it as motivation, **not proof**. The *direct* evidence is now
  ipsum's own Card B run: append-only plateau accuracy declines across drift epochs while the
  evicting store sustains. Lead with that.
- **Don't** evict on level noise, and **don't** rebuild from scratch on every wobble.

## 7. Measurement — the thesis stays the judge
- Headline: slope vs the **data-matched, abstraction-off control**; compounding =
  **sustained / decreasing post-drift recovery across successive drifts**, judged on *uncensored*
  plateau accuracy, not a censored recovery-time slope. (Cards I/B.)
- The append-only control *should* degrade across drift epochs while ipsum holds; that divergence
  *is* the result. (Motivated by Dohare; demonstrated on synth.)

---

## What ipsum deliberately does NOT adopt (the discipline)
- **No coding-theoretic recovery** (BCH / peeling / aliasing from SPEX & the Möbius line). That
  machinery needs *designed queries* ipsum doesn't have — and even *with* designed adaptive
  queries the L∞ price is still Θ(ns²) (van Doornmalen), so there's no free lunch to import. The
  shared **object** with Ramchandran's program is real; the recovery algorithm is not inherited —
  claiming it would be wrong.
- **No high-order interactions, no large predictor, no full Bayesian apparatus** where a running
  statistic suffices. The Andoni theorem is aspirational, not an import; ipsum's value is handling
  the regime where it *fails* (Boolean, correlated, non-uniform, drifting).
- **One of each.** One admission rule, one drift detector, one usefulness trace, one delay model.
  Resist a second of anything until the first is measured and shown insufficient.

## Map to the build
- **Candidate proposal** ← §1, §2  •  **Card A** ← §3 (e-LOND admission + positivity guard) and §4
- **Card B** ← §6  •  **Card C** ← §5  •  **Measurement** ← §7
The single most novel, ipsum-specific piece is the **§3 positivity guard under drift** — it's
where a known impossibility (high-dimensional overlap failure) becomes an actual line of code in a
*sequential, non-stationary* setting, which is the part no paper in the SPEX program or this
reading list solves.

---

## Extensions — parked until after the v1 compounding result
The "does NOT adopt" list is not all permanent. Keep all of this out of v1 so the compounding
result stays clean and attributable — then this is the v2 roadmap.

### Lead extension: active, uncertainty-aware coverage management (v2 headline)
Two ideas that are really one, targeting ipsum's hardest problem (non-uniform cube coverage — the
§3 guard) instead of merely tolerating it.
- **Per-coefficient uncertainty.** Maintain a posterior variance for each interaction coefficient,
  not just a point usefulness. This *unifies four current heuristics under one quantity*:
  admission ("confidently non-zero"), the positivity guard ("variance too high → defer"), the
  guard⟂drift resolution ("provisionally admit under a wide band post-drift"), and eviction
  ("usefulness decayed into the noise"). Likely **simpler**, not heavier. (Candidate: lightweight
  Bayesian sparse logistic / per-coefficient posterior.)
- **Active probing.** Spend a *small* CI budget to deliberately run extra tests on under-sampled
  regions of the file-cube — i.e., **restore overlap by design**. D'Amour et al. name designed
  collection (and trimming) as the remedies for high-dimensional overlap failure, so this is the
  literature-sanctioned fix, not an ad-hoc knob. It sits at the intersection of **optimal
  experimental design**, **active learning** (e.g., Zhang et al., "Active Learning for Optimal
  Intervention Design in Causal Models," Nat. Mach. Intell. 2023, arXiv:2209.04744), and
  **budgeted bandits**.
- **Why they're one idea:** uncertainty tells you *where* coverage is thin; probing is what you
  *do* about it. "You don't just suffer non-uniform coverage — you measure it and spend a budget
  to fix it." This is plausibly ipsum's most novel contribution and the deepest bridge back to
  Ramchandran's program (active sparse recovery under uncertainty) — the SPEX *principle* of
  designed sampling re-entering legitimately, without the BCH machinery that needs maskable
  queries CI can't provide. It is the right v2 headline and the strongest single thing to put in
  front of the professor.
- **Cost / why not v1:** adds a budget knob and an uncertainty model — both confound the clean
  abstraction-on/off comparison. Prove compounding first.

### Scoped, not eternal
- **Large predictor — product, not thesis.** A bigger model improves the *task* but destroys
  *attribution* (gains from abstractions vs. capacity). For an eventual product, use a strong
  predictor with the abstraction store as a feature layer — but then you're claiming a
  feature-engineering win, not compounding. Keep the small predictor until the thesis is proven.
- **Adaptive local degree — minor refinement.** The hard `k_max` cap (§1) is right because
  interactions are hierarchical. The only justified relaxation is *local*: allow degree to grow for
  a specific region **only** when a lower-order model demonstrably fails there AND coverage supports
  it. Low priority; never a global high-degree default.

### Stays excluded
- **The coding-theoretic recovery machinery** (BCH/peeling/aliasing) — needs arbitrary maskable
  queries CI can't provide; and even designed queries don't beat Θ(ns²). The *principle* survives
  as active probing above; the *algorithm* does not transfer. Out, permanently.

---

## Verified references (accuracy pass 2026-06-23)
- **Andoni, Panigrahy, Valiant, Zhang**, "Learning Sparse Polynomial Functions," SODA 2014 — continuous poly(n,k,2^d,1/ε); Boolean = noisy-parity-hard, `n^{Ω(d)}`. *[full text]*
- **Negahban & Shah**, "Learning Sparse Boolean Polynomials," Allerton 2012 — O(s²n) uniform samples, L1/Walsh. *[verified]*
- **van Doornmalen, Molina, Verdugo, Verschae**, "Tight L∞ Sample Complexity for Low-Degree and Sparse Boolean Polynomials," arXiv:2606.17319, 2026 — Θ(ns²) sparse / Θ(n^{d+1}) degree-d; lower bound vs adaptive designed queries. *[full text, now verified]*
- **Rosenbaum & Rubin**, "The Central Role of the Propensity Score," Biometrika 1983; **Hernán & Robins**, *Causal Inference: What If*, 2020 — positivity/overlap. *[verified]*
- **D'Amour, Ding, Feller, Lei, Sekhon**, "Overlap in Observational Studies with High-Dimensional Covariates," J. Econometrics 2021, arXiv:1711.02582 — strict overlap impossible as dimension grows. *[verified]*
- **Xu & Ramdas**, "Online multiple testing with e-values," AISTATS 2024, arXiv:2311.06412 (e-LOND) — FDR under arbitrary dependence; **Wang & Ramdas**, e-BH, arXiv:2009.02824. *[verified]*
- **Ramdas et al.** SAFFRON arXiv:1802.09098; **Tian & Ramdas** ADDIS arXiv:1905.11465; **Zrnic/Fisher et al.** arXiv:2110.08161 (FDR under positive dependence) — mFDR vs FDR. *[verified]*
- **Chapelle**, "Modeling Delayed Feedback in Display Advertising," KDD 2014 — survival-weighted EM. *[verified]*
- **Adams & MacKay**, "Bayesian Online Changepoint Detection," arXiv:0710.3742, 2007. *[verified]*
- **Dohare et al.**, "Loss of Plasticity in Deep Continual Learning," Nature 2024 (arXiv:2306.13812) — continual backprop; analog evidence only. *[verified]*
- **Mairal, Bach, Ponce, Sapiro**, "Online Learning for Matrix Factorization and Sparse Coding," JMLR 2010, arXiv:0908.0050 — sufficient-statistic update + forgetting factor. *[verified]*
- **Zhang et al.**, "Active Learning for Optimal Intervention Design in Causal Models," Nat. Mach. Intell. 2023, arXiv:2209.04744 — active-probing grounding (v2). *[new]*
- Co-change: **Kirbas et al. 2017**; **Oliva & Gerosa**; **Jia, Hassan & Zou** "CoRanker" arXiv:2411.19099. *[verified]*
