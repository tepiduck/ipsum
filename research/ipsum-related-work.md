# Related work for ipsum's open problems — deep-read notes

Full-text reads (by sub-agent), organized by the ipsum mechanism each addresses. Each
entry: precise result → mapping to ipsum → actionable takeaway → caveats. Access caveats
flagged inline. The **keystone (§0)** carries the most weight — and the deep read changed
the story, so read that note carefully.

---

## 0. KEYSTONE — observational sparse-interaction recovery (theory of ipsum's novelty)

**The headline correction from the deep read:** the clean positive results are for
*continuous* inputs; ipsum's *Boolean* setting is provably the hard side. Be honest about
this — it's the real research gap, not a solved import.

### 0a. Andoni, Panigrahy, Valiant, Zhang — "Learning Sparse Polynomial Functions" (SODA 2014). *Full text.*
- **Result:** learns k-sparse, degree-d, n-variate **real-valued** polynomials from **random
  samples** drawn from a **product distribution** (uniform on [−1,1]ⁿ or Gaussian), noise-
  tolerant. Sample complexity **poly(n, k, 2^d, 1/ε)** — polynomial in n and k, exponential
  only in interaction degree d. Algorithm is **NOT** ℓ1/compressed-sensing; it's
  "Growing-Basis": detect a variable's participation by correlating the **squared** response
  with squared basis polynomials, grow interactions greedily, subtract, recurse.
- **The crucial caveat (load-bearing):** the positive result is *specifically for continuous
  inputs*. Over the **Boolean cube {0,1}ⁿ** the same problem is "at least as hard as noisy
  parity" — no poly algorithm known, an `n^{Ω(d)}` barrier, even for k=1. ipsum's file-changed
  indicators ARE Boolean ⇒ ipsum lives on the hard side. Also requires a **known product
  distribution** (independent coordinates); real commits are heavily correlated/non-product;
  and assumes the cube is well-covered, whereas commits cover it extremely **non-uniformly**.
- **Borrow:** the grow-by-squared-correlation estimator (admit interactions by marginal gain
  rather than enumerating subsets); the intuition to **cap interaction degree aggressively**
  (cost explodes in d). Treat the theorem as aspirational, not a drop-in.

### 0b. Negahban & Shah — "Learning Sparse Boolean Polynomials" (Allerton 2012). *Full text.*
- **Result (this is the realistic one for ipsum):** f:{−1,1}ⁿ→ℝ that is **s-sparse in the
  Fourier/Walsh basis**, from **i.i.d. uniform** random samples, via **ℓ1 minimization /
  basis pursuit**. Sample complexity **m = O(s²n)** (or O(s² log N) if support is confined to
  N candidate subsets). Guarantee is **L2 (Parseval) coefficient error**, with explicit noise
  and approximate-sparsity terms. Works because uniform sampling makes Fourier columns
  pairwise-independent ⇒ incoherence `‖ÃᵀÃ−I‖∞ ≤ 4√(n/m)`.

### 0c. van Doornmalen, Molina, Verdugo, Verschae — "Tight L∞ Sample Complexity for Low-Degree and Sparse Boolean Polynomials" (arXiv:2606.17319, 2026). *Full text — VERIFIED (accuracy pass 2026-06-23).*
- **Result:** **L∞ recovery** (worst-case error of every coefficient), under σ-subgaussian
  noise: **Θ(ns²)** samples for s-sparse, **Θ(n^{d+1})** for degree-d (noiseless: ns and n^d).
  Lower bounds hold against **arbitrary adaptive learners** (the learner may *choose* each
  X_t∈{−1,1}ⁿ from past noisy evaluations), so the price is intrinsic; the matching upper bounds
  are nonadaptive. L∞ is the right metric because ipsum's admission rule acts on *individual*
  coefficients (`admit iff |α̂_j| > τ`), so you need every coefficient correct to ±ε.
- **Correction to earlier framing:** the model is an **adaptive query** model (norms defined under
  Unif({−1,1}ⁿ), recovery not restricted to i.i.d. uniform draws). Upshot for ipsum: **even
  designed adaptive queries cannot beat Θ(ns²)** — so observational ipsum is strictly harder, and
  there is no coding-theoretic shortcut to import.

### 0d. D'Amour, Ding, Feller, Lei, Sekhon — "Overlap in Observational Studies with High-Dimensional Covariates" (J. Econometrics 2021; arXiv:1711.02582). *Full text — VERIFIED.*
- **Result:** strict **overlap/positivity** becomes essentially *impossible* as covariate
  dimension grows — "these bounds become more restrictive as the dimension grows large";
  propensity scores are forced toward 0/1 and the problem concentrates. Remedies named: **trimming**
  and **designed collection** (sparsity helps).
- **Mapping:** this is the formal reason ipsum's §3 guard exists — a high-dimensional Boolean cube
  of file-indicators, non-uniformly covered, has *no overlap in most interaction regions by the
  curse of dimensionality*. The guard = the positivity assumption (Rosenbaum–Rubin 1983; Hernán &
  Robins 2020) applied per-coefficient. **Do not claim it as invented.** v2 active probing =
  "restore overlap by design," the remedy this paper names.

**Keystone takeaway for ipsum.** An s-sparse Boolean function is recoverable from **O(s²n)**
noisy random samples (0b), and to know each coefficient to ±ε uniformly (what admission needs)
the price is **Θ(ns²)** (0c) — so size the commit budget to the ns² regime and set the
admission threshold τ above the implied L∞ error. **But** every guarantee assumes **i.i.d.
uniform** sampling; ipsum's commits are **non-uniform, correlated, drifting, Boolean** — which
is exactly Andoni's hard side. *That gap is ipsum's actual contribution and the honest core of
the Ramchandran pitch:* designed-query recovery (their program) vs random-observational
recovery under non-uniform coverage + drift (ipsum).

---

## 1. Card A — Admission under uncertainty
### Foster & Stine, "α-investing" (JRSS-B 2008); Javanmard & Montanari (arXiv:1502.06197); Ramdas et al. **SAFFRON** (arXiv:1802.09098); Tian & Ramdas **ADDIS** (arXiv:1905.11465). *Full text (α-investing via JSM deck).*
- **Mechanism:** maintain "alpha-wealth" `W`; each test spends `α_j/(1−α_j)` on a non-rejection
  and earns a payout `ω` on a rejection; stop if wealth would go negative. Guarantees **mFDR ≤ α**
  at any stopping time (independence not needed for mFDR; full FDR needs independence). **LORD**
  renews `α_j` to its largest value after each discovery. **SAFFRON** only spends wealth on
  "candidate" small p-values (λ=0.5) and uses an adaptive null-proportion (Storey) estimate →
  much higher power when most hypotheses are null. **ADDIS** additionally discards large
  p-values, dominating SAFFRON exactly when nulls are abundant and conservative.
- **Mapping to ipsum:** each candidate abstraction = one hypothesis; null = "adds no held-out
  predictive value." Loop: (1) candidate's held-out LL gain Δ; (2) convert Δ→**valid p-value**
  via a CV/permutation test (shuffle the abstraction's feature, recompute Δ) — *not* the
  asymptotic χ²(2Δ), which is anti-conservative on data the model was tuned on; (3) compute α_t
  from SAFFRON/ADDIS; (4) **admit iff p_t ≤ α_t**, update wealth. Result: guaranteed long-run
  **false-admission rate ≤ α**.
- **Actionable:** implement **ADDIS or SAFFRON** (λ=0.5, γ_j∝j^−1.6, W₀=α/2, α≈0.1) — ipsum's
  pool is dominated by true nulls (most file-subsets are noise), which is exactly their sweet
  spot. This replaces the hand-tuned LL-gain threshold with a statistically principled gate.
- **Caveats / the important correction (accuracy pass):** overlapping/nested abstractions ⇒
  **strongly dependent** p-values. SAFFRON/ADDIS then control **mFDR only** (conditional
  super-uniformity); **full FDR needs independence or merely *local positive* dependence**
  (Zrnic/Fisher arXiv:2110.08161) — not the arbitrary dependence ipsum has. So **do not claim
  plain FDR** with p-value procedures here. The dependence-robust route is **e-LOND** (Xu &
  Ramdas, AISTATS 2024, arXiv:2311.06412), which consumes **e-values** and controls **FDR under
  arbitrary dependence** (online lift of e-BH, Wang–Ramdas arXiv:2009.02824). Natural e-value for
  "abstraction adds predictive value" = a **held-out likelihood-ratio test-martingale** (running
  product of with/without-abstraction predictive density ratios on held-out points = nonnegative
  martingale = valid e-value at any stopping time). Recommendation: **e-values + e-LOND**, or claim
  mFDR-only with p-values. Drift breaks exchangeability → re-evaluate on a rolling window.

---

## 2. Card B — Eviction / anti-staleness
### 2a. Adams & MacKay — "Bayesian Online Changepoint Detection" (arXiv:0710.3742, 2007). *Full text.*
- **Algorithm:** track the **run-length** posterior `P(r_t | x_{1:t})` (time since last
  changepoint) by message passing: `P(r_t,x_{1:t}) = Σ_{r_{t−1}} P(r_t|r_{t−1})·P(x_t|r_{t−1},x^{(r)})·P(r_{t−1},x_{1:t−1})`.
  The changepoint prior puts mass only on grow (r→r+1, prob 1−H) or reset (r→0, prob H), with
  hazard H (constant = 1/λ for a geometric gap prior). A conjugate predictive model (UPM) with
  incrementally-updated sufficient statistics makes it O(t)/step, O(E[r]) with tail pruning.
- **Mapping/actionable:** feed it one scalar per cycle — the abstraction store's **prediction
  error / residual**. While code is stable, run length grows; on drift the error distribution
  shifts and run-length mass **collapses to r=0** — a reset literally means "the world drifted."
  Use that as the eviction trigger: when P(r_t=0) crosses a threshold (or MAP run-length drops),
  **evict abstractions whose usefulness/support predates the detected changepoint** and enter a
  re-learning phase (the diffuse post-reset UPM = "fresh prior"). Gaussian UPM with gamma-prior
  variance is the natural fit. Pick λ from expected drift cadence.
- **Caveats:** targets *abrupt* shifts (gradual drift may never cleanly reset — shorten λ / watch
  declining MAP run length); univariate (multivariate needs a joint UPM or per-feature detectors);
  false-alarm/hazard tradeoff. Ties cleanly to "posterior becomes prior."

### 2b. Dohare, …, Sutton — "Loss of Plasticity in Deep Continual Learning" (Nature 2024; preprint arXiv:2306.13812). *Full text (preprint).*
- **Finding:** standard deep nets on long non-stationary task streams (Continual ImageNet — 2000
  binary tasks; Permuted MNIST; RL) **lose plasticity** — accuracy first rises then **falls to
  or below a linear/shallow baseline** (89%→~77% by task 2000), across architectures/optimizers.
  Correlates: growing **dead-unit** fraction (up to 25%), growing weight magnitude, declining
  **effective rank**.
- **Fix — continual backprop:** every step, reinitialize a fraction ρ of the **lowest-utility**
  units (utility = decaying running average of |activation|·Σ|outgoing weights|, mean-corrected),
  protecting freshly-added units for a maturity period m. Maintains plasticity ~indefinitely.
- **Mapping to ipsum:** (i) **independent evidence that accumulation-without-maintenance rots** —
  predicts ipsum's append-only control should plateau/decay, not compound (directly supports the
  thesis and Card B's headline). (ii) "reinitialize low-utility units → admit fresh ones" *is*
  ipsum's "evict low-usefulness abstractions, admit fresh candidates"; per-unit utility ↔ ipsum's
  **usefulness trace**; maturity threshold ↔ grace period for new abstractions.
- **Actionable/caveats:** adopt a **utility-based prune-and-renew** rule and the thesis-level claim
  that *maintaining plasticity requires eviction*. Surface health metrics (fraction stale/never-
  used, store diversity). Loose fit: they reinit weights in a fixed net (continuous), ipsum evicts
  discrete symbolic abstractions; authors flag the utility measure as **heuristic/local** ("doesn't
  consider effect on the overall function") — expect to iterate the eviction criterion.

---

## 3. Card C — Delayed, noisy credit assignment
### Chapelle — "Modeling Delayed Feedback in Display Advertising" (KDD 2014). *Full text.*
- **Model:** two jointly-fit GLMs — a conversion/outcome model `p(x)=σ(w_c·x)` and an
  **exponential delay** model `P(D=d|x,C=1)=λ(x)e^{−λ(x)d}`, `λ(x)=exp(w_d·x)`. The crux is the
  corrected likelihood for a **not-yet-converted** sample (Eq. 9):
  `P(Y=0|x,e) = (1−p(x)) + p(x)·e^{−λ(x)e}` — a mixture of "true negative" and "positive whose
  outcome hasn't arrived (survival D>e)." EM with hidden C: E-step weight `w_i = p(x)e^{−λ(x)e_i}`
  (=1 if already converted); M-step = weighted logistic + exponential/survival regression.
- **Mapping/actionable for ipsum:** a test that **hasn't failed yet** is an unlabeled
  positive-or-negative — **don't label it pass.** Fit `p(x)` (failure prob) and `λ(x)` (failure-
  *arrival-delay*) keyed on decision features (test, change-set, abstraction signature). At the
  credit/eligibility step, weight a not-yet-failed test by `w_i=p(x)e^{−λ(x)e_i}` = prob the
  failure is merely **pending**; reinforce abstractions with these weighted outcomes (hard
  positives for observed failures). When elapsed ≪ mean delay, the sample barely moves credit;
  when ≫ mean delay it counts as a clean negative. This removes the systematic optimism (naive
  baseline under-predicted positives by 21%).
- **Caveats:** exponential = constant hazard. **Weibull/Gamma** fix the hazard *shape* but stay
  **unimodal** — for genuinely *bimodal* CI delay (queue-wait mode + fixed-runtime mode) use a
  **mixture** (mixture of exponentials/Weibulls), not a single Weibull. **Flakiness ≠ delay** — a flaky failure is label *noise*,
  a separate axis needing a flake model on top. Needs a **max-wait window** after which
  absence-of-failure = true pass (Criteo used 30 days). Joint likelihood is non-convex.

---

## 4. Architecture — the abstraction store as a learned object
### Mairal, Bach, Ponce, Sapiro — "Online Learning for Matrix Factorization and Sparse Coding" (JMLR 2010; arXiv:0908.0050). *Full text.*
- **Method:** online loop — **encode** new sample `α_t=argmin ½‖x_t−D_{t−1}α‖²+λ‖α‖₁` (LARS);
  **accumulate sufficient statistics** `A_t←A_{t−1}+α_tα_tᵀ`, `B_t←B_{t−1}+x_tα_tᵀ`; **update
  dictionary** by block-coordinate descent on the surrogate `½Tr(DᵀDA_t)−Tr(DᵀB_t)` (per-atom
  closed form, Eq. 7). **Stores no past data** — only fixed-size A_t,B_t. Converges a.s. to a
  stationary point (surrogate/stochastic-approximation).
- **Mapping/actionable:** store ↔ dictionary, abstraction ↔ atom. The O(1)-memory recursion is
  ipsum's "cheap, compounding, recursive update" made formal: keep **bounded co-occurrence/usage
  summaries** instead of replaying commits. Borrow the encode↔update alternation; the **forgetting
  factor** `β_t=(1−1/t)^ρ` rescaling A_t,B_t to down-weight stale evidence (a principled **drift
  knob**); and **atom replacement** ("atoms never used → replace by random elements") = directly
  informing eviction (drop dead abstractions) + admission (seed new ones).
- **Caveats:** atoms are continuous/dense — **no inspectability analogue** (ipsum's symbolic-subset
  bet has no counterpart here); convergence assumes i.i.d. compact-support data, so drift breaks
  the guarantee (the forgetting variant is heuristic); objective non-convex. Analogy is the
  recursive sufficient-statistic *update pattern*, not a borrowed correctness proof.

---

## 5. Candidate proposal + signal validation (domain)
### Oliva & Gerosa (change-coupling chapter); Kirbas et al. 2017 (J. Softw. Evol. Proc.); Jia, Hassan & Zou "CoRanker" (arXiv:2411.19099). *Full text (open-access copies).*
- **Definitions/metrics:** evolutionary/logical coupling = artifacts that frequently change
  together in VCS history. Market-basket metrics per commit-as-transaction: **support** (co-change
  count), **confidence(A⇒B)=support(A∪B)/support(A)**, lift/conviction. Published thresholds:
  Zimmermann (support>1, conf>0.5); Bavota (≥2% of commits, conf≥0.8); outlier cutoffs Q3+1.5·IQR.
- **Evidence co-change→defects:** real but modest. Kirbas (176K files, 7 yr): significant EC↔defect
  in 59% of modules, global Spearman ρ≈.28, logistic **odds ratio ≈1.08** per extra coupling
  (controlling for commits/devs). Cataldo & Nambiar (189 projects): EC the most significant factor
  explaining defects. CoRanker: co-change captures dependencies static analysis misses (5/20 top
  pairs had no structural dependency).
- **Mapping/actionable for ipsum's `candidates()`:** propose file-subsets by **support AND
  confidence** (not support alone — frequency's top error is stale/over-frequent couplings).
  **Group at PR/MR level, not raw commits** (67% of true co-change pairs span commits within one
  PR). **Filter large commits** (>30 files = infra/merge noise) and merge commits (`--first-parent`).
  Handle **tangled commits** (link to issues, keyword-classify). A learning-to-rank over ~10 cheap
  features (path similarity, author Jaccard, code deps) beats raw frequency by ~4.7% NDCG@5.
- **Biggest gap both papers flag — time decay:** coupling accuracy declines after ~60 days. Add a
  **"days-since-last-co-change" recency feature**, weight recent co-changes higher, and **re-propose
  candidates on a ~60-day cadence** — which dovetails with Card B's drift handling. Down-weight
  candidates from sparse-history modules (EC predicts poorly there).

---

## Priorities (unchanged, sharpened)
- **Read first / load-bearing:** §0 — and note the Boolean-hardness correction; it *is* the research gap.
- **Most immediately actionable for the build:** §1 ADDIS/SAFFRON (Card A admission, with a CV/permutation p-value); §3 Chapelle delay model (Card C); §2a BOCPD trigger (Card B).
- **Strong supporting:** §2b Dohare (motivation + utility-based prune-and-renew), §4 Mairal (sufficient-statistic online update + forgetting factor), §5 co-change (candidate proposal + ~60-day re-proposal cadence).

## Access caveats
All read full-text. (Accuracy pass 2026-06-23: **van Doornmalen 2606.17319** §0c is now full-text
verified — Θ(ns²)/Θ(n^{d+1}) confirmed, adaptive-query model; the earlier rate-limit is resolved.)
**Kirbas 2017** read via the open-access repository copy (Wiley DOI gated).

## Accuracy-pass corrections (2026-06-23) — summary
- §0c van Doornmalen now verified; adaptive-query model (even designed queries can't beat Θ(ns²)).
- §0d D'Amour added: high-dimensional overlap/positivity failure — the formal grounding of the §3 guard.
- §1 admission: SAFFRON/ADDIS give **mFDR, not FDR**, under ipsum's dependent hypotheses; switch to
  **e-values + e-LOND** (arbitrary-dependence FDR) — see corrected §1 caveat.
- §3 delay: a single Weibull is unimodal; bimodal CI delay needs a **mixture**.
- §2b Dohare is **analog** evidence (dense nets ≠ symbolic store), not proof; Card B's own run is the direct evidence.
