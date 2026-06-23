# Ramchandran research map (for the outreach / SPEX direction)

Reference for approaching Kannan Ramchandran (UC Berkeley EECS). Compiled from
full-text reads of the arXiv papers below (Spectral Sparsity is abstract-only —
paywalled, no preprint). Keep this separate from ipsum's build; it serves the
*relationship*, not the thesis.

## The program is one idea
His recent ML work is a single thesis: **model interpretability = sparse spectral
recovery of feature interactions over the Boolean cube `{0,1}ⁿ`, using
coding-theoretic tools.** A model output is treated as a set function
`f: {0,1}ⁿ → ℝ`; its important "interactions" are the few nonzero coefficients of a
transform (Möbius or Fourier); structured query masks act as a sparse-graph/channel
code so aliasing + peeling decode the sparse support cheaply. Shapley/Banzhaf indices
are linear read-outs of that same spectrum.

**The invariant across every paper:** they assume you can **design queries** to a
**fixed** function. Observational data, online estimation, and drift appear only as
future-work asides. That corner is exactly where ipsum lives — so ipsum's wedge holds
against the *entire* program, not one paper.

## The papers

1. **Adaptive Sparse Möbius Transforms for Learning Polynomials** — arXiv:2602.06246
   (Erginbas, Kang, Polito, Ramchandran, 2026). **Best anchor for ipsum.** Targets an
   s-sparse degree-d Boolean polynomial `f:{0,1}ⁿ→ℝ` in the **AND basis** — literally
   ipsum's object (file-subset coefficients). "Adaptive" = adaptive query *design*
   (group testing), noiseless oracle; exact recovery in O(sd·log(n/d)) queries. Same
   group (Kang co-author). Object matches exactly; access model (designed, noiseless,
   static) is precisely what ipsum relaxes.

2. **Learning to Understand: Identifying Interactions via the Möbius Transform** —
   arXiv:2402.02631 (Kang, Erginbas, Butler, Pedarsani, Ramchandran, NeurIPS 2024).
   The **methodological seed**: sparse Möbius/Harsanyi recovery via aliasing +
   graph-peeling + a **group-testing** connection → O(Kn), or O(Kt log n) for degree-t.
   Non-adaptive batch *query*. Read it to understand the coding machinery the whole
   line inherits.

3. **SPEX: Scaling Feature Interaction Explanations for LLMs** — arXiv:2502.13870
   (Kang, Butler, Agarwal, Erginbas, Pedarsani, Ramchandran, 2025). The Fourier/
   Walsh-Hadamard, LLM-scale (n≈1000) instantiation; BCH-coded masks + message-passing.
   Switches Möbius→Fourier because Fourier is orthonormal (less noise amplification).
   Static, per-input, requires designed query access.

4. **ProxySPEX: Inference-Efficient Interpretability via Sparse Feature Interactions**
   — arXiv:2505.17495 (Butler, Agarwal, Kang, Erginbas, Yu, Ramchandran, NeurIPS 2025).
   Fits a **gradient-boosted-tree surrogate** to uniformly-sampled (subset, output)
   pairs, extracts the trees' exact sparse Fourier support, top-k. ~10× fewer
   inferences. The one method ipsum could actually borrow for its admission step
   (caveat: assumes *hierarchical* interactions; batch, not online).

5. **Spectral Sparsity: A Unifying Framework for Scalable Model Interpretability
   Using Codes** — IEEE BITS 2026 (Kang, Tsui, Erginbas, Butler, Aghazadeh,
   Ramchandran). The umbrella: Möbius & Fourier are two bases, SPEX/ProxySPEX two
   decoders, Shapley/Banzhaf linear read-outs — all one sparse-recovery problem.
   ⚠️ Paywalled, no preprint — abstract only. Read for program-level vocabulary.

6. **Why Do Multi-Agent LLM Systems Fail?** — arXiv:2503.13657 (Cemri, Pan, …,
   Ramchandran, Stoica, Gonzalez, Zaharia; NeurIPS 2025). Thematic, not technical.
   MAST taxonomy of 14 failure modes (spec/design, inter-agent misalignment,
   verification/termination); fix = durable memory/state. Loose-but-real bridge to
   ipsum's "agents don't compound." Don't overclaim — they study within-run failure,
   not compounding.

## Reading order
1 → 2 → 3 → 4 → (5 abstract) → 6. Read 1 (ASMT) and 3 (SPEX) closely enough to defend
the bridge live; 2 for the through-line; 4 for the borrowable trick; 6 for motivation.

## The bridge, stated honestly
**Object match is exact** across the whole program: ipsum's failure-vs-changed-files is
a sparse, low-degree set function over `{0,1}^n_files`, and the abstraction store is its
sparse interaction support. **What ipsum relaxes — and they all assume:** (a) designed
query access (ipsum only *observes* commits), (b) stationarity (ipsum drifts), (c) a
noiseless oracle (ipsum's outcomes are noisy/flaky). Claim the **shared object + the new
regime** — NOT the peeling/group-testing/BCH machinery, which ipsum does not inherit.

## Email anchor (paragraph 2 — re-anchored on ASMT)
> Your Adaptive Sparse Möbius Transform recovers sparse AND-basis interactions over
> {0,1}ⁿ by adaptively designing queries to a noiseless oracle. ipsum asks the inverse:
> can you recover that same support **observationally** — from passively-observed
> commits, under noise and drift — relaxing the designed-query and stationarity
> assumptions your near-optimal bounds rely on?

This stakes a new axis (observational + drift) instead of competing on query complexity,
and shows you've read past SPEX into his 2026 work.
