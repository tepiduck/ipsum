# Synthesis — the nine foundational papers

The thesis splits into two halves: **(A) a learned prior you can cheaply update**, and **(B) abstractions that compound and stay inspectable.** The literature solves half A cleanly and leaves half B — the novel part — open.

## Cluster 1 — "Prior as a network" (solved; reuse off the shelf)

- **MAML** (`01`) — the prior *is* a weight initialization; a few gradient steps = the posterior update. Use first-order MAML.
- **Grant et al.** (`02`) — proves MAML's init is literally an empirical-Bayes prior mean, and the inner optimizer is its covariance. The formal grounding that makes "the priors of ML training are themselves a network" a precise statement.
- **Neural Processes / CNP** (`03`) — encoder→mean-pooled latent→decoder; conditions on experience in one O(n) forward pass and emits uncertainty. Mean-pool gives the O(1) streaming update ("posterior becomes next context"). Incremental conditioning done correctly — not online weight surgery.
- **HyperNetworks** (`04`) — most literal "make the prior a network." Likely overkill for v1; substrate for later.
- **EWC** (`07`) — Fisher-weighted quadratic anchor = "posterior of task A becomes prior for task B." Drift/forgetting control and the recursive-update mechanism made concrete.

**Conclusion:** CNP-style amortized conditioning + an EWC-style anchor + FOMAML is a complete, buildable recipe for a cheaply-updated prior with uncertainty. No novelty needed here.

## Cluster 2 — "Abstractions that compound" (open; the contribution)

- **DreamCoder** (`05`) — closest prior art. Wake/sleep loop; admits an abstraction only if it compresses (MDL) solved tasks more than it costs. That admission criterion is the most reusable idea in the stack. But its machinery assumes **binary, immediate, exact** outcomes, and it never prunes the library.
- **Voyager** (`06`) — proof the abstraction store can be an inspectable code library updated with **zero weight training** (solo-dev-achievable). Its no-library-plateaus / full-library-keeps-climbing ablation is the empirical shape of the slope thesis. But retrieval is naive embedding similarity (no credit assignment) and skills are write-once (no revision, no eviction).
- **AlphaZero** (`08`) — purest compounding loop: learned prior + search + self-play under one clean reward; the Elo slope is the proof. But it needs a perfect zero-latency simulator and self-play, which CI does not provide. (NNUE contrast: NNUE = cheap inference on frozen weights; AlphaZero = the prior actually learns. The thesis is the AlphaZero side of that line.)

**Conclusion:** every paper that demonstrates compounding does it under conditions we won't have (exact/immediate outcomes, or a simulator). The shared gaps — **credit assignment, staleness/eviction, noisy+delayed outcomes** — are the research surface.

## The wedge

- **Predictive Test Selection** (`09`, Machalica et al., ICSE-SEIP 2019) — confirmed. XGBoost over (change, test) pairs; strongest features = build-dependency-graph distance + historical per-test failure rate. Targets: TestRecall > 0.95, ChangeRecall > 0.999, SelectionRate < 0.33 → ~3x fewer test runs. Retrained **weekly from scratch — zero online update, zero compounding.** That plateau is the bar.

## Solved vs. open

| Need | Status | Source |
|---|---|---|
| Prior as a network | Solved | MAML / HyperNetworks |
| Formal Bayesian grounding | Solved | Grant et al. |
| Cheap recursive/streaming update | Solved | Neural Processes; EWC |
| Forgetting/drift control | Solved-ish | EWC |
| Inspectable abstraction store w/o training | Demonstrated | Voyager |
| Abstraction admission (exact case) | Demonstrated | DreamCoder |
| Compounding loop shape | Demonstrated (simulator) | AlphaZero |
| **Credit assignment under noisy/delayed outcomes** | **OPEN** | contribution |
| **Abstraction eviction / anti-staleness** | **OPEN** | contribution |
| **Admission under uncertain likelihoods** | **OPEN** | contribution |
| v1 problem + baseline | Given | Predictive Test Selection |

## Three mechanisms to re-derive (the actual research)

1. **Admission under uncertainty** — DreamCoder's exact MDL → Bayesian model selection on held-out predictive likelihood.
2. **Delayed credit assignment** — eligibility traces / retrospective labeling to attribute a CI outcome (after de-flaking) to the abstractions that informed it.
3. **Eviction / staleness** — decay + eviction rule; EWC Fisher importance as a candidate "what to protect" signal.

Per-paper detail in `01`–`09` alongside this file.
