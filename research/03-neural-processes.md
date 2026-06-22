# Annotated Note: Conditional Neural Processes (CNPs)
**Paper:** Garnelo et al., "Conditional Neural Processes," ICML 2018 — arXiv:1807.01613

---

## Core idea

A CNP is a meta-learned, amortized function approximator. Rather than training a new model for each new task, it trains a single network across a distribution of tasks so that at test time it can condition on a small observation set O = {(x_i, y_i)} and immediately produce calibrated predictions for any target x — in a single forward pass, no re-training, no MCMC, no matrix inversion. It replaces the GP's analytic kernel machinery with a learned encoder + aggregator + decoder trained by maximizing the conditional log-likelihood of held-out targets given randomly sampled context subsets. The practical result: a model that carries domain-wide statistical structure as weights (the "prior") and updates its effective posterior just by feeding in new context points at inference time.

---

## Mechanism

**Architecture (three components):**

1. **Encoder h_θ : X × Y → R^d** — Each observation (x_i, y_i) is independently embedded into a fixed-dimensional representation r_i = h_θ(x_i, y_i). For low-dimensional inputs, h is an MLP; for images, convolutional layers are added.

2. **Symmetric aggregator ⊕** — The observation embeddings are pooled into a single representation r = (r_1 ⊕ ... ⊕ r_n), typically a mean: r = (1/n) Σ r_i. The key invariant: this operation is commutative and associative, so the result is permutation-invariant in O and the running aggregate can be updated in O(1) as new observations stream in (r_new = (n·r_old + r_{n+1}) / (n+1)).

3. **Decoder g_θ : X × R^d → Φ** — For each target x_t, the decoder takes (x_t, r) and outputs distribution parameters φ_t. For regression: (μ_t, σ²_t) of a Gaussian. For classification: class logits.

**Training objective:** Sample a function f ~ P, sample n observations and a subset N < n for context; maximize E[log Q_θ({y_i}_{i=0}^{n-1} | O_N, {x_i}_{i=0}^{n-1})]. This forces the model to score both observed and unobserved targets, training it to generalize from partial context.

**Runtime:** O(n + m) at test time, where n = context size, m = number of query points. This contrasts with GP inference at O((n+m)^3).

**Relation to GPs:** A GP specifies a prior over functions via an analytic kernel; inference is exact but cubic-cost. A CNP replaces the kernel with a learned encoder, the posterior update with a forward pass through the aggregator, and the marginal likelihood with an amortized variational objective. The CNP does *not* guarantee consistency of its conditional distributions (i.e., querying a different target set with the same context may not yield marginals consistent with any single prior process) — it trades that theoretical guarantee for scalability and flexibility.

**Latent variable extension (NP proper):** The factored CNP independently predicts each target, so it cannot produce coherent joint samples (it averages over modes). Adding a global latent z ~ Gaussian(r) fixes this: one sample of z generates a coherent prediction across all targets. Trained as a VAE with a context-conditioned prior p(z|O) and a target-conditioned posterior p(z|O,T). This is the full "Neural Process" (NP) variant.

---

## Why it matters for this project

Jeffrey's thesis is: expertise = a learned prior over "what matters in a domain," updated recursively and cheaply as experience arrives. CNPs are almost a literal implementation of that description — they just operate on function-observation tuples rather than domain episodes. The mapping is direct:

- **The trained encoder+aggregator weights are the learned prior.** They encode domain-wide statistics extracted from a training distribution of tasks/functions (analogous to "what matters" in a code domain — e.g., which test/feature co-occurrences predict failure).
- **Feeding new context points O is the cheap recursive update.** It does not touch θ; it runs a forward pass through h_θ and updates r via a running mean in O(1) per new observation. This is exactly the "cheap update of a learned prior" Jeffrey wants: no re-training, no retrieval overhead, just conditioning.
- **The amortized posterior r = aggregate(h_θ(O))** is the inspectable learned state after seeing experience. r can be interrogated, visualized, or used as a feature.
- **Compounding:** As more domain-specific experience arrives (more context points), r converges faster and predictions sharpen. The model's *rate* of improvement with incoming data is governed by what it learned during meta-training — this is structurally the "slope" advantage over a frozen LLM+retrieval system, which gets no smarter with domain experience.

**Concretely for the v1 wedge (predictive test selection):** A CNP trained on (commit-delta → test outcomes) pairs across many repos learns a prior over "which test/code feature co-occurrences predict failure." At deployment on a new repo, new commits O = {(Δ_i, pass/fail_i)} are appended to context in O(1); the decoder then scores candidate tests for the next commit. The baseline (frozen LLM+retrieval) has no such update mechanism — Jeffrey's system improves its prior on that specific repo's failure patterns with every commit, and the baseline doesn't.

---

## What to extract / reuse

1. **The encoder-aggregate-decode pattern** — adopt directly. For the test-selection domain: h_θ encodes (code-delta features, test outcome) pairs; aggregator maintains a running mean representation of the repo's observed failure pattern; g_θ decodes (candidate test features, r) → failure probability.

2. **Streaming O(1) update formula** — since r is a mean, new observations update r without storing history: r_new = (n·r_old + h_θ(x_{n+1}, y_{n+1})) / (n+1). This is the cheap recursive update. Implement exactly this.

3. **Training regime: random context masking** — training by subsampling context (N ~ uniform[0, n-1]) is the mechanism that forces the model to generalize from partial evidence. Use this to train the test-selection model on historical commit subsets.

4. **Uncertainty output** — the per-target variance σ² is a principled uncertainty estimate derived from context size and consistency. For test selection this becomes a selection criterion: pick tests where predicted failure probability is high *and* uncertainty is high (most informative). This is also the active exploration hook (Figure 3b in the paper shows uncertainty-guided selection beats random).

5. **Latent variable extension** — not needed for v1, but relevant if Jeffrey wants coherent joint selection (e.g., select a *set* of tests with correlated coverage, not independent per-test predictions).

---

## What to skip or ignore for our purposes

- **Image completion experiments** (Sections 4.1–4.2.2): illustrative proofs of concept, not informative for the code domain. The architecture lessons transfer but the experimental details don't.
- **Omniglot few-shot classification** (Section 4.3): same — demonstrates generality but isn't the target domain.
- **The GP comparison as a primary yardstick**: the relevant baseline for Jeffrey is frozen LLM+retrieval, not GPs. The CNP-vs-GP framing is the paper's frame, not ours.
- **The theoretical stochastic process framework** (Section 2.1, consistency requirements): CNPs explicitly *drop* the consistency guarantee, and that's fine. Don't get distracted trying to recover it.
- **Deep GP and Deep Kernel Learning references**: tangential detours into GP-centric literature; skip unless specifically evaluating kernel methods.

---

## Limitations & risks relevant to us

1. **No autonomous abstraction discovery.** The CNP architecture assumes the feature representation of observations (x_i, y_i) is pre-specified. In Jeffrey's system, the hard open problem is what to put in x (which code/commit features are "what matters"). CNPs do not solve this — they learn to aggregate representations of already-defined features. If the input featurization is wrong, the learned prior is wrong. This is Open Problem (1) in Jeffrey's list.

2. **The aggregation is lossy.** The mean pooling aggregator throws away ordering and individual structure. Two observation sets with the same mean embedding get the same prediction. For the test-selection domain, pathological cases (e.g., rare failure patterns masked by a majority of passing tests) may collapse in r. Alternatives: attention-based aggregators (used in later NP variants like Attentive NPs), but at higher cost.

3. **No credit assignment across observations.** The symmetric aggregator weights all context points equally. There is no mechanism to weight a highly informative commit (one that revealed a new failure mode) more than a routine one. Open Problem (3) in Jeffrey's list.

4. **Drift / staleness risk.** The trained θ is frozen after meta-training. If the domain shifts (e.g., a codebase's failure patterns change significantly after a major refactor), r will still be formed by passing observations through a stale h_θ. The CNP framework has no inner-loop weight update — it only updates the context buffer, not the encoder. Open Problem (4).

5. **Distribution shift in meta-training.** CNPs are only as good as the distribution of tasks P they were trained on. If GitHub public repos don't cover the structure of a specific enterprise codebase, the learned prior may not transfer. This is the key risk for the v1 wedge: define the meta-training distribution carefully.

6. **No proof of compounding.** The paper demonstrates that more context → better predictions *within a session*. It does not demonstrate that accumulated context across sessions improves the *rate* of future learning. The compounding thesis requires either periodic fine-tuning of θ (which re-opens the costly learning loop) or a smarter update scheme (e.g., continual learning / EWC). This is Open Problem (5).

7. **Factored output ignores inter-test correlation.** The base CNP predicts each test independently. In the test-selection domain, tests often have correlated coverage — selecting one makes another redundant. The latent variable extension partially addresses this but requires the full NP training setup (VAE + posterior conditioning).

---

## One-line takeaway

CNPs give you the cheapest possible implementation of "condition on experience, update the effective posterior, query for predictions" — the encoder-aggregate-decoder pattern with O(1) streaming updates is directly adoptable as the inference-time update mechanism in Jeffrey's expertise system, but it requires a pre-specified featurization, doesn't self-discover abstractions, and doesn't solve compounding across training episodes without periodic re-training of θ.
