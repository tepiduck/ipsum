# EWC: Overcoming Catastrophic Forgetting in Neural Networks
**Paper:** Kirkpatrick et al. (2017), "Overcoming Catastrophic Forgetting in Neural Networks," PNAS  
**arXiv:** 1612.00796

---

## Core idea

When a neural network learns a new task sequentially, gradient descent on the new task's loss freely overwrites weights that were critical for earlier tasks — this is catastrophic forgetting. EWC (Elastic Weight Consolidation) prevents this by adding a per-weight quadratic penalty that anchors each weight close to its old value, with stiffness proportional to how important that weight was to previously learned tasks. The result is that the network finds parameter settings that perform well on the new task while staying in the low-error region of the old task. This is analogous to synaptic consolidation in the mammalian neocortex, where important synapses become less plastic after skill acquisition.

---

## Mechanism

**Fisher Information as importance:** Importance of weight i to task A is measured by F_i, the i-th diagonal element of the Fisher information matrix. F has three key properties exploited here: (a) it equals the second derivative of the loss near a minimum (i.e., curvature — high curvature means moving that weight hurts performance a lot), (b) it can be computed from first-order gradients alone (cheap, scalable), and (c) it is guaranteed positive semi-definite. In practice, only the diagonal is used (factorized/mean-field approximation), making cost linear in number of parameters.

**The quadratic penalty (loss at task B given prior task A):**

```
L(θ) = L_B(θ) + Σ_i  (λ/2) · F_i · (θ_i − θ*_A,i)²
```

Each weight is pulled back toward its task-A optimal value θ*_A,i with spring constant λ·F_i. λ controls the old-vs-new trade-off. For a third task C, penalties for A and B can be combined into a single quadratic (sums of quadratics are quadratic).

**Sequential Bayesian interpretation (the clean version):** From Bayes' rule:

```
log p(θ | D) = log p(D_B | θ) + log p(θ | D_A) − log p(D_B)
```

When learning task B, the posterior over task A — p(θ|D_A) — becomes the prior. All information about task A is encoded in this posterior. EWC approximates p(θ|D_A) as a Laplace (Gaussian) approximation: mean = θ*_A, precision = diagonal Fisher F. Minimizing the EWC loss is then equivalent to MAP inference under this Gaussian prior. Each new task's posterior feeds forward as the prior for the next task — this is the sequential/recursive Bayes chain.

---

## Why it matters for this project

**Direct connection to drift/forgetting (open problem 4):** Jeffrey's system maintains a learned prior over "what matters in a domain." As it ingests new repositories and abstractions, gradient updates will tend to overwrite previously consolidated structure — the same catastrophic forgetting failure mode EWC was designed for. EWC's penalty mechanism is a concrete, tested solution: when updating the prior store after processing a new batch of repos, compute per-weight Fisher importances on the old corpus, then constrain the update. This directly addresses the drift/staleness open problem.

**"Posterior becomes next prior" = recursive Bayesian update (Jeffrey's framing):** EWC formalizes exactly the recursive-update structure Jeffrey is targeting. After task A, p(θ|D_A) is the posterior; it becomes the prior for task B, yielding p(θ|D_A, D_B); this becomes the prior for task C, and so on. Each cycle is cheap (recompute diagonal Fisher on the new data, add a penalty term). This is not a metaphor — it is the literal mathematical justification for EWC, and it maps 1:1 onto the thesis that expertise = a recursively-updated prior. The v1 wedge (predictive test selection on GitHub repos) could be framed as task A; every new batch of repos is the next task in the sequence.

**Preserving compounding structure (open problem 5):** A compounding system only compounds if early learned structure is not destroyed by later updates. EWC is a proved (empirically, on MNIST permutations and 10 Atari games simultaneously) mechanism for preserving that structure. It is the ingredient that would let the abstraction store actually improve monotonically rather than oscillate.

---

## What to extract / reuse

1. **The penalty formula itself** (equation 3 above) — directly applicable to any gradient-based update of the abstraction/knowledge store. If the store is parameterized as a neural model, this is drop-in.

2. **Diagonal Fisher as importance signal** — even if EWC's penalty is not used verbatim, Fisher-diagonal importance scores are a cheap, first-order proxy for "which parts of the learned prior actually matter." This is useful for selective freezing, pruning, or versioning of the prior.

3. **The sequential Bayes framing** — use it in the project writeup/thesis to give rigorous grounding to the "recursive prior update" claim. The derivation (pages 2-3 of the paper) is clean enough to cite directly.

4. **Task-specific gains/biases trick** (from Atari experiments, Appendix 4.2) — they added per-task scale and bias parameters at each layer, allowing task-specific calibration without full separate networks. Analogous to domain-specific adapters on a shared prior — worth noting for v1 where different repos/domains may need lightweight specialization on top of a shared abstraction layer.

5. **Fisher overlap metric** (Appendix 4.3, Frechet distance between normalized Fisher matrices) — a principled way to measure whether two tasks (or two domains of repos) share important weights, i.e., whether the prior is genuinely transferring vs. just co-located. Directly useful for ablation/validation: if Fisher overlap between test-prediction-related weights and abstraction weights is high, that is evidence the shared prior is doing real work.

---

## What to skip or ignore for our purposes

- **The Atari RL setup in detail** — DQN, experience replay buffers, task-recognition HMM: this is scaffolding specific to the RL domain. The RL machinery is not relevant to the v1 wedge (supervised/predictive test selection on static repos).
- **The neurobiological motivation** (dendritic spines, synaptic consolidation in mouse neocortex) — interesting backstory but adds no technical value for implementation.
- **The task-recognition module (Forget-Me-Not / HMM over observation models)** — this is for the setting where task boundaries are unknown. For our v1, we have explicit domain/repo boundaries, so this complexity is unnecessary.
- **The full Fisher matrix (off-diagonal)** — EWC's own approximation is diagonal; the off-diagonal structure is acknowledged to be an approximation weakness but out of scope even for EWC's authors. Do not try to use the full matrix; it is O(n²) and intractable at scale.
- **Permuted MNIST experiments** — the MNIST results are just proof-of-concept; they do not generalize any specific numerical insight to our domain.

---

## Limitations & risks relevant to us

1. **Diagonal Fisher underestimates uncertainty / over-protects some weights.** The paper explicitly notes (Fig 3C, page 6) that perturbing in the alleged null-space of the Fisher still hurts performance — meaning EWC over-confidently labels some weights as unimportant. For our system, this means some drift may be falsely permitted and some capacity may be unnecessarily frozen. The remedy they suggest (Bayesian neural networks, Blundell et al.) is expensive; a practical middle ground is periodic Fisher recomputation rather than single-point estimation.

2. **Fixed-capacity bottleneck.** EWC uses a fixed-size network and squeezes multiple tasks into shared capacity. The paper shows only modest error growth across many tasks, but this *will* saturate eventually. For our system, if the abstraction space keeps growing (new domains, new patterns), fixed capacity becomes a hard ceiling. Mitigation: plan for progressive expansion (see Progressive Neural Networks, Rusu et al. 2016, which the paper cites as the alternative).

3. **Quadratic penalty can stifle plasticity if λ is too large.** The λ hyperparameter is critical and problem-specific; too high and the system cannot update the prior at all (L2-like behavior), too low and forgetting recurs. For our recursive update setting, λ may need to decay as the system becomes more confident in early abstractions — not straightforward to automate.

4. **No autonomous abstraction discovery.** EWC assumes the task boundary is known (or estimated via a separate mechanism). It says nothing about how to form new abstractions or decide which experiences should constitute a "task." Open problem 1 (autonomous abstraction discovery) is entirely outside EWC's scope.

5. **Point-estimate posterior (Laplace approximation) is local.** The Fisher is computed at the current θ*_A, so the Gaussian prior is a local approximation valid near that point. If future training moves θ far from θ*_A before the penalty is applied, the approximation degrades. For our system with large batches of repos arriving infrequently, this could be significant — may want to compute Fisher incrementally.

6. **Does not handle abstract, compositional priors.** EWC works at the level of raw network weights. If the "prior" Jeffrey wants to learn is a structured, symbolic, or compositional object (e.g., a graph of abstractions), EWC's weight-space penalty may not translate directly without re-parameterization.

---

## One-line takeaway

EWC gives a principled, computationally cheap (O(n) per parameter, no replay needed) mechanism for recursive prior update in gradient-based systems: anchor important weights via Fisher-scaled quadratic penalties, which is exactly the sequential-Bayes "posterior becomes next prior" operation Jeffrey's thesis requires — use it as the anti-drift backbone of the abstraction store update loop.
