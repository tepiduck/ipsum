# 01 — MAML: Model-Agnostic Meta-Learning (Finn, Abbeel, Levine 2017)
> arXiv:1703.03400v3

---

## Core idea

MAML finds a single weight initialization θ such that a small number of gradient steps on any new task drawn from a task distribution p(T) produces strong performance — i.e., it trains the parameters to be *maximally sensitive to task-specific loss signals* rather than to directly minimize any one task. The weights themselves are the prior; the prior is not a separate module but is implicit in the geometry of the loss landscape induced by θ. Fast adaptation is the payoff: after meta-training, one to five gradient steps on K examples of a new task reliably generalizes.

---

## Mechanism

**Outer loop (meta-update):** Sample a batch of tasks Ti ~ p(T). For each Ti, compute an *inner* gradient step on K examples to get task-adapted parameters θ'i = θ − α ∇θ L_Ti(fθ). Then update θ via SGD/Adam on the *post-adaptation* losses summed across all Ti (Equation 1):

    θ ← θ − β ∇θ Σ L_Ti(f_{θ'i})

The outer gradient differentiates *through* the inner gradient step, requiring second-order derivatives (Hessian-vector products). A first-order approximation (omit second-order terms) recovers ~99% of the performance with ~33% faster compute, because ReLU nets are locally nearly linear.

**Inner loop (task adaptation):** One or a few vanilla SGD steps on K labeled examples for that task — no learned optimizer, no extra parameters, just gradient descent on the task loss.

**What is actually learned:** Only θ, the shared initialization. No extra parameters, no meta-network, no recurrent state. The structure of the model is unchanged; only the starting point in weight space is optimized so that it lies in a region that is sensitive and plastic to any task in p(T).

**Domains demonstrated:** Sinusoid regression (K=5–20), Omniglot/MiniImagenet few-shot classification (1-shot, 5-shot), continuous-control RL (2D navigation, half-cheetah, ant) using policy gradient + TRPO as meta-optimizer.

---

## Why it matters for this project

Jeffrey's thesis is: **the prior is itself a learned network, recursively updated, and the compounding of that prior over time is the moat.** MAML is the clearest existence proof that a *single parameter vector* can encode a domain prior that survives cheap gradient adaptation — the "prior as a network" is not metaphorical in MAML; it is literally θ. Key connections:

1. **Prior = initialization geometry.** MAML operationalizes the Bayesian framing: θ is a prior over functions such that a small amount of new data (the likelihood) updates it efficiently to a posterior θ'. This directly instantiates the "prior as a network" hypothesis in a concrete, gradient-based way.

2. **Rate of improvement, not level.** The core MAML claim is not "θ achieves high accuracy" but "fθ achieves good accuracy faster than any non-meta-trained initialization." That is exactly Jeffrey's framing: the slope (improvement rate given new experience) is what the prior buys, not the level. MAML gives a direct mathematical characterization of what it means to optimize for slope: maximizing loss sensitivity with respect to θ across p(T).

3. **Cheap update.** MAML's inner loop is just SGD — no Bayesian inference, no memory replay, no expensive retraining. One gradient step is the update. This is the cheapest possible recursive update of a learned prior. It answers open problem (2) in the simplest possible way.

4. **Task distribution = domain abstraction.** The outer loop works over p(T), which corresponds to what Jeffrey calls the "abstraction store": the shared structure latent across domain tasks. MAML learns that structure into θ without explicitly representing it as discrete abstractions — the abstraction is distributed across θ's geometry.

5. **Wedge relevance.** For predictive test selection: tasks Ti could be "given a commit diff and repo state, predict which tests fail." K examples = a small labeled set of recent CI runs. The meta-prior θ would capture cross-repo structure (test coupling patterns, module boundaries, typical change footprints). Adaptation at task time = a few gradient steps on a repo's recent history before predicting.

---

## What to extract / reuse

- **The two-loop structure** (outer loop = prior update, inner loop = task adaptation) is the canonical template for the learning loop Jeffrey needs. Use it verbatim or as a scaffold.
- **The loss-sensitivity framing** (Section 2.2): training θ to maximize ∂L_Ti/∂θ across tasks. This is the cleanest mathematical statement of "prior that makes you learn fast."
- **First-order MAML (FOMAML):** drop second derivatives, nearly same performance, 33% cheaper. Use FOMAML as the default — it removes the only significant computational bottleneck.
- **RL formulation (Algorithm 3):** the meta-RL setup maps almost directly to a scenario where "tasks" are repos/codebases and "rollouts" are CI execution traces. The REINFORCE-style credit assignment over trajectories may be adaptable to the test-selection setting where the reward is prediction accuracy.
- **The oracle upper bound trick:** always evaluate against an oracle that receives task identity as input. Adapt this as a debugging baseline: give the model the repo name/known metadata and measure how much performance is left on the table vs. the learned prior.
- **Continual improvement property:** MAML keeps improving with more gradient steps at test time, not just one. This confirms that a single-step cheap update is a lower bound, not the ceiling.

---

## What to skip or ignore for our purposes

- **Image classification benchmarks** (Omniglot, MiniImagenet): implementation details, hyperparameter tables, convolutional architectures. The classification experiments are existence proofs, not design templates for this domain.
- **Second-order Hessian implementation details** in TensorFlow: use FOMAML — the second-order pass is not needed and the paper itself shows it barely matters (Table 1, MiniImagenet FOMAML vs. full MAML: 48.07% vs. 48.70% 1-shot).
- **Task distribution stationarity assumption:** MAML assumes p(T) is fixed throughout meta-training. Jeffrey's setting involves a non-stationary distribution (GitHub ecosystem evolves). The mechanics still apply but the stationarity assumption must be actively monitored.
- **RL-specific trust region meta-optimizer (TRPO):** not relevant unless the v1 product uses a reinforcement learning formulation. Stick to supervised meta-learning for test selection.
- **Sinusoid regression experiments:** pedagogically useful for grokking MAML but not architecturally informative for the code/test domain.

---

## Limitations & risks relevant to us

**1. Cheap online update — real but bounded.**
The inner loop is cheap (one gradient step), but the *outer* loop requires a full backprop-through-gradient (second-order unless using FOMAML). FOMAML avoids this, but even FOMAML requires accumulating gradients over a batch of tasks before updating θ. This means the outer loop is not truly online — it's a mini-batch meta-update. For cheap *recursive* update of θ as new CI runs arrive, you need either: (a) very frequent small outer-loop updates (online meta-learning, not addressed in this paper), or (b) a different mechanism (e.g., Reptile, which approximates MAML's outer loop without differentiation through the gradient). MAML as stated does not solve open problem (2) for the streaming/online case.

**2. Credit assignment — diffuse, not discrete.**
MAML assigns credit entirely through the outer gradient, which propagates through the inner update step. Which components of θ get reinforced when a task goes well? All of them, proportionally to their effect on post-update loss. There is no mechanism for identifying *which substructure of θ* was responsible for fast adaptation on a particular task. This is a direct gap relative to open problem (3): MAML provides task-level credit assignment (this task loss improved) but not abstraction-level credit assignment (this abstraction was why). If Jeffrey needs an inspectable, discrete abstraction store, MAML's distributed θ is a black box.

**3. Drift and stationarity.**
MAML assumes p(T) is stationary during meta-training. In a real GitHub ecosystem, task distributions shift (new languages, new frameworks, changing test patterns). Catastrophic forgetting in θ is a real risk if you continually update the outer loop on new tasks without replay. The paper does not address this. Combined with the Kirkpatrick et al. (EWC) paper in the notes folder (1612.00796v2.pdf), this is the natural companion problem.

**4. Task boundary requirement.**
MAML requires clear task boundaries: you must know when one task ends and another begins to reset the inner loop. In a streaming CI setting, task boundaries (repo boundaries, or time-window boundaries within a repo) must be defined explicitly. If tasks bleed into each other, the inner loop has no clean signal to adapt to.

**5. K-shot assumption and cold-start.**
MAML needs K labeled examples at test time to run the inner loop before inference. For a new repo with zero CI history, you get no inner-loop update — you fall back to the raw prior θ. This is actually acceptable (the prior should still be useful) but it means the "fast adaptation" benefit kicks in only after K runs, not immediately.

**6. No autonomous abstraction discovery.**
MAML learns a single flat parameter vector θ. It does not autonomously discover discrete, compositional, or hierarchical abstractions (open problem 1). The representation inside θ may be broadly transferable, but you cannot inspect or reuse individual "abstractions" — the geometry is implicit. For the "inspectable prior" thesis, MAML is necessary but not sufficient.

**7. Ablation gap.**
Open problem (5): does compounding come from accumulated abstractions, or just more data/bigger model? MAML does not provide this ablation. The sinusoid and MiniImagenet experiments show MAML > pretraining, but both use the same amount of data. A clean ablation would hold data fixed and vary the structure of the prior. This is an open empirical question for Jeffrey's system.

---

## One-line takeaway

MAML proves that a weight initialization is a learnable, gradient-updatable prior over task distributions — and that optimizing for loss sensitivity (slope, not level) across tasks is the right outer objective — but it solves the cheapness and compounding problems only at the task-level granularity, not at the level of discrete, inspectable abstractions.

---

*Read: 2026-06-22. Source: 1703.03400v3.pdf (9 pages + appendices).*
