# AlphaZero — Annotated Note
**Paper:** "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
Silver et al., DeepMind, 2017. arXiv:1712.01815v1

---

## Core idea

AlphaZero replaces decades of hand-engineered chess/shogi knowledge — evaluation functions, opening books, endgame tablebases, alpha-beta heuristics — with a single deep neural network that jointly outputs a move-probability vector (policy) and a scalar position evaluation (value), trained purely by self-play reinforcement learning. The only domain knowledge fed in is the game rules. Starting from random weights, it surpasses Stockfish (the TCEC world champion) in chess within 4 hours and defeats all three world-champion programs across chess, shogi, and Go within 24 hours. The entire prior over "what matters" is the network; the network is the expertise.

---

## Mechanism

**Policy/value network as learned prior.**
A single network `f_θ(s) → (p, v)` encodes all learned strategic knowledge: `p` is a probability distribution over legal moves (the prior), `v` is the expected game outcome from position `s`. This pair is the compressed, generalized representation of everything the system has learned about good play.

**MCTS as search / amplification.**
At inference, MCTS runs 800 simulations per move. Each simulation traverses the tree by selecting moves with high prior probability, high value, and low visit count (UCB-style balancing of exploitation and exploration). MCTS returns improved move probabilities `π` — a "distilled" search result that is sharper than the raw network prior alone. The network guides search; search improves the policy. This is the amplification loop within a single decision.

**Self-play loop (the outer compounding loop).**
Games are generated continuously by the current network parameters. At game end, the terminal outcome `z ∈ {-1, 0, +1}` provides the training signal. The loss is:

```
l = (z - v)^2  −  π^T log p  +  c||θ||^2
```

Mean-squared error trains the value head; cross-entropy between MCTS visit counts `π` and network policy `p` trains the policy head. The updated `θ` is immediately used to generate the next batch of self-play games. There is no separate "best player" gating (unlike AlphaGo Zero) — the network is updated continuously, tightening the prior with every batch.

**Single, clean reward signal.**
The only supervision is the game outcome `z`. No intermediate rewards, no human labels, no domain-specific annotations. The signal is sparse (one value per game of ~30-100 moves) but unambiguous and verifiable. The network must credit-assign backward through the entire game implicitly — yet it works because the signal is ground-truth, not noisy proxy.

**Key contrast with Stockfish / NNUE:**
Stockfish's NNUE (Efficiently Updatable Neural Network) achieves fast *inference* by exploiting incremental position differences — it updates the hidden-layer activations cheaply as pieces move. But NNUE weights are trained **offline on fixed data** and then frozen. "Efficiently updatable" in NNUE refers to the forward-pass computation, not to learning. AlphaZero's prior is what Jeffrey's project cares about: a network whose *weights themselves* are updated from experience, recursively. NNUE is efficient inference of a static prior; AlphaZero is a prior that improves.

---

## Why it matters for this project

**This is the canonical instantiation of the thesis.**
Jeffrey's whole idea traces back to chess engines and the question: "why can't the priors of ML training be a network itself?" AlphaZero is the cleanest answer — the network *is* the prior, updated from experience, with no frozen handcrafted knowledge. The compounding dynamic is explicit: better prior → better MCTS search → better self-play games → better training signal → better prior. This is the loop. The slope of Elo improvement in Figure 1 (surpassing Stockfish in 4 hours, continuing to improve beyond) is the empirical evidence that the loop compounds.

**What is directly transferable:**
- The architectural pattern: a single model that jointly encodes "what to do" (policy/prior) and "how good is this" (value), trained end-to-end from outcome signals.
- The principle that cheap search at inference time (MCTS) can amplify a learned prior well beyond the prior's raw quality — i.e., you don't need a perfect prior, just one good enough to guide search.
- The continuous-update discipline: don't gate on "best player," just keep updating. Faster iteration = faster compounding.
- The loss formulation: joint policy + value training on a single outcome signal. In the software domain, the analog is: policy = "which tests to select" or "which code change matters," value = "is this a risky commit," outcome = "did CI fail."

**What does NOT transfer cleanly (and why):**
- **No perfect simulator.** AlphaZero's self-play loop requires a perfect, deterministic, zero-latency oracle that can answer "what happens if I make this move?" Software engineering has no equivalent. You cannot self-play commits. Real feedback (CI runs, code review, production failures) is slow (minutes to hours), noisy, and sometimes absent. This breaks the tight self-play loop entirely.
- **No unambiguous terminal signal.** Chess ends with +1/0/-1. A test failure is noisier: flaky tests, environment issues, incomplete coverage. The signal exists but is not ground-truth in the same sense.
- **Compute requirements are immense.** AlphaZero used 5,000 first-gen TPUs for self-play data generation and 64 second-gen TPUs for training. The compounding result required this scale. A solo-developer wedge needs orders-of-magnitude cheaper feedback loops.
- **State representation is discrete and bounded.** Board positions are finite combinatorial objects. Code commits + test suites are high-dimensional, variable-length, partially observed, and semantically ambiguous. The neural architecture for policy/value in AlphaZero exploits the grid structure of a board (convolutional planes). No equivalent inductive bias exists for code.

---

## What to extract / reuse

1. **The loop structure as a design template:** (learned prior) → (cheap search/exploration) → (outcome signal) → (prior update) → repeat. Map every component to the software domain explicitly before building.

2. **Joint policy + value head on a shared representation.** For predictive test selection: policy = probability distribution over tests, value = predicted probability that *any* test in the suite fails given the commit diff. Train both heads jointly; the value head gives you a confidence signal for when to expand test coverage vs. trust the prior.

3. **MCTS-style amplification idea — adapted.** In the absence of a game simulator, a weaker analog is: use the policy prior to *rank* tests, then use a fast static analysis or embedding-based filter as the "simulation" step, then re-rank. This is not true MCTS but preserves the "cheap amplification of a learned prior" structure.

4. **Continuous update (no gating).** AlphaZero's decision to drop the "best player" gate and update continuously is directly applicable: update the model incrementally as each CI result arrives, don't batch until you have "enough" data. Momentum in the prior compounds faster this way.

5. **The Elo curve as a proof-of-compounding metric.** For the v1 wedge, define an analogous metric — F1 on test selection vs. ground-truth failures, plotted over calendar time — and track its slope. The slope is the moat proof.

6. **Credit assignment intuition.** AlphaZero assigns credit to moves by training the value network to predict game outcomes, then using MCTS visit counts to re-weight the policy. In software: train a value model on commit-level CI outcomes, use it to back-weight which test selections were "responsible" for catching vs. missing failures.

---

## What to skip or ignore for our purposes

- The specific neural architecture details (convolutional planes, 8-step history encoding, 73-plane action representation for chess). These are chess-specific and have no software analog.
- The MCTS hyperparameters (Dirichlet noise scaling, 800 simulations, UCB constants). The specific values matter only for game domains.
- The comparison with alpha-beta search scalability (Figure 2). Relevant to understanding search theory in general but not transferable.
- Symmetry augmentation (rotation/reflection for Go). Board-specific.
- The example games and opening book analysis (Table 2). Domain flavor, not structural insight.
- The distinction between AlphaZero and AlphaGo Zero on binary vs. expected outcome and symmetry handling. Minor variant, not structurally important for us.

---

## Limitations and risks relevant to this project

**No simulator = no inner loop.**
The most dangerous gap. AlphaZero's power comes from generating millions of self-play games cheaply. In software-eng, a single CI run can take 10–60 minutes. At that rate, you cannot bootstrap a prior from scratch through self-play; you must start from a pre-trained model (e.g., a code LLM) and fine-tune incrementally on arriving CI data. This changes the problem from "learn from scratch" to "adapt a warm prior cheaply" — which is actually closer to MAML/meta-learning territory.

**Sparse and noisy outcome signal.**
Chess gives one clean signal per game. Real CI gives: flaky tests, environment-dependent failures, coverage gaps (a test doesn't fail but the bug is still there). The value head must be trained to handle label noise and delayed/missing feedback. Without explicit handling, the learned prior will drift toward "predict the environment, not the code."

**Credit assignment is harder.**
In chess, the MCTS visit counts provide a principled credit signal (which move probability deserved updating). In test selection, it's unclear which tests "caused" a pass or fail. A commit that adds 3 files might have the defect in 1; all selected tests that ran and passed still get the positive label. This risks rewarding lazy selectors.

**Drift / staleness.**
AlphaZero plays against itself — the distribution is always current. In software, the codebase, test suite, and failure modes change structurally over time (major refactors, new modules, deleted tests). A prior trained on month-1 data may be actively misleading in month-6. Need explicit mechanisms for detecting and handling distribution shift (EWC or replay buffers are the standard tools; see Kirkpatrick et al. in the reading list).

**Compute cost of rigorous ablation.**
AlphaZero's proof of compounding required 5,000 TPUs and 44 million self-play games. Demonstrating a comparable compounding result in software on a solo-developer budget requires a carefully scoped wedge (public GitHub repos, small test suites, fast CI) and a much simpler model. The risk is that the compounding effect only emerges at scale AlphaZero used, and is unobservable in a v1 experiment.

---

## One-line takeaway

AlphaZero proves that a jointly trained policy/value network, updated continuously from self-play outcomes with no handcrafted knowledge, is strictly superior to frozen expert-designed systems — but it requires a perfect simulator and clean terminal rewards that software engineering domains currently lack, making the key open problem for Jeffrey's project how to approximate the inner self-play loop with slow, noisy real-world CI signals.
