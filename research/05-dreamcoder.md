# 05 — DreamCoder: Growing Generalizable, Interpretable Knowledge with Wake-Sleep Bayesian Program Learning (Ellis et al. 2020)
> arXiv:2006.08381v1

---

## Core idea

DreamCoder is a program synthesis system that compounds expertise across tasks by jointly growing a symbolic library of reusable abstractions and training a neural recognition model to search that library efficiently. A "wake-sleep" cycle alternates between (a) solving tasks using the current library + model ("waking"), (b) compressing the solutions into new library primitives by MDL/Bayesian criterion ("abstraction sleep"), and (c) retraining the model on both remembered and imagined problems ("dreaming sleep"). Each iteration the library grows, programs get shorter, search gets faster, and harder tasks become tractable — genuine compounding from explicit symbolic structure, not parameter interpolation.

---

## Mechanism

### Wake phase

Given a mini-batch of tasks X, for each task x the system searches for the program rho that maximizes the posterior:

    rho_x = arg max P[rho|x, L]  proportional to  P[x|rho] * P[rho|L],
             subject to Q(rho|x) being large

Search is **neurally-guided enumeration**: programs are enumerated in decreasing order of probability under the recognition model Q(rho|x) and tested for correctness (binary P[x|rho] = 0 or 1) until a per-task timeout. A beam of k=5 programs with highest posterior probability is retained per task and used in subsequent sleep phases. The difficulty of enumeration is roughly breadth^depth (branching factor × program depth): the library shrinks depth; the neural model shrinks effective breadth.

### Sleep phase 1: Abstraction (library learning)

After waking, the library L is updated to compress the set of found programs. The exact optimization objective is:

    L* = arg max  P[L] * prod_{x in X}  max_{rho: refactoring of rho_x}  P[x|rho] * P[rho|L]
              L

where P[L] is a description-length prior (shorter library = higher prior). This is MDL: add a primitive only if it shortens the combined description of library + refactored programs by more than the cost of describing the new primitive itself. The system is not just extracting repeated syntactic subtrees — it allows **semantic refactoring** (rewriting programs into equivalent forms via up to 3 lambda-calculus evaluation steps) before finding common structure. This exposes reused semantic patterns that share no surface syntax.

**Algorithm:** The set of all refactorings is represented compactly using a novel data structure combining version space algebras and equivalence graphs, built via dynamic programming. ~10^6 nodes encode ~10^14 distinct refactorings. Candidate abstractions are extracted as shared subtrees, then greedily added to L in order of marginal MDL gain until no further improvement is possible.

### Sleep phase 2: Dreaming (recognition model retraining)

Q(rho|x) is retrained on:

- **Replays:** (task, program) pairs from actual waking solutions — prevents forgetting.
- **Fantasies:** Programs rho sampled from the current library L, then executed to produce a task x. The system then runs its own wake search on each fantasy task and trains Q to predict the MAP solution (not the sampled program). This teaches canonical symmetry-breaking: Q learns to prefer the simplest/shortest equivalent program.

Training uses a 50/50 mix. The MAP objective — E[log Q(best-rho | x)] — is used rather than full posterior training, which would require summing over exponentially many syntactically equivalent variants.

### The library / DSL

Starts from a small domain-appropriate set of primitives (map/fold/cons for lists; pen-control for LOGO; sequence primitives for physics). Representation: polymorphically typed lambda-calculus, including higher-order functions, conditionals, and recursive definitions. After learning, the library typically contains ~20 new named routines that directly correspond to interpretable concepts (filter, sort, nth-largest for list processing; stroke and arc macros for LOGO; vector cross-product for physics). Library depth tracks strongly with task accuracy (r = 0.79 across domains).

### The recognition model

Domain-dependent neural network (CNNs for image tasks, unspecified architecture for others). Inputs: task observation x (I/O pairs, images, etc.). Output: a distribution over programs Q(rho|x), used to rank candidates in enumeration order. Architecture injects domain inductive biases but is otherwise a standard discriminative network trained with gradient descent. The model is re-initialized and retrained from scratch each sleep cycle on the full set of replays + fantasies accumulated so far — there is no incremental / online update of Q; full retraining is the mechanism.

---

## Why it matters for this project

DreamCoder is the **canonical existence proof for symbolic compounding**: it demonstrates empirically and formally that a system can recursively discover abstractions that make future learning cheaper. Every open problem Jeffrey faces has a DreamCoder counterpart to examine:

1. **Abstraction formation (open problem 1):** DreamCoder's compression objective is the cleanest known criterion for deciding whether a candidate abstraction is worth keeping: does including it in the library reduce total description length? For Jeffrey's system, the analog is: does adding an abstraction to the "expertise prior" reduce the cost of predicting which tests break, across a corpus of CI runs? The MDL criterion is directly portable in principle.

2. **Cheap recursive update (open problem 2):** DreamCoder's library update is polynomial in program size (due to the version-space/e-graph data structure), not exponential in the number of tasks. The key enabling insight: represent the full set of refactorings compactly rather than enumerating them. Jeffrey's system needs an analogous cheap update that doesn't require reprocessing all historical CI data from scratch each cycle.

3. **Credit assignment (open problem 3):** DreamCoder sidesteps this entirely via exact execution: P[x|rho] is 0 or 1 with no ambiguity. This is the sharpest difference from Jeffrey's domain (see critical assessment below).

4. **Compounding = library depth × model accuracy:** The r=0.79 correlation between library depth and task accuracy, combined with ablations showing that removing either sleep phase degrades performance, gives a quantitative template for how to measure compounding in Jeffrey's system. The equivalent metric: do later repositories benefit faster from the accumulated abstraction library than early ones? Does the slope of improvement increase over time?

5. **Inspectability / interpretability:** DreamCoder's abstractions are named lambda-calculus functions — literally human-readable code. This is exactly the "explicit, symbolic, inspectable abstractions" moat Jeffrey is betting on. The LOGO experiments show the system rediscovering arc and stroke macros that match human turtle-graphics vocabulary. Jeffrey's system should produce analogous inspectable artifacts (e.g., "tests tagged fast+network+auth break when any file under src/auth/ changes"), not opaque embeddings.

---

## How well does the framing "DreamCoder but for noisy real-world outcomes" hold up?

**Critical assessment: the framing is directionally correct but architecturally deep — the differences are not cosmetic and require new mechanisms, not adaptations.**

DreamCoder's entire architecture rests on **three hard assumptions** that are violated in Jeffrey's domain:

### 1. Binary, immediate correctness signal

P[x|rho] = 0 or 1, testable in milliseconds by running code. This makes wake search decidable (stop when a solution is found), makes the beam of k solutions meaningful (they are all correct), makes semantic refactoring tractable (two programs are equivalent iff their outputs match on all inputs), and makes Q's MAP training objective well-posed (there is a canonical best program to converge on).

In Jeffrey's setting — predicting which tests fail on a given commit — the "correctness" of a prediction is revealed only after CI runs complete (minutes to hours, not milliseconds). Worse, the "correctness" of an abstraction like "auth-related tests break when auth.py changes" is statistical, not binary: true 80% of the time is useful but not certain. This transforms:

- Wake search: from enumeration-until-correct to a ranking/retrieval problem with no crisp stop signal.
- Compression objective: from MDL over exact-solution programs to an information-gain criterion over noisy statistical patterns — fundamentally different mathematics (expected MDL, Bayesian model selection with uncertain likelihoods, or empirical risk minimization).
- Semantic refactoring: undefined without deterministic execution — two prediction rules that disagree on 20% of cases cannot be declared equivalent.

### 2. Delay-free credit assignment

DreamCoder tests a program the moment it proposes it. There is no temporal gap between action (propose program) and outcome (does it pass?). Jeffrey's system faces delayed feedback: the abstraction "predicting test X fails" is credited or blamed only when CI completes, and by then the system may have formed many subsequent abstractions. Standard credit assignment techniques (eligibility traces, advantage estimation) are needed but have no DreamCoder counterpart.

### 3. Drift and distribution shift

DreamCoder assumes a stationary task distribution (the 109 list tasks are fixed throughout all 20 iterations). In production CI, code structure, test suites, and failure patterns evolve — an abstraction learned six months ago may be anti-informative today. DreamCoder has no staleness mechanism; the library grows monotonically and is never pruned (except by MDL cost at addition time). Jeffrey needs an explicit decay or eviction policy, which DreamCoder's theory does not address.

### What actually maps over

The framing works for the **high-level architecture** (wake: apply current abstractions to new data; sleep: compress patterns into library; repeat) and for the **motivation** (compounding expertise by reusing discovered structure). It fails at the **mechanism level**: the specific algorithms — version-space enumeration, semantic refactoring via execution equivalence, MAP recognition model training — assume noiseless, delay-free, deterministic outcomes throughout.

**The right adaptation:** Replace DreamCoder's MDL-over-exact-programs with a Bayesian model selection criterion over statistical prediction rules. Replace wake enumeration with probabilistic inference or retrieval. Replace semantic refactoring with a clustering or association rule mining step that operates on (commit, test-outcome) tuples. Replace the binary likelihood with a calibrated probabilistic likelihood (e.g., logistic regression over abstraction features). The wake-sleep structure is preserved; the internals must be re-derived.

---

## What to extract / reuse

1. **Wake-sleep loop structure**: alternate between (a) applying current abstractions to new experience and (b) compressing new patterns into the library. This is the fundamental template.

2. **MDL / Bayesian criterion for abstraction admission**: an abstraction earns its place only if it compresses description length more than it costs. In the noisy setting, adapt this to: an abstraction earns its place only if it reduces expected prediction error on held-out CI runs more than its complexity penalty. This prevents the library from bloating with spurious patterns.

3. **Replay + fantasy dreaming**: retraining the recognition model on both remembered successes and self-generated synthetic tasks is the mechanism for avoiding catastrophic forgetting and for generalizing beyond seen cases. Jeffrey should maintain a corpus of (commit-state, test-outcome) replays and, when possible, synthesize hypothetical commits from his abstraction library to train the predictor.

4. **Compounding metrics**: r=0.79 correlation between library depth and task accuracy, plus ablations removing each sleep phase. Jeffrey should instrument his system to measure: (a) does prediction accuracy improve monotonically over CI iterations? (b) does removing the abstraction-learning phase flatten the slope? (c) does a frozen LLM retrieval baseline show a flat or much shallower slope?

5. **Inspectability as a design constraint**: DreamCoder's abstractions are human-readable lambda expressions. Jeffrey should enforce an analogous constraint: every abstraction in the library must be expressible as a human-readable predicate or rule (e.g., "if file X changed AND tag Y then test Z likely fails"). This is both the moat and the correctness check.

---

## What to skip or ignore for our purposes

1. **The specific version-space / equivalence-graph data structure**: this is engineered for polynomial-time enumeration of *syntactically* equivalent program refactorings. It has no direct analog when abstractions are statistical patterns over noisy CI data.

2. **The fantasy dreaming mechanism in its DreamCoder form**: generating tasks by executing random programs and using exact outputs as training signal requires deterministic execution. Skip this; replace with sampling hypothetical commits from the abstraction library and simulating outcomes via a learned model.

3. **The typed lambda-calculus program representation**: specific to the program synthesis domain. Jeffrey's abstractions should be domain-specific predicates over commit features (changed files, test metadata, historical co-failure rates), not lambda-expressions.

4. **The neurally-guided enumeration search**: designed for program synthesis over a typed combinatorial space. Jeffrey's analog is retrieval / ranking over a library of statistical rules, which has different algorithms (vector retrieval, rule matching, BM25-style scoring).

5. **The 20-iteration convergence story**: DreamCoder converges in a small number of iterations because tasks are fixed and the library grows monotonically. Jeffrey's system must run continuously against a non-stationary stream; convergence is not the right frame.

---

## Limitations & risks relevant to us

### Credit assignment

DreamCoder's abstraction phase uses only programs that exactly solve their tasks (P[x|rho_x] = 1). This is trivial credit assignment — if the program ran and produced the right answer, it gets full credit. In Jeffrey's setting, a prediction rule gets partial, delayed, noisy credit. Two risks:

- **False positives in the library**: an abstraction that appears to improve accuracy during training (because of selection bias in the CI run sample) may be spurious. DreamCoder avoids this because only programs that demonstrably work are compressed. Jeffrey must use held-out validation sets and explicit statistical significance tests before admitting a new abstraction.
- **Stale abstractions**: DreamCoder never removes abstractions after they are admitted (MDL guarantees they compressed the training set at the time). If Jeffrey's codebase drifts, a valid historical abstraction becomes misleading. A decay function (weight an abstraction's compression benefit by recency) and explicit eviction when accuracy falls below threshold are needed. DreamCoder offers no template for this.

### Which abstractions to keep — the compression criterion under noise

DreamCoder's MDL criterion is exact: add the abstraction iff sum_x min_{refactoring} -log P[rho|L] decreases by more than the description length of the abstraction itself. In the noisy setting, the equivalent is: add the abstraction iff it improves expected log-likelihood on held-out CI outcomes, regularized by complexity. This is straightforward Bayesian model selection, but with finite samples and noise the estimate is uncertain. Small libraries on few repos will produce high-variance decisions. Jeffrey needs explicit confidence intervals (e.g., bootstrap or Bayesian credible intervals) before promoting a pattern to a named abstraction. DreamCoder, working with deterministic programs and thousands of tasks, never faces this variance problem.

### Scalability of the recognition model retraining

DreamCoder retrains Q from scratch every sleep cycle on the accumulated replay corpus. With 20 iterations and 109 tasks, this is tractable. For a production CI system with millions of CI runs accumulated over months, retraining from scratch each cycle is prohibitively expensive. Jeffrey needs an incremental / online update scheme for his predictor, which DreamCoder's full-batch-retrain approach does not address.

### Rigorous ablation proving compounding

DreamCoder's key ablation is: remove the abstraction sleep phase → library stays flat → accuracy stays flat. The r=0.79 correlation across domains is compelling. Jeffrey needs an equivalent experiment: run his system on a fresh set of repositories (held out during training), and compare the slope of prediction accuracy improvement over CI iterations against a frozen LLM+retrieval baseline that does not update its abstractions. If the slopes converge, there is no moat. This experiment must be designed upfront; DreamCoder's ablation protocol is the template.

---

## One-line takeaway

DreamCoder proves that symbolic, inspectable abstraction libraries can compound learning via MDL-driven compression in a wake-sleep loop, but its entire mechanism assumes noiseless, delay-free, binary correctness signals — Jeffrey's core contribution is re-deriving this loop for the statistical, delayed, noisy setting of real CI outcomes, which requires replacing exact-execution semantics with Bayesian model selection and adding staleness/eviction mechanisms DreamCoder never needed.
