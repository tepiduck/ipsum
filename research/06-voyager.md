# 06 — Voyager: An Open-Ended Embodied Agent with Large Language Models (Wang et al. 2023)
> arXiv:2305.16291v2

---

## Core idea

Voyager is a lifelong learning agent that plays Minecraft open-endedly by continuously proposing tasks, writing executable JavaScript skills to complete them, and storing those skills in a vector-indexed library for later reuse. There are no gradient updates: all learning is in the skill library and in the in-context prompt. The key bet is that if you store *verified, reusable programs* rather than raw episodes, capabilities compound — each new task can call skills already in the library, so the effective action vocabulary grows monotonically. The agent never forgets because forgetting is not a property of a key-value store.

---

## Mechanism

**1. Automatic curriculum.**
GPT-4 proposes the next task by looking at (a) the agent's current Minecraft state (inventory, biome, time, health, nearby blocks/entities), (b) a list of completed and failed tasks, and (c) self-generated Q&A context produced by GPT-3.5 querying a Minecraft wiki. The directive is explicit: "discover as many diverse things as possible; the next task should not be too hard." A warm-up schedule gradually exposes more state information as the agent matures. Ablation shows that a random curriculum drops item discovery by 93%; the automatic curriculum is the steering wheel.

**2. Skill library of executable code.**
Each successfully completed task is stored as a named JavaScript async function (e.g., `craftStoneShovel()`, `combatZombieWithSword()`). The key in the vector database is the embedding (GPT-3.5 `text-embedding-ada-002`) of a 6-sentence natural-language description of the function; the value is the raw code. At inference, the top-5 most relevant skills are retrieved via embedding similarity and included in the code-generation prompt. Skills are explicitly designed to be generic and reusable: the prompt instructs GPT-4 to "make it generic and reusable" and to check for required items rather than assuming inventory state. Complex skills call simpler ones, creating a compositional hierarchy.

**3. Iterative prompting with self-verification and environment feedback.**
Code generation is iterative, not one-shot. Each round: (a) execute the generated program in MineDojo/Mineflayer; (b) collect environment feedback (intermediate `bot.chat()` outputs, e.g., "I cannot make an iron chestplate because I need: 7 more iron ingots") and execution errors (interpreter exceptions); (c) feed both back to GPT-4 with chain-of-thought prompting for the next refinement round. After at most 4 rounds, a separate GPT-4 critic agent performs *self-verification*: given the agent's final state and the task specification, it returns `{success: bool, critique: str}`. Success commits the skill to the library; failure queues the task for retry or deferral. Self-verification is the single most important component (ablation: removing it cuts item discovery by 73%).

---

## Why it matters for this project

Voyager is the direct instantiation of the abstraction-compounding loop that this project aims to build. The **skill library is a concrete, inspectable prior over "what matters in the domain"** — it is not weights, it is not a retrieval corpus of raw episodes, it is a set of *verified, named abstractions* that can be called by future abstractions. This is exactly the structure Jeffrey's thesis needs to make concrete for software engineering (test selection, code understanding, etc.).

Critically, this is achievable for a solo developer **without training anything**: Voyager runs entirely on GPT-4 API calls and a vector database. No fine-tuning, no custom model. The engineering surface is: (1) define a structured state representation for your domain; (2) implement an LLM-driven curriculum (which tasks to try next); (3) implement execution + feedback collection; (4) implement a self-verification step (success/fail oracle); (5) maintain an embedding-indexed skill store. Steps 1–5 are all tractable solo-dev work.

The compounding effect is real in Voyager's ablation: **VOYAGER w/o Skill Library plateaus** while VOYAGER keeps climbing. This is the experimental grounding for the slope-over-time thesis. The moat is not a smarter model; it is the library that accumulates over time.

For the **predictive test selection wedge**: skills = learned test-selection heuristics (e.g., "when file X changes, always run tests Y, Z; skip W"), stored as inspectable rules, updated with each CI run's outcome, reused across repos. The Voyager loop maps onto: curriculum = "pick the next PR/commit to process"; skill = "test selection policy for file-pattern X"; verification = "did the CI run actually catch the regressions we predicted it would?" The library grows each CI run, not each training epoch.

---

## What to extract / reuse

- **Skill library architecture**: embedding-indexed key-value store; description as key, executable artifact as value; top-k retrieval at inference. Directly portable to a test selection setting where the "skill" is a selection predicate or policy.
- **Self-verification pattern**: a separate LLM call that acts as a critic, returning `{success, critique}` in structured JSON. Cheap, accurate, and tells you *when to commit* a new abstraction vs. when to keep refining. For test selection: success = "model's prediction matched actual CI outcome on held-out run."
- **Iterative prompting with execution feedback**: the REPL loop — generate, execute, observe, refine, commit. Works for any domain where you have an execution environment and observable outcomes. In the test selection domain, execution = running the test suite; observation = pass/fail vs. prediction.
- **Warm-up schedule / curriculum design**: start with easy/small tasks and escalate complexity. In test selection: start with small PRs on well-characterized modules before tackling cross-cutting refactors.
- **Compositional skill generation**: newer skills explicitly call older ones. In test selection: higher-level policies could call lower-level file-pattern selectors.
- **Ablation methodology**: the paper ablates each component independently (curriculum, skill library, each feedback type, model choice). Replicate this structure rigorously for any v1 build.

---

## What to skip or ignore for our purposes

- **Minecraft-specific engineering**: Mineflayer API wrappers, MineDojo setup, 3D spatial navigation, biome/entity perception. Entirely domain-specific.
- **The embodiment layer**: low-level motor control, pathfinding, bot resurrection logic. Not relevant to a software-domain agent.
- **Multimodal feedback / human-as-curriculum demo (Sec. 3.5)**: interesting but peripheral to the core loop; a future concern for a solo v1 build.
- **GPT-3.5 for auxiliary tasks (Q&A, wiki retrieval)**: cost-optimization trick for Minecraft. In a software domain the LLM context is already rich; this layer may be unnecessary.
- **The Minecraft-specific warm-up table (Table A.1)**: state-space warm-up details that are environment-specific.

---

## Limitations & risks relevant to us

**Does Voyager actually show compounding vs. just accumulation?**
Partially. The ablation (Fig. 9, VOYAGER w/o Skill Library plateaus) shows that the library is necessary for sustained growth. But the paper does not directly measure *rate of improvement per new skill added* vs. *total skills accumulated*. The curves in Fig. 1 show monotonic growth in item count — but it is not possible from the paper alone to distinguish "skills compound" from "skills accumulate without super-linear returns." For this project, we need an explicit metric: does accuracy on test-N improve when skill-N+k is added to the library?

**Credit assignment is absent.**
Voyager uses top-k embedding similarity for retrieval — there is no mechanism for tracking which skills were actually used in a successful episode, no reinforcement of "this skill was key," no decay for skills that are never retrieved. In the test selection domain, credit assignment matters: which abstraction correctly predicted a test failure? Without this, the library grows but does not *improve* its priors; it just accumulates. This is open problem (3) in the project.

**Drift and staleness are unaddressed.**
Skills are committed once and never revised. If the game updated its API (or, in our domain, if a codebase refactors), stale skills sit in the library indefinitely. Voyager gets away with this because Minecraft APIs are stable and the evaluation horizon is short (~160 prompting iterations). For long-lived software repos, drift is a first-class concern — open problem (4).

**No cheap recursive update of the prior.**
Adding a skill is O(1) (one embedding call), but there is no mechanism for existing skills to be *revised* based on downstream failures. If `smeltFiveRawIron()` changes to `smeltFiveRawIronV2()`, the old version lingers unless explicitly pruned. The project thesis requires a *recursive* update — new evidence should propagate back to update the learned prior, not just append to it.

**GPT-4 dependency is a cost risk.**
Ablation shows GPT-3.5 produces 5.7x fewer unique items. For a solo dev on a budget, this is a real constraint. Test selection is a structurally simpler code-generation problem than open-ended Minecraft, so GPT-3.5 / smaller models may be sufficient — but this needs empirical verification.

**Skill retrieval accuracy is not 100%.**
Top-5 retrieval accuracy is 96.5% (Table A.4), top-1 is 80.2%. In test selection, a wrong skill retrieved means a wrong selection heuristic is applied. The impact depends on whether wrong retrieval is conservative (include too many tests) or aggressive (miss a regression). Needs careful evaluation.

---

## One-line takeaway

Voyager proves that an LLM agent with a growing library of verified executable abstractions — no weight updates, just a vector DB and iterative prompting — can outperform all frozen baselines at the rate of new capability acquisition, but it leaves credit assignment, staleness, and recursive prior updates as open engineering problems that are central to this project's differentiated thesis.
