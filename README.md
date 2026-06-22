# ipsum

> Expertise is not the accumulation of facts. It is the accumulation of useful, inspectable abstractions — updated cheaply as experience arrives.

**ipsum** is a research project testing one hypothesis: a system that maintains an explicit, cheaply-updated **prior** over "what matters in a domain" can *compound* — its rate of improvement keeps climbing where a stateless model (even a large one with retrieval) plateaus.

The bet is on the **learning loop**, not model size. The headline result we're chasing is not a higher accuracy number; it's a **steeper slope over time** than a strong frozen baseline.

## The idea in one diagram

```
experience ──▶ update a learned prior (cheap, recursive)
                     │
                     ▼
            inspectable abstraction store ──▶ better decisions
                     ▲                              │
                     └──── outcomes ◀───────────────┘
```

A frozen LLM + retrieval starts every task cold. ipsum doesn't: it carries a per-domain prior that gets revised by each outcome, and the abstractions it forms are explicit objects you can read, not weights you can't.

## v1 wedge: predictive test selection

We instantiate the idea on one task with a clean, objective signal: **given a code change, predict which tests can fail, and run only those.**

- **Signal is unambiguous** — did the selected subset catch the failure? yes/no.
- **ROI is concrete** — CI minutes saved.
- **No enterprise sales** — train and measure on public GitHub repos.
- **There is a real baseline to beat:** Facebook's Predictive Test Selection (Machalica et al., ICSE-SEIP 2019) retrains an XGBoost model *weekly, from scratch* — zero online update, zero compounding. That plateau is the bar.

**Win condition:** on held-out commits, ipsum's selection quality (TestRecall at fixed SelectionRate) shows a *widening* gap over the **data-matched, abstraction-off control**. Beating the weekly-retrain baseline is useful; beating the data-matched control is the thesis.

## Architecture (see [DESIGN.md](DESIGN.md))

Three reusable pieces from the literature, assembled — we spend no novelty budget here:

- **Amortized prior** (Conditional Neural Process style): encode each experience, aggregate into a latent in one forward pass, decode predictions with uncertainty. Conditioning on new experience is an O(1) update — the cheap recursive step.
- **Consolidation** (EWC style): a Fisher-weighted anchor so the prior updates without catastrophically forgetting — "posterior of yesterday becomes prior for today."
- **Inspectable abstraction store** (DreamCoder / Voyager style): explicit, named abstractions admitted only when they earn their keep.

…plus the three things the literature does **not** solve for noisy, delayed, real-world outcomes — which is where the actual research is.

## The three open problems (the real work)

1. **Admission under uncertainty** — DreamCoder admits an abstraction by exact description-length compression. With statistical outcomes this must become Bayesian model selection: keep an abstraction iff it improves held-out predictive likelihood by more than its complexity cost.
2. **Delayed credit assignment** — CI outcomes land minutes-to-hours later and are noisy. Which abstraction(s) get reinforced for a good outcome?
3. **Eviction / anti-staleness** — code drifts, so abstractions go stale. DreamCoder and Voyager only ever *add*. ipsum must decay and evict.

## Repo layout

```
ipsum/
├── README.md            # this file
├── DESIGN.md            # architecture + the three open mechanisms + experiment design
├── RESEARCH.md          # methodology + synthetic-testbed experiment cards
├── INTERFACE.md         # backend/frontend JSON artifact contract
├── AGENTS.md            # Codex/backend instructions
├── CLAUDE.md            # Claude/frontend instructions
├── src/ipsum/           # prior / abstractions / consolidation / credit (skeletons)
├── experiments/         # the compounding-vs-baseline harness
├── data/                # RTPTorrent project selection + profiling script
├── frontend/            # local dashboard scaffold consuming experiment artifacts
├── research/            # paper notes + synthesis + dataset scouting
├── RESEARCH_LOG.md      # append-only experiment log
└── tests/
```

## Status

Early. Research scaffold, lit review, dataset scouting, backend/frontend agent boundaries, the frontend dashboard scaffold, and the synthetic-testbed interface are in place. Core mechanisms are still intentionally stubbed. The next backend step is to implement `src/ipsum/synth.py`, then validate Card A (admission under uncertainty) on synthetic data before touching RTPTorrent.

## Reading

Start with `AGENTS.md` (backend) or `CLAUDE.md` (frontend), then read `DESIGN.md`, `RESEARCH.md`, and `INTERFACE.md`. `research/` has the synthesis, annotated notes on the nine foundational papers, and dataset scouting for RTPTorrent. Start the literature pass with `research/00-synthesis.md`.

## License

MIT
