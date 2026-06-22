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

**Win condition:** on held-out commits, ipsum's selection quality (TestRecall at fixed SelectionRate) shows a *widening* gap over the weekly-retrain baseline from month 3 to month 6. The slope is the whole point.

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
├── src/ipsum/           # prior / abstractions / consolidation / credit (skeletons)
├── experiments/         # the compounding-vs-baseline harness
├── research/            # annotated notes on the 9 foundational papers
└── tests/
```

## Status

Early. Research scaffold + lit review complete; core mechanisms are stubs with defined interfaces. Roadmap is in DESIGN.md.

## Reading

`research/` has annotated notes on the nine papers this is built on — MAML, Grant et al. (hierarchical Bayes), Neural Processes, HyperNetworks, DreamCoder, Voyager, EWC, AlphaZero, and Predictive Test Selection. Start with `research/00-synthesis.md`.

## License

MIT
