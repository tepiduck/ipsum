# Design

This is the original architecture and research plan for ipsum. It is kept because
the pieces are still useful, even though v1 ended as a negative result on predictive
test selection.

For the final outcome, read [RESULTS.md](RESULTS.md). This file explains what we
were trying to build and why.

## The basic idea

Most predictors start each example from the same learned weights, plus whatever
context is retrieved at inference time. ipsum tried a different shape:

```text
experience -> update prior and abstraction store -> make better decisions next time
```

The bet was not "make the model bigger." The bet was that a system could carry a
small amount of domain-specific state forward:

- a learned prior,
- a set of explicit abstractions,
- and a cheap update rule after each outcome.

The measurement target was slope over time. If the system really compounds, its
quality should improve faster than a control that sees the same data but has no
abstraction store.

## Borrowed pieces

Several parts of the design were intentionally borrowed. They were not where the
novelty was supposed to be.

| Piece | Source of the idea | Why it was useful |
|---|---|---|
| Prior as network parameters | MAML | adaptation starts from a learned prior |
| Bayesian interpretation | Grant et al. 2018 | the initialization can be read as an empirical-Bayes prior |
| Fast conditioning | Conditional Neural Processes | new experience can update a latent summary cheaply |
| Consolidation | EWC | old knowledge can be protected during updates |
| Explicit abstractions | Voyager | reusable objects can be inspected and edited |
| Admission by usefulness | DreamCoder | abstractions should earn their cost |
| Compounding loop | AlphaZero | the proof is a learning curve, not a static score |

The intent was to assemble these pieces conservatively and spend the research effort
on the parts that are not already solved.

## The open mechanisms

### 1. Admission under uncertainty

DreamCoder admits abstractions when they compress solved tasks. That is clean when
outcomes are exact. CI outcomes are noisy and statistical, so ipsum used held-out
predictive log-likelihood instead:

```text
admit abstraction a if DeltaLL(a) - complexity(a) is positive enough
```

The v1 implementation used a simple held-out LL gate. The design notes also sketch
stronger versions: likelihood-ratio e-values, e-LOND for dependent candidate tests,
and a coverage guard for positivity failures.

### 2. Delayed and noisy credit

CI outcomes arrive later, and tests can be flaky. A useful system needs to know which
decision or abstraction should be credited when an outcome finally lands.

The planned version had three parts:

- an eligibility buffer for in-flight decisions,
- de-flaking before labels count,
- and outcome-weighted reinforcement of the abstractions that actually informed the
  decision.

v1 did not reach the full delayed-credit card.

### 3. Eviction and staleness

DreamCoder and Voyager mostly add abstractions. Codebases drift. An abstraction that
was useful last month can become noise later.

The store therefore tracks recent usefulness, decays it, and evicts abstractions when
their usefulness falls below their complexity cost. On synthetic drift this helped in
some settings, but not strongly enough to count as a finished mechanism.

## Reference architecture

```text
                 experience
                     |
                     v
        encoder(change, test, context)
                     |
                     v
              latent prior z_t
              /       |       \
             /        |        \
            v         v         v
 abstraction store   decoder   credit assigner
  admit / evict     P(fail)      delayed outcomes
            \         |         /
             \        v        /
              outcomes and updates
```

In v1, the practical center of gravity became the abstraction store and experiment
harness. The neural prior stack stayed more as design scaffolding than as the main
result.

## v1 task: predictive test selection

The first task was:

> Given a code change, rank tests by failure probability and run only the top subset.

The metric was TestRecall at a fixed SelectionRate. The control that mattered was a
data-matched system with the abstraction store turned off.

The hoped-for result was a widening gap:

```text
TestRecall(ipsum) - TestRecall(data-matched control)
```

That did not happen on the valid real-data run. See [RESULTS.md](RESULTS.md).

## Confounders the design tried to control

- **More data, not better abstractions.** This is why the data-matched,
  abstraction-off control is primary.
- **Bigger model, not better memory.** The predictor was kept small so any gain had to
  come from the abstraction store.
- **Flaky labels.** Outcomes were de-flaked before they counted.
- **Sparse changed-file coverage.** Real-data runs report the admission funnel so a
  zero-admit result can be diagnosed.

## Milestones

The intended order was:

1. build the synthetic testbed,
2. validate admission,
3. validate the measurement instrument with positive and negative controls,
4. test eviction under drift,
5. test the coverage guard,
6. move to RTPTorrent,
7. only then judge compounding on real data.

That order turned out to be the right call. It did not save the thesis on this task,
but it did keep the negative result from being a pile of ambiguous failures.
