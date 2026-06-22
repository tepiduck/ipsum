# AGENTS.md — backend agent guide (Codex)

You own the **backend**: the research mechanisms, the synthetic testbed, the data
pipeline, the baselines, and the experiment harness. You produce JSON run
artifacts; the frontend (a separate agent, `frontend/`, governed by `CLAUDE.md`)
reads them. **You never touch `frontend/`.**

## What this project is
ipsum tests one claim: a system that keeps an explicit, cheaply-updated,
**inspectable** prior over "what matters in a domain" can **compound** — its *rate*
of improvement (slope over time) beats a strong stateless baseline. The bet is the
learning loop, not model size. Read before coding:
`README.md` → `DESIGN.md` → `research/00-synthesis.md` → `RESEARCH.md`.

## Your scope
```
src/ipsum/
  synth.py         synthetic testbed w/ ground-truth oracles — BUILD THIS FIRST
  prior.py         amortized prior (CNP-style) — BORROWED, assemble don't innovate
  consolidation.py EWC anchor — BORROWED
  abstractions.py  admit / reinforce / evict — OPEN (RESEARCH.md cards A, B)
  credit.py        delayed/noisy credit assignment — OPEN (card C)
experiments/       compounding harness + weekly-retrain baseline; WRITES run artifacts
data/              RTPTorrent selection + profiler (dataset gitignored)
research/          paper notes; treat as background, not code
```
You do **not** own and must **not** edit: `frontend/`, `_private/`.

## The mental model you must hold
A research mechanism is **a claim to test, not a feature to ship.** Do not just
implement `admit()` / `decay_and_evict()` / `settle()` and call it done. Each must
be validated with an *isolating metric against a control*, on the synthetic
testbed, then logged in `RESEARCH_LOG.md`. Code without an experiment behind it is
not progress. Full method + experiment cards are in `RESEARCH.md`.

## Order of work
1. Implement `synth.py` (oracles kept strictly separate from `step()`). Cards A/B/C all depend on it.
2. Build the **instrument**: data loader for RTPTorrent, the weekly-retrain
   baseline (`experiments/baseline_weekly_retrain.py`), the **data-matched
   abstraction-off control**, and the slope harness (`experiments/compounding.py`).
3. Card A (admission) → Card B (eviction) → Card C (credit assignment). Degenerate
   case before the hard case (zero-delay before delayed; no-noise before noisy).
4. Wire a mechanism to RTPTorrent only after it passes on synth.

## You MUST emit run artifacts (see INTERFACE.md)
Every experiment run writes `experiments/runs/<run_id>/` with at least `meta.json`
and `slope.json`, plus `abstractions.json` / `metrics.json` / `events.json` when
the run has that data, and updates `experiments/runs/index.json`. **`INTERFACE.md`
is the source of truth for these shapes — do not invent fields; bump
`schema_version` if you must change them.** This is how the frontend sees your work;
a run that doesn't emit artifacts is invisible.

## Hard rules
- **`_private/` and `frontend/` are off-limits.** Never read, edit, or commit them.
- **Keep the predictor small.** Any improvement must come from accumulated
  abstractions, not capacity — a big model silently invalidates the thesis (DESIGN.md §5).
- **The control is "data-matched, abstraction-off,"** not just "weekly retrain."
  Beating weekly-retrain is table stakes; beating the data-matched control is the point.
- **Oracles never leak.** `synth.py`'s `true_*()` / `cause()` are evaluation-only.
- **De-flake before a CI outcome counts as a label.**
- **Keep smoke tests green;** new mechanisms get their own tests.

## Commands
```bash
pip install -e '.[dev]'              # core + pytest/ruff
pip install -e '.[experiments]'      # + sklearn/xgboost/pandas
pytest -q
ruff check .                         # line length 100
python data/profile_rtptorrent.py /path/to/rtptorrent
```

## Definition of done (a mechanism)
implementation + synth experiment reporting its isolating metric vs its control +
emitted run artifacts per INTERFACE.md + passing test + a dated `RESEARCH_LOG.md`
entry (keep/kill and why). Missing any → not done.
