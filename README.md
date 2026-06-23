# ipsum

ipsum was a small research project about one question:

> Can a system get better over time by keeping explicit, readable abstractions about
> a domain, instead of only retraining a predictor or retrieving more context?

The v1 answer, for the task we tried, was **no**.

That is still a useful result. The codebase now contains a synthetic testbed, a
real-data replay harness, experiment artifacts, and a fairly clear record of why this
particular task did not work.

## What we tested

The first target was predictive test selection for CI:

Given a code change, pick the tests most likely to fail, while running only a fixed
fraction of the suite.

The idea was that ipsum would learn reusable co-change abstractions, such as groups
of files or directories that tend to move together and affect related tests. If those
abstractions were useful, ipsum should improve faster than a control that saw the
same data but had the abstraction store turned off.

The key comparison was always:

```
ipsum
vs.
data-matched control with abstractions disabled
```

Beating a weekly retrain baseline is not enough. That mostly shows online learning
beats batch retraining. The thesis needed ipsum to beat the data-matched control.

It did not.

## Current status

v1 is complete and banked as a negative result. See [RESULTS.md](RESULTS.md).

Short version:

- The synthetic harness works and catches both real signal and fake wins.
- Admission, eviction, and coverage guards all produced useful diagnostics on synth,
  but none became a strong mechanism.
- On RTPTorrent real data, co-change abstractions did not add predictive value beyond
  simple historical test failure rates.
- okhttp was the meaningful real-data run: it had usable changed-file coverage and
  nonzero admissions, but ipsum still matched the data-matched control exactly.
- sonarqube was wired up as a second project, but its v1 run admitted nothing, so it
  remains an admission-starved diagnostic rather than a thesis result.

The failure mode is pretty concrete: for CI test selection, recent/historical failure
rate is already a very strong cheap feature. The abstractions were chasing real file
change signal, but that signal was mostly redundant once historical failure rate was
available.

## What is reusable

Even though the v1 task failed, a few pieces are worth keeping:

- `src/ipsum/synth.py`: a seeded synthetic CI world with ground-truth oracles.
- `experiments/`: slope and artifact harnesses for comparing an online system against
  a data-matched control.
- `data/rtptorrent.py`: loader for the actual RTPTorrent v1 CSV schema.
- `experiments/runs/`: JSON artifacts consumed by the dashboard.
- `RESEARCH_LOG.md`: append-only record of experiments and verdicts.
- `research/`: notes on the related work and dataset scouting.

## Repo layout

```text
ipsum/
├── README.md
├── RESULTS.md           # final v1 result and lessons
├── DESIGN.md            # original architecture and research plan
├── RESEARCH.md          # experiment cards and method notes
├── INTERFACE.md         # JSON artifact contract for the dashboard
├── AGENTS.md            # backend/Codex instructions
├── CLAUDE.md            # frontend/Claude instructions
├── src/ipsum/           # synthetic testbed and mechanism code
├── experiments/         # experiment harnesses and artifact writers
├── data/                # RTPTorrent loader and profiling notes
├── frontend/            # local dashboard scaffold
├── research/            # paper notes and synthesis
├── RESEARCH_LOG.md
└── tests/
```

## Running the project

Install in editable mode:

```bash
pip install -e '.[dev]'
pip install -e '.[experiments]'
```

Run the checks:

```bash
pytest -q
ruff check .
```

The project declares Python 3.10 support, so use Python 3.10 when validating final
changes.

## Reading order

For the final outcome, start with [RESULTS.md](RESULTS.md).

For the original research design, read:

1. [DESIGN.md](DESIGN.md)
2. [RESEARCH.md](RESEARCH.md)
3. [INTERFACE.md](INTERFACE.md)
4. [research/00-synthesis.md](research/00-synthesis.md)

For backend work, also read [AGENTS.md](AGENTS.md). For frontend work, read
[CLAUDE.md](CLAUDE.md).

## License

MIT
