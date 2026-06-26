# ipsum

`ipsum` is a research codebase for experimenting with explicit, human-readable abstractions in online learning systems.

The initial implementation focuses on predictive test selection for continuous integration (CI). Given a code change, the system ranks tests by their estimated likelihood of failure while operating under a fixed execution budget.

The repository includes a synthetic evaluation environment, real-data replay harnesses, experiment infrastructure, and implementations of the abstraction mechanisms explored during the project.

## Features

* Synthetic CI environment with configurable ground-truth structure
* Replay harness for RTPTorrent commit and test history
* Online experiment framework for comparing different learning strategies
* JSON artifact generation for experiment visualization
* Research notes and reproducible experiment logs

## Repository layout

```text
ipsum/
├── README.md
├── RESULTS.md           # experiment outcomes and discussion
├── DESIGN.md            # system architecture
├── RESEARCH.md          # methodology and experiment notes
├── INTERFACE.md         # artifact format
├── AGENTS.md            # backend development notes
├── CLAUDE.md            # frontend development notes
├── src/ipsum/           # core implementation
├── experiments/         # experiment runners
├── data/                # dataset loaders
├── frontend/            # dashboard scaffold
├── research/            # literature notes
├── RESEARCH_LOG.md      # experiment log
└── tests/
```

## Installation

The project targets Python 3.10.

Install the development dependencies:

```bash
pip install -e '.[dev]'
pip install -e '.[experiments]'
```

## Validation

Run the test suite:

```bash
pytest -q
```

Run linting:

```bash
ruff check .
```

## Running experiments

The repository contains two primary evaluation workflows:

* **Synthetic experiments**, which evaluate mechanisms in a controlled simulated CI environment.
* **Replay experiments**, which evaluate the same mechanisms against historical repository data using the RTPTorrent dataset.

Experiment outputs are written as JSON artifacts under `experiments/runs/`, where they can be consumed by the dashboard or inspected directly.

See `DESIGN.md` and `RESEARCH.md` for implementation details and experiment configuration.

## Documentation

* `DESIGN.md` — architecture and implementation
* `RESEARCH.md` — methodology and experiment setup
* `INTERFACE.md` — artifact schema
* `RESULTS.md` — experiment outcomes
* `RESEARCH_LOG.md` — chronological experiment record

## License

MIT
