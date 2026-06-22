"""Baseline to beat: Predictive Test Selection (Machalica et al., ICSE-SEIP 2019).

XGBoost over (change, test) pairs, retrained weekly FROM SCRATCH. No online
update, no compounding — this is the plateau ipsum's learning loop must out-slope.
See research/09-test-selection.md.

Strongest features reported: build-dependency-graph distance, historical per-test
failure rate, file-change-history windows (3/14/56 day), file extensions, project.
Labels must be DE-FLAKED before training (retry failures; keep only consistent ones).

This is a SKELETON. Wire to the dataset and an XGBoost model to run.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WeeklyRetrainBaseline:
    selection_rate_cap: float = 0.33
    _model: object | None = None
    _buffer: list = field(default_factory=list)

    def features(self, change, test) -> dict:
        """Build the (change, test) feature vector. TODO."""
        raise NotImplementedError

    def retrain(self) -> None:
        """Retrain XGBoost from scratch on the last ~3 months. TODO."""
        raise NotImplementedError

    def select(self, change) -> set:
        """Rank tests by predicted fail prob; take the top under the cap. TODO."""
        raise NotImplementedError

    def observe(self, outcome) -> None:
        """No-op except buffering — the baseline only learns at weekly retrain."""
        self._buffer.append(outcome)
