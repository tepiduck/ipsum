"""Baseline to beat: Predictive Test Selection (Machalica et al., ICSE-SEIP 2019).

The full paper baseline uses XGBoost over (change, test) features and retrains
weekly from scratch. This implementation wires the real data shape and feature
surface without requiring XGBoost at import time: ``retrain`` builds a small,
deterministic scoring model from the same feature families, so the harness can
run before the optional experiments dependency stack is installed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import floor


@dataclass(frozen=True)
class _Observation:
    cycle: int
    test_name: str
    failed: bool
    changed_files: frozenset[str]


@dataclass
class WeeklyRetrainBaseline:
    selection_rate_cap: float = 0.33
    retrain_interval: int = 80
    train_window: int = 240
    _model: dict[str, float] = field(default_factory=dict)
    _buffer: list[object] = field(default_factory=list)
    _tests: set[str] = field(default_factory=set)

    def features(self, change, test) -> dict[str, float]:
        """Build the (change, test) feature vector from observed history."""
        test_name = str(test)
        changed_files = _changed_files(change)
        cycle = _cycle(change)
        history = self._training_observations(cycle)
        test_history = [obs for obs in history if obs.test_name == test_name]
        related_history = [
            obs
            for obs in test_history
            if changed_files and obs.changed_files and changed_files & obs.changed_files
        ]
        return {
            "changed_file_count": float(len(changed_files)),
            "has_changed_files": float(bool(changed_files)),
            "java_file_fraction": _extension_fraction(changed_files, ".java"),
            "historical_failure_rate": _failure_rate(test_history),
            "test_seen_count": float(len(test_history)),
            "related_file_failure_rate": _failure_rate(related_history),
            "related_file_seen_count": float(len(related_history)),
            "recent_failure_rate_3": _failure_rate(_recent(test_history, cycle, 3)),
            "recent_failure_rate_14": _failure_rate(_recent(test_history, cycle, 14)),
            "recent_failure_rate_56": _failure_rate(_recent(test_history, cycle, 56)),
        }

    def retrain(self) -> None:
        """Retrain from scratch on the recent window."""
        if not self._buffer:
            self._model = {}
            return

        latest_cycle = max(_cycle(change) for change in self._buffer)
        history = self._training_observations(latest_cycle)
        by_test: dict[str, list[_Observation]] = {test: [] for test in self._tests}
        for obs in history:
            by_test.setdefault(obs.test_name, []).append(obs)

        self._model = {
            test_name: _failure_rate(observations)
            for test_name, observations in by_test.items()
        }

    def select(self, change) -> set[str]:
        """Rank tests by predicted fail probability under the current model."""
        candidates = set(self._tests)
        candidates.update(_outcome_test_names(change))
        if not candidates:
            return set()

        n_select = max(1, floor(len(candidates) * self.selection_rate_cap))
        ranked = sorted(
            candidates,
            key=lambda test: self._predict(change, test),
            reverse=True,
        )
        return set(ranked[:n_select])

    def observe(self, outcome) -> None:
        """Buffer an observed CI cycle; learning happens only on retrain."""
        self._buffer.append(outcome)
        self._tests.update(_outcome_test_names(outcome))
        cycle = _cycle(outcome)
        if cycle and cycle % self.retrain_interval == 0:
            self.retrain()

    def _predict(self, change, test_name: str) -> float:
        f = self.features(change, test_name)
        base = self._model.get(test_name, 0.05)
        score = (
            0.50 * base
            + 0.25 * f["related_file_failure_rate"]
            + 0.15 * f["recent_failure_rate_14"]
            + 0.10 * f["recent_failure_rate_56"]
        )
        if f["related_file_seen_count"] == 0.0:
            score *= 0.9
        return score

    def _training_observations(self, current_cycle: int) -> list[_Observation]:
        min_cycle = max(0, current_cycle - self.train_window)
        observations: list[_Observation] = []
        for change in self._buffer:
            if min_cycle <= _cycle(change) <= current_cycle:
                observations.extend(_observations(change))
        return observations


def _observations(change) -> list[_Observation]:
    changed_files = _changed_files(change)
    return [
        _Observation(
            cycle=_cycle(change),
            test_name=_test_name(outcome),
            failed=bool(outcome.failed),
            changed_files=changed_files,
        )
        for outcome in getattr(change, "outcomes", ())
    ]


def _outcome_test_names(change) -> set[str]:
    return {_test_name(outcome) for outcome in getattr(change, "outcomes", ())}


def _test_name(outcome) -> str:
    if hasattr(outcome, "test_name"):
        return str(outcome.test_name)
    return str(outcome.test)


def _changed_files(change) -> frozenset[str]:
    if hasattr(change, "changed_files"):
        return frozenset(str(file_id) for file_id in change.changed_files)
    if hasattr(change, "commit") and hasattr(change.commit, "changed_files"):
        return frozenset(str(file_id) for file_id in change.commit.changed_files)
    return frozenset()


def _cycle(change) -> int:
    return int(getattr(change, "cycle", 0))


def _failure_rate(observations: list[_Observation]) -> float:
    if not observations:
        return 0.05
    failures = sum(obs.failed for obs in observations)
    return (failures + 0.1) / (len(observations) + 2.0)


def _recent(
    observations: list[_Observation],
    current_cycle: int,
    window: int,
) -> list[_Observation]:
    min_cycle = max(0, current_cycle - window)
    return [obs for obs in observations if obs.cycle >= min_cycle]


def _extension_fraction(files: frozenset[str], suffix: str) -> float:
    if not files:
        return 0.0
    counts = Counter(file.rsplit(".", 1)[-1] for file in files if "." in file)
    suffix_key = suffix.removeprefix(".")
    return counts[suffix_key] / len(files)
