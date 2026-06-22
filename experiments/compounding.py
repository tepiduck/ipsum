"""The headline experiment: does ipsum COMPOUND vs. a frozen/retrained baseline?

We do not care about absolute accuracy. We care about the SLOPE of a quality
metric over time, and whether ipsum's slope exceeds the baseline's and the gap
WIDENS. See DESIGN.md section 5.

Metric: TestRecall at a fixed SelectionRate cap, on a rolling held-out future
window of commits.

This is a SKELETON harness. Plug in real systems + data to run it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Point:
    repo_age_days: float
    test_recall: float
    selection_rate: float


def evaluate_window(system, commits) -> Point:
    """Run ``system`` over a held-out window of commits; return its metric point.

    ``system`` must expose ``.select(change) -> set[test]`` and ``.observe(outcome)``.
    TODO: implement against the real dataset.
    """
    raise NotImplementedError


def compounding_curve(system, timeline) -> list[Point]:
    """Walk the timeline, letting the system learn online, sampling the metric.

    For the baseline (weekly retrain), ``.observe`` is a no-op between retrains.
    For ipsum, every observed outcome updates the prior + abstraction store.
    """
    raise NotImplementedError


def slope_gap(ipsum_curve: list[Point], baseline_curve: list[Point]) -> float:
    """The number that decides the thesis: does ipsum's slope exceed baseline's,
    and does the gap widen month 3 -> month 6? TODO: fit + report."""
    raise NotImplementedError


if __name__ == "__main__":
    print("ipsum compounding harness — skeleton. See DESIGN.md section 5.")
