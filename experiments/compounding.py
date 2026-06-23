"""Compounding harness with a data-matched abstraction-off control.

The real thesis metric is slope over time. This module provides the executable
instrument on synth first: every system sees the same stream and cumulative data,
but only ``ipsum`` can use admitted abstractions.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from math import floor
from pathlib import Path
from statistics import mean
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.rtptorrent import load_project_csv
from experiments.artifacts import make_run_id, utc_now, write_run
from experiments.baseline_weekly_retrain import WeeklyRetrainBaseline
from ipsum.abstractions import Abstraction, AbstractionStore
from ipsum.synth import Synth, SynthConfig


@dataclass(frozen=True)
class SelectionCommit:
    cycle: int
    changed_files: frozenset[object]


@dataclass(frozen=True)
class SelectionOutcome:
    test: int
    failed: bool


@dataclass(frozen=True)
class Cycle:
    cycle: int
    commit: SelectionCommit
    outcomes: tuple[SelectionOutcome, ...]


@dataclass(frozen=True)
class Point:
    cycle: int
    repo_age_days: float
    test_recall: float
    selection_rate: float


@dataclass(frozen=True)
class SynthCompoundingConfig:
    n_files: int = 96
    n_tests: int = 48
    n_clusters: int = 8
    cycles: int = 520
    eval_interval: int = 40
    eval_window: int = 80
    selection_rate_cap: float = 0.33
    p_hit: float = 0.9
    p_flaky: float = 0.01
    drift_schedule: tuple[int, ...] = ()
    seed: int = 300
    admission_interval: int = 40
    admission_warmup: int = 120
    validation_cycles: int = 40
    cochange_threshold: float = 0.22
    min_support: int = 4
    weekly_retrain_interval: int = 80
    weekly_retrain_window: int = 240


@dataclass(frozen=True)
class RTPTorrentCompoundingConfig:
    project_csv: str
    changes_csv: str | None = None
    max_cycles: int | None = None
    eval_interval: int = 200
    eval_window: int = 400
    selection_rate_cap: float = 0.33
    admission_interval: int = 200
    admission_warmup: int = 600
    validation_cycles: int = 200
    cochange_threshold: float = 0.08
    min_support: int = 3
    weekly_retrain_interval: int = 200
    weekly_retrain_window: int = 1200


@dataclass(frozen=True)
class PredictionExample:
    cycle: int
    changed_files: frozenset[object]
    test: int
    failed: bool


class DataMatchedControl:
    """Cumulative-data selector with no abstraction store."""

    system_name = "data_matched_control"

    def __init__(self, n_tests: int, selection_rate_cap: float) -> None:
        self.n_tests = n_tests
        self.selection_rate_cap = selection_rate_cap
        self.examples: list[PredictionExample] = []
        self._base_rates = [0.05 for _ in range(n_tests)]

    def select(self, change) -> set[int]:
        return _top_tests(self._base_rates, self.selection_rate_cap)

    def observe(self, cycle: Cycle) -> None:
        self.examples.extend(_examples(cycle))
        self._base_rates = _base_rates(self.examples, self.n_tests)


class WeeklyRetrainControl(DataMatchedControl):
    """Raw-rate selector retrained from scratch on a recent window."""

    system_name = "weekly_retrain"

    def __init__(
        self,
        n_tests: int,
        selection_rate_cap: float,
        *,
        retrain_interval: int,
        retrain_window: int,
    ) -> None:
        super().__init__(n_tests, selection_rate_cap)
        self.retrain_interval = retrain_interval
        self.retrain_window = retrain_window
        self._buffer: list[PredictionExample] = []

    def observe(self, cycle: Cycle) -> None:
        self._buffer.extend(_examples(cycle))
        if cycle.cycle % self.retrain_interval == 0:
            min_cycle = max(0, cycle.cycle - self.retrain_window)
            recent = [example for example in self._buffer if example.cycle >= min_cycle]
            self._base_rates = _base_rates(recent, self.n_tests)


class WeeklyRetrainHarnessAdapter:
    """Adapter for the real weekly-retrain baseline inside indexed harness cycles."""

    system_name = "weekly_retrain"

    def __init__(
        self,
        n_tests: int,
        selection_rate_cap: float,
        *,
        retrain_interval: int,
        retrain_window: int,
    ) -> None:
        self.n_tests = n_tests
        self._baseline = WeeklyRetrainBaseline(
            selection_rate_cap=selection_rate_cap,
            retrain_interval=retrain_interval,
            train_window=retrain_window,
        )

    def select(self, cycle: Cycle) -> set[int]:
        return {int(test) for test in self._baseline.select(cycle)}

    def observe(self, cycle: Cycle) -> None:
        self._baseline.observe(cycle)


class IpsumSelector(DataMatchedControl):
    """Small selector whose lift can only come from admitted abstractions."""

    system_name = "ipsum"

    def __init__(
        self,
        n_tests: int,
        selection_rate_cap: float,
        *,
        admission_interval: int,
        admission_warmup: int,
        validation_cycles: int,
        cochange_threshold: float,
        min_support: int,
    ) -> None:
        super().__init__(n_tests, selection_rate_cap)
        self.admission_interval = admission_interval
        self.admission_warmup = admission_warmup
        self.validation_cycles = validation_cycles
        self.store = AbstractionStore(
            cochange_threshold=cochange_threshold,
            min_support=min_support,
        )
        self._rates_by_abstraction: dict[str, list[float]] = {}

    def select(self, change) -> set[int]:
        changed_files = _changed_files(change)
        rates = [
            self._predict_with_store(changed_files, test)
            for test in range(self.n_tests)
        ]
        return _top_tests(rates, self.selection_rate_cap)

    def observe(self, cycle: Cycle) -> None:
        self.examples.extend(_examples(cycle))
        self.store.observe_commit(_changed_files(cycle))
        self._base_rates = _base_rates(self.examples, self.n_tests)
        if (
            cycle.cycle >= self.admission_warmup
            and cycle.cycle % self.admission_interval == 0
        ):
            self._readmit()

    def _readmit(self) -> None:
        validation_start = max(0, self.examples[-1].cycle - self.validation_cycles + 1)
        train = [example for example in self.examples if example.cycle < validation_start]
        heldout = [example for example in self.examples if example.cycle >= validation_start]
        if not train or not heldout:
            return

        self.store._items.clear()
        for candidate in self.store.candidates():
            ll_gain = _ll_for_abstractions([candidate], train, heldout, self.n_tests) - (
                _ll_for_abstractions([], train, heldout, self.n_tests)
            )
            self.store.admit(candidate, ll_gain)
        self._rates_by_abstraction = _abstraction_rates(list(self.store), self.examples, self.n_tests)

    def _predict_with_store(self, changed_files: frozenset[object], test: int) -> float:
        p = self._base_rates[test]
        for abstraction in self.store:
            files = _files(abstraction)
            if changed_files & files:
                p = max(p, self._rates_by_abstraction.get(abstraction.name, self._base_rates)[test])
        return p


def evaluate_window(system, commits: list[Cycle]) -> Point:
    """Run ``system`` over a held-out window without updating it."""
    failures = 0
    caught = 0
    selected = 0
    total_tests = 0
    for cycle in commits:
        chosen = system.select(cycle)
        failed_tests = {outcome.test for outcome in cycle.outcomes if outcome.failed}
        failures += len(failed_tests)
        caught += len(chosen & failed_tests)
        selected += len(chosen)
        total_tests += len(cycle.outcomes)
    return Point(
        cycle=commits[-1].cycle if commits else 0,
        repo_age_days=float(commits[-1].cycle if commits else 0),
        test_recall=caught / failures if failures else 1.0,
        selection_rate=selected / total_tests if total_tests else 0.0,
    )


def compounding_curve(
    system,
    timeline: list[Cycle],
    *,
    eval_interval: int = 40,
    eval_window: int = 80,
) -> list[Point]:
    """Walk the timeline, score pre-observe decisions, and sample rolling points."""
    decisions: list[tuple[int, int, int, int]] = []
    points: list[Point] = []
    for cycle in timeline:
        chosen = system.select(cycle)
        failed_tests = {outcome.test for outcome in cycle.outcomes if outcome.failed}
        decisions.append(
            (
                cycle.cycle,
                len(chosen & failed_tests),
                len(failed_tests),
                len(chosen),
            )
        )
        system.observe(cycle)
        if cycle.cycle and cycle.cycle % eval_interval == 0:
            window = decisions[-eval_window:]
            caught = sum(row[1] for row in window)
            failures = sum(row[2] for row in window)
            selected = sum(row[3] for row in window)
            total_tests = len(window) * system.n_tests
            points.append(
                Point(
                    cycle=cycle.cycle,
                    repo_age_days=float(cycle.cycle),
                    test_recall=caught / failures if failures else 1.0,
                    selection_rate=selected / total_tests if total_tests else 0.0,
                )
            )
    return points


def slope_gap(ipsum_curve: list[Point], baseline_curve: list[Point]) -> float:
    """Return slope(ipsum recall) - slope(baseline recall)."""
    return _slope(ipsum_curve) - _slope(baseline_curve)


def run_synth(config: SynthCompoundingConfig = SynthCompoundingConfig()) -> dict:
    timeline = _synth_timeline(config)
    systems = [
        WeeklyRetrainControl(
            config.n_tests,
            config.selection_rate_cap,
            retrain_interval=config.weekly_retrain_interval,
            retrain_window=config.weekly_retrain_window,
        ),
        DataMatchedControl(config.n_tests, config.selection_rate_cap),
        IpsumSelector(
            config.n_tests,
            config.selection_rate_cap,
            admission_interval=config.admission_interval,
            admission_warmup=config.admission_warmup,
            validation_cycles=config.validation_cycles,
            cochange_threshold=config.cochange_threshold,
            min_support=config.min_support,
        ),
    ]
    curves = {
        system.system_name: compounding_curve(
            system,
            timeline,
            eval_interval=config.eval_interval,
            eval_window=config.eval_window,
        )
        for system in systems
    }
    ipsum_system = systems[-1]
    assert isinstance(ipsum_system, IpsumSelector)

    created = utc_now()
    run_id = make_run_id("instrument", "synth", created)
    series = [
        {
            "cycle": point.cycle,
            "system": system_name,
            "value": point.test_recall,
        }
        for system_name, curve in curves.items()
        for point in curve
    ]
    metrics = {
        "ipsum_final_test_recall": curves["ipsum"][-1].test_recall,
        "data_matched_final_test_recall": curves["data_matched_control"][-1].test_recall,
        "weekly_retrain_final_test_recall": curves["weekly_retrain"][-1].test_recall,
        "ipsum_vs_data_matched_slope_gap": slope_gap(
            curves["ipsum"],
            curves["data_matched_control"],
        ),
        "ipsum_vs_weekly_slope_gap": slope_gap(curves["ipsum"], curves["weekly_retrain"]),
        "ipsum_final_selection_rate": curves["ipsum"][-1].selection_rate,
        "admitted_abstractions_final": float(len(ipsum_system.store)),
    }
    controls = {
        "data_matched_control_slope": _slope(curves["data_matched_control"]),
        "weekly_retrain_slope": _slope(curves["weekly_retrain"]),
    }
    admitted_cycle = _last_admission_cycle(config)
    abstractions = {
        "snapshots": [
            {
                "cycle": config.cycles - 1,
                "abstractions": [
                    {
                        "name": abstraction.name,
                        "complexity": abstraction.complexity,
                        "usefulness": abstraction.usefulness,
                        "admitted_cycle": admitted_cycle,
                        "evicted_cycle": None,
                        "payload_summary": _payload_summary(abstraction),
                    }
                    for abstraction in ipsum_system.store
                ],
            }
        ]
    }
    run_dir = write_run(
        run_id=run_id,
        card="instrument",
        dataset="synth",
        created=created,
        config=asdict(config),
        slope={
            "metric_name": "test_recall_at_selrate",
            "selection_rate_cap": config.selection_rate_cap,
            "series": series,
        },
        metrics=metrics,
        controls=controls,
        abstractions=abstractions,
        headline_metric=metrics["ipsum_vs_data_matched_slope_gap"],
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "metrics": metrics,
        "controls": controls,
    }


def run_rtptorrent(config: RTPTorrentCompoundingConfig) -> dict:
    timeline, n_tests = _rtptorrent_timeline(config)
    if not timeline:
        raise ValueError("RTPTorrent timeline is empty")

    systems = [
        WeeklyRetrainHarnessAdapter(
            n_tests,
            config.selection_rate_cap,
            retrain_interval=config.weekly_retrain_interval,
            retrain_window=config.weekly_retrain_window,
        ),
        DataMatchedControl(n_tests, config.selection_rate_cap),
        IpsumSelector(
            n_tests,
            config.selection_rate_cap,
            admission_interval=config.admission_interval,
            admission_warmup=config.admission_warmup,
            validation_cycles=config.validation_cycles,
            cochange_threshold=config.cochange_threshold,
            min_support=config.min_support,
        ),
    ]
    curves = {
        system.system_name: compounding_curve(
            system,
            timeline,
            eval_interval=config.eval_interval,
            eval_window=config.eval_window,
        )
        for system in systems
    }
    if any(not curve for curve in curves.values()):
        raise ValueError("not enough cycles to produce a compounding curve")

    ipsum_system = systems[-1]
    assert isinstance(ipsum_system, IpsumSelector)
    changed_file_coverage = sum(bool(cycle.commit.changed_files) for cycle in timeline) / len(timeline)

    created = utc_now()
    dataset = Path(config.project_csv).stem
    run_id = make_run_id("instrument", dataset, created)
    series = [
        {
            "cycle": point.cycle,
            "system": system_name,
            "value": point.test_recall,
        }
        for system_name, curve in curves.items()
        for point in curve
    ]
    metrics = {
        "ipsum_final_test_recall": curves["ipsum"][-1].test_recall,
        "data_matched_final_test_recall": curves["data_matched_control"][-1].test_recall,
        "weekly_retrain_final_test_recall": curves["weekly_retrain"][-1].test_recall,
        "ipsum_vs_data_matched_slope_gap": slope_gap(
            curves["ipsum"],
            curves["data_matched_control"],
        ),
        "ipsum_vs_weekly_slope_gap": slope_gap(curves["ipsum"], curves["weekly_retrain"]),
        "ipsum_final_selection_rate": curves["ipsum"][-1].selection_rate,
        "admitted_abstractions_final": float(len(ipsum_system.store)),
        "changed_file_coverage": changed_file_coverage,
        "n_cycles": float(len(timeline)),
        "n_tests": float(n_tests),
    }
    controls = {
        "data_matched_control_slope": _slope(curves["data_matched_control"]),
        "weekly_retrain_slope": _slope(curves["weekly_retrain"]),
    }
    run_dir = write_run(
        run_id=run_id,
        card="instrument",
        dataset=dataset,
        created=created,
        config=asdict(config),
        slope={
            "metric_name": "test_recall_at_selrate",
            "selection_rate_cap": config.selection_rate_cap,
            "series": series,
        },
        metrics=metrics,
        controls=controls,
        abstractions={
            "snapshots": [
                {
                    "cycle": timeline[-1].cycle,
                    "abstractions": [
                        {
                            "name": abstraction.name,
                            "complexity": abstraction.complexity,
                            "usefulness": abstraction.usefulness,
                            "admitted_cycle": _last_rtptorrent_admission_cycle(config, timeline[-1].cycle),
                            "evicted_cycle": None,
                            "payload_summary": _payload_summary(abstraction),
                        }
                        for abstraction in ipsum_system.store
                    ],
                }
            ]
        },
        headline_metric=metrics["ipsum_vs_data_matched_slope_gap"],
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "metrics": metrics,
        "controls": controls,
    }


def _synth_timeline(config: SynthCompoundingConfig) -> list[Cycle]:
    world = Synth(
        SynthConfig(
            n_files=config.n_files,
            n_tests=config.n_tests,
            n_clusters=config.n_clusters,
            p_hit=config.p_hit,
            p_flaky=config.p_flaky,
            drift_schedule=config.drift_schedule,
            seed=config.seed,
        )
    )
    timeline = []
    for _ in range(config.cycles):
        commit, outcomes = world.step()
        timeline.append(
            Cycle(
                commit.cycle,
                SelectionCommit(commit.cycle, frozenset(commit.changed_files)),
                tuple(SelectionOutcome(outcome.test, outcome.failed) for outcome in outcomes),
            )
        )
    return timeline


def _rtptorrent_timeline(config: RTPTorrentCompoundingConfig) -> tuple[list[Cycle], int]:
    cycles = load_project_csv(
        config.project_csv,
        max_cycles=config.max_cycles,
        changes_csv=config.changes_csv,
    )
    test_index: dict[str, int] = {}
    timeline: list[Cycle] = []
    for cycle in cycles:
        indexed_outcomes = []
        for outcome in cycle.outcomes:
            if outcome.test_name not in test_index:
                test_index[outcome.test_name] = len(test_index)
            indexed_outcomes.append(
                SelectionOutcome(
                    test=test_index[outcome.test_name],
                    failed=outcome.failed,
                )
            )
        timeline.append(
            Cycle(
                cycle=cycle.cycle,
                commit=SelectionCommit(
                    cycle=cycle.cycle,
                    changed_files=frozenset(cycle.changed_files),
                ),
                outcomes=tuple(indexed_outcomes),
            )
        )
    return timeline, len(test_index)


def _last_admission_cycle(config: SynthCompoundingConfig) -> int | None:
    if config.cycles <= config.admission_warmup:
        return None
    last_cycle = config.cycles - 1
    return last_cycle - (last_cycle % config.admission_interval)


def _last_rtptorrent_admission_cycle(
    config: RTPTorrentCompoundingConfig,
    last_cycle: int,
) -> int | None:
    if last_cycle < config.admission_warmup:
        return None
    return last_cycle - (last_cycle % config.admission_interval)


def _examples(cycle: Cycle) -> list[PredictionExample]:
    return [
        PredictionExample(
            cycle=cycle.cycle,
            changed_files=_changed_files(cycle),
            test=outcome.test,
            failed=outcome.failed,
        )
        for outcome in cycle.outcomes
    ]


def _top_tests(rates: list[float], selection_rate_cap: float) -> set[int]:
    n_select = max(1, floor(len(rates) * selection_rate_cap))
    ranked = sorted(range(len(rates)), key=lambda test: rates[test], reverse=True)
    return set(ranked[:n_select])


def _base_rates(examples: list[PredictionExample], n_tests: int) -> list[float]:
    counts = [[0, 0] for _ in range(n_tests)]
    for example in examples:
        counts[example.test][0] += int(example.failed)
        counts[example.test][1] += 1
    return [_smoothed(fails, total, 0.05) for fails, total in counts]


def _abstraction_rates(
    abstractions: list[Abstraction],
    examples: list[PredictionExample],
    n_tests: int,
) -> dict[str, list[float]]:
    base = _base_rates(examples, n_tests)
    rates: dict[str, list[float]] = {}
    for abstraction in abstractions:
        files = _files(abstraction)
        counts = [[0, 0] for _ in range(n_tests)]
        for example in examples:
            if example.changed_files & files:
                counts[example.test][0] += int(example.failed)
                counts[example.test][1] += 1
        rates[abstraction.name] = [
            _smoothed(fails, total, base[test])
            for test, (fails, total) in enumerate(counts)
        ]
    return rates


def _ll_for_abstractions(
    abstractions: list[Abstraction],
    train: list[PredictionExample],
    heldout: list[PredictionExample],
    n_tests: int,
) -> float:
    base = _base_rates(train, n_tests)
    if not abstractions:
        return _log_likelihood(heldout, lambda ex: base[ex.test])

    filesets = [_files(a) for a in abstractions]
    rates_by_abstraction: list[list[float]] = []
    for files in filesets:
        counts = [[0, 0] for _ in range(n_tests)]
        for example in train:
            if example.changed_files & files:
                counts[example.test][0] += int(example.failed)
                counts[example.test][1] += 1
        rates_by_abstraction.append(
            [_smoothed(fails, total, base[test]) for test, (fails, total) in enumerate(counts)]
        )

    return _log_likelihood(
        heldout,
        lambda ex: _predict_with_abstraction_rates(ex, base, filesets, rates_by_abstraction),
    )


def _log_likelihood(examples: list[PredictionExample], predict) -> float:
    from math import log

    eps = 1e-6
    total = 0.0
    for example in examples:
        p = min(1.0 - eps, max(eps, predict(example)))
        total += log(p) if example.failed else log(1.0 - p)
    return total / len(examples)


def _predict_with_abstraction_rates(
    example: PredictionExample,
    base: list[float],
    filesets: list[frozenset[object]],
    rates_by_abstraction: list[list[float]],
) -> float:
    p = base[example.test]
    for files, rates in zip(filesets, rates_by_abstraction, strict=True):
        if example.changed_files & files:
            p = max(p, rates[example.test])
    return p


def _smoothed(fails: int, total: int, prior: float) -> float:
    strength = 2.0
    return (fails + prior * strength) / (total + strength)


def _files(abstraction: Abstraction) -> frozenset[object]:
    payload = abstraction.payload
    if not isinstance(payload, dict):
        return frozenset()
    return frozenset(payload.get("files", []))


def _changed_files(change) -> frozenset[object]:
    if hasattr(change, "changed_files"):
        return frozenset(change.changed_files)
    if hasattr(change, "commit") and hasattr(change.commit, "changed_files"):
        return frozenset(change.commit.changed_files)
    return frozenset()


def _slope(curve: list[Point]) -> float:
    if len(curve) < 2:
        return 0.0
    xs = [point.cycle for point in curve]
    ys = [point.test_recall for point in curve]
    x_bar = mean(xs)
    y_bar = mean(ys)
    denom = sum((x - x_bar) ** 2 for x in xs)
    if denom == 0.0:
        return 0.0
    return sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys, strict=True)) / denom


def _payload_summary(abstraction: Abstraction) -> str:
    files = sorted(_files(abstraction))
    shown = ", ".join(str(file_id) for file_id in files[:10])
    suffix = "" if len(files) <= 10 else f", ... ({len(files)} files)"
    return f"files: {shown}{suffix}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtptorrent", help="Path to one RTPTorrent project CSV")
    parser.add_argument("--changes-csv", help="Optional changed-file metadata CSV")
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-window", type=int, default=None)
    return parser.parse_args()


def _run_from_cli(args: argparse.Namespace) -> dict:
    if args.rtptorrent:
        config = RTPTorrentCompoundingConfig(
            project_csv=args.rtptorrent,
            changes_csv=args.changes_csv,
            max_cycles=args.max_cycles,
            eval_interval=args.eval_interval or RTPTorrentCompoundingConfig.eval_interval,
            eval_window=args.eval_window or RTPTorrentCompoundingConfig.eval_window,
        )
        return run_rtptorrent(config)
    return run_synth()


if __name__ == "__main__":
    result = _run_from_cli(_parse_args())
    print(result["run_dir"])
    print(
        "ipsum_vs_data_matched_slope_gap="
        f"{result['metrics']['ipsum_vs_data_matched_slope_gap']:.6f}"
    )
    print(f"ipsum_final_test_recall={result['metrics']['ipsum_final_test_recall']:.3f}")
