"""RTPTorrent real-data compounding run.

V1 intentionally adds no new mechanism: it joins RTPTorrent's provided
job->commit and commit->patch CSVs, deflakes labels conservatively, then runs
weekly-retrain, data-matched control, and ipsum with the current admission +
eviction store.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from math import log
from pathlib import Path
from statistics import median
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.rtptorrent import (
    RTPTorrentCycle,
    RTPTorrentV1JoinStats,
    load_rtptorrent_v1_project,
)
from experiments.artifacts import make_run_id, utc_now, write_run
from experiments.compounding import (
    Cycle,
    DataMatchedControl,
    Point,
    PredictionExample,
    SelectionCommit,
    SelectionOutcome,
    _abstraction_rates,
    _base_rates,
    _changed_files,
    _examples,
    _files,
    _payload_summary,
    _slope,
    _top_tests,
    compounding_curve,
)
from ipsum.abstractions import Abstraction, AbstractionStore


SYSTEMS = ("weekly_retrain", "data_matched_control", "ipsum")


@dataclass(frozen=True)
class RTPTorrentRealDataConfig:
    dataset: str
    project_csv: str
    built_commits_csv: str
    patches_csv: str
    max_cycles: int | None = None
    max_changed_files: int = 30
    max_wait_cycles: int = 0
    eval_interval: int = 200
    eval_window: int = 400
    plateau_points: int = 3
    selection_rate_cap: float = 0.33
    change_granularity: str = "directory"
    change_path_depth: int = 3
    admission_interval: int = 200
    admission_warmup: int = 400
    validation_cycles: int = 120
    adaptation_window: int = 1000
    cochange_threshold: float = 0.02
    min_support: int = 2
    max_candidates: int = 256
    complexity_per_file: float = 0.0005
    decay: float = 0.99
    eviction_grace_cycles: int = 400
    abstraction_lift_weight: float = 0.35
    weekly_retrain_interval: int = 200
    weekly_retrain_window: int = 1200


@dataclass(frozen=True)
class StreamStats:
    raw_cycles: int
    loaded_cycles: int
    used_cycles: int
    jobs_with_commits: int
    jobs_with_changed_files: int
    dropped_large_change_cycles: int
    missing_commit_jobs: int
    missing_patch_jobs: int
    missing_changed_file_cycles: int
    unresolved_outcomes: int
    deflaked_outcomes: int
    skipped_outcomes: int
    positive_labels: int
    n_tests: int
    distinct_raw_changed_files: int
    distinct_change_tokens: int
    mean_raw_changed_files: float
    mean_change_tokens: float


class FastWeeklyRetrainControl(DataMatchedControl):
    """Weekly-retrain baseline refit from scratch on a recent raw-rate window."""

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
        if cycle.cycle % self.retrain_interval != 0:
            return
        min_cycle = max(0, cycle.cycle - self.retrain_window)
        recent = [example for example in self._buffer if example.cycle >= min_cycle]
        self._base_rates = _base_rates(recent, self.n_tests)


class FastDataMatchedControl(DataMatchedControl):
    """Cumulative abstraction-off control with incremental raw-rate updates."""

    system_name = "data_matched_control"

    def __init__(self, n_tests: int, selection_rate_cap: float) -> None:
        super().__init__(n_tests, selection_rate_cap)
        self._counts = [[0, 0] for _ in range(n_tests)]

    def observe(self, cycle: Cycle) -> None:
        examples = _examples(cycle)
        self.examples.extend(examples)
        _update_counts(self._counts, examples)
        self._base_rates = _rates_from_counts(self._counts)


class RealDataIpsumSelector(DataMatchedControl):
    """Current small ipsum store with admission, reinforcement, and eviction."""

    system_name = "ipsum"

    def __init__(self, n_tests: int, config: RTPTorrentRealDataConfig) -> None:
        super().__init__(n_tests, config.selection_rate_cap)
        self.config = config
        self.store = AbstractionStore(
            decay=config.decay,
            min_support=config.min_support,
            cochange_threshold=config.cochange_threshold,
            max_candidates=config.max_candidates,
            complexity_per_file=config.complexity_per_file,
        )
        self._rates_by_abstraction: dict[str, list[float]] = {}
        self._base_counts = [[0, 0] for _ in range(n_tests)]
        self._commits: list[tuple[int, frozenset[object]]] = []
        self._admitted_cycle: dict[str, int] = {}
        self._events: list[dict] = []
        self._admission_round = 0

    @property
    def events(self) -> list[dict]:
        return list(self._events)

    def funnel_metrics(self) -> dict[str, float]:
        funnels = [event for event in self._events if event["type"] == "admission_funnel"]
        ll_means = [event["ll_gain_mean"] for event in funnels if event["candidates_proposed"]]
        return {
            "admission_funnel_cycles": float(len(funnels)),
            "candidates_proposed_total": float(
                sum(event["candidates_proposed"] for event in funnels)
            ),
            "candidates_proposed_mean": _mean(
                event["candidates_proposed"] for event in funnels
            ),
            "candidates_passing_coverage_guard_total": float(
                sum(event["candidates_passing_coverage_guard"] for event in funnels)
            ),
            "candidates_passing_coverage_guard_mean": _mean(
                event["candidates_passing_coverage_guard"] for event in funnels
            ),
            "admitted_count_total": float(sum(event["admitted_count"] for event in funnels)),
            "admitted_count_mean": _mean(event["admitted_count"] for event in funnels),
            "ll_gain_mean_over_funnels": _mean(ll_means),
            "ll_gain_max_over_funnels": max(
                (event["ll_gain_max"] for event in funnels),
                default=0.0,
            ),
        }

    def select(self, change) -> set[int]:
        changed_files = _changed_files(change)
        rates = [
            self._predict_with_store(changed_files, test)
            for test in range(self.n_tests)
        ]
        return _top_tests(rates, self.selection_rate_cap)

    def observe(self, cycle: Cycle) -> None:
        examples = _examples(cycle)
        self.examples.extend(examples)
        changed_files = _changed_files(cycle)
        if changed_files:
            self._commits.append((cycle.cycle, changed_files))
        _update_counts(self._base_counts, examples)
        self._base_rates = _rates_from_counts(self._base_counts)
        self._reinforce(examples)
        self._evict(cycle.cycle)
        if (
            cycle.cycle >= self.config.admission_warmup
            and cycle.cycle % self.config.admission_interval == 0
        ):
            self._admit_recent(cycle.cycle)

    def _admit_recent(self, cycle: int) -> None:
        recent_start = max(0, cycle - self.config.adaptation_window + 1)
        validation_start = max(recent_start, cycle - self.config.validation_cycles + 1)
        train = self._examples_between(recent_start, validation_start - 1)
        heldout = self._examples_between(validation_start, cycle)
        if not train or not heldout:
            self._record_funnel(
                cycle=cycle,
                candidates_proposed=0,
                candidates_passing_coverage_guard=0,
                admitted_count=0,
                ll_gains=[],
                reason="insufficient_train_or_heldout",
            )
            return

        proposal_store = AbstractionStore(
            min_support=self.config.min_support,
            cochange_threshold=self.config.cochange_threshold,
            max_candidates=self.config.max_candidates,
            complexity_per_file=self.config.complexity_per_file,
        )
        for commit_cycle, changed_files in self._commits:
            if commit_cycle >= recent_start:
                proposal_store.observe_commit(changed_files)

        candidates = proposal_store.candidates()
        base_rates = _base_rates(train, self.n_tests)
        raw_ll = _ll_from_rates(heldout, base_rates)
        admitted = 0
        ll_gains: list[float] = []
        candidates_passing_coverage_guard = 0
        for candidate in candidates:
            if _candidate_support(candidate, self._commits, recent_start) >= self.config.min_support:
                candidates_passing_coverage_guard += 1
            candidate = self._versioned_candidate(candidate)
            ll_gain = _candidate_ll(candidate, train, heldout, base_rates, self.n_tests) - raw_ll
            ll_gains.append(ll_gain)
            if self.store.admit(candidate, ll_gain):
                self._admitted_cycle[candidate.name] = cycle
                admitted += 1
                self._events.append(
                    {
                        "cycle": cycle,
                        "type": "admit",
                        "system": self.system_name,
                        "name": candidate.name,
                        "detail": f"ll_gain={ll_gain:.6f}",
                    }
                )
        if admitted:
            self._refresh_rates(cycle)
        self._record_funnel(
            cycle=cycle,
            candidates_proposed=len(candidates),
            candidates_passing_coverage_guard=candidates_passing_coverage_guard,
            admitted_count=admitted,
            ll_gains=ll_gains,
            reason="ok",
        )
        self._admission_round += 1

    def _record_funnel(
        self,
        *,
        cycle: int,
        candidates_proposed: int,
        candidates_passing_coverage_guard: int,
        admitted_count: int,
        ll_gains: list[float],
        reason: str,
    ) -> None:
        self._events.append(
            {
                "cycle": cycle,
                "type": "admission_funnel",
                "system": self.system_name,
                "candidates_proposed": candidates_proposed,
                "candidates_passing_coverage_guard": candidates_passing_coverage_guard,
                "admitted_count": admitted_count,
                "ll_gain_min": min(ll_gains, default=0.0),
                "ll_gain_p50": median(ll_gains) if ll_gains else 0.0,
                "ll_gain_mean": _mean(ll_gains),
                "ll_gain_p90": _percentile(ll_gains, 0.9),
                "ll_gain_max": max(ll_gains, default=0.0),
                "reason": reason,
            }
        )

    def _versioned_candidate(self, candidate: Abstraction) -> Abstraction:
        return Abstraction(
            name=f"{candidate.name}_r{self._admission_round}",
            payload=candidate.payload,
            complexity=candidate.complexity,
            usefulness=candidate.usefulness,
        )

    def _reinforce(self, examples: list[PredictionExample]) -> None:
        for abstraction in list(self.store):
            rates = self._rates_by_abstraction.get(abstraction.name)
            if rates is None:
                continue
            gain = _cycle_ll_gain(abstraction, examples, self._base_rates, rates)
            if gain != 0.0:
                self.store.reinforce(abstraction.name, gain)

    def _evict(self, cycle: int) -> None:
        protected = {
            name
            for name, admitted in self._admitted_cycle.items()
            if cycle - admitted < self.config.eviction_grace_cycles
        }
        evicted = self.store.decay_and_evict(protected=protected)
        for name in evicted:
            self._rates_by_abstraction.pop(name, None)
            self._events.append(
                {
                    "cycle": cycle,
                    "type": "evict",
                    "system": self.system_name,
                    "name": name,
                }
            )

    def _refresh_rates(self, cycle: int) -> None:
        recent_start = max(0, cycle - self.config.adaptation_window + 1)
        recent = self._examples_between(recent_start, cycle)
        self._rates_by_abstraction.update(
            _abstraction_rates(list(self.store), recent, self.n_tests)
        )

    def _examples_between(self, start_cycle: int, end_cycle: int) -> list[PredictionExample]:
        if end_cycle < start_cycle:
            return []
        selected = []
        for example in reversed(self.examples):
            if example.cycle < start_cycle:
                break
            if example.cycle <= end_cycle:
                selected.append(example)
        selected.reverse()
        return selected

    def _predict_with_store(self, changed_files: frozenset[object], test: int) -> float:
        base = self._base_rates[test]
        lift = 0.0
        for abstraction in self.store:
            files = _files(abstraction)
            if changed_files & files:
                rates = self._rates_by_abstraction.get(abstraction.name, self._base_rates)
                lift += max(0.0, rates[test] - base)
        return min(1.0, base + self.config.abstraction_lift_weight * lift)


def run(config: RTPTorrentRealDataConfig) -> dict:
    raw_cycles, join_stats = load_rtptorrent_v1_project(
        config.project_csv,
        config.built_commits_csv,
        config.patches_csv,
        max_cycles=config.max_cycles,
        max_changed_files=config.max_changed_files,
    )
    timeline, stats = _timeline(raw_cycles, join_stats, config)
    if not timeline:
        raise ValueError("RTPTorrent stream is empty after changed-file/drop/deflake filters")

    systems = [
        FastWeeklyRetrainControl(
            stats.n_tests,
            config.selection_rate_cap,
            retrain_interval=config.weekly_retrain_interval,
            retrain_window=config.weekly_retrain_window,
        ),
        FastDataMatchedControl(stats.n_tests, config.selection_rate_cap),
        RealDataIpsumSelector(stats.n_tests, config),
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

    ipsum = systems[-1]
    assert isinstance(ipsum, RealDataIpsumSelector)
    metrics = _metrics(curves, stats, ipsum, config)
    controls = {
        "data_matched_control_slope": _slope(curves["data_matched_control"]),
        "weekly_retrain_slope": _slope(curves["weekly_retrain"]),
        "n_seeds": 1.0,
        "selection_rate_cap": config.selection_rate_cap,
    }

    created = utc_now()
    run_id = make_run_id("compounding", config.dataset, created)
    run_dir = write_run(
        run_id=run_id,
        card="compounding",
        dataset=config.dataset,
        created=created,
        config=asdict(config),
        slope={
            "metric_name": "test_recall_at_selrate",
            "selection_rate_cap": config.selection_rate_cap,
            "series": _series(curves),
            "gap_series": _gap_series(curves),
        },
        metrics=metrics,
        controls=controls,
        abstractions=_abstractions(ipsum, timeline[-1].cycle),
        events={"events": ipsum.events},
        headline_metric=metrics["ipsum_vs_data_matched_slope_gap"],
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "metrics": metrics,
        "controls": controls,
    }


def _timeline(
    cycles: list[RTPTorrentCycle],
    join_stats: RTPTorrentV1JoinStats,
    config: RTPTorrentRealDataConfig,
) -> tuple[list[Cycle], StreamStats]:
    test_index: dict[str, int] = {}
    timeline: list[Cycle] = []
    missing_changes = 0
    deflaked = 0
    skipped = 0
    positives = 0
    raw_file_counts: list[int] = []
    token_counts: list[int] = []
    raw_files_seen: set[str] = set()
    tokens_seen: set[str] = set()
    for raw in cycles:
        if not raw.changed_files:
            missing_changes += 1
        changed_tokens = _coarsen_changed_files(
            raw.changed_files,
            granularity=config.change_granularity,
            depth=config.change_path_depth,
        )
        raw_file_counts.append(len(raw.changed_files))
        token_counts.append(len(changed_tokens))
        raw_files_seen.update(raw.changed_files)
        tokens_seen.update(str(token) for token in changed_tokens)
        outcomes, local_stats = _deflaked_outcomes(raw)
        deflaked += local_stats["deflaked"]
        skipped += local_stats["skipped"]
        if not outcomes:
            continue
        indexed = []
        for test_name, failed in outcomes:
            if test_name not in test_index:
                test_index[test_name] = len(test_index)
            positives += int(failed)
            indexed.append(SelectionOutcome(test=test_index[test_name], failed=failed))
        timeline.append(
            Cycle(
                cycle=len(timeline),
                commit=SelectionCommit(
                    cycle=len(timeline),
                    changed_files=changed_tokens,
                ),
                outcomes=tuple(indexed),
            )
        )
    stats = StreamStats(
        raw_cycles=join_stats.raw_jobs,
        loaded_cycles=len(cycles),
        used_cycles=len(timeline),
        jobs_with_commits=join_stats.jobs_with_commits,
        jobs_with_changed_files=join_stats.jobs_with_changed_files,
        dropped_large_change_cycles=join_stats.dropped_large_change_jobs,
        missing_commit_jobs=join_stats.missing_commit_jobs,
        missing_patch_jobs=join_stats.missing_patch_jobs,
        missing_changed_file_cycles=missing_changes,
        unresolved_outcomes=0 if config.max_wait_cycles == 0 else 0,
        deflaked_outcomes=deflaked,
        skipped_outcomes=skipped,
        positive_labels=positives,
        n_tests=len(test_index),
        distinct_raw_changed_files=len(raw_files_seen),
        distinct_change_tokens=len(tokens_seen),
        mean_raw_changed_files=_mean(raw_file_counts),
        mean_change_tokens=_mean(token_counts),
    )
    return timeline, stats


def _deflaked_outcomes(cycle: RTPTorrentCycle) -> tuple[list[tuple[str, bool]], dict[str, int]]:
    by_test: dict[str, list[bool]] = {}
    skipped = 0
    for outcome in cycle.outcomes:
        if outcome.skipped and not outcome.failed:
            skipped += 1
            continue
        by_test.setdefault(outcome.test_name, []).append(outcome.failed)

    labels = []
    deflaked = 0
    for test_name, values in by_test.items():
        if any(value != values[0] for value in values):
            deflaked += len(values)
            continue
        if len(values) > 1:
            deflaked += len(values) - 1
        labels.append((test_name, values[0]))
    return labels, {"deflaked": deflaked, "skipped": skipped}


def _metrics(
    curves: dict[str, list[Point]],
    stats: StreamStats,
    ipsum: RealDataIpsumSelector,
    config: RTPTorrentRealDataConfig,
) -> dict[str, float]:
    ipsum_gap = slope_gap(curves["ipsum"], curves["data_matched_control"])
    weekly_gap = slope_gap(curves["ipsum"], curves["weekly_retrain"])
    gap_points = _gap_points(curves)
    plateau = _plateau_metrics(curves, config.plateau_points)
    slopes = {system: _slope(curve) for system, curve in curves.items()}
    metrics = {
        "n_seeds": 1.0,
        "ipsum_final_test_recall": curves["ipsum"][-1].test_recall,
        "data_matched_final_test_recall": curves["data_matched_control"][-1].test_recall,
        "weekly_retrain_final_test_recall": curves["weekly_retrain"][-1].test_recall,
        "ipsum_slope": slopes["ipsum"],
        "ipsum_slope_mean": slopes["ipsum"],
        "ipsum_slope_se": 0.0,
        "data_matched_control_slope": slopes["data_matched_control"],
        "data_matched_control_slope_mean": slopes["data_matched_control"],
        "data_matched_control_slope_se": 0.0,
        "weekly_retrain_slope": slopes["weekly_retrain"],
        "weekly_retrain_slope_mean": slopes["weekly_retrain"],
        "weekly_retrain_slope_se": 0.0,
        "ipsum_vs_data_matched_slope_gap": ipsum_gap,
        "ipsum_vs_data_matched_slope_gap_mean": ipsum_gap,
        "ipsum_vs_data_matched_slope_gap_se": 0.0,
        "ipsum_vs_weekly_slope_gap": weekly_gap,
        "ipsum_vs_weekly_slope_gap_mean": weekly_gap,
        "ipsum_vs_weekly_slope_gap_se": 0.0,
        "ipsum_vs_data_matched_gap_trend": _gap_trend(gap_points),
        "ipsum_vs_data_matched_gap_trend_se": 0.0,
        "ipsum_final_selection_rate": curves["ipsum"][-1].selection_rate,
        "max_selection_rate": max(
            point.selection_rate for curve in curves.values() for point in curve
        ),
        "admitted_abstractions_final": float(len(ipsum.store)),
        "raw_cycles": float(stats.raw_cycles),
        "loaded_cycles": float(stats.loaded_cycles),
        "used_cycles": float(stats.used_cycles),
        "n_tests": float(stats.n_tests),
        "positive_labels": float(stats.positive_labels),
        "jobs_with_commits": float(stats.jobs_with_commits),
        "jobs_with_changed_files": float(stats.jobs_with_changed_files),
        "job_commit_coverage": (
            stats.jobs_with_commits / stats.raw_cycles if stats.raw_cycles else 0.0
        ),
        "job_changed_file_coverage": (
            stats.jobs_with_changed_files / stats.raw_cycles if stats.raw_cycles else 0.0
        ),
        "changed_file_coverage": (
            (stats.used_cycles - stats.missing_changed_file_cycles) / stats.used_cycles
            if stats.used_cycles
            else 0.0
        ),
        "distinct_raw_changed_files": float(stats.distinct_raw_changed_files),
        "distinct_change_tokens": float(stats.distinct_change_tokens),
        "mean_raw_changed_files": stats.mean_raw_changed_files,
        "mean_change_tokens": stats.mean_change_tokens,
        "dropped_large_change_cycles": float(stats.dropped_large_change_cycles),
        "missing_commit_jobs": float(stats.missing_commit_jobs),
        "missing_patch_jobs": float(stats.missing_patch_jobs),
        "missing_changed_file_cycles": float(stats.missing_changed_file_cycles),
        "deflaked_outcomes": float(stats.deflaked_outcomes),
        "skipped_outcomes": float(stats.skipped_outcomes),
        "unresolved_outcomes": float(stats.unresolved_outcomes),
    }
    metrics.update(ipsum.funnel_metrics())
    metrics.update(plateau)
    return metrics


def slope_gap(ipsum_curve: list[Point], baseline_curve: list[Point]) -> float:
    return _slope(ipsum_curve) - _slope(baseline_curve)


def _plateau_metrics(curves: dict[str, list[Point]], plateau_points: int) -> dict[str, float]:
    metrics = {}
    for system, curve in curves.items():
        tail = curve[-plateau_points:] if plateau_points else curve
        metrics[f"{system}_plateau_test_recall"] = _mean(point.test_recall for point in tail)
        metrics[f"{system}_plateau_test_recall_se"] = 0.0
    metrics["ipsum_vs_data_matched_plateau_gap"] = (
        metrics["ipsum_plateau_test_recall"]
        - metrics["data_matched_control_plateau_test_recall"]
    )
    metrics["ipsum_vs_data_matched_plateau_gap_se"] = 0.0
    metrics["ipsum_vs_weekly_plateau_gap"] = (
        metrics["ipsum_plateau_test_recall"] - metrics["weekly_retrain_plateau_test_recall"]
    )
    metrics["ipsum_vs_weekly_plateau_gap_se"] = 0.0
    return metrics


def _series(curves: dict[str, list[Point]]) -> list[dict]:
    return [
        {
            "cycle": point.cycle,
            "system": system,
            "value": point.test_recall,
            "selection_rate": point.selection_rate,
            "repo_age_days": point.repo_age_days,
        }
        for system, curve in curves.items()
        for point in curve
    ]


def _gap_series(curves: dict[str, list[Point]]) -> list[dict]:
    return [
        {
            "cycle": cycle,
            "system": "ipsum_minus_data_matched_control",
            "value": gap,
        }
        for cycle, gap in _gap_points(curves)
    ]


def _gap_points(curves: dict[str, list[Point]]) -> list[tuple[int, float]]:
    by_baseline = {point.cycle: point for point in curves["data_matched_control"]}
    rows = []
    for point in curves["ipsum"]:
        baseline = by_baseline.get(point.cycle)
        if baseline is not None:
            rows.append((point.cycle, point.test_recall - baseline.test_recall))
    return rows


def _gap_trend(points: list[tuple[int, float]]) -> float:
    return _slope(
        [
            Point(cycle=cycle, repo_age_days=float(cycle), test_recall=gap, selection_rate=0.0)
            for cycle, gap in points
        ]
    )


def _abstractions(system: RealDataIpsumSelector, last_cycle: int) -> dict:
    return {
        "snapshots": [
            {
                "cycle": last_cycle,
                "abstractions": [
                    {
                        "name": abstraction.name,
                        "complexity": abstraction.complexity,
                        "usefulness": abstraction.usefulness,
                        "admitted_cycle": system._admitted_cycle.get(abstraction.name),
                        "evicted_cycle": None,
                        "payload_summary": _payload_summary(abstraction),
                    }
                    for abstraction in system.store
                ],
            }
        ]
    }


def _cycle_ll_gain(
    abstraction: Abstraction,
    examples: list[PredictionExample],
    base_rates: list[float],
    abstraction_rates: list[float],
) -> float:
    files = _files(abstraction)
    touched = [example for example in examples if example.changed_files & files]
    if not touched:
        return 0.0
    eps = 1e-6
    total = 0.0
    for example in touched:
        base = min(1.0 - eps, max(eps, base_rates[example.test]))
        improved = max(base, abstraction_rates[example.test])
        improved = min(1.0 - eps, max(eps, improved))
        total += (
            log(improved) - log(base)
            if example.failed
            else log(1.0 - improved) - log(1.0 - base)
        )
    return total / len(touched)


def _candidate_ll(
    candidate: Abstraction,
    train: list[PredictionExample],
    heldout: list[PredictionExample],
    base_rates: list[float],
    n_tests: int,
) -> float:
    files = _files(candidate)
    counts = [[0, 0] for _ in range(n_tests)]
    for example in train:
        if example.changed_files & files:
            counts[example.test][0] += int(example.failed)
            counts[example.test][1] += 1
    rates = [
        _smoothed_rate(fails, total, base_rates[test])
        for test, (fails, total) in enumerate(counts)
    ]
    eps = 1e-6
    total = 0.0
    for example in heldout:
        p = base_rates[example.test]
        if example.changed_files & files:
            p = max(p, rates[example.test])
        p = min(1.0 - eps, max(eps, p))
        total += log(p) if example.failed else log(1.0 - p)
    return total / len(heldout) if heldout else 0.0


def _ll_from_rates(examples: list[PredictionExample], rates: list[float]) -> float:
    eps = 1e-6
    total = 0.0
    for example in examples:
        p = min(1.0 - eps, max(eps, rates[example.test]))
        total += log(p) if example.failed else log(1.0 - p)
    return total / len(examples) if examples else 0.0


def _smoothed_rate(fails: int, total: int, prior: float) -> float:
    strength = 2.0
    return (fails + prior * strength) / (total + strength)


def _coarsen_changed_files(
    changed_files,
    *,
    granularity: str,
    depth: int,
) -> frozenset[str]:
    if granularity == "path":
        return frozenset(str(path) for path in changed_files)
    if depth < 1:
        raise ValueError("change_path_depth must be >= 1")
    tokens = set()
    for path in changed_files:
        text = str(path).strip("/")
        if not text:
            continue
        parts = [part for part in text.split("/") if part]
        if granularity == "directory":
            dirs = parts[:-1]
            token_parts = dirs[:depth] if dirs else parts[:1]
            tokens.add("/".join(token_parts))
        elif granularity == "java_package":
            tokens.add(_java_package_token(parts, depth))
        else:
            raise ValueError(
                "change_granularity must be one of: path, directory, java_package"
            )
    return frozenset(token for token in tokens if token)


def _java_package_token(parts: list[str], depth: int) -> str:
    if parts[-1].endswith(".java"):
        for marker in (("src", "main", "java"), ("src", "test", "java")):
            marker_len = len(marker)
            for idx in range(0, max(0, len(parts) - marker_len)):
                if tuple(parts[idx : idx + marker_len]) == marker:
                    package_parts = parts[idx + marker_len : -1]
                    if package_parts:
                        return ".".join(package_parts[:depth])
        dirs = parts[:-1]
        if dirs:
            return ".".join(dirs[-depth:])
    return "/".join((parts[:-1] or parts[:1])[:depth])


def _candidate_support(
    candidate: Abstraction,
    commits: list[tuple[int, frozenset[object]]],
    recent_start: int,
) -> int:
    files = _files(candidate)
    return sum(
        1
        for commit_cycle, changed_files in commits
        if commit_cycle >= recent_start and changed_files & files
    )


def _percentile(values, q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    pos = (len(values) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(values) - 1)
    weight = pos - lower
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _update_counts(counts: list[list[int]], examples: list[PredictionExample]) -> None:
    for example in examples:
        counts[example.test][0] += int(example.failed)
        counts[example.test][1] += 1


def _rates_from_counts(counts: list[list[int]]) -> list[float]:
    strength = 2.0
    prior = 0.05
    return [
        (fails + prior * strength) / (total + strength)
        for fails, total in counts
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("sling", "okhttp", "sonarqube"))
    parser.add_argument("--project-csv", required=True)
    parser.add_argument("--built-commits-csv", required=True)
    parser.add_argument("--patches-csv", required=True)
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=RTPTorrentRealDataConfig.eval_interval)
    parser.add_argument("--eval-window", type=int, default=RTPTorrentRealDataConfig.eval_window)
    parser.add_argument(
        "--change-granularity",
        choices=("path", "directory", "java_package"),
        default=RTPTorrentRealDataConfig.change_granularity,
    )
    parser.add_argument(
        "--change-depth",
        type=int,
        default=RTPTorrentRealDataConfig.change_path_depth,
    )
    parser.add_argument("--min-support", type=int, default=RTPTorrentRealDataConfig.min_support)
    parser.add_argument(
        "--cochange-threshold",
        type=float,
        default=RTPTorrentRealDataConfig.cochange_threshold,
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=RTPTorrentRealDataConfig.max_candidates,
    )
    parser.add_argument(
        "--admission-interval",
        type=int,
        default=RTPTorrentRealDataConfig.admission_interval,
    )
    parser.add_argument(
        "--admission-warmup",
        type=int,
        default=RTPTorrentRealDataConfig.admission_warmup,
    )
    parser.add_argument(
        "--validation-cycles",
        type=int,
        default=RTPTorrentRealDataConfig.validation_cycles,
    )
    parser.add_argument(
        "--adaptation-window",
        type=int,
        default=RTPTorrentRealDataConfig.adaptation_window,
    )
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> RTPTorrentRealDataConfig:
    return RTPTorrentRealDataConfig(
        dataset=args.dataset,
        project_csv=args.project_csv,
        built_commits_csv=args.built_commits_csv,
        patches_csv=args.patches_csv,
        max_cycles=args.max_cycles,
        eval_interval=args.eval_interval,
        eval_window=args.eval_window,
        change_granularity=args.change_granularity,
        change_path_depth=args.change_depth,
        min_support=args.min_support,
        cochange_threshold=args.cochange_threshold,
        max_candidates=args.max_candidates,
        admission_interval=args.admission_interval,
        admission_warmup=args.admission_warmup,
        validation_cycles=args.validation_cycles,
        adaptation_window=args.adaptation_window,
    )


if __name__ == "__main__":
    result = run(_config_from_args(_parse_args()))
    print(result["run_dir"])
    print(
        "ipsum_vs_data_matched_slope_gap="
        f"{result['metrics']['ipsum_vs_data_matched_slope_gap']:.6f}"
    )
