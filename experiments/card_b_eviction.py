"""Card B: eviction / anti-staleness on the synth drift harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import log
from pathlib import Path
from statistics import median
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    _ll_for_abstractions,
    _payload_summary,
    _slope,
    _top_tests,
)
from ipsum.abstractions import Abstraction, AbstractionStore
from ipsum.synth import Synth, SynthConfig


@dataclass(frozen=True)
class CardBEvictionConfig:
    n_files: int = 120
    n_tests: int = 60
    n_clusters: int = 10
    cycles: int = 1360
    drift_schedule: tuple[int, ...] = (200, 360, 520, 680, 840, 1000, 1160)
    selection_rate_cap: float = 0.33
    p_hit: float = 0.95
    p_flaky: float = 0.0
    seed: int = 733
    seeds: tuple[int, ...] = (733, 734, 735, 736, 737)
    admission_warmup: int = 240
    admission_interval: int = 40
    validation_cycles: int = 40
    adaptation_window: int = 140
    cochange_threshold: float = 0.18
    min_support: int = 3
    max_candidates: int = 96
    decay: float = 0.99
    complexity_per_file: float = 0.0005
    eviction_margin: float = 0.0
    eviction_grace_cycles: int = 80
    drift_suspicion_window: int = 30
    drift_suspicion_drop: float = 0.16
    drift_suspicion_cooldown: int = 40
    recovery_window: int = 30
    pre_drift_window: int = 40
    post_drift_window: int = 120
    recovery_epsilon: float = 0.04
    plateau_window: int = 40
    stable_margin: float = 0.02
    min_mean_plateau_advantage: float = 0.05
    min_plateau_trend_advantage: float = 0.0
    min_eviction_precision: float = 0.7
    min_eviction_recall: float = 0.5
    abstraction_lift_weight: float = 0.35
    stale_pre_jaccard: float = 0.45
    stale_drop_jaccard: float = 0.08


@dataclass(frozen=True)
class OracleDrift:
    cycle: int
    before_clusters: tuple[frozenset[int], ...]
    after_clusters: tuple[frozenset[int], ...]


class NoStoreControl(DataMatchedControl):
    system_name = "no_store"


class EvictionSelector(DataMatchedControl):
    """Small abstraction selector whose only variant is eviction on/off."""

    def __init__(
        self,
        *,
        system_name: str,
        n_tests: int,
        selection_rate_cap: float,
        evict: bool,
        config: CardBEvictionConfig,
    ) -> None:
        super().__init__(n_tests, selection_rate_cap)
        self.system_name = system_name
        self.evict = evict
        self.config = config
        self.store = AbstractionStore(
            decay=config.decay,
            min_support=config.min_support,
            cochange_threshold=config.cochange_threshold,
            max_candidates=config.max_candidates,
            complexity_per_file=config.complexity_per_file,
        )
        self._rates_by_abstraction: dict[str, list[float]] = {}
        self._commits: list[tuple[int, frozenset[object]]] = []
        self._events: list[dict] = []
        self._admission_round = 0
        self._birth_clusters: dict[str, tuple[frozenset[int], ...]] = {}
        self._birth_files: dict[str, frozenset[object]] = {}
        self._admitted_cycle: dict[str, int] = {}
        self._evicted_cycle: dict[str, int] = {}
        self._last_clusters: tuple[frozenset[int], ...] = ()
        self._recent_recalls: list[tuple[int, float]] = []
        self._drift_suspected_until = -1

    @property
    def events(self) -> list[dict]:
        return list(self._events)

    def set_oracle_clusters(self, clusters: tuple[frozenset[int], ...]) -> None:
        self._last_clusters = clusters

    def observe_recall(self, cycle: int, recall: float) -> None:
        """Update a no-oracle drift suspicion trace from recent recall drops."""
        window = self.config.drift_suspicion_window
        self._recent_recalls.append((cycle, recall))
        if cycle <= self._drift_suspected_until:
            return
        if len(self._recent_recalls) < 2 * window:
            return
        previous = [value for _, value in self._recent_recalls[-2 * window : -window]]
        recent = [value for _, value in self._recent_recalls[-window:]]
        if _mean(previous) - _mean(recent) < self.config.drift_suspicion_drop:
            return
        suspected_until = cycle + self.config.drift_suspicion_cooldown
        self._drift_suspected_until = suspected_until
        self._events.append(
            {
                "cycle": cycle,
                "type": "drift_suspect",
                "system": self.system_name,
                "detail": f"until={suspected_until}",
            }
        )

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
        self._commits.append((cycle.cycle, changed_files))
        self._base_rates = _base_rates(self.examples, self.n_tests)
        self._reinforce(examples)
        if self.evict:
            self._evict(cycle.cycle)
        if (
            cycle.cycle >= self.config.admission_warmup
            and cycle.cycle % self.config.admission_interval == 0
        ):
            self._admit_recent(cycle.cycle)

    def _admit_recent(self, cycle: int) -> None:
        recent_start = max(0, cycle - self.config.adaptation_window + 1)
        validation_start = max(recent_start, cycle - self.config.validation_cycles + 1)
        train = [
            example
            for example in self.examples
            if recent_start <= example.cycle < validation_start
        ]
        heldout = [
            example
            for example in self.examples
            if validation_start <= example.cycle <= cycle
        ]
        if not train or not heldout:
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

        for candidate in proposal_store.candidates():
            candidate = self._versioned_candidate(candidate)
            ll_gain = _ll_for_abstractions([candidate], train, heldout, self.n_tests) - (
                _ll_for_abstractions([], train, heldout, self.n_tests)
            )
            if self.store.admit(candidate, ll_gain):
                self._birth_clusters[candidate.name] = self._last_clusters
                self._birth_files[candidate.name] = _files(candidate)
                self._admitted_cycle[candidate.name] = cycle
                self._events.append(
                    {
                        "cycle": cycle,
                        "type": "admit",
                        "system": self.system_name,
                        "name": candidate.name,
                        "detail": f"ll_gain={ll_gain:.6f}",
                    }
                )
        self._refresh_rates()
        self._admission_round += 1

    def _versioned_candidate(self, candidate: Abstraction) -> Abstraction:
        return Abstraction(
            name=f"{candidate.name}_r{self._admission_round}",
            payload=candidate.payload,
            complexity=candidate.complexity,
            usefulness=candidate.usefulness,
        )

    def _reinforce(self, examples: list[PredictionExample]) -> None:
        if not self.store:
            return
        for abstraction in list(self.store):
            rates = self._rates_by_abstraction.get(abstraction.name)
            if rates is None:
                continue
            gain = _cycle_ll_gain(
                abstraction,
                examples,
                self._base_rates,
                rates,
            )
            if gain != 0.0:
                self.store.reinforce(abstraction.name, gain)

    def _evict(self, cycle: int) -> None:
        protected = {
            name
            for name, admitted in self._admitted_cycle.items()
            if cycle - admitted < self.config.eviction_grace_cycles
        }
        if cycle <= self._drift_suspected_until:
            protected = set()
        evicted = self.store.decay_and_evict(protected=protected)
        for name in evicted:
            self._rates_by_abstraction.pop(name, None)
            self._evicted_cycle[name] = cycle
            self._events.append(
                {
                    "cycle": cycle,
                    "type": "evict",
                    "system": self.system_name,
                    "name": name,
                }
            )

    def _refresh_rates(self) -> None:
        recent_start = max(0, self.examples[-1].cycle - self.config.adaptation_window + 1)
        recent = [example for example in self.examples if example.cycle >= recent_start]
        new_rates = _abstraction_rates(list(self.store), recent, self.n_tests)
        self._rates_by_abstraction.update(new_rates)

    def _predict_with_store(self, changed_files: frozenset[object], test: int) -> float:
        base = self._base_rates[test]
        lift = 0.0
        for abstraction in self.store:
            files = _files(abstraction)
            if changed_files & files:
                rates = self._rates_by_abstraction.get(abstraction.name, self._base_rates)
                lift += max(0.0, rates[test] - base)
        return min(1.0, base + self.config.abstraction_lift_weight * lift)


def run(config: CardBEvictionConfig = CardBEvictionConfig()) -> dict:
    seeds = config.seeds or (config.seed,)
    seed_results = [_run_single_seed(replace(config, seed=seed), seed) for seed in seeds]
    metrics = _aggregate_metrics(seed_results, config)
    plateau_series = _aggregate_plateau_series(seed_results)
    recovery_summary_series = _aggregate_recovery_summary_series(seed_results)
    representative = seed_results[0]

    created = utc_now()
    run_id = make_run_id("B", "synth", created)
    run_dir = write_run(
        run_id=run_id,
        card="B",
        dataset="synth",
        created=created,
        config=asdict(config),
        slope={
            "metric_name": "post_drift_plateau_accuracy",
            "selection_rate_cap": config.selection_rate_cap,
            "series": plateau_series,
            "recovery_summary_series": recovery_summary_series,
        },
        metrics=metrics,
        controls={
            "min_mean_plateau_advantage": config.min_mean_plateau_advantage,
            "min_plateau_trend_advantage": config.min_plateau_trend_advantage,
            "min_eviction_precision": config.min_eviction_precision,
            "min_eviction_recall": config.min_eviction_recall,
            "stable_margin": config.stable_margin,
        },
        abstractions=representative["abstractions"],
        events={"events": representative["events"]},
        headline_metric=metrics["mean_plateau_advantage_mean"],
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "metrics": metrics,
        "recovery": representative["recovery"],
        "plateaus": representative["plateaus"],
        "seed_results": seed_results,
    }


def _run_single_seed(config: CardBEvictionConfig, seed: int) -> dict:
    timeline, drift_oracles, initial_clusters = _synth_timeline_with_oracles(config)
    systems = [
        NoStoreControl(config.n_tests, config.selection_rate_cap),
        EvictionSelector(
            system_name="append_only",
            n_tests=config.n_tests,
            selection_rate_cap=config.selection_rate_cap,
            evict=False,
            config=config,
        ),
        EvictionSelector(
            system_name="ipsum_evict",
            n_tests=config.n_tests,
            selection_rate_cap=config.selection_rate_cap,
            evict=True,
            config=config,
        ),
    ]
    per_cycle = _run_systems(systems, timeline, drift_oracles, initial_clusters)
    recovery = {
        name: _recovery_by_drift(series, config)
        for name, series in per_cycle.items()
    }
    plateaus = {
        name: _post_drift_plateaus(series, config)
        for name, series in per_cycle.items()
    }
    stable = _stable_period_accuracy(per_cycle, config)
    quality = _eviction_quality(systems[-1], drift_oracles, config)
    evict_plateau_trend = _trend(plateaus["ipsum_evict"], "plateau_accuracy")
    append_plateau_trend = _trend(plateaus["append_only"], "plateau_accuracy")
    plateau_trend_advantage = evict_plateau_trend - append_plateau_trend
    plateau_advantages = [
        evict["plateau_accuracy"] - append["plateau_accuracy"]
        for evict, append in zip(
            plateaus["ipsum_evict"],
            plateaus["append_only"],
            strict=True,
        )
    ]
    mean_plateau_advantage = _mean(plateau_advantages)
    stable_delta = stable["ipsum_evict"] - stable["append_only"]

    metrics = {
        "seed": float(seed),
        "n_drift_epochs": float(len(config.drift_schedule)),
        "ipsum_evict_plateau_trend": evict_plateau_trend,
        "append_only_plateau_trend": append_plateau_trend,
        "plateau_trend_advantage": plateau_trend_advantage,
        "mean_plateau_advantage": mean_plateau_advantage,
        "ipsum_evict_stable_accuracy": stable["ipsum_evict"],
        "append_only_stable_accuracy": stable["append_only"],
        "stable_accuracy_delta": stable_delta,
        "mean_eviction_latency": (
            _mean(quality["latencies"]) if quality["latencies"] else float(config.cycles)
        ),
        "eviction_precision": quality["precision"],
        "eviction_recall": quality["recall"],
    }
    for system_name, rows in recovery.items():
        metrics[f"{system_name}_recovered_fraction"] = _recovered_fraction(rows)
        metrics[f"{system_name}_median_recovery_time_recovered"] = (
            _median_recovered_time(rows)
        )
        for row in rows:
            suffix = row["epoch"]
            metrics[f"{system_name}_recovery_time_epoch_{suffix}"] = row["recovery_time"]
            metrics[f"{system_name}_recovered_epoch_{suffix}"] = float(row["recovered"])
    for system_name, rows in plateaus.items():
        for row in rows:
            metrics[f"{system_name}_plateau_accuracy_epoch_{row['epoch']}"] = row[
                "plateau_accuracy"
            ]

    event_rows = [
        {"cycle": drift, "type": "drift", "detail": f"epoch={idx}"}
        for idx, drift in enumerate(config.drift_schedule, start=1)
    ]
    for system in systems:
        if isinstance(system, EvictionSelector):
            event_rows.extend(system.events)
    event_rows.sort(key=lambda event: (event["cycle"], event["type"], event.get("name", "")))

    return {
        "seed": seed,
        "metrics": metrics,
        "recovery": recovery,
        "plateaus": plateaus,
        "events": event_rows,
        "abstractions": _abstraction_snapshot(systems[-1], config),
    }


def _synth_timeline_with_oracles(
    config: CardBEvictionConfig,
) -> tuple[list[Cycle], list[OracleDrift], tuple[frozenset[int], ...]]:
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
    timeline: list[Cycle] = []
    drifts: list[OracleDrift] = []
    initial_clusters = tuple(world.true_clusters())
    drift_set = set(config.drift_schedule)
    for _ in range(config.cycles):
        before = tuple(world.true_clusters())
        commit, outcomes = world.step()
        after = tuple(world.true_clusters())
        if commit.cycle in drift_set:
            drifts.append(OracleDrift(commit.cycle, before, after))
        timeline.append(
            Cycle(
                cycle=commit.cycle,
                commit=SelectionCommit(commit.cycle, frozenset(commit.changed_files)),
                outcomes=tuple(SelectionOutcome(outcome.test, outcome.failed) for outcome in outcomes),
            )
        )
    return timeline, drifts, initial_clusters


def _aggregate_metrics(seed_results: list[dict], config: CardBEvictionConfig) -> dict:
    per_seed = [result["metrics"] for result in seed_results]
    metrics = {
        "n_seeds": float(len(seed_results)),
        "n_drift_epochs": float(len(config.drift_schedule)),
    }
    for field in [
        "mean_plateau_advantage",
        "plateau_trend_advantage",
        "stable_accuracy_delta",
        "eviction_precision",
        "eviction_recall",
        "mean_eviction_latency",
        "append_only_recovered_fraction",
        "ipsum_evict_recovered_fraction",
        "append_only_median_recovery_time_recovered",
        "ipsum_evict_median_recovery_time_recovered",
    ]:
        values = [row[field] for row in per_seed]
        metrics[f"{field}_mean"] = _mean(values)
        metrics[f"{field}_variance"] = _variance(values)

    for system_name in ["no_store", "append_only", "ipsum_evict"]:
        for epoch in range(1, len(config.drift_schedule) + 1):
            field = f"{system_name}_plateau_accuracy_epoch_{float(epoch)}"
            values = [row[field] for row in per_seed]
            metrics[f"{field}_mean"] = _mean(values)
            metrics[f"{field}_variance"] = _variance(values)
            recovery_field = f"{system_name}_recovered_epoch_{float(epoch)}"
            metrics[f"{recovery_field}_mean"] = _mean(row[recovery_field] for row in per_seed)

    for seed_result in seed_results:
        seed = seed_result["seed"]
        metrics[f"seed_{seed}_mean_plateau_advantage"] = seed_result["metrics"][
            "mean_plateau_advantage"
        ]
        metrics[f"seed_{seed}_eviction_precision"] = seed_result["metrics"][
            "eviction_precision"
        ]
        metrics[f"seed_{seed}_eviction_recall"] = seed_result["metrics"]["eviction_recall"]

    metrics["card_b_passed"] = float(
        metrics["mean_plateau_advantage_mean"] >= config.min_mean_plateau_advantage
        and metrics["plateau_trend_advantage_mean"] >= config.min_plateau_trend_advantage
        and metrics["stable_accuracy_delta_mean"] >= -config.stable_margin
        and metrics["eviction_precision_mean"] >= config.min_eviction_precision
        and metrics["eviction_recall_mean"] >= config.min_eviction_recall
    )
    return metrics


def _aggregate_plateau_series(seed_results: list[dict]) -> list[dict]:
    rows: list[dict] = []
    systems = ["no_store", "append_only", "ipsum_evict"]
    n_epochs = len(seed_results[0]["plateaus"]["append_only"])
    for system_name in systems:
        for epoch_idx in range(n_epochs):
            values = [
                result["plateaus"][system_name][epoch_idx]["plateau_accuracy"]
                for result in seed_results
            ]
            rows.append(
                {
                    "cycle": float(epoch_idx + 1),
                    "system": system_name,
                    "value": _mean(values),
                    "variance": _variance(values),
                }
            )
    return rows


def _aggregate_recovery_summary_series(seed_results: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for system_name in ["no_store", "append_only", "ipsum_evict"]:
        recovered_values = [
            result["metrics"][f"{system_name}_recovered_fraction"]
            for result in seed_results
        ]
        median_values = [
            result["metrics"][f"{system_name}_median_recovery_time_recovered"]
            for result in seed_results
        ]
        rows.append(
            {
                "cycle": 1,
                "system": f"{system_name}_recovered_fraction",
                "value": _mean(recovered_values),
                "variance": _variance(recovered_values),
            }
        )
        rows.append(
            {
                "cycle": 1,
                "system": f"{system_name}_median_recovery_time_recovered",
                "value": _mean(median_values),
                "variance": _variance(median_values),
            }
        )
    return rows


def _run_systems(
    systems: list[DataMatchedControl],
    timeline: list[Cycle],
    drifts: list[OracleDrift],
    initial_clusters: tuple[frozenset[int], ...],
) -> dict[str, list[float]]:
    drift_after = {drift.cycle: drift.after_clusters for drift in drifts}
    latest_clusters = initial_clusters
    per_cycle: dict[str, list[float]] = {system.system_name: [] for system in systems}
    for cycle in timeline:
        if cycle.cycle in drift_after:
            latest_clusters = drift_after[cycle.cycle]
        for system in systems:
            if isinstance(system, EvictionSelector):
                system.set_oracle_clusters(latest_clusters)
            chosen = system.select(cycle)
            failed_tests = {outcome.test for outcome in cycle.outcomes if outcome.failed}
            recall = len(chosen & failed_tests) / len(failed_tests) if failed_tests else 1.0
            per_cycle[system.system_name].append(recall)
            if isinstance(system, EvictionSelector):
                system.observe_recall(cycle.cycle, recall)
            system.observe(cycle)
    return per_cycle


def _recovery_by_drift(
    series: list[float],
    config: CardBEvictionConfig,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for epoch, drift_cycle in enumerate(config.drift_schedule, start=1):
        pre_start = max(0, drift_cycle - config.pre_drift_window)
        pre_accuracy = _mean(series[pre_start:drift_cycle])
        target = max(0.0, pre_accuracy - config.recovery_epsilon)
        search_end = min(len(series), drift_cycle + config.post_drift_window)
        recovered_at = search_end
        recovered = False
        for cycle in range(drift_cycle, search_end):
            window = series[cycle : min(search_end, cycle + config.recovery_window)]
            if len(window) < config.recovery_window:
                break
            if _mean(window) >= target:
                recovered_at = cycle
                recovered = True
                break
        rows.append(
            {
                "epoch": float(epoch),
                "drift_cycle": float(drift_cycle),
                "pre_accuracy": pre_accuracy,
                "target_accuracy": target,
                "recovery_time": float(recovered_at - drift_cycle),
                "recovered": recovered,
            }
        )
    return rows


def _post_drift_plateaus(
    series: list[float],
    config: CardBEvictionConfig,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for epoch, drift_cycle in enumerate(config.drift_schedule, start=1):
        start = min(len(series), drift_cycle + config.post_drift_window - config.plateau_window)
        end = min(len(series), drift_cycle + config.post_drift_window)
        rows.append(
            {
                "epoch": float(epoch),
                "drift_cycle": float(drift_cycle),
                "plateau_accuracy": _mean(series[start:end]),
            }
        )
    return rows


def _stable_period_accuracy(
    per_cycle: dict[str, list[float]],
    config: CardBEvictionConfig,
) -> dict[str, float]:
    excluded: set[int] = set()
    for drift in config.drift_schedule:
        excluded.update(range(drift, min(config.cycles, drift + config.post_drift_window)))
    stable_indices = [
        idx
        for idx in range(config.admission_warmup, config.cycles)
        if idx not in excluded
    ]
    return {
        name: _mean(series[idx] for idx in stable_indices)
        for name, series in per_cycle.items()
    }


def _eviction_latencies(
    system: EvictionSelector,
    drifts: list[OracleDrift],
    config: CardBEvictionConfig,
) -> tuple[list[float], float]:
    quality = _eviction_quality(system, drifts, config)
    return quality["latencies"], quality["recall"]


def _eviction_quality(
    system: EvictionSelector,
    drifts: list[OracleDrift],
    config: CardBEvictionConfig,
) -> dict:
    evictions = [
        event
        for event in system.events
        if event["type"] == "evict"
    ]
    stale_event_evictions = 0
    classified_evictions = 0
    drift_by_cycle = sorted(drifts, key=lambda drift: drift.cycle)
    for event in evictions:
        name = event["name"]
        birth_clusters = system._birth_clusters.get(name)
        if birth_clusters is None:
            continue
        latest_drift = _latest_drift_before(drift_by_cycle, event["cycle"])
        classified_evictions += 1
        if latest_drift is not None and _is_stale(
            name,
            system,
            birth_clusters,
            latest_drift.after_clusters,
            config,
        ):
            stale_event_evictions += 1

    stale_evictions = 0
    stale_total = 0
    latencies: list[float] = []
    for drift in drifts:
        stale_names = [
            name
            for name, birth_clusters in system._birth_clusters.items()
            if system._admitted_cycle[name] < drift.cycle
            and system._evicted_cycle.get(name, config.cycles + 1) >= drift.cycle
            if _is_stale(name, system, birth_clusters, drift.after_clusters, config)
        ]
        stale_total += len(stale_names)
        first_latency = None
        for name in stale_names:
            evicted = next(
                (
                    event
                    for event in evictions
                    if event["name"] == name and event["cycle"] >= drift.cycle
                ),
                None,
            )
            if evicted is None:
                continue
            stale_evictions += 1
            latency = float(evicted["cycle"] - drift.cycle)
            first_latency = latency if first_latency is None else min(first_latency, latency)
        if first_latency is not None:
            latencies.append(first_latency)
    recall = stale_evictions / stale_total if stale_total else 1.0
    precision = stale_event_evictions / classified_evictions if classified_evictions else 1.0
    return {
        "latencies": latencies,
        "precision": precision,
        "recall": recall,
        "classified_evictions": classified_evictions,
        "stale_evictions": stale_event_evictions,
        "stale_total": stale_total,
    }


def _latest_drift_before(drifts: list[OracleDrift], cycle: int) -> OracleDrift | None:
    latest = None
    for drift in drifts:
        if drift.cycle > cycle:
            break
        latest = drift
    return latest


def _is_stale(
    name: str,
    system: EvictionSelector,
    before_clusters: tuple[frozenset[int], ...],
    after_clusters: tuple[frozenset[int], ...],
    config: CardBEvictionConfig,
) -> bool:
    files = {int(file_id) for file_id in system._birth_files.get(name, frozenset())}
    pre = _best_jaccard(files, before_clusters)
    post = _best_jaccard(files, after_clusters)
    return pre >= config.stale_pre_jaccard and pre - post >= config.stale_drop_jaccard


def _best_jaccard(files: set[int], clusters: tuple[frozenset[int], ...]) -> float:
    if not files:
        return 0.0
    return max((_jaccard(files, set(cluster)) for cluster in clusters), default=0.0)


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


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
    total = 0.0
    eps = 1e-6
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


def _trend(rows: list[dict[str, float]], field: str) -> float:
    return _slope(
        [
            Point(
                cycle=int(row["epoch"]),
                repo_age_days=row["epoch"],
                test_recall=row[field],
                selection_rate=0.0,
            )
            for row in rows
        ]
    )


def _abstraction_snapshot(system: EvictionSelector, config: CardBEvictionConfig) -> dict:
    return {
        "snapshots": [
            {
                "cycle": config.cycles - 1,
                "abstractions": [
                    {
                        "name": abstraction.name,
                        "complexity": abstraction.complexity,
                        "usefulness": abstraction.usefulness,
                        "admitted_cycle": None,
                        "evicted_cycle": None,
                        "payload_summary": _payload_summary(abstraction),
                    }
                    for abstraction in system.store
                ],
            }
        ]
    }


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _variance(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    avg = _mean(values)
    return sum((value - avg) ** 2 for value in values) / len(values)


def _recovered_fraction(rows: list[dict]) -> float:
    return _mean(float(row["recovered"]) for row in rows)


def _median_recovered_time(rows: list[dict]) -> float:
    recovered = [row["recovery_time"] for row in rows if row["recovered"]]
    return float(median(recovered)) if recovered else 0.0


if __name__ == "__main__":
    result = run()
    print(result["run_dir"])
    print(f"card_b_passed={result['metrics']['card_b_passed']:.0f}")
    print(
        "mean_plateau_advantage="
        f"{result['metrics']['mean_plateau_advantage_mean']:.6f}"
    )
