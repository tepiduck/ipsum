"""Card I: instrument self-check with positive and negative controls."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.artifacts import make_run_id, utc_now, write_run
from experiments.compounding import (
    Cycle,
    DataMatchedControl,
    IpsumSelector,
    Point,
    PredictionExample,
    WeeklyRetrainControl,
    _abstraction_rates,
    _changed_files,
    _examples,
    _last_admission_cycle,
    _ll_for_abstractions,
    _payload_summary,
    _slope,
    _synth_timeline,
    compounding_curve,
)
from ipsum.abstractions import AbstractionStore


@dataclass(frozen=True)
class InstrumentSelfCheckConfig:
    n_files: int = 96
    n_tests: int = 48
    n_clusters: int = 8
    cycles: int = 760
    eval_interval: int = 40
    eval_window: int = 80
    selection_rate_cap: float = 0.33
    p_hit: float = 0.9
    p_flaky: float = 0.01
    drift_schedule: tuple[int, ...] = (160, 320, 480, 640)
    seed: int = 412
    admission_interval: int = 40
    admission_warmup: int = 280
    validation_cycles: int = 40
    cochange_threshold: float = 0.18
    min_support: int = 3
    weekly_retrain_interval: int = 80
    weekly_retrain_window: int = 240
    positive_adaptation_window: int = 120
    positive_min_slope_gap: float = 0.0004
    positive_min_half_gap_delta: float = 0.04
    positive_min_late_over_mid_gap: float = 0.02
    negative_max_abs_slope_gap: float = 1e-12
    negative_max_abs_gap: float = 1e-12


class DisabledIpsumControl(DataMatchedControl):
    """Byte-for-byte data-matched control behavior, named as ipsum."""

    system_name = "ipsum"


class DriftAdaptiveIpsumSelector(IpsumSelector):
    """Positive-control selector with only abstraction-store adaptation changed.

    The data stream and base cumulative rates are identical to the data-matched
    control. The planted edge is that admitted abstraction rates are refreshed
    from the recent drift epoch, so the store can relearn structure faster than
    cumulative raw test rates after repeated synth drifts.
    """

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
        adaptation_window: int,
    ) -> None:
        super().__init__(
            n_tests,
            selection_rate_cap,
            admission_interval=admission_interval,
            admission_warmup=admission_warmup,
            validation_cycles=validation_cycles,
            cochange_threshold=cochange_threshold,
            min_support=min_support,
        )
        self.adaptation_window = adaptation_window
        self._commits: list[tuple[int, frozenset[object]]] = []
        self._cochange_threshold = cochange_threshold
        self._min_support = min_support

    def observe(self, cycle: Cycle) -> None:
        self.examples.extend(_examples(cycle))
        self._commits.append((cycle.cycle, _changed_files(cycle)))
        self._base_rates = _base_rates_like_control(self.examples, self.n_tests)
        if (
            cycle.cycle >= self.admission_warmup
            and cycle.cycle % self.admission_interval == 0
        ):
            self._readmit()

    def _readmit(self) -> None:
        last_cycle = self.examples[-1].cycle
        recent_start = max(0, last_cycle - self.adaptation_window + 1)
        validation_start = max(recent_start, last_cycle - self.validation_cycles + 1)
        train = [
            example
            for example in self.examples
            if recent_start <= example.cycle < validation_start
        ]
        heldout = [
            example
            for example in self.examples
            if validation_start <= example.cycle <= last_cycle
        ]
        if not train or not heldout:
            return

        store = AbstractionStore(
            cochange_threshold=self._cochange_threshold,
            min_support=self._min_support,
        )
        for cycle, changed_files in self._commits:
            if cycle >= recent_start:
                store.observe_commit(changed_files)
        for candidate in store.candidates():
            ll_gain = _ll_for_abstractions([candidate], train, heldout, self.n_tests) - (
                _ll_for_abstractions([], train, heldout, self.n_tests)
            )
            store.admit(candidate, ll_gain)
        self.store = store
        recent_examples = [
            example
            for example in self.examples
            if recent_start <= example.cycle <= last_cycle
        ]
        self._rates_by_abstraction = _abstraction_rates(
            list(self.store),
            recent_examples,
            self.n_tests,
        )


def score_test_recall_at_selection_rate(
    cycles: list[Cycle],
    selections: list[set[int]],
    n_tests: int,
) -> Point:
    """Score TestRecall and SelectionRate for precomputed selections."""
    if len(cycles) != len(selections):
        raise ValueError("cycles and selections must have the same length")
    failures = 0
    caught = 0
    selected = 0
    for cycle, chosen in zip(cycles, selections, strict=True):
        failed_tests = {outcome.test for outcome in cycle.outcomes if outcome.failed}
        failures += len(failed_tests)
        caught += len(chosen & failed_tests)
        selected += len(chosen)
    total_tests = len(cycles) * n_tests
    return Point(
        cycle=cycles[-1].cycle if cycles else 0,
        repo_age_days=float(cycles[-1].cycle if cycles else 0),
        test_recall=caught / failures if failures else 1.0,
        selection_rate=selected / total_tests if total_tests else 0.0,
    )


def run_both(config: InstrumentSelfCheckConfig = InstrumentSelfCheckConfig()) -> dict:
    positive = _run_mode(config, mode="positive")
    negative = _run_mode(config, mode="negative")
    delta = positive["metrics"]["ipsum_vs_data_matched_slope_gap"] - negative["metrics"][
        "ipsum_vs_data_matched_slope_gap"
    ]
    positive["metrics"]["pos_vs_neg_slope_gap_delta"] = delta
    negative["metrics"]["pos_vs_neg_slope_gap_delta"] = delta
    positive["metrics"]["instrument_self_check_passed"] = float(
        positive["metrics"]["positive_control_passed"]
        and negative["metrics"]["negative_control_passed"]
        and delta > config.positive_min_slope_gap
    )
    negative["metrics"]["instrument_self_check_passed"] = positive["metrics"][
        "instrument_self_check_passed"
    ]

    created = utc_now()
    positive_dir = _write_mode_run(config, positive, created=created)
    negative_dir = _write_mode_run(config, negative, created=created)
    return {
        "positive": {**positive, "run_dir": str(positive_dir)},
        "negative": {**negative, "run_dir": str(negative_dir)},
    }


def _run_mode(config: InstrumentSelfCheckConfig, *, mode: str) -> dict:
    timeline = _synth_timeline(_as_synth_config(config))
    if mode not in {"positive", "negative"}:
        raise ValueError(f"unknown self-check mode: {mode}")
    systems = [
        WeeklyRetrainControl(
            config.n_tests,
            config.selection_rate_cap,
            retrain_interval=config.weekly_retrain_interval,
            retrain_window=config.weekly_retrain_window,
        ),
        DataMatchedControl(config.n_tests, config.selection_rate_cap),
        (
            DriftAdaptiveIpsumSelector(
                config.n_tests,
                config.selection_rate_cap,
                admission_interval=config.admission_interval,
                admission_warmup=config.admission_warmup,
                validation_cycles=config.validation_cycles,
                cochange_threshold=config.cochange_threshold,
                min_support=config.min_support,
                adaptation_window=config.positive_adaptation_window,
            )
            if mode == "positive"
            else DisabledIpsumControl(config.n_tests, config.selection_rate_cap)
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
        raise ValueError("not enough cycles to produce self-check curves")

    ipsum_curve = curves["ipsum"]
    control_curve = curves["data_matched_control"]
    post_warmup_curves = {
        system: _post_warmup_curve(curve, config.admission_warmup)
        for system, curve in curves.items()
    }
    if any(not curve for curve in post_warmup_curves.values()):
        raise ValueError("not enough post-warmup points to produce self-check slopes")
    gap_series = [
        {"cycle": ip.cycle, "value": ip.test_recall - ctrl.test_recall}
        for ip, ctrl in zip(ipsum_curve, control_curve, strict=True)
    ]
    gap_values = [point["value"] for point in gap_series]
    early_gap = _segment_mean(gap_values, 0, 1 / 3)
    mid_gap = _segment_mean(gap_values, 1 / 3, 2 / 3)
    late_gap = _segment_mean(gap_values, 2 / 3, 1)
    first_half_gap = _segment_mean(gap_values, 0, 1 / 2)
    second_half_gap = _segment_mean(gap_values, 1 / 2, 1)
    half_gap_delta = second_half_gap - first_half_gap
    late_over_mid_gap = late_gap - mid_gap
    gap_widened = (
        half_gap_delta >= config.positive_min_half_gap_delta
        and late_over_mid_gap >= config.positive_min_late_over_mid_gap
    )
    full_weekly_slope = _slope(curves["weekly_retrain"])
    full_control_slope = _slope(control_curve)
    full_ipsum_slope = _slope(ipsum_curve)
    full_slope_gap = full_ipsum_slope - full_control_slope
    weekly_slope = _slope(post_warmup_curves["weekly_retrain"])
    control_slope = _slope(post_warmup_curves["data_matched_control"])
    ipsum_slope = _slope(post_warmup_curves["ipsum"])
    slope_gap_value = ipsum_slope - control_slope
    max_selection_rate = max(
        point.selection_rate for curve in curves.values() for point in curve
    )
    positive_passed = (
        mode == "positive"
        and slope_gap_value > config.positive_min_slope_gap
        and gap_widened
        and max_selection_rate <= config.selection_rate_cap + 1e-9
    )
    negative_passed = (
        mode == "negative"
        and abs(slope_gap_value) <= config.negative_max_abs_slope_gap
        and abs(half_gap_delta) <= config.negative_max_abs_gap
        and abs(late_gap) <= config.negative_max_abs_gap
        and max_selection_rate <= config.selection_rate_cap + 1e-9
    )
    ipsum_system = systems[-1]
    abstractions = list(ipsum_system.store) if isinstance(ipsum_system, IpsumSelector) else []
    return {
        "mode": mode,
        "curves": curves,
        "gap_series": gap_series,
        "abstractions": abstractions,
        "metrics": {
            "slope_start_cycle": float(config.admission_warmup),
            "weekly_retrain_slope": weekly_slope,
            "data_matched_control_slope": control_slope,
            "ipsum_slope": ipsum_slope,
            "ipsum_vs_data_matched_slope_gap": slope_gap_value,
            "full_weekly_retrain_slope": full_weekly_slope,
            "full_data_matched_control_slope": full_control_slope,
            "full_ipsum_slope": full_ipsum_slope,
            "full_ipsum_vs_data_matched_slope_gap": full_slope_gap,
            "early_gap": early_gap,
            "mid_gap": mid_gap,
            "late_gap": late_gap,
            "first_half_gap": first_half_gap,
            "second_half_gap": second_half_gap,
            "half_gap_delta": half_gap_delta,
            "late_over_mid_gap": late_over_mid_gap,
            "final_gap": gap_series[-1]["value"],
            "gap_widened": float(gap_widened),
            "max_selection_rate": max_selection_rate,
            "admitted_abstractions_final": float(len(abstractions)),
            "positive_control_passed": float(positive_passed),
            "negative_control_passed": float(negative_passed),
        },
    }


def _write_mode_run(
    config: InstrumentSelfCheckConfig,
    result: dict,
    *,
    created,
) -> Path:
    mode = result["mode"]
    card = f"instrument-{'poscontrol' if mode == 'positive' else 'negcontrol'}"
    run_id = make_run_id(card, "synth", created)
    curves: dict[str, list[Point]] = result["curves"]
    series = [
        {"cycle": point.cycle, "system": system, "value": point.test_recall}
        for system, curve in curves.items()
        for point in curve
    ]
    abstractions = {
        "snapshots": [
            {
                "cycle": config.cycles - 1,
                "abstractions": [
                    {
                        "name": abstraction.name,
                        "complexity": abstraction.complexity,
                        "usefulness": abstraction.usefulness,
                        "admitted_cycle": _last_admission_cycle(_as_synth_config(config)),
                        "evicted_cycle": None,
                        "payload_summary": _payload_summary(abstraction),
                    }
                    for abstraction in result["abstractions"]
                ],
            }
        ]
    }
    return write_run(
        run_id=run_id,
        card=card,
        dataset="synth",
        created=created,
        config={**asdict(config), "mode": mode},
        slope={
            "metric_name": "test_recall_at_selrate",
            "selection_rate_cap": config.selection_rate_cap,
            "series": series,
            "gap_series": result["gap_series"],
        },
        metrics=result["metrics"],
        controls={
            "positive_min_slope_gap": config.positive_min_slope_gap,
            "positive_min_half_gap_delta": config.positive_min_half_gap_delta,
            "positive_min_late_over_mid_gap": config.positive_min_late_over_mid_gap,
            "negative_max_abs_slope_gap": config.negative_max_abs_slope_gap,
            "negative_max_abs_gap": config.negative_max_abs_gap,
        },
        abstractions=abstractions,
        headline_metric=result["metrics"]["ipsum_vs_data_matched_slope_gap"],
    )


def _as_synth_config(config: InstrumentSelfCheckConfig):
    from experiments.compounding import SynthCompoundingConfig

    fields = SynthCompoundingConfig.__dataclass_fields__
    values = {
        name: getattr(config, name)
        for name in fields
        if hasattr(config, name)
    }
    return replace(SynthCompoundingConfig(), **values)


def _segment_mean(values: list[float], start: float, end: float) -> float:
    if not values:
        return 0.0
    lo = int(len(values) * start)
    hi = int(len(values) * end)
    if end == 1:
        hi = len(values)
    hi = max(lo + 1, hi)
    segment = values[lo:hi]
    return sum(segment) / len(segment)


def _post_warmup_curve(curve: list[Point], warmup_cycle: int) -> list[Point]:
    return [point for point in curve if point.cycle >= warmup_cycle]


def _base_rates_like_control(
    examples: list[PredictionExample],
    n_tests: int,
) -> list[float]:
    from experiments.compounding import _base_rates

    return _base_rates(examples, n_tests)


if __name__ == "__main__":
    result = run_both()
    print(result["positive"]["run_dir"])
    print(result["negative"]["run_dir"])
    print(
        "positive_gap="
        f"{result['positive']['metrics']['ipsum_vs_data_matched_slope_gap']:.6f}"
    )
    print(
        "negative_gap="
        f"{result['negative']['metrics']['ipsum_vs_data_matched_slope_gap']:.6f}"
    )
