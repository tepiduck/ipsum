"""Card D coverage-skew sweep.

This is the keystone evidence check for the coarse positivity guard: the guard's
benefit should scale with coverage skew, not just clear one operating point.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import sqrt
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.artifacts import make_run_id, utc_now, write_run
from experiments.card_d_coverage import CardDConfig, _mean, _run_single_seed, _variance


SWEEP_METRICS = (
    "strict_thin_heldout_ll_delta",
    "provisional_thin_heldout_ll_delta",
    "strict_thin_undercovered_harm_reduction",
    "provisional_thin_undercovered_harm_reduction",
)


@dataclass(frozen=True)
class SkewLevel:
    name: str
    severity: float
    coverage_skew: tuple[float, ...]


@dataclass(frozen=True)
class CardDSkewSweepConfig:
    base: CardDConfig = CardDConfig()
    levels: tuple[SkewLevel, ...] = (
        SkewLevel("uniform", 0.0, (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
        SkewLevel("mild", 1.0, (3.0, 2.5, 2.0, 1.5, 1.0, 0.8, 0.7, 0.6)),
        SkewLevel("medium", 2.0, (6.0, 5.0, 4.0, 3.0, 1.0, 0.7, 0.5, 0.4)),
        SkewLevel("strong", 3.0, (10.0, 8.0, 6.0, 4.0, 0.7, 0.5, 0.35, 0.25)),
        SkewLevel("extreme", 4.0, (16.0, 12.0, 9.0, 6.0, 0.5, 0.35, 0.22, 0.14)),
    )
    monotone_tolerance: float = 0.001
    min_extreme_provisional_harm_ci_lower: float = 0.0
    min_skew_slope: float = 0.0005


def run(config: CardDSkewSweepConfig = CardDSkewSweepConfig()) -> dict:
    level_results = [_run_level(config.base, level) for level in config.levels]
    metrics = _sweep_metrics(config, level_results)
    controls = _controls(config)

    created = utc_now()
    run_id = make_run_id("D-skew-sweep", "synth", created)
    run_dir = write_run(
        run_id=run_id,
        card="D",
        dataset="synth",
        created=created,
        config={
            "base": asdict(config.base),
            "levels": [asdict(level) for level in config.levels],
            "monotone_tolerance": config.monotone_tolerance,
            "min_extreme_provisional_harm_ci_lower": (
                config.min_extreme_provisional_harm_ci_lower
            ),
            "min_skew_slope": config.min_skew_slope,
        },
        slope={
            "metric_name": "coverage_skew_sweep",
            "selection_rate_cap": 0.33,
            "series": _series(level_results),
        },
        metrics=metrics,
        controls=controls,
        headline_metric=metrics["provisional_thin_undercovered_harm_reduction_skew_slope"],
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "metrics": metrics,
        "controls": controls,
        "level_results": level_results,
    }


def _run_level(base: CardDConfig, level: SkewLevel) -> dict:
    seeds = base.seeds or (base.seed,)
    seed_results = [
        _run_single_seed(
            replace(base, seed=seed, coverage_skew=level.coverage_skew),
            seed,
        )
        for seed in seeds
    ]
    return {
        "level": level,
        "seed_results": seed_results,
        "summary": _level_summary(seed_results),
    }


def _level_summary(seed_results: list[dict]) -> dict[str, dict[str, float | str]]:
    summary = {}
    for metric in SWEEP_METRICS:
        values = [result["metrics"][metric] for result in seed_results]
        mean = _mean(values)
        se = _standard_error(values)
        negative_seeds = [
            str(result["seed"])
            for result in seed_results
            if result["metrics"][metric] < 0.0
        ]
        summary[metric] = {
            "mean": mean,
            "variance": _variance(values),
            "se": se,
            "ci95_low": mean - 1.96 * se,
            "ci95_high": mean + 1.96 * se,
            "negative_seed_count": float(len(negative_seeds)),
            "negative_seeds": ",".join(negative_seeds),
        }
    return summary


def _sweep_metrics(config: CardDSkewSweepConfig, level_results: list[dict]) -> dict[str, float]:
    metrics: dict[str, float] = {
        "n_seeds": float(len(config.base.seeds or (config.base.seed,))),
        "n_skew_levels": float(len(level_results)),
    }
    for result in level_results:
        level = result["level"]
        metrics[f"{level.name}_severity"] = level.severity
        for metric, summary in result["summary"].items():
            prefix = f"{level.name}_{metric}"
            metrics[f"{prefix}_mean"] = float(summary["mean"])
            metrics[f"{prefix}_se"] = float(summary["se"])
            metrics[f"{prefix}_ci95_low"] = float(summary["ci95_low"])
            metrics[f"{prefix}_ci95_high"] = float(summary["ci95_high"])
            metrics[f"{prefix}_negative_seed_count"] = float(summary["negative_seed_count"])

    for metric in SWEEP_METRICS:
        x = [result["level"].severity for result in level_results]
        y = [float(result["summary"][metric]["mean"]) for result in level_results]
        slopes = _adjacent_slopes(x, y)
        metrics[f"{metric}_skew_slope"] = _linear_slope(x, y)
        metrics[f"{metric}_min_adjacent_slope"] = min(slopes) if slopes else 0.0
        metrics[f"{metric}_monotone_non_decreasing"] = float(
            all(slope >= -config.monotone_tolerance for slope in slopes)
        )

    extreme = level_results[-1]["summary"]
    metrics["extreme_provisional_harm_ci_excludes_zero"] = float(
        float(extreme["provisional_thin_undercovered_harm_reduction"]["ci95_low"])
        > config.min_extreme_provisional_harm_ci_lower
    )
    metrics["extreme_provisional_thin_ll_ci_excludes_zero"] = float(
        float(extreme["provisional_thin_heldout_ll_delta"]["ci95_low"]) > 0.0
    )
    metrics["card_d_keystone_demonstrated"] = float(
        metrics["provisional_thin_undercovered_harm_reduction_skew_slope"]
        >= config.min_skew_slope
        and bool(metrics["provisional_thin_undercovered_harm_reduction_monotone_non_decreasing"])
        and bool(metrics["extreme_provisional_harm_ci_excludes_zero"])
    )
    return metrics


def _series(level_results: list[dict]) -> list[dict]:
    rows = []
    for result in level_results:
        level = result["level"]
        for metric, summary in result["summary"].items():
            rows.append(
                {
                    "cycle": level.severity,
                    "system": metric,
                    "skew_level": level.name,
                    "value": summary["mean"],
                    "se": summary["se"],
                    "ci95_low": summary["ci95_low"],
                    "ci95_high": summary["ci95_high"],
                    "negative_seed_count": summary["negative_seed_count"],
                    "negative_seeds": summary["negative_seeds"],
                }
            )
    return rows


def _controls(config: CardDSkewSweepConfig) -> dict[str, float]:
    return {
        "n_seeds_required": 5.0,
        "n_seeds": float(len(config.base.seeds or (config.base.seed,))),
        "monotone_tolerance": config.monotone_tolerance,
        "min_extreme_provisional_harm_ci_lower": (
            config.min_extreme_provisional_harm_ci_lower
        ),
        "min_skew_slope": config.min_skew_slope,
    }


def _standard_error(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return sqrt(_variance(values) / len(values))


def _linear_slope(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return 0.0
    x_mean = _mean(x)
    y_mean = _mean(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    if denominator == 0.0:
        return 0.0
    numerator = sum((x_value - x_mean) * (y_value - y_mean) for x_value, y_value in zip(x, y))
    return numerator / denominator


def _adjacent_slopes(x: list[float], y: list[float]) -> list[float]:
    slopes = []
    for idx in range(1, len(x)):
        delta = x[idx] - x[idx - 1]
        slopes.append((y[idx] - y[idx - 1]) / delta if delta else 0.0)
    return slopes


if __name__ == "__main__":
    result = run()
    print(result["run_dir"])
    print(
        "card_d_keystone_demonstrated="
        f"{result['metrics']['card_d_keystone_demonstrated']:.0f}"
    )
    print(
        "provisional_harm_skew_slope="
        f"{result['metrics']['provisional_thin_undercovered_harm_reduction_skew_slope']:.6f}"
    )
