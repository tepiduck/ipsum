"""Card D: coarse positivity / coverage guard on the synth cube."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import ceil, sqrt
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.artifacts import make_run_id, utc_now, write_run
from experiments.card_a_admission import (
    Example,
    _base_rates,
    _cluster_scores,
    _collect,
    _files,
    _ll_gain_lower_bound,
    _log_likelihood,
    _payload_summary,
    _smoothed,
)
from ipsum.abstractions import Abstraction, AbstractionStore
from ipsum.synth import Synth, SynthConfig


GUARD_SYSTEMS = ("no_guard", "strict_guard", "provisional_guard")


@dataclass(frozen=True)
class CoverageDecision:
    admitted: bool
    provisional: bool
    coverage_count: int
    standard_error: float


@dataclass(frozen=True)
class CardDConfig:
    n_files: int = 96
    n_tests: int = 48
    n_clusters: int = 8
    train_cycles: int = 360
    heldout_cycles: int = 120
    drift_post_cycles: int = 240
    admission_interval: int = 20
    p_hit: float = 0.9
    p_flaky: float = 0.01
    seed: int = 901
    seeds: tuple[int, ...] = (901, 902, 903, 904, 905)
    coverage_skew: tuple[float, ...] = (10.0, 8.0, 6.0, 4.0, 0.7, 0.5, 0.35, 0.25)
    cochange_threshold: float = 0.16
    min_support: int = 2
    max_candidates: int = 192
    complexity_per_file: float = 0.0015
    gain_confidence_z: float = 1.0
    coverage_epsilon: float = 0.10
    provisional_epsilon: float = 0.20
    coverage_fraction: float = 0.5
    min_true_pair_precision: float = 0.8
    min_false_rate_reduction: float = 0.15
    min_provisional_false_rate_reduction: float = 0.02
    min_thin_ll_gain: float = 0.0005
    max_provisional_latency_extra: float = 40.0
    max_provisional_latency_ratio: float = 2.0
    provisional_predictor_weight: float = 0.25
    min_strict_undercovered_harm_reduction: float = 0.0005
    min_provisional_undercovered_harm_reduction: float = 0.0001


def coverage_count(
    candidate: Abstraction,
    commits: list[frozenset[int]],
    *,
    coverage_fraction: float = 0.5,
) -> int:
    """Coarse v1 support coverage: commits touching the candidate region."""
    files = _files(candidate)
    if not files:
        return 0
    min_hits = max(1, int(ceil(len(files) * coverage_fraction)))
    return sum(1 for commit in commits if len(commit & files) >= min_hits)


def coverage_decision(
    *,
    mode: str,
    candidate: Abstraction,
    commits: list[frozenset[int]],
    ll_gain: float,
    min_count: int,
    provisional_min_count: int,
    coverage_fraction: float = 0.5,
) -> CoverageDecision:
    """Apply the v1 count/SE guard; no posterior or active probing."""
    count = coverage_count(candidate, commits, coverage_fraction=coverage_fraction)
    se = sqrt(0.25 / count) if count else float("inf")
    if ll_gain <= candidate.complexity:
        return CoverageDecision(False, False, count, se)
    if mode == "no_guard":
        return CoverageDecision(True, False, count, se)
    if mode == "strict_guard":
        return CoverageDecision(count >= min_count, False, count, se)
    if mode == "provisional_guard":
        if count >= min_count:
            return CoverageDecision(True, False, count, se)
        provisional = count >= provisional_min_count
        return CoverageDecision(provisional, provisional, count, se)
    raise ValueError(f"unknown coverage mode: {mode}")


def run(config: CardDConfig = CardDConfig()) -> dict:
    seeds = config.seeds or (config.seed,)
    seed_results = [_run_single_seed(replace(config, seed=seed), seed) for seed in seeds]
    metrics = _aggregate_metrics(seed_results, config)
    controls = _controls(config, seed_results)
    representative = seed_results[0]

    created = utc_now()
    run_id = make_run_id("D", "synth", created)
    slope = {
        "metric_name": "coverage_guard_thin_heldout_ll",
        "selection_rate_cap": 0.33,
        "series": _aggregate_system_series(seed_results, "thin_heldout_ll", config.train_cycles),
        "undercovered_harm_series": _aggregate_system_series(
            seed_results,
            "thin_undercovered_harm",
            config.train_cycles,
        ),
        "latency_series": _aggregate_latency_series(
            seed_results,
            config.train_cycles + config.heldout_cycles,
        ),
    }
    abstractions = {
        "snapshots": [
            {
                "cycle": config.train_cycles,
                "abstractions": [
                    {
                        "name": abstraction.name,
                        "complexity": abstraction.complexity,
                        "usefulness": abstraction.usefulness,
                        "admitted_cycle": config.train_cycles,
                        "evicted_cycle": None,
                        "payload_summary": _payload_summary(abstraction),
                    }
                    for abstraction in representative["per_system"]["provisional_guard"]["admitted"]
                ],
            }
        ]
    }
    events = {
        "events": [
            {
                "cycle": config.train_cycles,
                "type": "admit",
                "system": mode,
                "name": abstraction.name,
                "detail": (
                    "provisional"
                    if abstraction.name in representative["per_system"][mode]["provisional_names"]
                    else "confirmed"
                ),
            }
            for mode in GUARD_SYSTEMS
            for abstraction in representative["per_system"][mode]["admitted"]
        ]
    }
    run_dir = write_run(
        run_id=run_id,
        card="D",
        dataset="synth",
        created=created,
        config=asdict(config),
        slope=slope,
        metrics=metrics,
        controls=controls,
        abstractions=abstractions,
        events=events,
        headline_metric=metrics["provisional_thin_undercovered_harm_reduction_mean"],
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "metrics": metrics,
        "controls": controls,
        "seed_results": seed_results,
    }


def _run_single_seed(config: CardDConfig, seed: int) -> dict:
    world = Synth(_synth_config(config, drift_schedule=()))
    train = _collect(world, config.train_cycles)
    heldout = _collect(world, config.heldout_cycles)
    true_clusters = world.true_clusters()
    thick_ids, thin_ids = _coverage_region_ids(config)
    candidates = _propose_candidates(config, train)
    commits = _commits_from_examples(train)
    min_count = _min_count(config.coverage_epsilon)
    provisional_min_count = _min_count(config.provisional_epsilon)
    gains = {
        candidate.name: _ll_gain_lower_bound(
            candidate,
            train,
            heldout,
            config.n_tests,
            confidence_z=config.gain_confidence_z,
        )
        for candidate in candidates
    }

    per_system = {
        mode: _evaluate_mode(
            mode,
            config,
            candidates,
            gains,
            commits,
            train,
            heldout,
            true_clusters,
            thick_ids,
            thin_ids,
            min_count,
            provisional_min_count,
        )
        for mode in GUARD_SYSTEMS
    }
    latency = _post_drift_latency(config, min_count, provisional_min_count)
    metrics = _seed_metrics(per_system, latency, config)
    return {
        "seed": seed,
        "per_system": per_system,
        "latency": latency,
        "metrics": metrics,
    }


def _evaluate_mode(
    mode: str,
    config: CardDConfig,
    candidates: list[Abstraction],
    gains: dict[str, float],
    commits: list[frozenset[int]],
    train: list[Example],
    heldout: list[Example],
    true_clusters: list[frozenset[int]],
    thick_ids: set[int],
    thin_ids: set[int],
    min_count: int,
    provisional_min_count: int,
) -> dict:
    admitted = []
    provisional_names = set()
    decisions = {}
    predictor_weights: dict[str, float] = {}
    for candidate in candidates:
        decision = coverage_decision(
            mode=mode,
            candidate=candidate,
            commits=commits,
            ll_gain=gains[candidate.name],
            min_count=min_count,
            provisional_min_count=provisional_min_count,
            coverage_fraction=config.coverage_fraction,
        )
        decisions[candidate.name] = decision
        if decision.admitted:
            admitted.append(candidate)
            predictor_weights[candidate.name] = _predictor_weight(decision, config)
            if decision.provisional:
                provisional_names.add(candidate.name)

    weighted_admitted = [
        (abstraction, predictor_weights[abstraction.name])
        for abstraction in admitted
        if predictor_weights[abstraction.name] > 0.0
    ]
    undercovered_used = {
        abstraction.name
        for abstraction in admitted
        if (
            predictor_weights[abstraction.name] > 0.0
            and decisions[abstraction.name].coverage_count < min_count
        )
    }
    without_undercovered = [
        (abstraction, weight)
        for abstraction, weight in weighted_admitted
        if abstraction.name not in undercovered_used
    ]

    overall_ll = _ll_for_weighted_abstractions(weighted_admitted, train, heldout, config.n_tests)
    thick_heldout = _examples_in_regions(heldout, true_clusters, thick_ids)
    thin_heldout = _examples_in_regions(heldout, true_clusters, thin_ids)
    thick_ll = _ll_for_weighted_abstractions(
        weighted_admitted,
        train,
        thick_heldout,
        config.n_tests,
    )
    thin_ll = _ll_for_weighted_abstractions(
        weighted_admitted,
        train,
        thin_heldout,
        config.n_tests,
    )
    overall_ll_without_undercovered = _ll_for_weighted_abstractions(
        without_undercovered,
        train,
        heldout,
        config.n_tests,
    )
    thin_ll_without_undercovered = _ll_for_weighted_abstractions(
        without_undercovered,
        train,
        thin_heldout,
        config.n_tests,
    )
    overall_undercovered_effect = overall_ll - overall_ll_without_undercovered
    thin_undercovered_effect = thin_ll - thin_ll_without_undercovered
    false_thin, admitted_thin = _false_thin_counts(
        admitted,
        true_clusters,
        thin_ids,
        decisions,
        min_count,
        config,
    )
    precision, recall, f1 = _cluster_scores(admitted, true_clusters)
    return {
        "admitted": admitted,
        "provisional_names": provisional_names,
        "predictor_weights": predictor_weights,
        "overall_heldout_ll": overall_ll,
        "thick_heldout_ll": thick_ll,
        "thin_heldout_ll": thin_ll,
        "overall_undercovered_ll_effect": overall_undercovered_effect,
        "thin_undercovered_ll_effect": thin_undercovered_effect,
        "overall_undercovered_harm": max(0.0, -overall_undercovered_effect),
        "thin_undercovered_harm": max(0.0, -thin_undercovered_effect),
        "undercovered_used_count": float(len(undercovered_used)),
        "false_admissions_thin": float(false_thin),
        "admitted_thin": float(admitted_thin),
        "false_admission_rate_thin": false_thin / admitted_thin if admitted_thin else 0.0,
        "cluster_precision": precision,
        "cluster_recall": recall,
        "cluster_f1": f1,
        "provisional_count": float(len(provisional_names)),
    }


def _seed_metrics(
    per_system: dict[str, dict],
    latency: dict[str, float],
    config: CardDConfig,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    no_guard = per_system["no_guard"]
    for mode, result in per_system.items():
        metrics[f"{mode}_admitted_count"] = float(len(result["admitted"]))
        metrics[f"{mode}_provisional_count"] = result["provisional_count"]
        metrics[f"{mode}_overall_heldout_ll"] = result["overall_heldout_ll"]
        metrics[f"{mode}_thick_heldout_ll"] = result["thick_heldout_ll"]
        metrics[f"{mode}_thin_heldout_ll"] = result["thin_heldout_ll"]
        metrics[f"{mode}_overall_undercovered_ll_effect"] = result[
            "overall_undercovered_ll_effect"
        ]
        metrics[f"{mode}_thin_undercovered_ll_effect"] = result[
            "thin_undercovered_ll_effect"
        ]
        metrics[f"{mode}_overall_undercovered_harm"] = result["overall_undercovered_harm"]
        metrics[f"{mode}_thin_undercovered_harm"] = result["thin_undercovered_harm"]
        metrics[f"{mode}_undercovered_used_count"] = result["undercovered_used_count"]
        metrics[f"{mode}_false_admission_rate_thin"] = result["false_admission_rate_thin"]
        metrics[f"{mode}_cluster_precision"] = result["cluster_precision"]
        metrics[f"{mode}_cluster_recall"] = result["cluster_recall"]
        metrics[f"{mode}_cluster_f1"] = result["cluster_f1"]
        metrics[f"{mode}_post_drift_admission_latency"] = latency[mode]

    metrics["strict_false_admission_rate_thin_reduction"] = (
        no_guard["false_admission_rate_thin"]
        - per_system["strict_guard"]["false_admission_rate_thin"]
    )
    metrics["provisional_false_admission_rate_thin_reduction"] = (
        no_guard["false_admission_rate_thin"]
        - per_system["provisional_guard"]["false_admission_rate_thin"]
    )
    metrics["strict_thin_undercovered_harm_reduction"] = (
        no_guard["thin_undercovered_harm"]
        - per_system["strict_guard"]["thin_undercovered_harm"]
    )
    metrics["provisional_thin_undercovered_harm_reduction"] = (
        no_guard["thin_undercovered_harm"]
        - per_system["provisional_guard"]["thin_undercovered_harm"]
    )
    metrics["strict_overall_undercovered_harm_reduction"] = (
        no_guard["overall_undercovered_harm"]
        - per_system["strict_guard"]["overall_undercovered_harm"]
    )
    metrics["provisional_overall_undercovered_harm_reduction"] = (
        no_guard["overall_undercovered_harm"]
        - per_system["provisional_guard"]["overall_undercovered_harm"]
    )
    metrics["strict_thin_heldout_ll_delta"] = (
        per_system["strict_guard"]["thin_heldout_ll"] - no_guard["thin_heldout_ll"]
    )
    metrics["provisional_thin_heldout_ll_delta"] = (
        per_system["provisional_guard"]["thin_heldout_ll"] - no_guard["thin_heldout_ll"]
    )
    metrics["provisional_latency_extra_vs_no_guard"] = (
        latency["provisional_guard"] - latency["no_guard"]
    )
    metrics["strict_latency_extra_vs_no_guard"] = latency["strict_guard"] - latency["no_guard"]
    latency_denominator = max(float(config.admission_interval), latency["no_guard"])
    metrics["provisional_latency_ratio_vs_no_guard"] = (
        latency["provisional_guard"] / latency_denominator
    )
    metrics["strict_latency_ratio_vs_no_guard"] = latency["strict_guard"] / latency_denominator
    metrics["coverage_epsilon"] = config.coverage_epsilon
    metrics["coverage_provisional_epsilon"] = config.provisional_epsilon
    return metrics


def _aggregate_metrics(seed_results: list[dict], config: CardDConfig) -> dict[str, float]:
    per_seed = [result["metrics"] for result in seed_results]
    metrics = {"n_seeds": float(len(seed_results))}
    for field in sorted(per_seed[0]):
        values = [row[field] for row in per_seed]
        metrics[f"{field}_mean"] = _mean(values)
        metrics[f"{field}_variance"] = _variance(values)

    for seed_result in seed_results:
        seed = seed_result["seed"]
        metrics[f"seed_{seed}_provisional_thin_undercovered_harm_reduction"] = (
            seed_result["metrics"]["provisional_thin_undercovered_harm_reduction"]
        )
        metrics[f"seed_{seed}_strict_thin_heldout_ll_delta"] = seed_result["metrics"][
            "strict_thin_heldout_ll_delta"
        ]

    metrics["card_d_passed"] = float(
        metrics["strict_false_admission_rate_thin_reduction_mean"]
        >= config.min_false_rate_reduction
        and metrics["strict_thin_heldout_ll_delta_mean"] >= config.min_thin_ll_gain
        and metrics["provisional_thin_undercovered_harm_reduction_mean"]
        >= config.min_provisional_undercovered_harm_reduction
        and metrics["provisional_latency_extra_vs_no_guard_mean"]
        <= config.max_provisional_latency_extra
        and metrics["provisional_latency_ratio_vs_no_guard_mean"]
        <= config.max_provisional_latency_ratio
    )
    return metrics


def _controls(config: CardDConfig, seed_results: list[dict]) -> dict[str, float]:
    min_count = _min_count(config.coverage_epsilon)
    provisional_min_count = _min_count(config.provisional_epsilon)
    no_guard_false_rates = [
        result["per_system"]["no_guard"]["false_admission_rate_thin"]
        for result in seed_results
    ]
    no_guard_harms = [
        result["per_system"]["no_guard"]["thin_undercovered_harm"]
        for result in seed_results
    ]
    return {
        "coverage_min_count": float(min_count),
        "coverage_provisional_min_count": float(provisional_min_count),
        "provisional_predictor_weight": config.provisional_predictor_weight,
        "min_false_rate_reduction": config.min_false_rate_reduction,
        "min_thin_ll_gain": config.min_thin_ll_gain,
        "min_provisional_undercovered_harm_reduction": (
            config.min_provisional_undercovered_harm_reduction
        ),
        "no_guard_false_admission_rate_thin_mean": _mean(no_guard_false_rates),
        "no_guard_thin_undercovered_harm_mean": _mean(no_guard_harms),
    }


def _aggregate_system_series(
    seed_results: list[dict],
    field: str,
    cycle: int,
) -> list[dict]:
    rows = []
    for mode in GUARD_SYSTEMS:
        values = [result["per_system"][mode][field] for result in seed_results]
        rows.append(
            {
                "cycle": cycle,
                "system": mode,
                "value": _mean(values),
                "variance": _variance(values),
            }
        )
    return rows


def _aggregate_latency_series(seed_results: list[dict], cycle: int) -> list[dict]:
    rows = []
    for mode in GUARD_SYSTEMS:
        values = [result["latency"][mode] for result in seed_results]
        rows.append(
            {
                "cycle": cycle,
                "system": mode,
                "value": _mean(values),
                "variance": _variance(values),
            }
        )
    return rows


def _post_drift_latency(
    config: CardDConfig,
    min_count: int,
    provisional_min_count: int,
) -> dict[str, float]:
    drift_cycle = config.train_cycles + config.heldout_cycles
    world = Synth(_synth_config(config, drift_schedule=(drift_cycle,)))
    _collect(world, drift_cycle)
    latency = {mode: float(config.drift_post_cycles) for mode in GUARD_SYSTEMS}
    post_examples: list[Example] = []
    for offset in range(
        config.admission_interval,
        config.drift_post_cycles + 1,
        config.admission_interval,
    ):
        post_examples.extend(_collect(world, config.admission_interval))
        if len(post_examples) < config.admission_interval:
            continue
        train = post_examples[:-config.admission_interval] or post_examples
        heldout = post_examples[-config.admission_interval:]
        candidates = _propose_candidates(config, train)
        commits = _commits_from_examples(train)
        gains = {
            candidate.name: _ll_gain_lower_bound(
                candidate,
                train,
                heldout,
                config.n_tests,
                confidence_z=config.gain_confidence_z,
            )
            for candidate in candidates
        }
        for mode in GUARD_SYSTEMS:
            if latency[mode] < config.drift_post_cycles:
                continue
            for candidate in candidates:
                decision = coverage_decision(
                    mode=mode,
                    candidate=candidate,
                    commits=commits,
                    ll_gain=gains[candidate.name],
                    min_count=min_count,
                    provisional_min_count=provisional_min_count,
                    coverage_fraction=config.coverage_fraction,
                )
                if decision.admitted and _predictor_weight(decision, config) > 0.0:
                    latency[mode] = float(offset)
                    break
    return latency


def _ll_for_weighted_abstractions(
    weighted_abstractions: list[tuple[Abstraction, float]],
    train: list[Example],
    heldout: list[Example],
    n_tests: int,
) -> float:
    if not heldout:
        return 0.0
    base = _base_rates(train, n_tests)
    active = [
        (abstraction, weight)
        for abstraction, weight in weighted_abstractions
        if weight > 0.0
    ]
    if not active:
        return _log_likelihood(heldout, lambda ex: base[ex.test])

    filesets = [_files(abstraction) for abstraction, _ in active]
    weights = [weight for _, weight in active]
    rates_by_abstraction: list[list[float]] = []
    for files in filesets:
        hit_counts = [[0, 0] for _ in range(n_tests)]
        for example in train:
            if example.changed_files & files:
                hit_counts[example.test][0] += int(example.failed)
                hit_counts[example.test][1] += 1
        rates_by_abstraction.append(
            [_smoothed(fails, total, base[test]) for test, (fails, total) in enumerate(hit_counts)]
        )

    return _log_likelihood(
        heldout,
        lambda ex: _predict_with_weighted_abstractions(
            ex,
            base,
            filesets,
            rates_by_abstraction,
            weights,
        ),
    )


def _predict_with_weighted_abstractions(
    example: Example,
    base: list[float],
    filesets: list[frozenset[int]],
    rates_by_abstraction: list[list[float]],
    weights: list[float],
) -> float:
    p = base[example.test]
    for files, rates, weight in zip(filesets, rates_by_abstraction, weights, strict=True):
        if example.changed_files & files:
            weighted_lift = base[example.test] + weight * (rates[example.test] - base[example.test])
            p = max(p, weighted_lift)
    return p


def _predictor_weight(decision: CoverageDecision, config: CardDConfig) -> float:
    if not decision.admitted:
        return 0.0
    if decision.provisional:
        return config.provisional_predictor_weight
    return 1.0


def _propose_candidates(config: CardDConfig, examples: list[Example]) -> list[Abstraction]:
    store = AbstractionStore(
        min_support=config.min_support,
        cochange_threshold=config.cochange_threshold,
        max_candidates=config.max_candidates,
        complexity_per_file=config.complexity_per_file,
    )
    for commit in _commits_from_examples(examples):
        store.observe_commit(commit)
    return store.candidates()


def _commits_from_examples(examples: list[Example]) -> list[frozenset[int]]:
    commits = {}
    for example in examples:
        commits.setdefault(example.cycle, example.changed_files)
    return list(commits.values())


def _false_thin_counts(
    admitted: list[Abstraction],
    true_clusters: list[frozenset[int]],
    thin_ids: set[int],
    decisions: dict[str, CoverageDecision],
    min_count: int,
    config: CardDConfig,
) -> tuple[int, int]:
    false_count = 0
    admitted_thin = 0
    for abstraction in admitted:
        best_cluster, _ = _best_cluster(_files(abstraction), true_clusters)
        thin_region = (
            best_cluster in thin_ids
            or decisions[abstraction.name].coverage_count < min_count
        )
        if not thin_region:
            continue
        admitted_thin += 1
        pair_precision = _candidate_pair_precision(_files(abstraction), true_clusters)
        if decisions[abstraction.name].coverage_count < min_count or (
            pair_precision < config.min_true_pair_precision
        ):
            false_count += 1
    return false_count, admitted_thin


def _examples_in_regions(
    examples: list[Example],
    true_clusters: list[frozenset[int]],
    region_ids: set[int],
) -> list[Example]:
    return [
        example
        for example in examples
        if _dominant_cluster(example.changed_files, true_clusters) in region_ids
    ]


def _dominant_cluster(files: frozenset[int], true_clusters: list[frozenset[int]]) -> int:
    overlaps = [len(files & cluster) for cluster in true_clusters]
    return max(range(len(overlaps)), key=lambda idx: overlaps[idx])


def _best_cluster(files: frozenset[int], true_clusters: list[frozenset[int]]) -> tuple[int, float]:
    scores = [_jaccard(files, cluster) for cluster in true_clusters]
    best = max(range(len(scores)), key=lambda idx: scores[idx])
    return best, scores[best]


def _candidate_pair_precision(files: frozenset[int], true_clusters: list[frozenset[int]]) -> float:
    file_list = sorted(files)
    if len(file_list) < 2:
        return 1.0
    total = 0
    same_cluster = 0
    for idx, left in enumerate(file_list):
        for right in file_list[idx + 1 :]:
            total += 1
            if any(left in cluster and right in cluster for cluster in true_clusters):
                same_cluster += 1
    return same_cluster / total if total else 1.0


def _jaccard(left: frozenset[int], right: frozenset[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _coverage_region_ids(config: CardDConfig) -> tuple[set[int], set[int]]:
    ordered = sorted(
        range(config.n_clusters),
        key=lambda idx: config.coverage_skew[idx],
        reverse=True,
    )
    split = config.n_clusters // 2
    return set(ordered[:split]), set(ordered[split:])


def _min_count(epsilon: float) -> int:
    return int(ceil(0.25 / (epsilon * epsilon)))


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _variance(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    avg = _mean(values)
    return sum((value - avg) ** 2 for value in values) / len(values)


def _synth_config(config: CardDConfig, *, drift_schedule: tuple[int, ...]) -> SynthConfig:
    return SynthConfig(
        n_files=config.n_files,
        n_tests=config.n_tests,
        n_clusters=config.n_clusters,
        p_hit=config.p_hit,
        p_flaky=config.p_flaky,
        drift_schedule=drift_schedule,
        coverage_skew=config.coverage_skew,
        seed=config.seed,
    )


if __name__ == "__main__":
    result = run()
    print(result["run_dir"])
    print(f"card_d_passed={result['metrics']['card_d_passed']:.0f}")
    print(
        "strict_false_admission_rate_thin_reduction_mean="
        f"{result['metrics']['strict_false_admission_rate_thin_reduction_mean']:.3f}"
    )
    print(
        "provisional_thin_undercovered_harm_reduction_mean="
        f"{result['metrics']['provisional_thin_undercovered_harm_reduction_mean']:.6f}"
    )
