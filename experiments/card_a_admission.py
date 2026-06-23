"""Card A: admission under uncertainty on the synthetic testbed.

Hypothesis: candidates admitted by held-out predictive log-likelihood gain minus
complexity recover true synth clusters better than naive accumulation, and
improve held-out likelihood over a no-abstraction predictor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from math import log, sqrt
from pathlib import Path
from statistics import mean
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.artifacts import make_run_id, utc_now, write_run
from ipsum.abstractions import Abstraction, AbstractionStore
from ipsum.synth import Synth, SynthConfig


@dataclass(frozen=True)
class Example:
    cycle: int
    changed_files: frozenset[int]
    test: int
    failed: bool


@dataclass(frozen=True)
class CardAConfig:
    n_files: int = 96
    n_tests: int = 48
    n_clusters_sweep: tuple[int, ...] = (4, 8, 12, 16)
    train_cycles: int = 480
    heldout_cycles: int = 160
    p_hit: float = 0.9
    p_flaky: float = 0.01
    seed: int = 100
    cochange_threshold: float = 0.18
    min_support: int = 3
    max_candidates: int = 256
    complexity_per_file: float = 0.0015
    admission_threshold: float = 0.0
    gain_confidence_z: float = 1.0
    min_average_rejection_fraction: float = 0.2
    min_f1_margin_vs_admit_everything: float = 0.05


def run(config: CardAConfig = CardAConfig()) -> dict:
    per_setting = []
    final_store: AbstractionStore | None = None
    final_candidates: list[Abstraction] = []

    for offset, n_clusters in enumerate(config.n_clusters_sweep):
        result = _run_one(config, n_clusters=n_clusters, seed=config.seed + offset)
        per_setting.append(result)
        final_store = result["store"]
        final_candidates = result["candidates"]

    metrics = _mean_metrics(per_setting)
    controls = _mean_controls(per_setting)
    for result in per_setting:
        prefix = f"n_clusters_{result['n_clusters']}"
        metrics[f"{prefix}_cluster_f1"] = result["metrics"]["cluster_f1"]
        metrics[f"{prefix}_held_out_ll_gain"] = result["metrics"]["held_out_ll_gain"]
        metrics[f"{prefix}_rejection_fraction"] = result["metrics"]["rejection_fraction"]
        metrics[f"{prefix}_passes_granularity_gate"] = float(result["passes_granularity_gate"])
        controls[f"{prefix}_admit_everything_cluster_f1"] = result["controls"][
            "admit_everything_cluster_f1"
        ]
        controls[f"{prefix}_f1_margin_vs_admit_everything"] = result["metrics"][
            "cluster_f1"
        ] - result["controls"]["admit_everything_cluster_f1"]

    metrics["min_cluster_f1_margin_vs_admit_everything"] = min(
        result["metrics"]["cluster_f1"] - result["controls"]["admit_everything_cluster_f1"]
        for result in per_setting
    )
    metrics["average_rejection_fraction"] = metrics["rejection_fraction"]
    metrics["granularity_gate_passed"] = float(
        all(result["passes_granularity_gate"] for result in per_setting)
    )
    metrics["admission_gate_passed"] = float(
        bool(metrics["granularity_gate_passed"])
        and metrics["average_rejection_fraction"] >= config.min_average_rejection_fraction
    )

    created = utc_now()
    run_id = make_run_id("A", "synth", created)
    final_abstractions = list(final_store or [])
    abstractions = {
        "snapshots": [
            {
                "cycle": config.train_cycles,
                "abstractions": [
                    {
                        "name": a.name,
                        "complexity": a.complexity,
                        "usefulness": a.usefulness,
                        "admitted_cycle": config.train_cycles,
                        "evicted_cycle": None,
                        "payload_summary": _payload_summary(a),
                    }
                    for a in final_abstractions
                ],
            }
        ]
    }
    events = {
        "events": [
            {"cycle": config.train_cycles, "type": "admit", "name": a.name}
            for a in final_abstractions
        ]
    }
    slope = {
        "metric_name": "card_a_cluster_f1",
        "selection_rate_cap": 0.33,
        "note": "Card A controls at one training horizon, not the compounding slope plot.",
        "series": [
            {
                "cycle": config.train_cycles,
                "system": "weekly_retrain",
                "condition": "admit_everything_control",
                "value": controls["admit_everything_cluster_f1"],
            },
            {
                "cycle": config.train_cycles,
                "system": "data_matched_control",
                "condition": "no_abstraction_control",
                "value": controls["no_abstraction_cluster_f1"],
            },
            {
                "cycle": config.train_cycles,
                "system": "ipsum",
                "condition": "ll_gain_admission",
                "value": metrics["cluster_f1"],
            },
        ],
    }

    run_dir = write_run(
        run_id=run_id,
        card="A",
        dataset="synth",
        created=created,
        config={**asdict(config), "candidate_count_final": len(final_candidates)},
        slope=slope,
        metrics=metrics,
        controls=controls,
        abstractions=abstractions,
        events=events,
        headline_metric=metrics["min_cluster_f1_margin_vs_admit_everything"],
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "metrics": metrics,
        "controls": controls,
    }


def _run_one(config: CardAConfig, *, n_clusters: int, seed: int) -> dict:
    synth_cfg = SynthConfig(
        n_files=config.n_files,
        n_tests=config.n_tests,
        n_clusters=n_clusters,
        p_hit=config.p_hit,
        p_flaky=config.p_flaky,
        seed=seed,
    )
    world = Synth(synth_cfg)
    train = _collect(world, config.train_cycles)
    heldout = _collect(world, config.heldout_cycles)

    store = AbstractionStore(
        min_support=config.min_support,
        cochange_threshold=config.cochange_threshold,
        max_candidates=config.max_candidates,
        complexity_per_file=config.complexity_per_file,
        admission_threshold=config.admission_threshold,
    )
    for example in train:
        if example.test == 0:
            store.observe_commit(example.changed_files)

    candidates = store.candidates()
    raw_ll = _ll_for_abstractions([], train, heldout, synth_cfg.n_tests)
    gains = {
        candidate.name: _ll_gain_lower_bound(
            candidate,
            train,
            heldout,
            synth_cfg.n_tests,
            confidence_z=config.gain_confidence_z,
        )
        for candidate in candidates
    }
    for candidate in candidates:
        store.admit(candidate, gains[candidate.name])
    admitted = list(store)
    admitted_ll = _ll_for_abstractions(admitted, train, heldout, synth_cfg.n_tests)
    admit_everything_ll = _ll_for_abstractions(candidates, train, heldout, synth_cfg.n_tests)

    cluster_precision, cluster_recall, cluster_f1 = _cluster_scores(
        admitted,
        world.true_clusters(),
    )
    admit_all_precision, admit_all_recall, admit_all_f1 = _cluster_scores(
        candidates,
        world.true_clusters(),
    )
    rejection_fraction = 1.0 - (len(admitted) / len(candidates) if candidates else 0.0)
    f1_margin = cluster_f1 - admit_all_f1
    held_out_ll_gain = admitted_ll - raw_ll
    passes_granularity_gate = (
        f1_margin >= config.min_f1_margin_vs_admit_everything
        and held_out_ll_gain > 0.0
    )

    return {
        "n_clusters": n_clusters,
        "store": store,
        "candidates": candidates,
        "metrics": {
            "cluster_precision": cluster_precision,
            "cluster_recall": cluster_recall,
            "cluster_f1": cluster_f1,
            "held_out_ll_gain": held_out_ll_gain,
            "admitted_count": float(len(admitted)),
            "candidate_count": float(len(candidates)),
            "rejection_fraction": rejection_fraction,
        },
        "controls": {
            "no_abstraction_cluster_f1": 0.0,
            "no_abstraction_ll_per_obs": raw_ll,
            "admit_everything_cluster_precision": admit_all_precision,
            "admit_everything_cluster_recall": admit_all_recall,
            "admit_everything_cluster_f1": admit_all_f1,
            "admit_everything_ll_gain": admit_everything_ll - raw_ll,
        },
        "passes_granularity_gate": passes_granularity_gate,
    }


def _collect(world: Synth, n_cycles: int) -> list[Example]:
    examples: list[Example] = []
    for _ in range(n_cycles):
        commit, outcomes = world.step()
        examples.extend(
            Example(
                cycle=outcome.cycle_decided,
                changed_files=commit.changed_files,
                test=outcome.test,
                failed=outcome.failed,
            )
            for outcome in outcomes
        )
    return examples


def _ll_gain(
    candidate: Abstraction,
    train: list[Example],
    heldout: list[Example],
    n_tests: int,
    raw_ll: float | None = None,
) -> float:
    if raw_ll is None:
        raw_ll = _ll_for_abstractions([], train, heldout, n_tests)
    return _ll_for_abstractions([candidate], train, heldout, n_tests) - raw_ll


def _ll_gain_lower_bound(
    candidate: Abstraction,
    train: list[Example],
    heldout: list[Example],
    n_tests: int,
    *,
    confidence_z: float,
) -> float:
    diffs = _ll_gain_diffs(candidate, train, heldout, n_tests)
    gain = sum(diffs) / len(diffs)
    if len(diffs) < 2:
        return gain
    variance = sum((diff - gain) ** 2 for diff in diffs) / (len(diffs) - 1)
    standard_error = sqrt(variance / len(diffs))
    return gain - confidence_z * standard_error


def _ll_gain_diffs(
    candidate: Abstraction,
    train: list[Example],
    heldout: list[Example],
    n_tests: int,
) -> list[float]:
    base = _base_rates(train, n_tests)
    files = _files(candidate)
    hit_counts = [[0, 0] for _ in range(n_tests)]
    for example in train:
        if example.changed_files & files:
            hit_counts[example.test][0] += int(example.failed)
            hit_counts[example.test][1] += 1
    hit_rates = [_smoothed(fails, total, base[test]) for test, (fails, total) in enumerate(hit_counts)]

    eps = 1e-6
    diffs = []
    for example in heldout:
        p0 = min(1.0 - eps, max(eps, base[example.test]))
        p1 = hit_rates[example.test] if example.changed_files & files else p0
        p1 = min(1.0 - eps, max(eps, p1))
        raw_ll = log(p0) if example.failed else log(1.0 - p0)
        abstraction_ll = log(p1) if example.failed else log(1.0 - p1)
        diffs.append(abstraction_ll - raw_ll)
    return diffs


def _ll_for_abstractions(
    abstractions: list[Abstraction],
    train: list[Example],
    heldout: list[Example],
    n_tests: int,
) -> float:
    base = _base_rates(train, n_tests)
    if not abstractions:
        return _log_likelihood(heldout, lambda ex: base[ex.test])

    filesets = [_files(a) for a in abstractions]
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
        lambda ex: _predict_with_abstractions(ex, base, filesets, rates_by_abstraction),
    )


def _predict_with_abstractions(
    example: Example,
    base: list[float],
    filesets: list[frozenset[int]],
    rates_by_abstraction: list[list[float]],
) -> float:
    p = base[example.test]
    for files, rates in zip(filesets, rates_by_abstraction, strict=True):
        if example.changed_files & files:
            p = max(p, rates[example.test])
    return p


def _base_rates(examples: list[Example], n_tests: int) -> list[float]:
    counts = [[0, 0] for _ in range(n_tests)]
    for example in examples:
        counts[example.test][0] += int(example.failed)
        counts[example.test][1] += 1
    return [_smoothed(fails, total, 0.05) for fails, total in counts]


def _smoothed(fails: int, total: int, prior: float) -> float:
    strength = 2.0
    return (fails + prior * strength) / (total + strength)


def _log_likelihood(examples: list[Example], predict) -> float:
    eps = 1e-6
    total = 0.0
    for example in examples:
        p = min(1.0 - eps, max(eps, predict(example)))
        total += log(p) if example.failed else log(1.0 - p)
    return total / len(examples)


def _cluster_scores(
    abstractions: list[Abstraction],
    true_clusters: list[frozenset[int]],
) -> tuple[float, float, float]:
    predicted_pairs = _pairs([_files(a) for a in abstractions])
    true_pairs = _pairs(true_clusters)
    if not predicted_pairs:
        return 0.0, 0.0, 0.0
    true_positive = len(predicted_pairs & true_pairs)
    precision = true_positive / len(predicted_pairs)
    recall = true_positive / len(true_pairs) if true_pairs else 0.0
    return precision, recall, _f1(precision, recall)


def _pairs(groups: list[frozenset[int]]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for group in groups:
        pairs.update(combinations(sorted(group), 2))
    return pairs


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _files(a: Abstraction) -> frozenset[int]:
    payload = a.payload
    if not isinstance(payload, dict):
        return frozenset()
    return frozenset(int(file_id) for file_id in payload.get("files", []))


def _mean_metrics(per_setting: list[dict]) -> dict[str, float]:
    keys = per_setting[0]["metrics"].keys()
    return {key: mean(result["metrics"][key] for result in per_setting) for key in keys}


def _mean_controls(per_setting: list[dict]) -> dict[str, float]:
    keys = per_setting[0]["controls"].keys()
    return {key: mean(result["controls"][key] for result in per_setting) for key in keys}


def _payload_summary(a: Abstraction) -> str:
    files = sorted(_files(a))
    shown = ", ".join(str(file_id) for file_id in files[:10])
    suffix = "" if len(files) <= 10 else f", ... ({len(files)} files)"
    return f"files: {shown}{suffix}"


if __name__ == "__main__":
    result = run()
    print(result["run_dir"])
    print(f"cluster_f1={result['metrics']['cluster_f1']:.3f}")
    print(f"held_out_ll_gain={result['metrics']['held_out_ll_gain']:.4f}")
