"""Verify and compact fresh 0.17 evidence without changing recorded verdicts."""

import argparse
import json
from collections import Counter
from pathlib import Path

from validate_check_v2 import MODELS

from faultline.check_execution import committed
from faultline.native import file_hash
from faultline.storage import atomic_json, read_json


def compact(root, cohort, model):
    verdict = read_json(root / "verdict.json")
    identity = read_json(root / "check.json")
    if verdict["execution_state"] != "complete" or verdict["policy_version"] != 2:
        raise ValueError(f"Unfinished or incompatible check: {cohort}/{model}")
    rows = []
    hashes = []
    for index in range(identity["planned"]):
        batch = root / f"batch-{index:04d}"
        primary = committed(batch, "primary-attempt-*/commit.json", "primary.json")
        row = committed(batch, "attempt-*/commit.json", "evidence.json")
        if primary is None or row is None or row["batch"] != index or primary["batch"] != index:
            raise ValueError("Missing committed primary or diagnostic batch")
        for side in ("baseline", "candidate"):
            if row[side]["status"] != primary[side]["status"] or row[side].get(
                "comparison"
            ) != primary[side].get("comparison"):
                raise ValueError("Advisory changed a primary observation")
        rows.append(row)
        hashes += [
            file_hash(max(batch.glob(pattern)))
            for pattern in ("primary-attempt-*/commit.json", "attempt-*/commit.json")
        ]
    return {
        "cohort": cohort,
        "model": model,
        "verdict": verdict["verdict"],
        "exit_code": verdict["exit_code"],
        "injected_fault": verdict["injected_fault"],
        "scope": verdict["scope"],
        "metrics": verdict["metrics"],
        "pre_existing_batches": verdict["pre_existing_differences"]["batches"],
        "pre_existing_failed_targets": {
            side: verdict["pre_existing_differences"][f"{side}_failed_targets"]
            for side in ("baseline", "candidate")
        },
        "reason_counts": dict(Counter(r["code"] for r in verdict["reasons"])),
        "primary_counts": {
            side: dict(Counter(r[side]["status"] for r in rows))
            for side in ("baseline", "candidate")
        },
        "accuracy_shift_replicated": verdict["accuracy_shift_replicated"],
        "source_sha256": identity["source_sha256"],
        "bundle_hash": identity["bundle_hash"],
        "settings": identity["settings"],
        "baseline_settings": identity["baseline_settings"],
        "allocation": identity["allocation"],
        "cache_policy": identity.get("cache_policy", "per_batch"),
        "environments": {
            side: {
                "python": identity[f"{side}_environment"]["python"],
                "packages": identity[f"{side}_environment"]["packages"],
            }
            for side in ("baseline", "candidate")
        },
        "evidence_sha256": {
            name: file_hash(root / name) for name in ("check.json", "evidence.json", "verdict.json")
        },
        "commits_sha256": hashes,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "controls", "old-costs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    pairs = [
        compact(args.root / "held-out" / cohort / name, cohort, name)
        for name, _, _, _ in MODELS
        for cohort in ("same_version", "upgrade")
    ]
    models = []
    for name, _, _, _ in MODELS:
        metadata = read_json(args.root / "prepared" / name / "source/model.json")
        models.append(
            {
                "name": name,
                **{
                    key: metadata[key]
                    for key in (
                        "provider",
                        "architecture",
                        "repository",
                        "revision",
                        "weights",
                        "weights_url",
                        "weights_sha256",
                        "inputs_sha256",
                        "parameter_count",
                        "batch_size",
                        "sequence_length",
                        "input_sources",
                    )
                    if key in metadata
                },
            }
        )
    old = {(row["model"], row["level"]): row for row in read_json(args.old_costs)["measurements"]}
    costs = []
    for model in ("resnet50", "bert", "gpt2"):
        row = compact(args.root / "costs/cost" / model, "cost", model)
        previous = old[(model, "standard")]
        row["v016"] = {
            key: previous[key]
            for key in ("verdict", "command_wall_s", "peak_worker_rss_mb", "scope", "source_sha256")
        }
        row["wall_ratio_v016_over_v017"] = (
            previous["command_wall_s"] / row["metrics"]["command_wall_s"]
        )
        row["complete_comparison"] = row["scope"]["completed_batches"] == 10 and row[
            "accuracy_shift_replicated"
        ].get("decision_complete", False)
        costs.append(row)
    controls = [
        compact(args.controls / name, "controlled_injection", name)
        for name in ("relu-as-neg", "sum-as-mean", "gradient-scale")
    ]
    counts = {
        cohort: dict(Counter(r["verdict"] for r in pairs if r["cohort"] == cohort))
        for cohort in ("same_version", "upgrade")
    }
    result = {
        "schema_version": 1,
        "advisory_only": True,
        "counts": counts,
        "planned_pairs": 12,
        "pairs": pairs,
        "models": models,
        "costs": costs,
        "controls": controls,
        "analysis_sha256": file_hash(__file__),
        "note": "Fresh CPU checks on six new checkpoint variants, with fixed v2 policy. Ten primary batches per pair; ordered diagnostics may stop with a verified negative certificate. Common sources/weights limit independence. Not a false-positive rate. Historical and exp_u20 cohorts are separately identified as retained-evidence replay.",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "validation-017.json", result)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
