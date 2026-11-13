"""Post hoc v1/v2 replay of preserved 0.16 cohorts. Executes no model workers."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from summarize_check_calibration import collect, compact

from faultline.native import file_hash
from faultline.storage import atomic_json, read_json
from faultline.verdict import summarize


def compare(cohort, model, rows, planned, **kwargs):
    original = compact(cohort, model, rows, planned, **kwargs)
    v2 = summarize(
        rows,
        planned,
        mode="environment_comparison",
        level=kwargs.get("level", "standard"),
        replication=original["accuracy_shift_replicated"],
        policy="v2",
    )
    # This compact replay does not publish the original large worker reports.
    # Link to the retained public cohort evidence instead of inventing report.html.
    for item in v2["pre_existing_differences"]["details"]:
        item["links"] = ["calibration-016.json", "rescore-017.csv"]
    return {
        **original,
        "v1": original["verdict"],
        "v2": v2["verdict"],
        "v2_reason_counts": dict(Counter(r["code"] for r in v2["reasons"])),
        "pre_existing_differences": v2["pre_existing_differences"],
        "post_hoc_v2": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "same-version",
        "cross-version",
        "history",
        "controls",
        "exp-inputs",
        "original",
        "policy",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    args.allow_partial = False
    before = file_hash(args.original)
    original = read_json(args.original)
    pairs = collect(args, summarize_pair=compare)
    old = {(p["cohort"], p["model"]): p for p in original["pairs"]}
    if len(old) != len(pairs):
        raise ValueError("Preserved cohort denominator changed")
    for pair in pairs:
        previous = old[(pair["cohort"], pair["model"])]
        for field in (
            "verdict",
            "primary_counts",
            "reason_counts",
            "accuracy_shift_replicated",
            "evidence_sha256",
        ):
            if pair[field] != previous[field]:
                raise ValueError(f"v1 evidence changed: {pair['cohort']}/{pair['model']}/{field}")
    if file_hash(args.original) != before:
        raise ValueError("Preserved calibration changed")
    counts = {
        cohort: {
            policy: dict(Counter(p[policy] for p in pairs if p["cohort"] == cohort))
            for policy in ("v1", "v2")
        }
        for cohort in sorted({p["cohort"] for p in pairs})
    }
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(
        args.output / "rescore-017.json",
        {
            "schema_version": 1,
            "post_hoc_v2": True,
            "advisory_only": True,
            "original_sha256": before,
            "policy_sha256": file_hash(args.policy),
            "analysis_sha256": file_hash(__file__),
            "counts": counts,
            "pairs": pairs,
            "note": "Policy-only replay. No model reruns, no simulated early stopping, no changed primary results or thresholds. Not a false-positive rate.",
        },
    )
    with (args.output / "rescore-017.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "cohort",
                "model",
                "v1",
                "v2",
                "pre_existing_batches",
                "primary_batches",
                "accuracy_batches",
                "planned",
                "post_hoc_v2",
            ],
        )
        writer.writeheader()
        for pair in pairs:
            writer.writerow(
                {
                    **{key: pair[key] for key in ("cohort", "model", "v1", "v2", "post_hoc_v2")},
                    "pre_existing_batches": pair["pre_existing_differences"]["batches"],
                    "primary_batches": pair["scope"]["completed_batches"],
                    "accuracy_batches": pair["accuracy_shift_replicated"]["completed"],
                    "planned": pair["scope"]["planned_batches"],
                }
            )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
