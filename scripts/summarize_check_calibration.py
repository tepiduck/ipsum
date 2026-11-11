"""Apply the frozen check policy to recorded calibration evidence without rerunning models."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from faultline.native import file_hash
from faultline.replication import accuracy_shift_replicated, batch_score
from faultline.storage import atomic_json, read_json
from faultline.sweep_models import MODELS
from faultline.verdict import summarize


def verified_attempts(root):
    rows, hashes = [], []
    for batch in sorted(root.glob("batch-*")):
        commits = sorted(batch.glob("attempt-*/commit.json"))
        if not commits:
            continue
        commit = commits[-1]
        for name, expected in read_json(commit)["files"].items():
            path = (commit.parent / name).resolve()
            if not path.is_relative_to(commit.parent.resolve()) or file_hash(path) != expected:
                raise ValueError("Calibration evidence changed")
        hashes.append(file_hash(commit))
        index = int(batch.name.split("-")[-1])
        if (commit.parent / "evidence.json").exists():
            rows.append(read_json(commit.parent / "evidence.json"))
        else:
            row = {"batch": index, "diagnostics": {}}
            for side in ("baseline", "candidate"):
                raw = read_json(commit.parent / side / "outcome.json")
                observation = raw.get("primary_observation")
                if observation and raw["status"] in {"resource_limit", "timeout", "crash"}:
                    row[side] = {**raw, **observation}
                    row[side].pop("primary_observation", None)
                    row["diagnostics"][side] = [
                        {
                            "status": "diagnostic_skipped_memory"
                            if raw["status"] == "resource_limit"
                            else "diagnostic_unavailable",
                            "message": f"Retained legacy primary checkpoint; outer diagnostic outcome was {raw['status']}.",
                        }
                    ]
                else:
                    row[side] = raw
            rows.append(row)
    return rows, hashes


def compact(
    cohort,
    model,
    rows,
    planned,
    *,
    level="standard",
    expected=None,
    hashes=(),
    note=None,
    replication=None,
):
    replication = replication or accuracy_shift_replicated(
        [{"batch": r["batch"], **batch_score(r["baseline"], r["candidate"])} for r in rows], planned
    )
    verdict = summarize(
        rows, planned, mode="environment_comparison", level=level, replication=replication
    )
    hit = (
        None
        if expected is None
        else verdict["verdict"] != "REGRESSION"
        if expected == "not REGRESSION"
        else verdict["verdict"] == expected
    )
    return {
        "cohort": cohort,
        "model": model,
        "level": level,
        "expected": expected,
        "verdict": verdict["verdict"],
        "expectation_met": hit,
        "scope": verdict["scope"],
        "injected_fault": verdict["injected_fault"],
        "primary_counts": {
            side: dict(Counter(r[side]["status"] for r in rows))
            for side in ("baseline", "candidate")
        },
        "reason_counts": dict(Counter(r["code"] for r in verdict["reasons"])),
        "note_counts": dict(Counter(r["code"] for r in verdict["notes"])),
        "accuracy_shift_replicated": replication,
        "evidence_sha256": list(hashes),
        "note": note,
    }


def collect(args):
    rows = []
    same_entries = {}
    same_origins = {}
    for filename in ("complete.json", "progress.json"):
        if (args.same_version / filename).exists():
            records = read_json(args.same_version / filename)
            same_entries = {r["model"]: Path(r["evidence"]) for r in records}
            same_origins = {r["model"]: r.get("origin") for r in records}
            break
    for model in MODELS:
        same = same_entries.get(model, args.same_version / "models" / model)
        observations, hashes = verified_attempts(same)
        rows.append(
            compact(
                "same_version",
                model,
                observations,
                10,
                expected="SAFE",
                hashes=hashes,
                note=same_origins.get(model)
                or "Planned independent same-version pair; execution provenance is recorded on completion.",
            )
        )
        observations, hashes = verified_attempts(args.cross_version / "cases" / model)
        rows.append(
            compact(
                "upgrade",
                model,
                observations,
                10,
                hashes=hashes,
                note="Policy replay of recorded full-diagnostic sweep workers; original outcomes remain unchanged. Two repeats, AOT capture.",
            )
        )
    historical = read_json(args.history / "summary.json")
    for case in historical["cases"]:
        root = args.history / "cases" / case["id"] / "comparison"
        paths = [root / side / "outcome.json" for side in ("baseline", "candidate")]
        observation = {
            "batch": 0,
            **{
                side: read_json(path)
                for side, path in zip(("baseline", "candidate"), paths, strict=True)
            },
        }
        rows.append(
            compact(
                "historical",
                case["id"],
                [observation],
                1,
                level="standard" if case["gradients"] else "quick",
                expected="REGRESSION",
                hashes=[file_hash(p) for p in paths],
                note="Retrospective policy calibration on retained historical-issue fixtures, not a new discovery or a fresh 0.16 model execution.",
            )
        )
    for case in read_json(args.controls / "complete.json"):
        observations, hashes = verified_attempts(Path(case["evidence"]))
        rows.append(
            compact(
                "controlled_injection",
                case["name"],
                observations,
                1,
                level=case["level"],
                expected="REGRESSION",
                hashes=hashes,
                note="Explicitly injected compiler/gradient fault; harness validation only.",
            )
        )
    old = read_json(args.exp_inputs)
    for model, batches in old["batches"].items():
        scores = [
            {
                "batch": row["batch"],
                "status": row["status"],
                "paired_targets": row.get("gradients"),
                "median_log2_growth": row["compiled"]["median_log2_growth"]["value"],
                "median_status": row["compiled"]["median_log2_growth"]["status"],
            }
            for row in batches
        ]
        replicated = accuracy_shift_replicated(scores, 20)
        rows.append(
            compact(
                "exp_u20",
                model,
                [],
                20,
                expected="not REGRESSION",
                replication=replicated,
                hashes=[file_hash(args.exp_inputs)],
                note="Retrospective accuracy-only rule replay from 0.15; batch primary checks were not recorded by that measurement script. INCONCLUSIVE is not SAFE; the replicated accuracy rule does not flag.",
            )
        )
    if not args.allow_partial and any(
        r["scope"]["completed_batches"] < 10
        for r in rows
        if r["cohort"] in {"same_version", "upgrade"}
    ):
        # Completed worker failures count in the denominator and are publishable; absent
        # committed attempts require explicit acknowledgement as a partial campaign.
        for model in MODELS:
            same = same_entries.get(model, args.same_version / "models" / model)
            if (
                len(verified_attempts(same)[0]) != 10
                or len(verified_attempts(args.cross_version / "cases" / model)[0]) != 10
            ):
                raise ValueError(
                    "Calibration has missing committed batches; use --allow-partial to report them explicitly"
                )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("same-version", "cross-version", "history", "controls", "exp-inputs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    rows = collect(args)
    args.output.mkdir(parents=True, exist_ok=False)
    counts = {
        cohort: dict(Counter(r["verdict"] for r in rows if r["cohort"] == cohort))
        for cohort in sorted({r["cohort"] for r in rows})
    }
    atomic_json(
        args.output / "calibration.json",
        {
            "schema_version": 1,
            "policy_version": 1,
            "advisory_only": True,
            "analysis_sha256": file_hash(__file__),
            "partial_acknowledged": args.allow_partial,
            "counts": counts,
            "pairs": rows,
            "note": "Calibration targets are expectations, not policy inputs. All planned models remain represented. This is not a false-positive rate or a compiler-defect claim.",
        },
    )
    with (args.output / "calibration.csv").open("w", newline="", encoding="utf-8") as stream:
        columns = [
            "cohort",
            "model",
            "level",
            "verdict",
            "expected",
            "expectation_met",
            "completed_batches",
            "planned_batches",
            "completed_accuracy_batches",
            "worsened",
            "improved",
            "ties",
            "fraction_worsened",
            "median_log2_growth",
            "median_status",
            "sign_test_p",
            "accuracy_shift_replicated",
        ]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            accuracy = row["accuracy_shift_replicated"]
            writer.writerow(
                {
                    **{k: row[k] for k in columns[:6]},
                    "completed_batches": row["scope"]["completed_batches"],
                    "planned_batches": row["scope"]["planned_batches"],
                    "completed_accuracy_batches": accuracy["completed"],
                    **{k: accuracy[k] for k in columns[9:]},
                }
            )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
