"""Verify and compact 0.18 memory-scheduling evidence without rescoring verdicts."""

import argparse
from collections import Counter
from pathlib import Path

from validate_check_memory import CELLS

from faultline.check_execution import committed, row_hash
from faultline.native import file_hash
from faultline.storage import atomic_json, read_json


def compact(root, model, jobs):
    evidence = read_json(root / "evidence.json")
    verdict = read_json(root / "verdict.json")
    configuration = read_json(root / "check.json")
    if verdict["execution_state"] != "complete":
        raise ValueError("Check did not finish")
    hashes = []
    primary_attempts = []
    for row in evidence["batches"]:
        batch = root / f"batch-{row['batch']:04d}"
        primary = committed(batch, "primary-attempt-*/commit.json", "primary.json")
        primary_attempts.append(primary)
        hashes.append(file_hash(max(batch.glob("primary-attempt-*/commit.json"))))
        retry = committed(batch, "serial-retry-*/commit.json", "primary.json")
        if retry:
            if retry["resource_retry"]["original_sha256"] != row_hash(primary):
                raise ValueError("Retry does not reference its original primary")
            primary = retry
            primary_attempts.append(retry)
            hashes.append(file_hash(max(batch.glob("serial-retry-*/commit.json"))))
        diagnostic = committed(batch, "attempt-*/commit.json", "evidence.json")
        if diagnostic != row:
            raise ValueError("Published evidence differs from its committed row")
        hashes.append(file_hash(max(batch.glob("attempt-*/commit.json"))))
        for side in ("baseline", "candidate"):
            if row[side]["status"] != primary[side]["status"] or row[side].get(
                "comparison"
            ) != primary[side].get("comparison"):
                raise ValueError("Diagnostic changed a primary observation")
    schedule = evidence["scheduling"]
    if jobs == "auto":
        recorded = committed(root / "schedule", "commit.json", "schedule.json")
        if any(schedule.get(key) != value for key, value in recorded.items()):
            raise ValueError("Report has a different measured schedule")
    wall = verdict["metrics"]["command_wall_s"]
    phases = [
        phase
        for row in evidence["batches"]
        for items in row["diagnostics"].values()
        for phase in items
    ]
    return {
        "model": model,
        "requested_jobs": jobs,
        "verdict": verdict["verdict"],
        "scope": verdict["scope"],
        "metrics": verdict["metrics"],
        "aggregate_worker_seconds": {
            "primary_including_retries": sum(
                row[side].get("worker_wall_s", 0)
                for row in primary_attempts
                for side in ("baseline", "candidate")
            ),
            "diagnostics_including_retries": sum(
                phase.get("supervision", {}).get("wall_s", phase.get("worker_wall_s", 0))
                for phase in phases
            ),
            "note": "Summed worker durations, not elapsed time; primary workers may overlap.",
        },
        "scheduling": schedule,
        "within_15_minutes_and_decisive": wall <= 900
        and verdict["verdict"] in {"SAFE", "REGRESSION"},
        "accuracy_shift_replicated": verdict["accuracy_shift_replicated"],
        "primary_counts": {
            side: dict(Counter(row[side]["status"] for row in evidence["batches"]))
            for side in ("baseline", "candidate")
        },
        "graph_capture": {
            side: {
                "compiled_graphs": sum(
                    row[side].get("graph_count", 0) for row in evidence["batches"]
                ),
                "nonportable_captures": sum(
                    not graph.get("portable", False)
                    for row in evidence["batches"]
                    for graph in row[side].get("graphs", [])
                ),
                "capture_errors": sorted(
                    {
                        graph["capture_error"]
                        for row in evidence["batches"]
                        for graph in row[side].get("graphs", [])
                        if graph.get("capture_error")
                    }
                ),
            }
            for side in ("baseline", "candidate")
        },
        "diagnostic_counts": {
            side: dict(
                Counter(
                    phase["status"]
                    for row in evidence["batches"]
                    for phase in row["diagnostics"].get(side, [])
                )
            )
            for side in ("baseline", "candidate")
        },
        "reason_counts": dict(Counter(row["code"] for row in verdict["reasons"])),
        "bundle_hash": configuration["bundle_hash"],
        "source_sha256": configuration["source_sha256"],
        "settings": configuration["settings"],
        "baseline_settings": configuration["baseline_settings"],
        "environments": {
            side: {key: configuration[f"{side}_environment"][key] for key in ("python", "packages")}
            for side in ("baseline", "candidate")
        },
        "evidence_sha256": {
            name: file_hash(root / name)
            for name in ("evidence.json", "verdict.json", "check.json", "memory-budget.json")
        },
        "commits_sha256": hashes,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Keep completed summaries; choose a new output path")
    rows = [compact(args.root / f"{model}-jobs-{jobs}", model, jobs) for model, jobs in CELLS]
    equivalence = {}
    for model in ("bert", "gpt2"):
        paired = [
            read_json(args.root / f"{model}-jobs-{jobs}" / "evidence.json")["batches"]
            for jobs in ("auto", "1")
        ]
        observations = [
            [
                {
                    side: {key: row[side].get(key) for key in ("status", "comparison")}
                    for side in ("baseline", "candidate")
                }
                for row in batches
            ]
            for batches in paired
        ]
        cells = [row for row in rows if row["model"] == model]
        equivalence[model] = {
            "same_primary_statuses_and_comparisons": observations[0] == observations[1],
            "same_accuracy_distribution": cells[0]["accuracy_shift_replicated"]["distribution"]
            == cells[1]["accuracy_shift_replicated"]["distribution"],
            "same_verdict": cells[0]["verdict"] == cells[1]["verdict"],
            "note": "Fresh auto and explicit-one-job runs; auto selected one job on this host.",
        }
    atomic_json(
        args.output,
        {
            "version": "0.18.0",
            "policy_version": 2,
            "measurements": rows,
            "serial_comparisons": equivalence,
        },
    )


if __name__ == "__main__":
    main()
