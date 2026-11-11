"""Batch-level advisory for paired input_batch_accuracy.py measurements.

This implements the predeclared 0.15 investigation rule, not a replacement for
Faultline's existing accuracy_shift thresholds or primary comparison status.
The manifest is a list of {batch, parent, child}; missing runs stay in N=20.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

EXPECTED_BATCHES = 20


def log_growth(parent, child):
    if not all(
        isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in (parent, child)
    ):
        raise ValueError("RMSE must be finite and nonnegative")
    if parent == child == 0:
        return 0.0
    if parent == 0:
        return math.inf
    if child == 0:
        return -math.inf
    return math.log2(child) - math.log2(parent)


def encode(value):
    if math.isnan(value):
        return {"status": "indeterminate", "value": None}
    if math.isinf(value):
        return {"status": "positive_infinity" if value > 0 else "negative_infinity", "value": None}
    return {"status": "finite", "value": value}


def decode(record):
    status = record["status"]
    if status == "finite" and math.isfinite(record["value"]):
        return record["value"]
    if status in ("positive_infinity", "negative_infinity") and record["value"] is None:
        return math.inf if status == "positive_infinity" else -math.inf
    raise ValueError("Invalid or indeterminate statistic")


def ratio(value):
    try:
        return 2.0**value
    except OverflowError:
        return math.inf


def paired_batch(parent, child):
    for key in (
        "schema",
        "identity_sha256",
        "inputs_sha256",
        "worker_sha256",
        "threads",
        "input_layouts",
        "observations_per_path",
    ):
        if key not in parent or parent[key] != child.get(key):
            raise ValueError(f"Paired measurement identity differs: {key}")
    if sorted(parent["absent_gradients"]) != sorted(child["absent_gradients"]):
        raise ValueError("Absent gradients differ")
    targets = []
    for result in (parent, child):
        rows = {r["parameter"]: r for r in result["gradients"]}
        if not rows or len(rows) != len(result["gradients"]):
            raise ValueError("Empty or duplicate gradient targets")
        targets.append(rows)
    before, after = targets
    if before.keys() != after.keys():
        raise ValueError("Present gradient sets differ")
    summary = {
        "status": "complete",
        "gradients": len(before),
        "fp64_reference_hashes_equal": all(
            before[k]["reference_sha256"] == after[k]["reference_sha256"] for k in before
        ),
    }
    for path in ("compiled", "eager"):
        values = [log_growth(before[k][path + "_rmse"], after[k][path + "_rmse"]) for k in before]
        score = statistics.median(values)
        summary[path] = {"median_log2_growth": encode(score), "growth": encode(ratio(score))}
        if math.isnan(score):
            summary["status"] = "inconclusive"
    return summary


def binomial_tail(worse, trials):
    if not 0 <= worse <= trials:
        raise ValueError("Invalid sign counts")
    return sum(math.comb(trials, k) for k in range(worse, trials + 1)) / 2**trials


def summarize_batches(rows):
    by_id = {}
    for row in rows:
        batch = row["batch"]
        if type(batch) is not int or not 0 <= batch < EXPECTED_BATCHES or batch in by_id:
            raise ValueError("Invalid or duplicate batch ID")
        by_id[batch] = row
    summary = {"expected_batches": EXPECTED_BATCHES, "unit_of_replication": "input_batch"}
    for path in ("compiled", "eager"):
        values, missing = [], []
        for batch in range(EXPECTED_BATCHES):
            row = by_id.get(batch)
            if row is None or row.get("status") != "complete":
                missing.append(batch)
                continue
            try:
                values.append(decode(row[path]["median_log2_growth"]))
            except (KeyError, TypeError, ValueError):
                missing.append(batch)
        worse = sum(value > 0 for value in values)
        better = sum(value < 0 for value in values)
        ties = len(values) - worse - better
        score = statistics.median(values) if values else math.nan
        if missing or math.isnan(score):
            decision = "inconclusive"
        elif worse >= 16 and score >= math.log2(1.25):
            decision = "systematic_across_inputs"
        elif worse <= 13:
            decision = "not_systematic"
        else:
            decision = "mixed"
        summary[path] = {
            "conclusive_batches": len(values),
            "inconclusive_batch_ids": missing,
            "worse": worse,
            "better": better,
            "ties": ties,
            "sign_trials": worse + better,
            "descriptive_p_value": binomial_tail(worse, worse + better),
            "median_log2_growth": encode(score),
            "median_growth": encode(ratio(score)),
            "decision": decision,
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest", type=Path, help="JSON list of paired paths relative to this file"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file to preserve evidence")
    rows = []
    for item in json.loads(args.manifest.read_text()):
        try:
            paths = [args.manifest.parent / item[key] for key in ("parent", "child")]
            row = paired_batch(*(json.loads(path.read_text()) for path in paths))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row = {"status": "inconclusive", "error": f"{type(exc).__name__}: {exc}"}
        rows.append({"batch": item["batch"], **row})
    result = {"advisory_only": True, "summary": summarize_batches(rows), "batches": rows}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
