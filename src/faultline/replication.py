"""Batch-level accuracy evidence. Never used by the primary oracle or reducer."""

import math
import statistics

from .accuracy import sign_test

POLICY = {
    "minimum_completed_batches": 10,
    "fraction_worsened_gte": 0.8,
    "median_growth_gte": 1.25,
}


def encoded(value):
    if value is None or math.isnan(value):
        return {"median_log2_growth": None, "median_status": "missing"}
    return {
        "median_log2_growth": value if math.isfinite(value) else None,
        "median_status": "finite"
        if math.isfinite(value)
        else "positive_infinity"
        if value > 0
        else "negative_infinity",
    }


def decoded(row):
    return {"positive_infinity": math.inf, "negative_infinity": -math.inf}.get(
        row.get("median_status"), row.get("median_log2_growth")
    )


def batch_score(baseline, candidate):
    sides = []
    for result in (baseline, candidate):
        evidence = result.get("fp64_advisory", {})
        if (
            result.get("status") not in {"pass", "gradient_mismatch", "numerical_mismatch"}
            or evidence.get("status") != "complete"
        ):
            return {"status": "incomplete", **encoded(None)}
        sides.append(
            {
                row["target"]: row.get("candidate", {}).get("rmse")
                for row in evidence.get("outputs", [])
                if ":gradient:parameter:" in row["target"]
                and not row["target"].endswith(":present")
                and row.get("reference_dtype") == "torch.float64"
            }
        )
    a, b = sides
    if not a or a.keys() != b.keys():
        return {"status": "incomplete", **encoded(None)}
    logs = []
    for target in a:
        x, y = a[target], b[target]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in (x, y)):
            return {"status": "incomplete", **encoded(None)}
        logs.append(
            0.0
            if x == y
            else math.inf
            if x == 0
            else -math.inf
            if y == 0
            else math.log2(y / x)
            if 0 < y / x < math.inf
            else math.log2(y) - math.log2(x)
        )
    median = statistics.median(logs)
    return {
        "status": "incomplete" if math.isnan(median) else "complete",
        "paired_targets": len(logs),
        **encoded(median),
    }


def accuracy_shift_replicated(rows, planned):
    """Rows have distinct integer batch IDs and one score per independent input batch."""
    if type(planned) is not int or planned < 1:
        raise ValueError("A positive planned batch denominator is required")
    ids = [row["batch"] for row in rows]
    if len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i < planned for i in ids):
        raise ValueError("Duplicate or unplanned input batch")
    scores = [decoded(row) for row in rows if row.get("status") == "complete"]
    if any(value is None or math.isnan(value) for value in scores):
        raise ValueError("Completed batch requires a determinate score")
    worse, better, ties = (
        sum(value > 0 for value in scores),
        sum(value < 0 for value in scores),
        sum(value == 0 for value in scores),
    )
    median = statistics.median(scores) if scores else None
    fraction = worse / len(scores) if scores else None
    flag = bool(
        len(scores) >= POLICY["minimum_completed_batches"]
        and fraction >= POLICY["fraction_worsened_gte"]
        and median >= math.log2(POLICY["median_growth_gte"])
    )
    return {
        "accuracy_shift_replicated": flag,
        "labels": ["accuracy_shift_replicated"] if flag else [],
        "planned": planned,
        "completed": len(scores),
        "missing": planned - len(scores),
        "coverage": len(scores) / planned,
        "worsened": worse,
        "improved": better,
        "ties": ties,
        "fraction_worsened": fraction,
        "sign_test_p": sign_test(worse, better),
        **encoded(median),
        "policy": dict(POLICY),
        "distribution": rows,
        "note": "Advisory only. Batch sign p-value is descriptive; common weights and input sources do not establish statistical independence. Missing batches remain in planned coverage.",
    }
