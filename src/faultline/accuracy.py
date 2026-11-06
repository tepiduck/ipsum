"""Cross-environment accuracy evidence, independent of the primary upgrade gate."""

import math

# Declared before inspecting the 0.10 GPT-2 target-level results. Do not tune to a sweep.
ACCURACY_POLICY = {
    "candidate_ratio_gt": 2.0,
    "ratio_growth_gte": 2.0,
    "fraction_gte": 0.2,
    "denominator": "union of output/gradient targets failing in either environment",
}


def error_ratio(record):
    eager = record.get("eager", {}).get("rmse")
    compiled = record.get("candidate", {}).get("rmse")
    base = {"eager_rmse": eager, "compiled_rmse": compiled, "value": None}
    if any(
        not isinstance(x, (float, int)) or not math.isfinite(x) or x < 0 for x in (eager, compiled)
    ):
        return {
            "eager_rmse": eager
            if isinstance(eager, (float, int)) and math.isfinite(eager)
            else None,
            "compiled_rmse": compiled
            if isinstance(compiled, (float, int)) and math.isfinite(compiled)
            else None,
            "value": None,
            "status": "missing_or_nonfinite",
        }
    if eager == 0:
        return {**base, "status": "both_exact" if compiled == 0 else "infinite"}
    ratio = compiled / eager
    return {
        **base,
        "status": "finite" if math.isfinite(ratio) else "missing_or_nonfinite",
        "value": ratio if math.isfinite(ratio) else None,
        **({"reason": "ratio_overflow"} if not math.isfinite(ratio) else {}),
    }


def accuracy_comparison(baseline, candidate):
    """Report all failing targets, including unavailable references, without mutating results."""
    primary = [
        result if "comparison" in result else result.get("primary_observation", {})
        for result in (baseline, candidate)
    ]
    failed = {
        row["target"]
        for result in primary
        for row in result.get("comparison", {}).get("outputs", [])
        if not row["matches"]
        and not row["target"].startswith("state:")
        and ":input_after:" not in row["target"]
    }
    sides = {}
    for name, result in (("baseline", baseline), ("candidate", candidate)):
        evidence = result.get("fp64_advisory", {})
        sides[name] = {
            row["target"]: row
            for row in evidence.get("outputs", [])
            if evidence.get("status") == "complete"
            and row.get("reference_dtype") == "torch.float64"
        }
    rows = []
    available = matched = 0
    for target in sorted(failed):
        a = error_ratio(sides["baseline"].get(target, {}))
        b = error_ratio(sides["candidate"].get(target, {}))
        row = {"target": target, "baseline": a, "candidate": b, "qualifies": False}
        if "missing_or_nonfinite" in (a["status"], b["status"]):
            row["reason"] = "missing_fp64_evidence"
        else:
            available += 1
            worse = b["status"] == "infinite" or (
                b["status"] == "finite" and b["value"] > ACCURACY_POLICY["candidate_ratio_gt"]
            )
            doubled = a["status"] != "infinite" and (
                a["status"] == "both_exact"
                or b["status"] == "infinite"
                or (
                    b["status"] == "finite"
                    and b["value"] >= ACCURACY_POLICY["ratio_growth_gte"] * a["value"]
                )
            )
            row["qualifies"] = worse and doubled
            row["reason"] = (
                "baseline_already_infinite"
                if a["status"] == "infinite"
                else "baseline_both_exact"
                if a["status"] == "both_exact"
                else "ratio_thresholds"
            )
        matched += int(row["qualifies"])
        rows.append(row)
    fraction = matched / len(failed) if failed else 0.0
    qualifies = bool(failed) and fraction >= ACCURACY_POLICY["fraction_gte"]
    primary_available = all("comparison" in r for r in primary)
    return {
        "schema_version": 1,
        "status": "complete"
        if primary_available and available == len(failed)
        else "partial"
        if available
        else "unavailable",
        "primary_comparisons_available": primary_available,
        "labels": ["accuracy_regression"] if qualifies else [],
        "accuracy_regression": qualifies,
        "matched": matched,
        "total": len(failed),
        "available": available,
        "missing": len(failed) - available,
        "fraction": fraction,
        "policy": dict(ACCURACY_POLICY),
        "targets": rows,
        "note": "Advisory only; the primary classification and exit code are unchanged. "
        "Ratios use each environment's own eager and fp64 execution. Inspect absolute RMSE too. "
        "Missing references stay in the denominator. Both-exact baseline followed by candidate "
        "ratio >2 (or infinity) qualifies; an already-infinite baseline cannot establish growth. "
        "This is a regression candidate, not a compiler defect confirmation.",
    }
