"""Cross-environment accuracy evidence, independent of the primary upgrade gate."""

import math
import statistics

# Declared before inspecting the 0.10 GPT-2 target-level results. Do not tune to a sweep.
ACCURACY_POLICY = {
    "candidate_ratio_gt": 2.0,
    "ratio_growth_gte": 2.0,
    "fraction_gte": 0.2,
    "denominator": "union of output/gradient targets failing in either environment",
}

# GPT-2 motivated this policy. Fixed in docs/v011-policy.json before the held-out sweep.
SHIFT_POLICY = {
    "fraction_gte": 0.75,
    "median_growth_gte": 1.25,
    "one_sided_sign_p_lte": 0.01,
    "minimum_paired_targets": 8,
    "minimum_coverage": 0.8,
}


def sign_test(worse, better):
    """Exact one-sided paired sign test; equal-error ties are excluded."""
    n = worse + better
    return sum(math.comb(n, k) for k in range(worse, n + 1)) / (1 << n) if n else 1.0


def accuracy_shift(baseline, candidate):
    """Distribution of absolute compiled/fp64 errors, independently of failure labels."""
    primary = [
        r if "comparison" in r else r.get("primary_observation", {}) for r in (baseline, candidate)
    ]
    targets = {
        row["target"]
        for result in primary
        for row in result.get("comparison", {}).get("outputs", [])
        if not row["target"].startswith("state:")
        and ":input_after:" not in row["target"]
        and not (":gradient:" in row["target"] and row["target"].endswith(":present"))
    }
    sides = []
    for result in (baseline, candidate):
        evidence = result.get("fp64_advisory", {})
        sides.append(
            {
                row["target"]: row.get("candidate", {}).get("rmse")
                for row in evidence.get("outputs", [])
                if evidence.get("status") == "complete"
                and row.get("reference_dtype") == "torch.float64"
            }
        )
    rows, logs = [], []
    worse = better = ties = 0
    for target in sorted(targets):
        a, b = (side.get(target) for side in sides)
        valid = all(isinstance(x, (float, int)) and math.isfinite(x) and x >= 0 for x in (a, b))
        row = {
            "target": target,
            "baseline_compiled_rmse": a if valid else None,
            "candidate_compiled_rmse": b if valid else None,
            "log2_growth": None,
            "status": "missing_or_nonfinite",
        }
        if valid:
            worse += b > a
            better += b < a
            ties += b == a
            value = (
                0.0
                if a == b
                else math.inf
                if a == 0
                else -math.inf
                if b == 0
                else math.log2(b) - math.log2(a)
            )
            logs.append(value)
            row.update(
                status="finite"
                if math.isfinite(value)
                else "baseline_exact"
                if a == 0
                else "candidate_exact",
                log2_growth=value if math.isfinite(value) else None,
            )
        rows.append(row)
    paired, total = len(logs), len(targets)
    median = statistics.median(logs) if logs else None
    median_status = (
        "missing"
        if median is None
        else "finite"
        if math.isfinite(median)
        else "positive_infinity"
        if median == math.inf
        else "negative_infinity"
        if median == -math.inf
        else "indeterminate"
    )
    fraction = worse / paired if paired else None
    coverage = paired / total if total else 0.0
    p = sign_test(worse, better)
    sufficient = (
        all("comparison" in r for r in primary)
        and paired >= SHIFT_POLICY["minimum_paired_targets"]
        and coverage >= SHIFT_POLICY["minimum_coverage"]
    )
    flagged = bool(
        sufficient
        and fraction >= SHIFT_POLICY["fraction_gte"]
        and median >= math.log2(SHIFT_POLICY["median_growth_gte"])
        and p <= SHIFT_POLICY["one_sided_sign_p_lte"]
    )
    return {
        "schema_version": 1,
        "status": "complete"
        if paired == total and total
        else "partial"
        if paired
        else "unavailable",
        "accuracy_shift": flagged,
        "interpretation": "single_input, descriptive only",
        "labels": ["accuracy_shift"] if flagged else [],
        "paired": paired,
        "total": total,
        "coverage": coverage,
        "worsened": worse,
        "improved": better,
        "ties": ties,
        "fraction_worsened": fraction,
        "median_log2_growth": median if median_status == "finite" else None,
        "median_status": median_status,
        "sign_test_p": p,
        "sufficient_evidence": sufficient,
        "policy": dict(SHIFT_POLICY),
        "targets": rows,
        "note": "Advisory accuracy shift, not a compiler defect. GPT-2 motivated the fixed thresholds. "
        "Targets share computations; the sign-test p-value is descriptive, not a calibrated model-level probability. "
        "Missing references remain in coverage; equal-error ties remain in the fraction and median. "
        "The 0.10 application is retrospective; held-out validation is recorded separately.",
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
