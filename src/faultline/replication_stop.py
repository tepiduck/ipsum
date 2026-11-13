"""Conservative stopping certificates for the unchanged batch replication rule."""

import math
import statistics

from .replication import POLICY, accuracy_shift_replicated, decoded


def stopping_certificate(rows, planned):
    # Validate before reasoning about a prefix, including extended-real scores.
    accuracy_shift_replicated(rows, planned)
    if (
        planned < POLICY["minimum_completed_batches"]
        or not rows
        or [row["batch"] for row in rows] != list(range(len(rows)))
        or any(row.get("status") != "complete" for row in rows)
    ):
        return None
    scores = [decoded(row) for row in rows]
    remaining = planned - len(scores)
    if not remaining:
        return None

    def flags(values):
        median = statistics.median(values)
        # An indeterminate infinity median cannot prove a positive outcome.
        return (
            len(values) >= POLICY["minimum_completed_batches"]
            and sum(x > 0 for x in values) / len(values) >= POLICY["fraction_worsened_gte"]
            and not math.isnan(median)
            and median >= math.log2(POLICY["median_growth_gte"])
        )

    bounds = []
    for extra in range(remaining + 1):
        best = scores + [math.inf] * extra
        worst = scores + [-math.inf] * extra
        # For an indeterminate median, positive infinity is a conservative upper bound.
        possible = flags(best)
        if len(best) >= 10 and math.isnan(statistics.median(best)):
            possible = sum(x > 0 for x in best) / len(best) >= POLICY["fraction_worsened_gte"]
        bounds.append(
            {"additional_complete": extra, "can_flag": possible, "must_flag": flags(worst)}
        )
    negative = not any(bound["can_flag"] for bound in bounds)
    positive = len(scores) >= 10 and all(bound["must_flag"] for bound in bounds)
    if not (negative or positive):
        return None
    return {
        "version": 1,
        "outcome": "flagged" if positive else "not_flagged",
        "reason": "replicated_rule_fixed_for_every_remaining_completion",
        "prefix_batches": [row["batch"] for row in rows],
        "bounds": bounds,
    }


def replication_evidence(rows, planned, *, certificate=None):
    result = accuracy_shift_replicated(rows, planned)
    if certificate is not None and certificate != stopping_certificate(rows, planned):
        raise ValueError("Invalid accuracy stopping certificate")
    result.update(
        attempted=len(rows),
        deliberately_skipped=planned - len(rows) if certificate else 0,
        decision_complete=bool(certificate) or (result["completed"] == planned and planned >= 10),
        stopping=certificate,
    )
    return result


def decision_complete(evidence, planned):
    """Do not trust a caller-supplied completeness bit or an unverified certificate."""
    if evidence.get("planned") != planned:
        return False
    certificate = evidence.get("stopping")
    if certificate:
        rows = evidence.get("distribution", [])
        return certificate == stopping_certificate(rows, planned)
    return evidence.get("completed") == planned and planned >= 10 and not evidence.get("missing")
