"""Two-environment change attribution; preserves the original single-env policy."""

from .portable import REDUCIBLE
from .replication import decoded
from .replication_stop import decision_complete
from .verdict import EXIT_CODES, assessment, summarize


def unstable_difference(outcome):
    if outcome.get("status") != "unstable_candidate":
        return False
    return any(
        any(not value.get("matches", False) for value in outcome.get(key, {}).get("outputs", []))
        for key in ("cold_comparison", "comparison")
    )


def summarize_upgrade(rows, planned, *, mode, level, replication=None, artifacts=None):
    # Reuse v1 validation, scope, artifact contract, diagnostic notes and injection labels.
    result = summarize(
        rows, planned, mode=mode, level=level, replication=replication, artifacts=artifacts
    )
    result.update(schema_version=2, policy_version=2)
    reasons, notes, differences = [], [], []
    flags = set()

    def reason(verdict, code, message, batch=None):
        flags.add(verdict)
        reasons.append({"code": code, "message": message, "batch": batch})

    for row in rows:
        index = row["batch"]
        baseline, candidate = row.get("baseline", {}), row.get("candidate", {})
        ba, ca = assessment(baseline), assessment(candidate)
        if baseline.get("status") == "pass":
            if candidate.get("status") in REDUCIBLE and not ca["explained"]:
                reason(
                    "REGRESSION",
                    "candidate_only_failure",
                    "Passing baseline, stable candidate failure.",
                    index,
                )
            elif unstable_difference(candidate):
                reason(
                    "NEEDS_REVIEW",
                    "unstable_candidate_difference",
                    "Passing baseline; candidate-only numerical differences are unstable across repeats.",
                    index,
                )
        for side, outcome, current in (("baseline", baseline, ba), ("candidate", candidate, ca)):
            if not current["conclusive"]:
                if (
                    side == "candidate"
                    and baseline.get("status") == "pass"
                    and unstable_difference(candidate)
                ):
                    continue
                reason(
                    "INCONCLUSIVE",
                    f"{side}_{outcome.get('status', 'missing')}",
                    f"{side.capitalize()} primary execution is {outcome.get('status', 'missing')}.",
                    index,
                )
        if baseline.get("status") in REDUCIBLE and candidate.get("status") in REDUCIBLE:
            differences.append(
                {
                    "batch": index,
                    "baseline_status": baseline["status"],
                    "candidate_status": candidate["status"],
                    "baseline_failed_targets": sum(
                        not x.get("matches", False)
                        for x in baseline.get("comparison", {}).get("outputs", [])
                    ),
                    "candidate_failed_targets": sum(
                        not x.get("matches", False)
                        for x in candidate.get("comparison", {}).get("outputs", [])
                    ),
                    "links": [
                        f"report.html#batch-{index}-{side}" for side in ("baseline", "candidate")
                    ],
                }
            )
        elif baseline.get("status") in REDUCIBLE and candidate.get("status") == "pass":
            notes.append(
                {
                    "code": "baseline_only_difference",
                    "batch": index,
                    "message": "Baseline compile/eager discrepancy is absent in the candidate.",
                }
            )
    notes.extend(result["notes"])
    if len(rows) < planned:
        reason(
            "INCONCLUSIVE",
            "missing_batches",
            f"{planned - len(rows)} planned batches lack primary results.",
        )
    replication = result["accuracy_shift_replicated"]
    if level != "quick":
        if replication.get("accuracy_shift_replicated"):
            reason(
                "REGRESSION",
                "accuracy_shift_replicated",
                "Accuracy degradation replicated across input batches.",
            )
        elif not decision_complete(replication, planned):
            score = decoded(replication)
            trend = replication.get("completed", 0) > 0 and score is not None and score > 0
            reason(
                "NEEDS_REVIEW" if trend else "INCONCLUSIVE",
                "replication_unavailable",
                "Required accuracy evidence is incomplete"
                + (
                    " and observed scores trend worse."
                    if trend
                    else "; no decisive change attribution."
                ),
            )
    verdict = next(
        (v for v in ("REGRESSION", "INCONCLUSIVE", "NEEDS_REVIEW") if v in flags), "SAFE"
    )
    result.update(
        verdict=verdict,
        exit_code=EXIT_CODES[verdict],
        reasons=reasons,
        notes=notes,
        pre_existing_differences={
            "message": "Pre-existing compile/eager differences, not attributable to this change.",
            "batches": len(differences),
            "baseline_failed_targets": sum(d["baseline_failed_targets"] for d in differences),
            "candidate_failed_targets": sum(d["candidate_failed_targets"] for d in differences),
            "details": differences,
        },
    )
    return result
