"""Versioned, advisory check summaries; no primary oracle or reducer changes."""

from .portable import REDUCIBLE

EXIT_CODES = {"SAFE": 0, "REGRESSION": 1, "NEEDS_REVIEW": 2, "INCONCLUSIVE": 3}
POLICY_VERSION = 1
EXPLANATIONS = {
    "comparable_to_eager_against_fp64",
    "eager_control_sensitive",
    "explained_by_discontinuity",
}


def assessment(outcome):
    """Interpret a recorded primary result and its verified numerical context."""
    context = outcome.get("numerical_context", {})
    triage = context.get("triage", {})
    contrary = bool(triage.get("contrary_evidence")) or (
        triage.get("classification") == "unexplained_contrary_evidence"
    )
    failed = {
        row["target"]
        for row in outcome.get("comparison", {}).get("outputs", [])
        if not row.get("matches", False)
    }
    covered = False
    coverage = context.get("label_coverage", {})
    for label in set(context.get("labels", [])) & EXPLANATIONS:
        key = "discontinuity_downstream" if label == "explained_by_discontinuity" else label
        evidence = coverage.get(key, {})
        if (
            failed
            and set(evidence.get("targets", [])) >= failed
            and (
                label != "explained_by_discontinuity" or evidence.get("backward_same_inputs_agree")
            )
        ):
            covered = True
    explained = (
        outcome.get("status") == "gradient_mismatch"
        and triage.get("classification") == "explained"
        and covered
        and not contrary
        and all(":gradient:" in target for target in failed)
    )
    return {
        "conclusive": outcome.get("status") in {"pass", *REDUCIBLE},
        "explained": explained,
        "contrary": contrary,
        "acceptable": not contrary and (outcome.get("status") == "pass" or explained),
    }


def summarize(rows, planned, *, mode, level, replication=None, artifacts=None):
    if mode not in {"single_environment", "environment_comparison"}:
        raise ValueError("Unknown check mode")
    if level not in {"quick", "standard", "thorough"}:
        raise ValueError("Unknown check level")
    if type(planned) is not int or planned < 1:
        raise ValueError("Check needs a positive planned batch count")
    ids = [row["batch"] for row in rows]
    if len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i < planned for i in ids):
        raise ValueError("Invalid or duplicate check batch")
    reasons, notes = [], []
    flags = set()
    complete = 0

    def reason(verdict, code, message, batch=None):
        flags.add(verdict)
        reasons.append({"code": code, "message": message, "batch": batch})

    for row in rows:
        index = row["batch"]
        candidate = row.get("candidate", {})
        baseline = row.get("baseline", {}) if mode == "environment_comparison" else None
        ca = assessment(candidate)
        ba = assessment(baseline) if baseline is not None else None
        if ca["conclusive"] and (ba is None or ba["conclusive"]):
            complete += 1
        if (
            baseline is not None
            and baseline.get("status") == "pass"
            and candidate.get("status") in REDUCIBLE
            and not ca["explained"]
        ):
            reason(
                "REGRESSION",
                "candidate_only_failure",
                "Passing baseline, stable candidate failure.",
                index,
            )
        for side, outcome, current in (("candidate", candidate, ca), ("baseline", baseline, ba)):
            if outcome is None:
                continue
            if not current["conclusive"]:
                reason(
                    "INCONCLUSIVE",
                    f"{side}_{outcome.get('status', 'missing')}",
                    f"{side.capitalize()} primary execution is {outcome.get('status', 'missing')}.",
                    index,
                )
            elif current["contrary"]:
                reason(
                    "NEEDS_REVIEW",
                    "contrary_evidence",
                    f"{side.capitalize()} has contrary numerical evidence.",
                    index,
                )
            elif not current["acceptable"]:
                reason(
                    "NEEDS_REVIEW",
                    "unexplained_failure",
                    f"{side.capitalize()} {outcome['status']} is not fully explained.",
                    index,
                )
            elif current["explained"]:
                notes.append(
                    {
                        "code": "explained_gradient_mismatch",
                        "batch": index,
                        "side": side,
                        "message": "Gradient-only mismatch fully explained by recorded diagnostics; primary status retained.",
                    }
                )
            for diagnostic in row.get("diagnostics", {}).get(side, []):
                if diagnostic.get("status") != "complete":
                    notes.append(
                        {
                            "code": diagnostic.get("status", "diagnostic_unavailable"),
                            "batch": index,
                            "side": side,
                            "message": diagnostic.get(
                                "message", "Optional diagnostic unavailable."
                            ),
                        }
                    )
    if len(rows) < planned:
        reason(
            "INCONCLUSIVE",
            "missing_batches",
            f"{planned - len(rows)} planned batches lack primary results.",
        )
    replication = replication or {"status": "not_requested" if level == "quick" else "unavailable"}
    if mode == "environment_comparison" and level != "quick":
        if replication.get("accuracy_shift_replicated"):
            reason(
                "REGRESSION",
                "accuracy_shift_replicated",
                "Accuracy degradation replicated across input batches.",
            )
        elif replication.get("completed", 0) < 10 or replication.get("missing", 0):
            reason(
                "NEEDS_REVIEW",
                "replication_unavailable",
                "Fewer than ten complete batch scores or missing required accuracy evidence.",
            )
    verdict = next(
        (v for v in ("REGRESSION", "INCONCLUSIVE", "NEEDS_REVIEW") if v in flags), "SAFE"
    )
    return {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "execution_state": "complete",
        "verdict": verdict,
        "exit_code": EXIT_CODES[verdict],
        "advisory_only": True,
        "scope": {
            "mode": mode,
            "level": level,
            "planned_batches": planned,
            "completed_batches": complete,
            "gradients": level != "quick",
            "statement": "Limited to the frozen inputs, selected settings and requested checks; not a correctness or model-quality guarantee.",
        },
        "reasons": reasons,
        "notes": notes,
        "accuracy_shift_replicated": replication,
        "injected_fault": any(
            row.get("injected_fault")
            or any(row.get(side, {}).get("injected_fault") for side in ("baseline", "candidate"))
            for row in rows
        ),
        "artifacts": artifacts
        or {"report": "report.html", "evidence": "evidence.json", "reproducers": []},
    }
