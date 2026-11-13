"""Readable advisory evidence for the check HTML report; no verdict decisions."""

import html
import math


def number(value):
    return f"{value:.4g}" if isinstance(value, (int, float)) and math.isfinite(value) else "—"


def scheduling_html(evidence):
    schedule = evidence.get("scheduling")
    if not schedule:
        return ""
    rows = [
        ("Requested / chosen jobs", f"{schedule['requested_jobs']} / {schedule['effective_jobs']}"),
        (
            "Total candidate / baseline budget (MiB)",
            f"{schedule['candidate_total_rss_mb']} / {schedule['baseline_total_rss_mb']}",
        ),
        ("Measured first-batch peak (MiB)", number(schedule.get("measured_peak_rss_mb"))),
        ("Scheduling reason", schedule["reason"]),
        ("Primary serial retries", str(len(schedule["serial_retries"]))),
    ]
    table = "".join(
        f"<tr><th>{html.escape(key)}</th><td>{html.escape(value)}</td></tr>" for key, value in rows
    )
    return f"<h2>Memory scheduling</h2><table>{table}</table><p>Full evidence retains the measured allocation and every memory retry. A successful retry does not erase the original attempt.</p>"


def accuracy_html(evidence):
    if "completed" not in evidence:
        message = (
            "Not requested for this mode and level."
            if evidence.get("status") in {"not_requested", "not_applicable"}
            else "Batch accuracy evidence is unavailable; see the recorded reasons."
        )
        return f"<h2>Replicated accuracy</h2><p>{message}</p>"
    median = evidence.get("median_log2_growth")
    status = evidence.get("median_status")
    growth = (
        f"{2**median:.4g}×"
        if status == "finite" and isinstance(median, (int, float)) and -1000 < median < 1000
        else f"2^{number(median)}×"
        if status == "finite"
        else "unbounded (baseline error is zero)"
        if status == "positive_infinity"
        else "zero (candidate error is zero)"
        if status == "negative_infinity"
        else "unavailable"
    )
    values = [
        ("Complete batch scores", f"{evidence['completed']}/{evidence.get('planned', '?')}"),
        (
            "Worse / better / tied batches",
            " / ".join(str(evidence.get(k, 0)) for k in ("worsened", "improved", "ties")),
        ),
        ("Median batch error growth", growth),
        ("Batch sign-test p-value (descriptive)", number(evidence.get("sign_test_p"))),
        ("Replicated rule flagged", "yes" if evidence.get("accuracy_shift_replicated") else "no"),
    ]
    if "attempted" in evidence:
        values.extend(
            [
                (
                    "Attempted / deliberately skipped accuracy batches",
                    f"{evidence['attempted']} / {evidence['deliberately_skipped']}",
                ),
                (
                    "Accuracy decision complete",
                    "yes" if evidence.get("decision_complete") else "no",
                ),
                (
                    "Stop reason",
                    (evidence.get("stopping") or {}).get(
                        "reason", "All requested accuracy evaluations attempted or still running"
                    ),
                ),
            ]
        )
    table = "".join(
        f"<tr><th>{html.escape(name)}</th><td>{html.escape(value)}</td></tr>"
        for name, value in values
    )
    return (
        "<h2>Replicated accuracy</h2><p>Compiled gradient error against fp64, candidate versus "
        "baseline. Each input batch contributes one median across paired parameter gradients. "
        "Gradient targets within a batch are correlated; shared weights and source material "
        "also limit batch independence.</p>"
        f"<table>{table}</table><p>The fixed rule requires at least ten complete batches, at least "
        "80% worsening, and median growth of at least 1.25×. Missing scores remain in coverage. "
        "A flag is advisory evidence, not a compiler-defect claim.</p>"
    )


def finding_html(batch, side, result):
    failed = [
        row
        for row in result.get("comparison", {}).get("outputs", [])
        if not row.get("matches", False)
    ]
    context = result.get("numerical_context", {})
    triage = context.get("triage", {})
    if not failed and not triage.get("contrary_evidence"):
        return ""
    precision = {row["target"]: row for row in result.get("fp64_advisory", {}).get("outputs", [])}
    # Put positive noise-floor evidence first, keeping every original observation in JSON.
    failed.sort(
        key=lambda row: not precision.get(row["target"], {}).get("farther_than_noise_floor", False)
    )
    rows = []
    for observation in failed[:10]:
        target = observation["target"]
        fp64 = precision.get(target, {})
        values = [
            target,
            number(fp64.get("eager", {}).get("rmse")),
            number(fp64.get("candidate", {}).get("rmse")),
            number(fp64.get("noise_floor", {}).get("value")),
            str(fp64.get("accuracy_label") or "unavailable"),
        ]
        rows.append(
            "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in values) + "</tr>"
        )
    reasons = (
        ", ".join(triage.get("contrary_evidence", []))
        or "No positive contrary diagnostic recorded."
    )
    coverage = (
        "; ".join(
            f"{label}: {row.get('matched', 0)}/{row.get('total', len(failed))}"
            for label, row in context.get("label_coverage", {}).items()
        )
        or "No explanation coverage recorded."
    )
    return (
        f"<details><summary>Batch {batch}, {html.escape(side)}: {len(failed)} failing targets</summary>"
        f"<p>{html.escape(reasons)}</p><p>{html.escape(coverage)}</p>"
        "<table><tr><th>Target</th><th>Eager RMSE vs fp64</th><th>Compiled RMSE vs fp64</th>"
        "<th>Noise floor</th><th>Accuracy label</th></tr>"
        f"{''.join(rows)}</table><p>Showing {min(10, len(failed))}/{len(failed)} failing targets; "
        "the full evidence JSON retains every target and diagnostic. These values do not change "
        "the primary comparison or its tolerances.</p></details>"
    )
