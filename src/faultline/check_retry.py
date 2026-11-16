"""Execution selection for one explicit GPU deterministic retry; no oracle changes."""

import html
import re
from dataclasses import replace

UNSTABLE = {"unstable_reference", "unstable_candidate"}
MESSAGE = "default execution was nondeterministic on this device; verdict uses --deterministic"


def needed(row, sides, enabled):
    return enabled and any(
        settings.device.startswith("cuda")
        and not settings.deterministic
        and row.get(side, {}).get("status") in UNSTABLE
        for side, _, settings in sides
    )


def effective(settings, row):
    return (
        replace(settings, deterministic=True)
        if "deterministic_retry" in row and settings.device.startswith("cuda")
        else settings
    )


def unavailable(outcome):
    if outcome.get("status") in UNSTABLE:
        return f"deterministic execution remains {outcome['status']}"
    error = str(outcome.get("error", outcome.get("message", "")))
    if re.search(
        r"does not have (?:a )?deterministic implementation|no deterministic implementation|"
        r"deterministic (?:algorithms?|implementations?) (?:are|is) not (?:available|supported)",
        error,
        re.IGNORECASE,
    ):
        return error
    return None


def notes(rows):
    result = []
    for row in rows:
        retry = row.get("deterministic_retry")
        if not retry:
            continue
        result.append({"code": "deterministic_retry", "batch": row["batch"], "message": MESSAGE})
        for side, reason in retry["unavailable"].items():
            result.append(
                {
                    "code": "deterministic_retry_unavailable",
                    "batch": row["batch"],
                    "side": side,
                    "message": f"{side} has no eligible deterministic primary execution: {reason}",
                }
            )
    return result


def retry_html(rows):
    entries = []
    for row in rows:
        retry = row.get("deterministic_retry")
        if not retry:
            continue
        for side, outcome in retry["outcomes"].items():
            values = (
                row["batch"],
                side,
                retry["original_results"][side]["status"],
                outcome["status"],
                retry["unavailable"].get(side, "deterministic attempt selected"),
            )
            entries.append(
                "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in values) + "</tr>"
            )
    if not entries:
        return ""
    return (
        "<h2>GPU deterministic retries</h2><p>"
        + MESSAGE
        + ". Original and retry outcomes are retained "
        "in the evidence and immutable attempt directories. A retry without a stable supported execution "
        "does not supply a primary result to the verdict.</p><table><tr><th>Batch</th><th>Environment</th>"
        "<th>Default status</th><th>Deterministic status</th><th>Selection</th></tr>"
        + "".join(entries)
        + "</table>"
    )
