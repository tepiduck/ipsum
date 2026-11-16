"""GitHub Actions adapter. Watch CLI verdicts and exit codes remain unchanged."""

import argparse
import json
import os
import sys
import time

from .storage import atomic_json, read_json
from .watch import _watch_command, add_watch_command
from .watch_support import HistoryIncompatible


def annotation(message, *, error=False):
    # Workflow-command escaping, including untrusted exception/reason text.
    escaped = str(message).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::{'error' if error else 'warning'}::{escaped}", flush=True)


def selected_records(root):
    path = root / "history.json"
    if not path.exists():
        return []
    history = read_json(path)
    keys = ("date", "model", "reference")
    planned = {tuple(row[k] for k in keys) for row in history["planned"]}
    return [r for r in history["records"] if tuple(r[k] for k in keys) in planned]


def run(args):
    root = args.output.resolve()
    notes, error, archive = [], None, None
    try:
        try:
            code = _watch_command(args)
        except HistoryIncompatible as exc:
            # Only an identity mismatch permits rotation. Corruption/modified frozen
            # evidence remains INCONCLUSIVE, without discarding or trusting it.
            archive = root.with_name(f"{root.name}-archived-{time.time_ns()}")
            root.rename(archive)
            notes.append(f"Incompatible restored history: {exc}. Preserved at {archive.name}.")
            code = _watch_command(args)
    except Exception as exc:
        # CI boundary: an orchestration exception is not a model REGRESSION.
        code, error = 3, f"{type(exc).__name__}: {exc}"
        notes.append(error)
    try:
        rows = selected_records(root)
        if (root / "active-plan.json").exists():
            marker = read_json(root / "active-plan.json")
            plan = read_json(root / marker["file"])
            availability = plan.get("availability", {})
            if availability.get("fallback"):
                notes.append(availability["reason"])
        for row in rows:
            if row["verdict"] in {"NEEDS_REVIEW", "INCONCLUSIVE", "REGRESSION"}:
                notes.append(
                    f"{row['model']} {row['date']} vs {row['reference']}: {row['verdict']}: "
                    + json.dumps(row.get("reasons", []), ensure_ascii=True)
                )
        if code == 3 and not error:
            history = read_json(root / "history.json")
            if history["state"] != "complete":
                notes.append(
                    f"Watch incomplete: {history['state']}. See retained interruption logs."
                )
    except (ValueError, OSError, KeyError) as exc:
        rows = []
        notes.append(f"Summary unavailable: {exc}")
    result = {
        "schema_version": 1,
        "watch_exit_code": code,
        "workflow_exit_code": 1 if code == 1 else 0,
        "outcome": {0: "SAFE", 1: "REGRESSION", 2: "NEEDS_REVIEW", 3: "INCONCLUSIVE"}[code],
        "notes": notes,
        "records": rows,
        "archived_history": archive.name if archive else None,
    }
    atomic_json(root / "ci-result.json", result)
    return result


def publish(result):
    outcome = result["outcome"]
    warning = outcome in {"INCONCLUSIVE", "NEEDS_REVIEW"}
    lines = [f"## {'⚠️ ' if warning else ''}Nightly watch: {outcome}", ""]
    if warning:
        lines.append(
            "**This green job is not a SAFE result. Investigation or another run is needed.**"
        )
    lines.extend(["", "| Model | Nightly | Reference | Verdict |", "|---|---|---|---|"])
    for row in result["records"]:
        lines.append(f"| {row['model']} | {row['date']} | {row['reference']} | {row['verdict']} |")
    for note in result["notes"]:
        lines.append("\n- " + str(note).replace("\n", " "))
        annotation(note, error=outcome == "REGRESSION")
    if warning and not result["notes"]:
        annotation(
            f"{outcome}: watch returned exit {result['watch_exit_code']}; see retained reports."
        )
    lines.append("\nReports, preparation logs and any archived history are in this job's artifact.")
    summary = "\n".join(lines) + "\n"
    print(summary, flush=True)
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(summary)
    if path := os.environ.get("GITHUB_OUTPUT"):
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(f"exit_code={result['watch_exit_code']}\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_watch_command(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(["watch", *(sys.argv[1:] if argv is None else argv)])
    publish(run(args))
    # The workflow's final step alone maps recorded exit 1 to a red job.


if __name__ == "__main__":
    main()
