"""Two-interpreter regression gates with baseline-preserving automatic reduction."""

import html
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from .evaluate import Settings
from .export import export_case
from .lowering import freeze
from .portable import REDUCIBLE, same_failure
from .reduce import minimize
from .runner import run_case
from .storage import atomic_json


def interpreter(path):
    resolved = shutil.which(str(path)) or str(Path(path).expanduser().resolve())
    if not Path(resolved).is_file():
        raise ValueError(f"Python interpreter does not exist: {path}")
    return resolved


def classification(baseline, candidate):
    a, b = baseline["status"], candidate["status"]
    if a not in {"pass", *REDUCIBLE} or b not in {"pass", *REDUCIBLE}:
        return "inconclusive"
    if a == "pass":
        return "agreement" if b == "pass" else "regression"
    if b == "pass":
        return "not_observed"
    return "recurring" if same_failure(baseline, candidate) else "changed_failure"


def gate_status(counts):
    if counts.get("regression", 0) or counts.get("changed_failure", 0):
        return 1
    return 2 if counts.get("inconclusive", 0) else 0


def reduce_regression(
    case,
    baseline_settings,
    candidate_settings,
    baseline_python,
    candidate_python,
    output,
    expected,
    max_evaluations=64,
    progress=print,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    calls = 0
    last_baseline = None

    def oracle(proposal):
        nonlocal calls, last_baseline
        calls += 1
        directory = output / "attempts" / f"{calls:04d}"
        last_baseline = run_case(
            proposal, baseline_settings, directory / "baseline", python=baseline_python
        )
        if last_baseline["status"] != "pass":
            return {
                **last_baseline,
                "status": "baseline_rejected",
                "fingerprint": "baseline_rejected",
            }
        candidate = run_case(
            proposal, candidate_settings, directory / "candidate", python=candidate_python
        )
        candidate["baseline_status"] = last_baseline["status"]
        if progress:
            progress(
                f"regression reduction {calls}: {proposal.complexity()} -> {candidate['status']}"
            )
        return candidate

    reduced, evidence = minimize(case, oracle, max_evaluations, expected=expected)
    evidence["baseline_confirmation"] = last_baseline
    evidence["predicate"] = "baseline passes AND candidate retains original failure"
    atomic_json(output / "reduction.json", evidence)
    export_case(
        reduced,
        candidate_settings,
        output / "candidate-reproducer",
        evidence["confirmation"],
        environment=evidence["confirmation"].get("worker_environment"),
    )
    export_case(
        reduced,
        baseline_settings,
        output / "baseline-reproducer",
        last_baseline,
        environment=last_baseline.get("worker_environment"),
    )
    return evidence


def compare_environments(
    cases,
    output,
    baseline_python,
    candidate_python=sys.executable,
    settings=None,
    candidate_settings=None,
    reduce=False,
    max_evaluations=64,
    progress=print,
):
    settings = settings or Settings()
    candidate_settings = candidate_settings or settings
    settings.validate()
    candidate_settings.validate()
    # A different backend can be deliberate, but the comparison policy must match.
    a, b = settings.to_dict(), candidate_settings.to_dict()
    if {k: v for k, v in a.items() if k != "backend"} != {
        k: v for k, v in b.items() if k != "backend"
    }:
        raise ValueError("baseline and candidate must use the same execution/comparison policy")
    if not cases or max_evaluations < 2:
        raise ValueError("comparison requires cases and a reduction budget >= 2")
    baseline_python, candidate_python = interpreter(baseline_python), interpreter(candidate_python)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema_version": 1,
        "baseline_python": baseline_python,
        "candidate_python": candidate_python,
        "settings": a,
        "candidate_settings": b,
        "cases": [],
        "counts": {},
        "exit_code": 0,
        "injected_fault": any(
            s.backend.startswith("mutant:") for s in (settings, candidate_settings)
        ),
        "note": "Correctness gate, not a latency benchmark. Not-observed findings are not proof of fixes.",
    }
    for index, case in enumerate(cases):
        case = freeze(case)
        key = f"{index:04d}-{case.digest()[:12]}"
        directory = output / "cases" / key
        atomic_json(directory / "case.json", case.to_dict())
        baseline = run_case(case, settings, directory / "baseline", python=baseline_python)
        candidate = run_case(
            case, candidate_settings, directory / "candidate", python=candidate_python
        )
        label = classification(baseline, candidate)
        record = {
            "id": key,
            "tag": case.tag,
            "classification": label,
            "baseline": baseline,
            "candidate": candidate,
        }
        if progress:
            progress(f"{key}: {baseline['status']} -> {candidate['status']} ({label})")
        if reduce and label == "regression":
            try:
                evidence = reduce_regression(
                    case,
                    settings,
                    candidate_settings,
                    baseline_python,
                    candidate_python,
                    directory / "reduced",
                    candidate,
                    max_evaluations,
                    progress,
                )
                record["reduction"] = {
                    "original_complexity": evidence["original_complexity"],
                    "reduced_complexity": evidence["reduced_complexity"],
                    "evaluations": evidence["evaluations"],
                    "baseline_confirmed": evidence["baseline_confirmation"]["status"] == "pass",
                }
            except (ValueError, RuntimeError, OSError) as exc:
                record["reduction_error"] = str(exc)
        summary["cases"].append(record)
        summary["counts"] = dict(Counter(r["classification"] for r in summary["cases"]))
        summary["exit_code"] = gate_status(summary["counts"])
        atomic_json(output / "comparison.json", summary)
        write_comparison_report(output, summary)
    return summary


def write_comparison_report(directory, summary):
    rows = []
    for item in summary["cases"]:
        key = html.escape(item["id"], quote=True)
        links = f'<a href="cases/{key}/baseline/outcome.json">baseline</a> · <a href="cases/{key}/candidate/outcome.json">candidate</a>'
        if "reduction" in item:
            links += f' · <a href="cases/{key}/reduced/candidate-reproducer/repro.py">minimal reproducer</a>'
        rows.append(
            f"<tr><td>{html.escape(item['tag'])}</td><td>{html.escape(item['classification'])}</td>"
            f"<td>{html.escape(item['baseline']['status'])}</td><td>{html.escape(item['candidate']['status'])}</td><td>{links}</td></tr>"
        )
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Faultline environment comparison</title>
<style>body{{font:16px system-ui;margin:40px;max-width:1200px}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #ddd}}code{{white-space:pre-wrap}}</style>
<h1>Compiler environment comparison</h1><p>{html.escape(summary["note"])}</p>
<p>Controlled fault injection: {summary["injected_fault"]}</p>
<pre>{html.escape(json.dumps(summary["counts"], indent=2))}</pre><a href="comparison.json">Full comparison and environments</a>
<table><thead><tr><th>Case</th><th>Classification</th><th>Baseline</th><th>Candidate</th><th>Evidence</th></tr></thead><tbody>{"".join(rows)}</tbody></table></html>"""
    (Path(directory) / "report.html").write_text(document, encoding="utf-8")
