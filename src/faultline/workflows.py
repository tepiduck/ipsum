"""End-to-end workflows shared by the command line and terminal interface."""

from pathlib import Path

from .campaign import run_campaign
from .evaluate import Settings
from .export import export_case
from .finding import load_finding
from .generate import reduction_fixture
from .reduce import minimize
from .report import write_report
from .runner import run_case
from .storage import atomic_json


def run_reduction(case, settings, output, max_evaluations=64, finding=None, progress=print):
    settings.validate()
    if max_evaluations < 2:
        raise ValueError("reduction needs at least two evaluations")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if finding:
        atomic_json(output / "source.json", finding.source_record())
    calls = 0

    def oracle(candidate):
        nonlocal calls
        calls += 1
        if progress:
            progress(f"evaluate {calls}: complexity={candidate.complexity()}")
        return run_case(candidate, settings, output / "attempts" / f"{calls:04d}")

    def accepted(event):
        if event["accepted"]:
            if progress:
                progress(f"accepted: {event['proposal']}")
            atomic_json(output / "latest-accepted.json", event)

    reduced, evidence = minimize(
        case,
        oracle,
        max_evaluations,
        accepted,
        expected=finding.outcome if finding else None,
    )
    atomic_json(output / "reduction.json", evidence)
    repro = export_case(reduced, settings, output / "reproducer", evidence["confirmation"])
    return repro, evidence


def run_demo(output, progress=print):
    """Demonstrate detection and reduction with an explicitly injected sum/mean fault."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    fixture = reduction_fixture()
    control = run_campaign(
        output / "control", [fixture], Settings(backend="eager"), progress=progress
    )
    write_report(output / "control")
    if control["counts"] != {"pass": 1}:
        raise RuntimeError("Demo eager control did not pass; inspect control/report.html")
    faults = run_campaign(
        output / "injected", [fixture], Settings(backend="mutant:sum-as-mean"), progress=progress
    )
    report = write_report(output / "injected")
    if faults["counts"] != {"numerical_mismatch": 1}:
        raise RuntimeError("Demo did not detect its controlled fault; inspect injected/report.html")
    saved = load_finding(output / "injected" / "cases" / faults["outcomes"][0]["id"])
    repro, evidence = run_reduction(
        saved.case, saved.settings, output / "reduced", 48, finding=saved, progress=progress
    )
    result = {
        "schema_version": 1,
        "injected_fault": True,
        "original_complexity": evidence["original_complexity"],
        "reduced_complexity": evidence["reduced_complexity"],
        "evaluations": evidence["evaluations"],
        "report": report.relative_to(output).as_posix(),
        "reproducer": repro.relative_to(output).as_posix(),
        "note": "Controlled sum/mean mutation; not an upstream compiler bug.",
    }
    atomic_json(output / "demo.json", result)
    return result
