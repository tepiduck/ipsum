"""Run the falsifiable Faultline validation experiment and emit reviewable evidence."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from faultline.campaign import run_campaign
from faultline.environment import describe
from faultline.evaluate import Settings
from faultline.export import export_case
from faultline.finding import load_finding, replay_finding
from faultline.generate import corpus, generate, reduction_fixture
from faultline.lowering import freeze
from faultline.reduce import minimize
from faultline.report import write_report
from faultline.runner import run_case
from faultline.storage import atomic_json, read_json


def validate(output, compiler=False):
    output.mkdir(parents=True, exist_ok=False)
    checks = []
    suites = [
        ("eager-control", "eager", corpus(), "pass"),
        ("sum-mutant", "mutant:sum-as-mean", [corpus()[0]], "numerical_mismatch"),
        ("relu-mutant", "mutant:relu-as-neg", [corpus()[0]], "numerical_mismatch"),
        ("softmax-mutant", "mutant:softmax-axis", [corpus()[1]], "numerical_mismatch"),
        ("nan-mutant", "mutant:nan", [corpus()[0]], "numerical_mismatch"),
        ("unstable-control", "mutant:flaky", [corpus()[0]], "unstable_candidate"),
        ("exception-control", "mutant:crash", [corpus()[0]], "compile_error"),
        ("worker-crash-control", "mutant:worker-exit", [corpus()[0]], "crash"),
    ]
    evidence = {
        "schema_version": 1,
        "environment": describe(),
        "checks": checks,
        "claim": "Controlled fault detection and reduction; not discovery of real compiler bugs.",
    }
    for name, backend, cases, expected in suites:
        result = run_campaign(output / name, cases, Settings(backend=backend, timeout_s=180))
        write_report(output / name)
        checks.append(
            {
                "name": name,
                "expected": {expected: len(cases)},
                "observed": result["counts"],
                "passed": result["counts"] == {expected: len(cases)},
            }
        )
        atomic_json(output / "validation.json", evidence)
    call = 0
    settings = Settings(backend="mutant:sum-as-mean")

    def oracle(case):
        nonlocal call
        call += 1
        print(f"reduction evaluation {call}: {case.complexity()}", flush=True)
        return run_case(case, settings, output / "reduction" / "attempts" / f"{call:04d}")

    case, reduction = minimize(freeze(reduction_fixture()), oracle, 48)
    atomic_json(output / "reduction" / "reduction.json", reduction)
    repro = export_case(
        case, settings, output / "reduction" / "reproducer", reduction["confirmation"]
    )
    execution = subprocess.run(
        [sys.executable, "-I", str(repro.resolve())],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    reproduced = json.loads(execution.stdout.splitlines()[-1]) if execution.stdout else {}
    checks.append(
        {
            "name": "independent-standalone-reproducer",
            "passed": execution.returncode == 1
            and reproduced.get("status") == "numerical_mismatch"
            and reproduced.get("reproduces_expected") is True,
            "observed": reproduced,
        }
    )
    checks.append(
        {
            "name": "reduction-retains-target-operation",
            "passed": len(case.nodes) == 1 and case.nodes[0].op == "sum",
            "original_complexity": reduction["original_complexity"],
            "reduced_complexity": reduction["reduced_complexity"],
        }
    )
    saved = load_finding(output / "reduction" / "reproducer")
    original = {path.name: path.read_bytes() for path in saved.directory.iterdir()}
    replay = replay_finding(saved, output / "saved-finding-replay")
    checks.append(
        {
            "name": "saved-finding-replay-preserves-policy-and-evidence",
            "passed": replay["reproduces_expected"]
            and read_json(output / "saved-finding-replay" / "job.json")["settings"]
            == saved.settings.to_dict()
            and {path.name: path.read_bytes() for path in saved.directory.iterdir()} == original,
            "observed": replay,
        }
    )
    exception = load_finding(next((output / "exception-control" / "cases").iterdir()))
    exception_repro = export_case(
        exception.case, exception.settings, output / "exception-reproducer", exception.outcome
    )
    execution = subprocess.run(
        [sys.executable, "-I", str(exception_repro.resolve())],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    observed = json.loads(execution.stdout.splitlines()[-1]) if execution.stdout else {}
    checks.append(
        {
            "name": "standalone-exception-confirmation",
            "passed": execution.returncode == 1
            and observed.get("reproduces_expected") is True
            and observed.get("exception_confirmations") == exception.settings.repeats,
            "observed": observed,
        }
    )
    resume_dir = output / "resume-control"
    run_campaign(resume_dir, corpus()[:2], Settings(backend="eager"), stop_after=1)
    first = next((resume_dir / "cases").glob("*/outcome.json"))
    before = first.read_bytes()
    resumed = run_campaign(resume_dir, resume=True)
    checks.append(
        {
            "name": "resume-preserves-completed-evidence",
            "passed": first.read_bytes() == before and resumed["completed"] == 2,
        }
    )
    if compiler:
        compiled = run_campaign(
            output / "inductor",
            [*corpus(), *(generate(i) for i in range(12))],
            Settings(backend="inductor", timeout_s=180),
        )
        write_report(output / "inductor")
        evidence["real_compiler"] = {
            "counts": compiled["counts"],
            "failure_families": compiled["failure_families"],
            "note": "Tolerance exceedances are unconfirmed candidates.",
        }
        checks.append(
            {
                "name": "compiler-toolchain-operational",
                "passed": set(compiled["counts"])
                <= {"pass", "numerical_mismatch", "structural_mismatch"},
            }
        )
    evidence["validation_passed"] = all(check["passed"] for check in checks)
    atomic_json(output / "validation.json", evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", action="store_true")
    args = parser.parse_args()
    result = validate(args.output, args.compiler)
    print(json.dumps({"passed": result["validation_passed"], "checks": len(result["checks"])}))
    return 0 if result["validation_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
