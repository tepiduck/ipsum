"""Backend ladder over one frozen workload, with conservative attribution."""

import sys
from dataclasses import replace
from pathlib import Path

from .compare_envs import interpreter
from .evaluate import Settings
from .native import NativeJob, verify_native
from .native_artifacts import copy_workload, export_native, load_native_run, write_native_report
from .portable import REDUCIBLE, fingerprint, same_failure
from .runner import run_case
from .storage import atomic_json

LADDER = ("eager", "aot_eager", "aot_eager_decomp_partition", "inductor")
BOUNDARIES = (
    "Dynamo capture or captured-graph execution",
    "AOTAutograd tracing or functionalization",
    "decomposition or partitioning",
    "Inductor lowering or code generation",
)


def symptom(result):
    # Only backend identity is normalized. This compares observed symptoms, not root causes.
    return fingerprint({**result, "backend": "diagnostic"})


def interpret_ladder(outcomes):
    if len(outcomes) != len(LADDER):
        raise ValueError("A complete backend ladder is required")
    if any(x["status"] not in {"pass", *REDUCIBLE} for x in outcomes):
        return {"status": "inconclusive", "boundary": None, "exit_code": 2}
    failures = [i for i, x in enumerate(outcomes) if x["status"] != "pass"]
    if not failures:
        return {"status": "agreement", "boundary": None, "exit_code": 0}
    first = failures[0]
    if failures != list(range(first, len(LADDER))):
        return {"status": "non_monotonic", "boundary": None, "exit_code": 1}
    if len({symptom(outcomes[i]) for i in failures}) != 1:
        return {"status": "different_failures", "boundary": None, "exit_code": 1}
    return {
        "status": "observed_boundary",
        "boundary": BOUNDARIES[first],
        "first_failing_backend": LADDER[first],
        "exit_code": 1,
    }


def diagnose_native(directory, output, settings=None, *, python=None, progress=print):
    directory, output = Path(directory), Path(output)
    expected, original = None, None
    if (directory / "evidence.json").exists():
        if settings is not None:
            raise ValueError("Saved runs retain their execution policy; overrides are not accepted")
        original, settings, expected = load_native_run(directory)
        if original.only_graph is not None or original.only_aot is not None:
            raise ValueError("Diagnose the full workload before graph isolation")
        bundle = original.bundle
    else:
        bundle = directory
        settings = settings or Settings()
    settings.validate()
    if verify_native(bundle)["kind"] != "workload":
        raise ValueError("The backend ladder needs an original-model bundle, not a detached graph")
    python = interpreter(python or sys.executable)
    output.mkdir(parents=True, exist_ok=False)
    copied = copy_workload(bundle, output / "workload")
    job = NativeJob.create(copied, requests=original.requests if original else None)
    summary = {
        "schema_version": 1,
        "kind": "backend_ladder",
        "settings": settings.to_dict(),
        "cases": [],
        "status": "in_progress",
        "exit_code": 2,
        "injected_fault": settings.backend.startswith("mutant:"),
        "aot_capture_for_ladder": False,
        "note": "Observed backend boundary, not proof of root cause. All rungs run separately against eager. The ladder uses the normal backend pipeline without diagnostic AOT capture. Inductor compile modes apply only to the Inductor rung.",
    }
    if original:
        anchor = run_case(
            replace(original, bundle=str(copied)), settings, output / "confirmation", python=python
        )
        summary["source_confirmation"] = anchor
        summary["source_reproduced"] = (
            anchor["status"] == "pass"
            if expected["status"] == "pass"
            else same_failure(expected, anchor)
        )
    results = []
    for index, backend in enumerate(LADDER):
        policy = replace(
            settings,
            backend=backend,
            compile_mode=settings.compile_mode if backend == "inductor" else "default",
        )
        target = output / "rungs" / f"{index}-{backend}"
        result = run_case(job, policy, target / "run", python=python)
        export_native(job, policy, target / "reproducer", result)
        write_native_report(
            target,
            {"schema_version": 1, "status": result["status"], "backend": backend, "result": result},
        )
        results.append(result)
        summary["cases"].append(
            {
                "id": backend,
                "status": result["status"],
                "graph_count": result.get("graph_count"),
                "report": f"rungs/{index}-{backend}/report.html",
                "settings": policy.to_dict(),
                "result": result,
            }
        )
        atomic_json(output / "summary.json", summary)
        if progress:
            progress(f"{backend}: {result['status']}", flush=True)
    summary.update(interpret_ladder(results))
    if original and not summary["source_reproduced"]:
        summary.update(status="source_failure_not_reproduced", boundary=None, exit_code=2)
    elif (
        original
        and expected["status"] in REDUCIBLE
        and (results[-1]["status"] not in REDUCIBLE or symptom(anchor) != symptom(results[-1]))
    ):
        summary.update(status="source_symptom_not_observed", boundary=None, exit_code=2)
    write_native_report(output, summary)
    return summary
