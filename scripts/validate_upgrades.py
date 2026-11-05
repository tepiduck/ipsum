"""Reproduce the model, backward, trace, and environment-gate experiments."""

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from faultline.campaign import run_campaign
from faultline.compare_envs import compare_environments
from faultline.environment import describe
from faultline.evaluate import Settings
from faultline.export import export_case
from faultline.generate import corpus
from faultline.ir import Case, Input, Node
from faultline.lowering import freeze
from faultline.model_source import capture_factory
from faultline.reduce import minimize
from faultline.runner import run_case
from faultline.storage import atomic_json
from faultline.trace import Trace


def validate(output, compiler=False, baseline_python=None):
    output.mkdir(parents=True, exist_ok=False)
    evidence = {
        "schema_version": 1,
        "environment": describe(),
        "checks": [],
        "claim": "Harness validation with controlled faults; no confirmed compiler bug.",
    }

    def record(name, passed, **data):
        evidence["checks"].append({"name": name, "passed": passed, **data})
        atomic_json(output / "validation.json", evidence)
        print(f"{name}: {'PASS' if passed else 'FAIL'}", flush=True)

    source = Path(__file__).resolve().parents[1] / "examples" / "models.py"
    models = [capture_factory(f"{source}:{name}") for name in ("mlp", "convolution", "attention")]
    for case, name in zip(models, ("mlp", "convolution", "attention"), strict=True):
        atomic_json(output / f"{name}.json", case.to_dict())
    settings = Settings(backend="inductor" if compiler else "eager", gradients=True, timeout_s=240)
    result = run_campaign(output / "models", models, settings)
    record(
        "captured-model-forward-and-backward",
        result["counts"] == {"pass": 3},
        counts=result["counts"],
        backend=settings.backend,
    )
    gradient = run_case(
        models[0], replace(settings, backend="mutant:gradient-scale"), output / "wrong-gradient"
    )
    forward = [
        x for x in gradient["comparison"]["outputs"] if not x["target"].startswith("gradient:")
    ]
    record(
        "identical-forward-wrong-backward",
        gradient["status"] == "gradient_mismatch" and all(x["matches"] for x in forward),
        status=gradient["status"],
    )
    trace = Trace(
        tuple(
            freeze(
                Case(
                    (Input("x", (n, 2), values=(1.0,) * (n * 2)),),
                    (Node("y", "sin", ("x",), {}),),
                    ("y",),
                )
            )
            for n in (5, 2, 7, 3, 6, 2, 8)
        ),
        "controlled-sequence",
    )
    trace_settings = Settings(backend="mutant:sequence-cache", repeats=2)
    isolated = [
        run_case(Trace((step,)), trace_settings, output / "isolated" / str(i))["status"]
        for i, step in enumerate(trace.steps)
    ]
    record("isolated-requests-do-not-trigger-sequence-fault", set(isolated) == {"pass"})
    attempts = 0

    def oracle(proposal):
        nonlocal attempts
        attempts += 1
        return run_case(proposal, trace_settings, output / "trace-reduction" / str(attempts))

    reduced, reduction = minimize(trace, oracle, 32)
    atomic_json(output / "trace-reduction" / "reduction.json", reduction)
    sizes = [step.inputs[0].shape[0] for step in reduced.steps]
    record(
        "whole-trace-reduction",
        sizes == [2, 3, 2],
        original_requests=7,
        reduced_batch_sizes=sizes,
        evaluations=reduction["evaluations"],
    )
    repro = export_case(
        reduced, trace_settings, output / "trace-reproducer", reduction["confirmation"]
    )
    execution = subprocess.run(
        [sys.executable, "-I", str(repro.resolve())],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    observed = json.loads(execution.stdout.splitlines()[-1]) if execution.stdout else {}
    record(
        "standalone-trace-reproducer",
        execution.returncode == 1 and observed.get("reproduces_expected"),
    )
    captured_trace = capture_factory(f"{source}:requests", trace=True)
    outcome = run_case(captured_trace, replace(settings, dynamic=True), output / "dynamic-trace")
    record(
        "dynamic-shape-trace-forward-and-backward",
        outcome["status"] == "pass",
        status=outcome["status"],
    )
    baseline_python = baseline_python or sys.executable
    gate = compare_environments(
        models, output / "agreement-gate", baseline_python, sys.executable, settings, progress=print
    )
    record(
        "environment-agreement-control", gate["counts"] == {"agreement": 3}, counts=gate["counts"]
    )
    base = Settings(backend="eager", repeats=2)
    gate = compare_environments(
        corpus()[:1],
        output / "injected-gate",
        baseline_python,
        sys.executable,
        base,
        replace(base, backend="mutant:sum-as-mean"),
        reduce=True,
        max_evaluations=8,
    )
    record(
        "new-regression-gate-and-baseline-preserving-reduction",
        gate["exit_code"] == 1
        and gate["cases"][0].get("reduction", {}).get("baseline_confirmed") is True,
        counts=gate["counts"],
    )
    evidence["validation_passed"] = all(check["passed"] for check in evidence["checks"])
    atomic_json(output / "validation.json", evidence)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", action="store_true")
    parser.add_argument("--baseline-python")
    args = parser.parse_args()
    result = validate(args.output, args.compiler, args.baseline_python)
    sys.exit(0 if result["validation_passed"] else 1)
