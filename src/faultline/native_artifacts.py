"""Durable native workload runs, environment gates, and standalone bundles."""

import html
import json
import shutil
import sys
from pathlib import Path

from .compare_envs import classification, gate_status, interpreter
from .evaluate import Settings
from .native import NativeJob, file_hash, verify_native
from .portable import REDUCIBLE, same_failure
from .runner import run_case
from .storage import atomic_json, read_json


def copy_workload(source, target):
    source, target = Path(source), Path(target)
    manifest = verify_native(source)
    target.mkdir(parents=True, exist_ok=False)
    for name in ["workload.json", *manifest["files"]]:
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, destination)
    verify_native(target)
    return target


def export_native(job, settings, output, expected=None):
    job.validate()
    settings.validate()
    if expected and (
        expected.get("case_hash") != job.digest() or expected.get("backend") != settings.backend
    ):
        raise ValueError("Saved outcome does not belong to this workload/backend")
    if expected:
        if (
            expected.get("compile_mode", "default") != settings.compile_mode
            or expected.get("autocast") != settings.autocast
        ):
            raise ValueError("Saved outcome has a different compiler/autocast policy")
        for name in (
            "device",
            "gradients",
            "dynamic",
            "repeats",
            "rtol",
            "atol",
            "compile_mode",
            "autocast",
        ):
            if name in expected and expected[name] != getattr(settings, name):
                raise ValueError(f"Saved outcome has a different {name} policy")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    bundle = copy_workload(job.bundle, output / "workload")
    source = Path(__file__).parent
    for name in (
        "native_runtime.py",
        "native_graph.py",
        "native_aot.py",
        "native_compile.py",
        "portable.py",
    ):
        shutil.copyfile(source / name, output / name)
    shutil.copyfile(source / "native_repro.py", output / "repro.py")
    manifest = verify_native(bundle)
    names = [
        "repro.py",
        "native_runtime.py",
        "native_graph.py",
        "native_aot.py",
        "native_compile.py",
        "portable.py",
        "workload/workload.json",
        *["workload/" + name for name in manifest["files"]],
    ]
    atomic_json(
        output / "reproducer.json",
        {
            "schema_version": 1,
            "settings": settings.to_dict(),
            "expected": expected,
            "only_graph": job.only_graph,
            "aot": job.aot,
            "only_aot": job.only_aot,
            "requests": job.requests,
            "files": {name: file_hash(output / name) for name in names},
        },
    )
    (output / "README.txt").write_text(
        "Run: python repro.py\nRequires PyTorch and the factory's dependencies; no Faultline install.\n"
        "This bundle contains executable Python, exact model weights, inputs, and environment metadata.\n"
        "Use trusted bundles. Review all files before sharing. No resource supervisor in standalone mode.\n"
        "Exit 0: expected agreement; 1: expected stable failure; 2: inconclusive or different outcome.\n",
        encoding="utf-8",
    )
    return output / "repro.py"


def write_native_report(output, summary):
    output = Path(output)
    atomic_json(output / "summary.json", summary)
    body = html.escape(json.dumps(summary, indent=2))
    links = ['<a href="summary.json">JSON result</a>']
    if (output / "workload" / "workload.json").exists():
        links.append('<a href="workload/workload.json">Workload manifest</a>')
    if (output / "run" / "aot-graphs.json").exists():
        links.append('<a href="run/aot-graphs.json">AOT graph index</a>')
    for name in ("candidate", "baseline", "run"):
        if (output / name).exists():
            links.append(f'<a href="{name}/outcome.json">{name} outcome</a>')
    table = ""
    if summary.get("cases"):
        rows = []
        for case in summary["cases"]:
            label = html.escape(case["id"])
            if case.get("report"):
                label = f'<a href="{html.escape(case["report"], quote=True)}">{label}</a>'
            rows.append(f"<tr><td>{label}</td><td>{html.escape(case['status'])}</td></tr>")
        table = "<table><tr><th>Workload</th><th>Outcome</th></tr>" + "".join(rows) + "</table>"
    (output / "report.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Faultline model comparison</title>'
        "<style>body{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 20px}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}a{margin-right:20px}</style>"
        "<h1>Model comparison</h1>" + " ".join(links) + table + f"<pre>{body}</pre>",
        encoding="utf-8",
    )


def run_native(
    bundle,
    output,
    settings,
    *,
    python=None,
    expected=None,
    only_graph=None,
    requests=None,
    aot=False,
    only_aot=None,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    copied = copy_workload(bundle, output / "workload")
    job = NativeJob.create(
        copied, only_graph=only_graph, requests=requests, aot=aot, only_aot=only_aot
    )
    atomic_json(output / "settings.json", settings.to_dict())
    atomic_json(
        output / "selection.json",
        {"only_graph": only_graph, "requests": requests, "aot": aot, "only_aot": only_aot},
    )
    result = run_case(job, settings, output / "run", python=interpreter(python or sys.executable))
    atomic_json(
        output / "evidence.json",
        {
            "schema_version": 1,
            "files": {
                name: file_hash(output / name)
                for name in (
                    "settings.json",
                    "selection.json",
                    "workload/workload.json",
                    "run/job.json",
                    "run/outcome.json",
                )
            },
        },
    )
    code = 0 if result["status"] == "pass" else 1 if result["status"] in REDUCIBLE else 2
    summary = {
        "schema_version": 1,
        "status": result["status"],
        "exit_code": code,
        "graph_count": result.get("graph_count", 0),
        "aot_graph_count": len(result.get("aot_graphs", [])),
        "observation_mode": result.get("observation_mode", "model_outputs_and_gradients"),
        "parameter_count": result.get("parameter_count"),
        "injected_fault": settings.backend.startswith("mutant:"),
        "case_hash": job.digest(),
    }
    if expected:
        summary["replay"] = (
            "reproduced"
            if same_failure(expected, result)
            else "not_observed"
            if result["status"] == "pass"
            else "changed_or_inconclusive"
        )
    export_native(job, settings, output / "reproducer", result)
    write_native_report(output, summary)
    return summary


def load_native_run(directory):
    directory = Path(directory)
    evidence = read_json(directory / "evidence.json")
    required = {
        "settings.json",
        "selection.json",
        "workload/workload.json",
        "run/job.json",
        "run/outcome.json",
    }
    if evidence.get("schema_version") != 1 or set(evidence.get("files", {})) != required:
        raise ValueError("Invalid saved run evidence manifest")
    for name, digest in evidence["files"].items():
        if file_hash(directory / name) != digest:
            raise ValueError(f"Saved run {name} changed")
    selection = read_json(directory / "selection.json")
    job = NativeJob.create(directory / "workload", **selection)
    settings = Settings(**read_json(directory / "settings.json"))
    outcome = read_json(directory / "run" / "outcome.json")
    if outcome["case_hash"] != job.digest():
        raise ValueError("Saved run workload or selection changed")
    saved_job = read_json(directory / "run" / "job.json")
    if saved_job["settings"] != settings.to_dict():
        raise ValueError("Saved run settings changed")
    return job, settings, outcome


def compare_native_environments(
    bundle,
    output,
    baseline_python,
    candidate_python,
    settings,
    candidate_settings=None,
    reduce=False,
    max_evaluations=32,
    aot=False,
):
    candidate_settings = candidate_settings or settings
    a, b = settings.to_dict(), candidate_settings.to_dict()
    if {k: v for k, v in a.items() if k != "backend"} != {
        k: v for k, v in b.items() if k != "backend"
    }:
        raise ValueError("Baseline and candidate comparison policies must match")
    baseline_python, candidate_python = interpreter(baseline_python), interpreter(candidate_python)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    job = NativeJob.create(copy_workload(bundle, output / "workload"), aot=aot)
    baseline = run_case(job, settings, output / "baseline", python=baseline_python)
    candidate = run_case(job, candidate_settings, output / "candidate", python=candidate_python)
    label = classification(baseline, candidate)
    summary = {
        "schema_version": 1,
        "classification": label,
        "exit_code": gate_status({label: 1}),
        "baseline": baseline,
        "candidate": candidate,
        "injected_fault": any(
            s.backend.startswith("mutant:") for s in (settings, candidate_settings)
        ),
    }
    export_native(job, candidate_settings, output / "candidate-reproducer", candidate)
    export_native(job, settings, output / "baseline-reproducer", baseline)
    write_native_report(output, summary)
    if reduce and label == "regression":
        from .native_reduce import reduce_native

        summary["reduction"] = reduce_native(
            job,
            candidate_settings,
            candidate,
            output / "reduction",
            candidate_python=candidate_python,
            baseline_settings=settings,
            baseline_python=baseline_python,
            max_evaluations=max_evaluations,
        )
        write_native_report(output, summary)
    return summary
