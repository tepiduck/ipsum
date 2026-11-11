"""User-facing check coordinator; primary workers and verdict policy stay separate."""

import contextlib
import copy
import html
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

from .batches import verify_batches
from .bisect_env import supervise
from .bootstrap import python_command
from .check_report import accuracy_html, finding_html
from .compare_envs import interpreter
from .evaluate import Settings
from .native import NativeJob, file_hash
from .native_artifacts import export_native
from .native_runtime import native_failure_key
from .portable import REDUCIBLE
from .replication import accuracy_shift_replicated, batch_score
from .runner import run_case
from .storage import atomic_json, campaign_lock, provenance, read_json
from .verdict import assessment, summarize

PRESETS = {
    "quick": {"batches": 3, "gradients": False, "repeats": 2},
    "standard": {"batches": 10, "gradients": True, "repeats": 2},
    "thorough": {"batches": 10, "gradients": True, "repeats": 3},
}


@contextlib.contextmanager
def compiler_cache(directory):
    key = "TORCHINDUCTOR_CACHE_DIR"
    previous = os.environ.get(key)
    os.environ[key] = str(Path(directory).resolve())
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def diagnostic(job, settings, primary, directory, python, *, control=False):
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(
        directory / "job.json",
        {
            "bundle": job.bundle,
            "case_hash": job.digest(),
            "settings": settings.to_dict(),
            "primary": primary,
            "control": control,
        },
    )
    command = python_command(
        python,
        "from faultline.check_diagnostics import main;main()",
        [
            "--job",
            str(directory / "job.json"),
            "--output",
            str(directory / "result.json"),
        ],
    )
    observed = supervise(
        command,
        directory / "supervision",
        timeout=settings.timeout_s,
        max_rss_mb=settings.max_rss_mb,
    )
    if observed["status"] != "complete":
        status = (
            "diagnostic_skipped_memory"
            if observed["status"] == "resource_limit"
            else "diagnostic_skipped_timeout"
            if observed["status"] == "timeout"
            else "diagnostic_error"
        )
        return {
            "status": status,
            "message": "Diagnostic skipped (memory); primary result retained."
            if status == "diagnostic_skipped_memory"
            else f"Diagnostic {observed['status']}; primary result retained.",
            "supervision": observed,
        }
    result = read_json(directory / "result.json")
    return {**result, "supervision": observed}


def enriched(primary, diagnostic_result):
    result = copy.deepcopy(primary)
    if diagnostic_result.get("status") == "complete" and diagnostic_result.get(
        "replay_matches_primary"
    ):
        for key in ("fp64_advisory", "numerical_context"):
            if key in diagnostic_result:
                result[key] = diagnostic_result[key]
    return result


def inspect_side(bundle, directory, settings, python, level, *, paired):
    job = NativeJob.create(bundle)
    primary = run_case(job, settings, directory / "primary", python=python)
    result, phases = primary, []
    if level != "quick" and primary["status"] in {
        "pass",
        "gradient_mismatch",
        "numerical_mismatch",
    }:
        if paired or primary["status"] != "pass" or level == "thorough":
            evidence = diagnostic(
                job, settings, primary, directory / "fp64", python, control=level == "thorough"
            )
            phases.append(evidence)
            result = enriched(primary, evidence)
        if (
            level == "standard"
            and primary["status"] == "gradient_mismatch"
            and not assessment(result)["acceptable"]
        ):
            evidence = diagnostic(
                job, settings, primary, directory / "control", python, control=True
            )
            phases.append(evidence)
            # Preserve completed fp64 evidence if a later optional control is unavailable.
            if evidence.get("status") == "complete" and evidence.get("replay_matches_primary"):
                result = enriched(result, evidence)
        if level == "thorough" or (
            primary["status"] == "gradient_mismatch"
            and not assessment(result)["acceptable"]
            and not assessment(result)["contrary"]
        ):
            mask_settings = replace(settings, numerics=True, fp64_advisory=True)
            mask = run_case(
                NativeJob.create(bundle, aot=True),
                mask_settings,
                directory / "masks",
                python=python,
            )
            if (
                mask["status"] in {"pass", "gradient_mismatch", "numerical_mismatch"}
                and mask["status"] == primary["status"]
                and native_failure_key(mask["comparison"])
                == native_failure_key(primary["comparison"])
            ):
                result = enriched(
                    result,
                    {
                        "status": "complete",
                        "replay_matches_primary": True,
                        **{k: mask[k] for k in ("fp64_advisory", "numerical_context") if k in mask},
                    },
                )
                phases.append(
                    {
                        "status": "complete",
                        "phase": "masks",
                        "peak_tree_rss_mb": mask.get("peak_tree_rss_mb"),
                        "worker_wall_s": mask.get("worker_wall_s"),
                    }
                )
            else:
                status = (
                    "diagnostic_skipped_memory"
                    if mask["status"] == "resource_limit"
                    else "diagnostic_skipped_timeout"
                    if mask["status"] == "timeout"
                    else "diagnostic_replay_unavailable"
                )
                phases.append(
                    {
                        "status": status,
                        "phase": "masks",
                        "message": f"Mask diagnostic {mask['status']}; primary retained.",
                        "peak_tree_rss_mb": mask.get("peak_tree_rss_mb"),
                        "worker_wall_s": mask.get("worker_wall_s"),
                    }
                )
    return result, phases


def report(output, verdict, evidence):
    atomic_json(output / "evidence.json", evidence)
    atomic_json(output / "verdict.json", verdict)
    reasons = "".join(f"<li>{html.escape(r['message'])}</li>" for r in verdict["reasons"])
    notes = "".join(f"<li>{html.escape(r['message'])}</li>" for r in verdict["notes"])
    batches = []
    findings = []
    for row in evidence.get("batches", []):
        for side in ("baseline", "candidate"):
            if side not in row:
                continue
            result = row[side]
            findings.append(finding_html(row["batch"], side, result))
            context = result.get("numerical_context", {})
            phases = row.get("diagnostics", {}).get(side, [])
            values = [
                str(row["batch"]),
                side,
                result["status"],
                ", ".join(context.get("labels", [])) or "—",
                ", ".join(p.get("status", "unavailable") for p in phases) or "not requested",
            ]
            batches.append(
                "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in values) + "</tr>"
            )
    reproduction = "".join(
        f'<li><a href="{html.escape(path, quote=True)}">{html.escape(path)}</a></li>'
        for path in verdict["artifacts"]["reproducers"]
    )
    scope = verdict["scope"]
    configuration = evidence.get("configuration", {})
    environments = []
    for side in ("baseline", "candidate"):
        environment = configuration.get(f"{side}_environment")
        if environment:
            settings = configuration.get(
                "baseline_settings" if side == "baseline" else "settings", {}
            )
            environments.append(
                html.escape(
                    f"{side}: PyTorch {environment.get('packages', {}).get('torch', 'unknown')}, "
                    f"{settings.get('backend', 'unknown')}, {settings.get('compile_mode', 'default')}, "
                    f"{settings.get('device', 'unknown')}"
                )
            )
    (output / "report.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Faultline check</title>'
        "<style>body{font:16px system-ui;max-width:1100px;margin:3rem auto;padding:1rem}li{margin:.5rem 0}table{border-collapse:collapse}td,th{padding:.5rem;border:1px solid #ccc;text-align:left}td{overflow-wrap:anywhere;max-width:30rem}details{margin:1rem 0}pre{white-space:pre-wrap}</style>"
        f"<h1>{verdict['verdict']}</h1><p>{html.escape(verdict['scope']['statement'])}</p>"
        f"<p>{html.escape(scope['mode'])}; {html.escape(scope['level'])}; "
        f"{scope['completed_batches']}/{scope['planned_batches']} conclusive primary batches. "
        f"Execution: {verdict.get('execution_state', 'complete')}.</p>"
        f"<p>{'<br>'.join(environments)}</p>"
        '<p><a href="verdict.json">Verdict JSON</a> · <a href="evidence.json">Full evidence</a></p>'
        f"<h2>Reasons</h2><ul>{reasons or '<li>No blocking evidence in the requested checks.</li>'}</ul>"
        f"<h2>Notes</h2><ul>{notes}</ul>"
        f"{accuracy_html(verdict['accuracy_shift_replicated'])}"
        f"<h2>Batch evidence</h2><table><tr><th>Batch</th><th>Environment</th><th>Primary status</th><th>Explanation labels</th><th>Diagnostics</th></tr>{''.join(batches)}</table>"
        f"<h2>Finding details</h2>{''.join(findings) or '<p>No failing primary targets.</p>'}"
        f"<h2>Reproducers</h2><ul>{reproduction or '<li>No failing primary workload exported.</li>'}</ul>"
        "<p>Primary statuses are unchanged. Inspect the full evidence for explanation coverage, contrary evidence, input hashes, settings and environment versions.</p>",
        encoding="utf-8",
    )


def run_check(
    bundle,
    output,
    *,
    candidate_python=None,
    baseline_python=None,
    level="standard",
    settings=None,
    baseline_settings=None,
    batches=None,
    resume=False,
):
    from .sweep import runtime_environment

    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    preset = PRESETS[level]
    settings = settings or Settings(
        gradients=preset["gradients"], repeats=preset["repeats"], timeout_s=600, max_rss_mb=12288
    )
    baseline_settings = baseline_settings or settings
    candidate_python = interpreter(candidate_python or sys.executable)
    baseline_python = interpreter(baseline_python) if baseline_python else None
    mode = "environment_comparison" if baseline_python else "single_environment"
    manifest = verify_batches(bundle) if (bundle / "batches.json").exists() else None
    planned = min(manifest["planned"], batches or preset["batches"]) if manifest else 1
    entries = (
        {r["batch"]: bundle / r["bundle"] for r in manifest["batches"] if r["batch"] < planned}
        if manifest
        else {0: bundle}
    )
    identity = {
        "schema_version": 1,
        "mode": mode,
        "level": level,
        "planned": planned,
        "source_sha256": provenance()["faultline_source_sha256"],
        "bundle_hash": file_hash(bundle / ("batches.json" if manifest else "workload.json")),
        "candidate_python": candidate_python,
        "baseline_python": baseline_python,
        "candidate_environment": runtime_environment(candidate_python),
        "baseline_environment": runtime_environment(baseline_python) if baseline_python else None,
        "settings": settings.to_dict(),
        "baseline_settings": baseline_settings.to_dict(),
    }
    started = time.monotonic()
    with campaign_lock(output):
        if (output / "check.json").exists():
            if not resume or read_json(output / "check.json") != identity:
                raise ValueError(
                    "Check exists or its source, inputs, environments or settings changed; choose a new output directory."
                )
        else:
            atomic_json(output / "check.json", identity)
        rows = []
        for index, source in sorted(entries.items()):
            if provenance()["faultline_source_sha256"] != identity["source_sha256"]:
                raise ValueError(
                    "Source changed during check; retain this run and choose a new output directory."
                )
            root = output / f"batch-{index:04d}"
            commits = sorted(root.glob("attempt-*/commit.json"))
            if commits:
                directory = commits[-1].parent
                for name, digest in read_json(commits[-1])["files"].items():
                    target = (directory / name).resolve()
                    if not target.is_relative_to(directory) or file_hash(target) != digest:
                        raise ValueError("Committed check evidence changed")
                row = read_json(directory / "evidence.json")
            else:
                directory = root / f"attempt-{len(list(root.glob('attempt-*'))) + 1:04d}"
                row = {"batch": index, "diagnostics": {}}
                for side, python, config in (
                    ("baseline", baseline_python, baseline_settings),
                    ("candidate", candidate_python, settings),
                ):
                    if python:
                        with compiler_cache(output / "compiler-cache" / side):
                            row[side], row["diagnostics"][side] = inspect_side(
                                source,
                                directory / side,
                                config,
                                python,
                                level,
                                paired=bool(baseline_python),
                            )
                atomic_json(directory / "evidence.json", row)
                files = {
                    p.relative_to(directory).as_posix(): file_hash(p)
                    for p in directory.rglob("*.json")
                }
                atomic_json(directory / "commit.json", {"files": files})
            rows.append(row)
            scores = (
                [{"batch": r["batch"], **batch_score(r["baseline"], r["candidate"])} for r in rows]
                if baseline_python
                else []
            )
            replication = (
                accuracy_shift_replicated(scores, planned)
                if baseline_python and level != "quick"
                else {"status": "not_applicable"}
            )
            reproducer = output / "reproducer"
            if not reproducer.exists() and any(
                not assessment(row.get(side, {}))["acceptable"]
                and row.get(side, {}).get("status") in REDUCIBLE
                for side in ("baseline", "candidate")
            ):
                side = next(
                    side
                    for side in ("candidate", "baseline")
                    if row.get(side, {}).get("status") in REDUCIBLE
                    and not assessment(row[side])["acceptable"]
                )
                export_native(
                    NativeJob.create(source),
                    settings if side == "candidate" else baseline_settings,
                    reproducer,
                    expected=read_json(directory / side / "primary/outcome.json"),
                )
            artifacts = {
                "report": "report.html",
                "evidence": "evidence.json",
                "reproducers": ["reproducer/repro.py"]
                if (reproducer / "reproducer.json").exists()
                else [],
            }
            verdict = summarize(
                rows, planned, mode=mode, level=level, replication=replication, artifacts=artifacts
            )
            peaks = [
                s.get("peak_tree_rss_mb", 0)
                for r in rows
                for s in (r.get("baseline", {}), r["candidate"])
            ]
            peaks += [
                p.get("peak_tree_rss_mb", p.get("supervision", {}).get("peak_tree_rss_mb", 0)) or 0
                for r in rows
                for phases in r["diagnostics"].values()
                for p in phases
            ]
            verdict["metrics"] = {
                "invocation_wall_s": time.monotonic() - started,
                "peak_worker_rss_mb": max([0, *peaks]),
            }
            verdict["execution_state"] = "running"
            report(output, verdict, {"configuration": identity, "batches": rows})
        if not rows:
            verdict = summarize([], planned, mode=mode, level=level)
            report(output, verdict, {"configuration": identity, "batches": []})
        verdict["execution_state"] = "complete"
        report(output, verdict, {"configuration": identity, "batches": rows})
    return verdict
