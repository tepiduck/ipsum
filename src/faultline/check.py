"""User-facing check coordinator; primary workers and verdict policy stay separate."""

import contextlib
import copy
import html
import os
from dataclasses import replace
from pathlib import Path

from .bisect_env import supervise
from .bootstrap import python_command
from .check_report import accuracy_html, finding_html, scheduling_html
from .native import NativeJob
from .native_runtime import native_failure_key
from .runner import run_case
from .storage import atomic_json, read_json
from .verdict import assessment

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


def diagnostic(job, settings, primary, directory, python, *, control=False, env=None):
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
        watch_parent=True,
    )
    observed = supervise(
        command,
        directory / "supervision",
        timeout=settings.timeout_s,
        max_rss_mb=settings.max_rss_mb,
        env=env,
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


def inspect_side(
    bundle,
    directory,
    settings,
    python,
    level,
    *,
    paired,
    primary=None,
    env=None,
    accuracy=True,
    explain=True,
):
    job = NativeJob.create(bundle)
    if primary is None:
        primary = run_case(job, settings, directory / "primary", python=python, env=env)
    result, phases = primary, []
    if level != "quick" and primary["status"] in {
        "pass",
        "gradient_mismatch",
        "numerical_mismatch",
    }:
        if accuracy and (paired or primary["status"] != "pass" or level == "thorough"):
            evidence = diagnostic(
                job,
                settings,
                primary,
                directory / "fp64",
                python,
                control=level == "thorough",
                env=env,
            )
            phases.append(evidence)
            result = enriched(primary, evidence)
        if (
            explain
            and level == "standard"
            and primary["status"] == "gradient_mismatch"
            and not assessment(result)["acceptable"]
        ):
            evidence = diagnostic(
                job, settings, primary, directory / "control", python, control=True, env=env
            )
            phases.append(evidence)
            # Preserve completed fp64 evidence if a later optional control is unavailable.
            if evidence.get("status") == "complete" and evidence.get("replay_matches_primary"):
                result = enriched(result, evidence)
        if explain and (
            level == "thorough"
            or (
                primary["status"] == "gradient_mismatch"
                and not assessment(result)["acceptable"]
                and not assessment(result)["contrary"]
            )
        ):
            mask_settings = replace(settings, numerics=True, fp64_advisory=True)
            mask = run_case(
                NativeJob.create(bundle, aot=True),
                mask_settings,
                directory / "masks",
                python=python,
                env=env,
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
                f'<tr id="batch-{row["batch"]}-{side}">'
                + "".join(f"<td>{html.escape(value)}</td>" for value in values)
                + "</tr>"
            )
    reproduction = "".join(
        f'<li><a href="{html.escape(path, quote=True)}">{html.escape(path)}</a></li>'
        for path in verdict["artifacts"]["reproducers"]
    )
    scope = verdict["scope"]
    existing = verdict.get("pre_existing_differences")
    existing_html = ""
    if existing:
        links = " ".join(
            f'<a href="#batch-{item["batch"]}-candidate">Batch {item["batch"]}</a>'
            for item in existing["details"]
        )
        existing_html = (
            f"<h2>Pre-existing differences</h2><p>{html.escape(existing['message'])} "
            f"{existing['batches']} batches; {existing['baseline_failed_targets']} baseline / "
            f"{existing['candidate_failed_targets']} candidate failed targets (summed over batches).</p><p>{links}</p>"
        )
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
        f"{scheduling_html(evidence)}"
        '<p><a href="verdict.json">Verdict JSON</a> · <a href="evidence.json">Full evidence</a></p>'
        f"{existing_html}<h2>Change-related findings / blocking evidence</h2><ul>{reasons or '<li>No blocking evidence in the requested checks.</li>'}</ul>"
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
    verdict_policy="v2",
    jobs=None,
    early_stop=True,
):
    from .check_execution import execute

    return execute(
        bundle,
        output,
        candidate_python=candidate_python,
        baseline_python=baseline_python,
        level=level,
        settings=settings,
        baseline_settings=baseline_settings,
        batches=batches,
        resume=resume,
        verdict_policy=verdict_policy,
        jobs=jobs,
        early_stop=early_stop,
    )
