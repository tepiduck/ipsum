"""Resumable parallel primary checks and ordered, bounded accuracy diagnostics."""

import copy
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from queue import Queue

from .batches import verify_batches
from .compare_envs import interpreter
from .evaluate import Settings
from .native import NativeJob, file_hash
from .native_artifacts import export_native
from .portable import REDUCIBLE
from .replication import batch_score
from .replication_stop import replication_evidence, stopping_certificate
from .storage import atomic_json, campaign_lock, provenance, read_json
from .verdict import assessment, summarize


def default_jobs():
    cores = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    )
    return max(1, min(4, cores // 2))


def committed(root, pattern, filename):
    commits = sorted(root.glob(pattern))
    if not commits:
        return None
    directory = commits[-1].parent.resolve()
    files = read_json(commits[-1])["files"]
    if filename not in files:
        raise ValueError("Committed check evidence changed: missing manifest entry")
    for name, digest in files.items():
        target = (directory / name).resolve()
        if (
            not target.is_relative_to(directory)
            or not target.is_file()
            or file_hash(target) != digest
        ):
            raise ValueError("Committed check evidence changed")
    return read_json(directory / filename)


def commit(directory, filename, row):
    atomic_json(directory / filename, row)
    atomic_json(
        directory / "commit.json",
        {
            "files": {
                p.relative_to(directory).as_posix(): file_hash(p)
                for p in directory.rglob("*.json")
                if p.name != "commit.json"
            }
        },
    )


def execute(
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
    from . import check
    from .sweep import runtime_environment

    if verdict_policy not in {"v1", "v2"}:
        raise ValueError("Unknown verdict policy")
    if jobs is not None and (type(jobs) is not int or jobs < 1):
        raise ValueError("--jobs must be a positive integer")
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    preset = check.PRESETS[level]
    settings = settings or Settings(
        gradients=preset["gradients"], repeats=preset["repeats"], timeout_s=600, max_rss_mb=12288
    )
    baseline_settings = baseline_settings or settings
    candidate_python = interpreter(candidate_python or sys.executable)
    baseline_python = interpreter(baseline_python) if baseline_python else None
    mode = "environment_comparison" if baseline_python else "single_environment"
    policy = verdict_policy if baseline_python else "v1"
    manifest = verify_batches(bundle) if (bundle / "batches.json").exists() else None
    planned = min(manifest["planned"], batches or preset["batches"]) if manifest else 1
    entries = (
        {r["batch"]: bundle / r["bundle"] for r in manifest["batches"] if r["batch"] < planned}
        if manifest
        else {0: bundle}
    )
    effective_jobs = min(jobs or default_jobs(), planned)
    if effective_jobs > 1 and (settings.threads != 1 or baseline_settings.threads != 1):
        raise ValueError(
            "Parallel batch workers require --threads 1; use --jobs 1 for another thread count"
        )
    allocation = {
        "requested_jobs": jobs,
        "effective_jobs": effective_jobs,
        "candidate_total_rss_mb": settings.max_rss_mb,
        "baseline_total_rss_mb": baseline_settings.max_rss_mb,
    }
    settings = replace(settings, max_rss_mb=settings.max_rss_mb // effective_jobs)
    baseline_settings = replace(
        baseline_settings, max_rss_mb=baseline_settings.max_rss_mb // effective_jobs
    )
    settings.validate()
    baseline_settings.validate()
    identity = {
        "schema_version": 2,
        "mode": mode,
        "level": level,
        "planned": planned,
        "verdict_policy": policy,
        "early_stop": early_stop,
        "allocation": allocation,
        "cache_policy": "exclusive_lane_v1",
        "source_sha256": provenance()["faultline_source_sha256"],
        "bundle_hash": file_hash(bundle / ("batches.json" if manifest else "workload.json")),
        "candidate_python": candidate_python,
        "baseline_python": baseline_python,
        "candidate_environment": runtime_environment(candidate_python),
        "baseline_environment": runtime_environment(baseline_python) if baseline_python else None,
        "settings": settings.to_dict(),
        "baseline_settings": baseline_settings.to_dict(),
    }
    sides = [
        ("baseline", baseline_python, baseline_settings),
        ("candidate", candidate_python, settings),
    ]
    sides = [side for side in sides if side[1]]
    started = time.monotonic()
    rows, scores = [], []
    certificate = None
    accuracy_required = bool(baseline_python) and level != "quick"

    def unchanged():
        if provenance()["faultline_source_sha256"] != identity["source_sha256"]:
            raise ValueError(
                "Source changed during check; retain this run and choose a new output directory."
            )

    def environment(index, side):
        return {
            "TORCHINDUCTOR_CACHE_DIR": str(
                output / "compiler-cache" / f"lane-{index % effective_jobs:04d}" / side
            )
        }

    def primary(entry):
        index, source = entry
        unchanged()
        root = output / f"batch-{index:04d}"
        existing = committed(root, "primary-attempt-*/commit.json", "primary.json")
        if existing is not None:
            return existing
        directory = root / f"primary-attempt-{len(list(root.glob('primary-attempt-*'))) + 1:04d}"
        row = {"batch": index, "diagnostics": {}}
        for side, python, config in sides:
            row[side] = check.run_case(
                NativeJob.create(source),
                config,
                directory / side / "primary",
                python=python,
                env=environment(index, side),
            )
        unchanged()
        commit(directory, "primary.json", row)
        return row

    def publish(current, state):
        replication = (
            replication_evidence(scores, planned, certificate=certificate)
            if accuracy_required
            else {"status": "not_applicable"}
        )
        artifacts = {
            "report": "report.html",
            "evidence": "evidence.json",
            "reproducers": ["reproducer/repro.py"]
            if (output / "reproducer/reproducer.json").exists()
            else [],
        }
        verdict = summarize(
            current,
            planned,
            mode=mode,
            level=level,
            policy=policy,
            replication=replication,
            artifacts=artifacts,
        )
        peaks = [
            r.get(side, {}).get("peak_tree_rss_mb", 0) or 0 for r in current for side, _, _ in sides
        ]
        peaks += [
            p.get("peak_tree_rss_mb", p.get("supervision", {}).get("peak_tree_rss_mb", 0)) or 0
            for r in current
            for phases in r.get("diagnostics", {}).values()
            for p in phases
        ]
        verdict["metrics"] = {
            "invocation_wall_s": time.monotonic() - started,
            "peak_worker_rss_mb": max([0, *peaks]),
        }
        verdict["execution_state"] = state
        check.report(output, verdict, {"configuration": identity, "batches": current})
        return verdict

    with campaign_lock(output):
        if (output / "check.json").exists():
            if not resume or read_json(output / "check.json") != identity:
                raise ValueError(
                    "Check exists or its source, inputs, environments or settings changed; choose a new output directory."
                )
        else:
            atomic_json(output / "check.json", identity)
        primary_rows = {}
        updates = Queue()

        def lane(entries):
            try:
                for entry in entries:
                    updates.put(primary(entry))
            finally:
                updates.put(None)

        # One sequential owner per cache lane. Completion order affects progress only;
        # diagnostics and exports always consume frozen batch order.
        with ThreadPoolExecutor(max_workers=effective_jobs) as executor:
            futures = [
                executor.submit(
                    lane,
                    [
                        entry
                        for entry in sorted(entries.items())
                        if entry[0] % effective_jobs == index
                    ],
                )
                for index in range(effective_jobs)
            ]
            finished = 0
            while finished < effective_jobs:
                row = updates.get()
                if row is None:
                    finished += 1
                else:
                    primary_rows[row["batch"]] = row
                    publish([primary_rows[i] for i in sorted(primary_rows)], "running")
            for future in futures:
                future.result()
        primaries = [primary_rows[i] for i in sorted(primary_rows)]
        for raw in primaries:
            unchanged()
            index = raw["batch"]
            source, root = entries[index], output / f"batch-{index:04d}"
            row = committed(root, "attempt-*/commit.json", "evidence.json")
            if row is None:
                directory = root / f"attempt-{len(list(root.glob('attempt-*'))) + 1:04d}"
                row = copy.deepcopy(raw)
                evaluate_accuracy = accuracy_required and certificate is None
                row["accuracy_evaluated"] = evaluate_accuracy
                for side, python, config in sides:
                    explain = (
                        policy == "v1"
                        or level == "thorough"
                        or (
                            side == "candidate"
                            and raw.get("baseline", {}).get("status") == "pass"
                            and raw[side]["status"] == "gradient_mismatch"
                        )
                    )
                    row[side], row["diagnostics"][side] = check.inspect_side(
                        source,
                        directory / side,
                        config,
                        python,
                        level,
                        paired=bool(baseline_python),
                        primary=raw[side],
                        env=environment(index, side),
                        accuracy=evaluate_accuracy or explain,
                        explain=explain,
                    )
                unchanged()
                commit(directory, "evidence.json", row)
            if row.get("accuracy_evaluated") != (accuracy_required and certificate is None):
                raise ValueError("Committed accuracy schedule changed")
            rows.append(row)
            if row["accuracy_evaluated"]:
                scores.append({"batch": index, **batch_score(row["baseline"], row["candidate"])})
                if early_stop and policy == "v2":
                    certificate = stopping_certificate(scores, planned)
            reproducer = output / "reproducer"
            if not reproducer.exists():
                side = next(
                    (
                        s
                        for s in ("candidate", "baseline")
                        if row.get(s, {}).get("status") in REDUCIBLE
                        and not assessment(row[s])["acceptable"]
                    ),
                    None,
                )
                if side:
                    export_native(
                        NativeJob.create(source),
                        settings if side == "candidate" else baseline_settings,
                        reproducer,
                        expected=raw[side],
                    )
            publish(rows + primaries[len(rows) :], "running")
        return publish(rows, "complete")
