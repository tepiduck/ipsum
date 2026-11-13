"""Resumable parallel primary checks and ordered, bounded accuracy diagnostics."""

import copy
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from queue import Queue

from .batches import verify_batches
from .check_resources import cpu_jobs, measured_schedule, memory_budget, memory_limited
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
    return cpu_jobs()


def row_hash(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


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
    jobs = "auto" if jobs is None else jobs
    if jobs != "auto" and (type(jobs) is not int or jobs < 1):
        raise ValueError("--jobs must be auto or a positive integer")
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    preset = check.PRESETS[level]
    settings = settings or Settings(
        gradients=preset["gradients"],
        repeats=preset["repeats"],
        timeout_s=600,
        max_rss_mb=memory_budget(output, resume=resume)["chosen_mb"],
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
    maximum_jobs = min(default_jobs(), planned)
    if jobs == "auto" and (settings.threads != 1 or baseline_settings.threads != 1):
        maximum_jobs = 1
    effective_jobs = min(jobs, planned) if jobs != "auto" else 1
    if effective_jobs > 1 and (settings.threads != 1 or baseline_settings.threads != 1):
        raise ValueError(
            "Parallel batch workers require --threads 1; use --jobs 1 for another thread count"
        )
    allocation = {
        "requested_jobs": jobs,
        "effective_jobs": effective_jobs if jobs != "auto" else None,
        "maximum_auto_jobs": maximum_jobs if jobs == "auto" else None,
        "candidate_total_rss_mb": settings.max_rss_mb,
        "baseline_total_rss_mb": baseline_settings.max_rss_mb,
    }
    settings.validate()
    baseline_settings.validate()
    identity = {
        "schema_version": 3,
        "mode": mode,
        "level": level,
        "planned": planned,
        "verdict_policy": policy,
        "early_stop": early_stop,
        "allocation": allocation,
        "cache_policy": "measured_exclusive_lane_v2",
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
    schedule = {"mode": jobs, "effective_jobs": effective_jobs, "reason": "explicit_jobs"}

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

    def primary(entry, *, full_budget=False):
        index, source = entry
        unchanged()
        root = output / f"batch-{index:04d}"
        existing = committed(root, "primary-attempt-*/commit.json", "primary.json")
        if existing is not None:
            return existing
        directory = root / f"primary-attempt-{len(list(root.glob('primary-attempt-*'))) + 1:04d}"
        row = {
            "batch": index,
            "diagnostics": {},
            "worker_jobs": 1 if full_budget else effective_jobs,
        }
        for side, python, config in sides:
            config = replace(config, max_rss_mb=config.max_rss_mb // row["worker_jobs"])
            config.validate()
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

    def retry_primary(raw):
        if raw["worker_jobs"] == 1 or not any(memory_limited(raw[s]) for s, _, _ in sides):
            return raw
        index = raw["batch"]
        root = output / f"batch-{index:04d}"
        existing = committed(root, "serial-retry-*/commit.json", "primary.json")
        if existing is not None:
            if existing["resource_retry"]["original_sha256"] != row_hash(raw):
                raise ValueError("Committed memory retry has a different primary")
            return existing
        directory = root / f"serial-retry-{len(list(root.glob('serial-retry-*'))) + 1:04d}"
        row = {
            "batch": index,
            "diagnostics": {},
            "worker_jobs": 1,
            "resource_retry": {
                "reason": "parallel_memory_limit",
                "original_sha256": row_hash(raw),
                "original_statuses": {s: raw[s]["status"] for s, _, _ in sides},
                "original_peaks_mb": {s: raw[s].get("peak_tree_rss_mb") for s, _, _ in sides},
                "attempt": directory.relative_to(output).as_posix(),
                "full_budget_mb": {s: config.max_rss_mb for s, _, config in sides},
            },
        }
        for side, python, config in sides:
            row[side] = check.run_case(
                NativeJob.create(entries[index]),
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
        scheduling = {
            **schedule,
            **{key: value for key, value in allocation.items() if key != "effective_jobs"},
            "serial_retries": [
                {"batch": r["batch"], **r["resource_retry"]}
                for r in current
                if "resource_retry" in r
            ],
        }
        if (output / "memory-budget.json").exists():
            scheduling["budget_selection"] = read_json(output / "memory-budget.json")
        check.report(
            output,
            verdict,
            {"configuration": identity, "scheduling": scheduling, "batches": current},
        )
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
        if jobs == "auto" and entries:
            first = min(entries)
            warmup = primary((first, entries[first]), full_budget=True)
            primary_rows[first] = warmup
            expected = measured_schedule(
                warmup, min(settings.max_rss_mb, baseline_settings.max_rss_mb), maximum_jobs
            )
            recorded = committed(output / "schedule", "commit.json", "schedule.json")
            if recorded is not None and recorded != expected:
                raise ValueError("Committed memory schedule changed")
            schedule = expected
            if recorded is None:
                commit(output / "schedule", "schedule.json", schedule)
            effective_jobs = schedule["effective_jobs"]
            print(
                f"Memory scheduling: first-batch peak {schedule['measured_peak_rss_mb']:.0f} MiB; {effective_jobs} job(s) ({schedule['reason']}).",
                flush=True,
            )
            publish([warmup], "running")
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
                        if entry[0] % effective_jobs == index and entry[0] not in primary_rows
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
        # The pool is drained before any full-budget retry: no competing owned workers.
        primaries = [retry_primary(primary_rows[i]) for i in sorted(primary_rows)]
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
                    # Auto diagnostics are sequential and can use the full budget.
                    divisor = 1 if jobs == "auto" or raw["worker_jobs"] == 1 else effective_jobs
                    diagnostic_config = replace(config, max_rss_mb=config.max_rss_mb // divisor)
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
                        diagnostic_config,
                        python,
                        level,
                        paired=bool(baseline_python),
                        primary=raw[side],
                        env=environment(index, side),
                        accuracy=evaluate_accuracy or explain,
                        explain=explain,
                    )
                    if divisor > 1 and any(
                        memory_limited(phase) for phase in row["diagnostics"][side]
                    ):
                        retry, phases = check.inspect_side(
                            source,
                            directory / side / "serial-retry",
                            config,
                            python,
                            level,
                            paired=bool(baseline_python),
                            primary=raw[side],
                            env=environment(index, side),
                            accuracy=evaluate_accuracy or explain,
                            explain=explain,
                        )
                        for key in ("fp64_advisory", "numerical_context"):
                            if key not in retry and key in row[side]:
                                retry[key] = row[side][key]
                        row[side] = retry
                        row["diagnostics"][side].append(
                            {"status": "serial_memory_retry", "full_budget_mb": config.max_rss_mb}
                        )
                        row["diagnostics"][side].extend(phases)
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
