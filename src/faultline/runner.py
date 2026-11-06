"""Supervise one isolated worker, including its compiler subprocess tree."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

from .storage import atomic_json, read_json


def stop_tree(process, observed=None):
    # Only the process started by this invocation and its descendants are targeted.
    try:
        descendants = psutil.Process(process.pid).children(recursive=True)
    except psutil.Error:
        descendants = []
    descendants = list(
        {child.pid: child for child in [*descendants, *(observed or {}).values()]}.values()
    )
    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for child in reversed(descendants):
        try:
            child.kill()
        except psutil.Error:
            pass
    if process.poll() is None:
        process.kill()
    process.wait()
    psutil.wait_procs(descendants, timeout=2)


def tree_rss(pid, observed=None):
    try:
        parent = psutil.Process(pid)
        processes = [parent, *parent.children(recursive=True)]
        if observed is not None:
            observed.update({child.pid: child for child in processes[1:]})
    except psutil.Error:
        return 0
    total = 0
    for proc in processes:
        try:
            total += proc.memory_info().rss
        except psutil.Error:
            pass
    return total


def run_case(case, settings, directory, *, python=None):
    case.validate()
    settings.validate()
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    job_path, raw_path = directory / "job.json", directory / "worker-result.json"
    # A prior incomplete attempt cannot provide a stale result for this attempt.
    raw_path.unlink(missing_ok=True)
    atomic_json(
        job_path,
        {
            "worker_protocol": 2,
            "case": case.to_dict(),
            "settings": settings.to_dict(),
            "parent": [os.getpid(), psutil.Process().create_time()],
        },
    )
    environment = os.environ.copy()
    source = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
    environment["OMP_NUM_THREADS"] = str(settings.threads)
    environment["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    environment["PYTHONHASHSEED"] = str(case.seed)
    flags = (
        {"creationflags": subprocess.CREATE_NO_WINDOW}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    status, peak = None, 0
    observed = {}
    start = time.monotonic()
    with (
        (directory / "stdout.log").open("wb") as stdout,
        (directory / "stderr.log").open("wb") as stderr,
    ):
        process = subprocess.Popen(
            [
                str(python or sys.executable),
                "-m",
                "faultline.worker",
                "--job",
                str(job_path),
                "--result",
                str(raw_path),
            ],
            stdout=stdout,
            stderr=stderr,
            env=environment,
            cwd=directory,
            **flags,
        )
        try:
            while process.poll() is None:
                peak = max(peak, tree_rss(process.pid, observed))
                if time.monotonic() - start > settings.timeout_s:
                    status = "timeout"
                    break
                if peak > settings.max_rss_mb * 1024**2:
                    status = "resource_limit"
                    break
                time.sleep(0.05)
        finally:
            # Also clean up any compiler subprocesses remaining after the worker exits.
            stop_tree(process, observed)
    base = {
        "schema_version": 1,
        "case_hash": case.digest(),
        "backend": settings.backend,
        "device": settings.device,
        "injected_fault": settings.backend.startswith("mutant:"),
        "gradients": settings.gradients,
        "dynamic": settings.dynamic,
        "compile_mode": settings.compile_mode,
        "autocast": settings.autocast,
        "trace": case.to_dict().get("kind") in {"trace", "native"},
        "worker_wall_s": time.monotonic() - start,
        "peak_tree_rss_mb": peak / 1024**2,
        "worker_returncode": process.returncode,
    }
    if status is not None:
        result = {**base, "status": status, "fingerprint": status}
    elif process.returncode != 0:
        result = {**base, "status": "crash", "fingerprint": f"crash:{process.returncode}"}
    elif not raw_path.exists():
        result = {
            **base,
            "status": "harness_error",
            "fingerprint": "harness_error",
            "error": "worker exited without a result",
        }
    else:
        try:
            payload = read_json(raw_path)
            if payload.get("worker_protocol") != 2:
                raise ValueError("worker protocol mismatch")
            if payload["status"] != "harness_error" and payload["case_hash"] != case.digest():
                raise ValueError("worker returned a different case")
            result = {**payload, **base}
        except (ValueError, KeyError) as exc:
            result = {
                **base,
                "status": "harness_error",
                "fingerprint": "harness_error",
                "error": str(exc),
            }
    atomic_json(directory / "outcome.json", result)
    return result
