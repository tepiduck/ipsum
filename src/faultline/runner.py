"""Supervise one isolated worker, including its compiler subprocess tree."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

from .storage import atomic_json, read_json


def process_identity(pid):
    """Linux start ticks avoid wall-clock/WSL boot-time corrections changing a live PID."""
    if sys.platform == "linux":
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat[stat.rindex(")") + 2 :].split()
        return {"linux_start_ticks": int(fields[19])}
    return {"create_time": psutil.Process(pid).create_time()}


def parent_alive(pid, identity):
    try:
        parent = psutil.Process(pid)
        if not parent.is_running() or parent.status() == psutil.STATUS_ZOMBIE:
            return False
        if isinstance(identity, (float, int)):  # Older worker protocol records.
            return parent.create_time() == identity
        return process_identity(pid) == identity
    except (psutil.Error, OSError):
        return False


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


def run_case(case, settings, directory, *, python=None, env=None):
    case.validate()
    settings.validate()
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    job_path, raw_path = directory / "job.json", directory / "worker-result.json"
    # A prior incomplete attempt cannot provide a stale result for this attempt.
    raw_path.unlink(missing_ok=True)
    (directory / "primary-result.json").unlink(missing_ok=True)
    atomic_json(
        job_path,
        {
            "worker_protocol": 2,
            "case": case.to_dict(),
            "settings": settings.to_dict(),
            "parent": [os.getpid(), process_identity(os.getpid())],
        },
    )
    environment = os.environ.copy()
    environment.update(env or {})
    environment["OMP_NUM_THREADS"] = str(settings.threads)
    environment["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    environment["PYTHONHASHSEED"] = str(case.seed)
    if settings.deterministic:
        from .portable import deterministic_environment

        environment.update(deterministic_environment())
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
        from .bootstrap import python_command

        process = subprocess.Popen(
            python_command(
                python or sys.executable,
                "import runpy;runpy.run_module('faultline.worker',run_name='__main__')",
                [
                    "--job",
                    str(job_path),
                    "--result",
                    str(raw_path),
                ],
            ),
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
        "fp64_requested": settings.fp64_advisory or settings.numerics,
        "numerics_requested": settings.numerics,
        "deterministic": settings.deterministic,
        "trace": case.to_dict().get("kind") in {"trace", "native"},
        "worker_wall_s": time.monotonic() - start,
        "peak_tree_rss_mb": peak / 1024**2,
        "worker_returncode": process.returncode,
    }
    if status is not None:
        result = {**base, "status": status, "fingerprint": status}
        checkpoint = directory / "primary-result.json"
        if (settings.fp64_advisory or settings.numerics) and checkpoint.exists():
            primary = read_json(checkpoint)
            if primary.get("case_hash") == case.digest():
                result["primary_observation"] = primary
                result["fp64_advisory"] = {"status": status}
                if settings.numerics:
                    result["numerical_context"] = {"status": status}
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
