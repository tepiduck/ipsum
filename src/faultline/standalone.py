"""Supervise an exported script without putting Faultline on its import path."""

import os
import subprocess
import time
from pathlib import Path

from .runner import stop_tree, tree_rss
from .storage import atomic_json, read_json


def verify_standalone(script, python, output, settings):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["OMP_NUM_THREADS"] = str(settings.threads)
    environment["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    flags = (
        {"creationflags": subprocess.CREATE_NO_WINDOW}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    observed, peak, status = {}, 0, None
    started = time.monotonic()
    with (
        (output / "stdout.json").open("wb") as stdout,
        (output / "stderr.log").open("wb") as stderr,
    ):
        process = subprocess.Popen(
            [str(python), str(Path(script).resolve())],
            cwd=output,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            **flags,
        )
        try:
            while process.poll() is None:
                peak = max(peak, tree_rss(process.pid, observed))
                if time.monotonic() - started > settings.timeout_s:
                    status = "timeout"
                    break
                if peak > settings.max_rss_mb * 1024**2:
                    status = "resource_limit"
                    break
                time.sleep(0.05)
        finally:
            stop_tree(process, observed)
    try:
        result = read_json(output / "stdout.json") if status is None else {"status": status}
    except (ValueError, OSError):
        result = {"status": "invalid_standalone_result"}
    if not isinstance(result, dict):
        result = {"status": "invalid_standalone_result"}
    expected_code = 0 if result.get("status") == "pass" else 1
    record = {
        "status": result.get("status"),
        "verified": process.returncode == expected_code
        and result.get("reproduces_expected") is True
        and status is None,
        "returncode": process.returncode,
        "wall_s": time.monotonic() - started,
        "peak_tree_rss_mb": peak / 1024**2,
        "result": result,
    }
    atomic_json(output / "outcome.json", record)
    return record
