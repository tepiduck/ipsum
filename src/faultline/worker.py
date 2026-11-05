"""Private subprocess entry point; never load arbitrary Python or pickle inputs."""

import argparse
import os
import threading
import time
from pathlib import Path

import psutil

from .storage import atomic_json, read_json


def watch_parent(pid, created):
    """Do not leave compiler children running after an abruptly killed coordinator."""
    while True:
        time.sleep(1)
        try:
            parent = psutil.Process(pid)
            if parent.create_time() == created and parent.is_running():
                continue
        except psutil.Error:
            pass
        for child in psutil.Process().children(recursive=True):
            try:
                child.kill()
            except psutil.Error:
                pass
        os._exit(125)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    job = read_json(args.job)
    if "parent" in job:
        threading.Thread(target=watch_parent, args=tuple(job["parent"]), daemon=True).start()
    # Import torch only after the coordinator has started enforcing resource limits.
    from .environment import describe
    from .evaluate import Settings, evaluate
    from .trace import program_from_dict

    try:
        if job.get("worker_protocol", 2) != 2:
            raise ValueError("unsupported worker protocol")
        result = evaluate(program_from_dict(job["case"]), Settings(**job["settings"]))
        result["worker_environment"] = describe()
    except Exception as exc:
        import traceback

        result = {
            "schema_version": 1,
            "status": "harness_error",
            "fingerprint": "harness_error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    result["worker_protocol"] = 2
    atomic_json(args.result, result)


if __name__ == "__main__":
    main()
