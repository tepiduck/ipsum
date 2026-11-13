"""Measure 0.18 auto scheduling on retained frozen CPU transformer/vision batches."""

import argparse
import json
import time
from pathlib import Path

from faultline.bisect_env import supervise
from faultline.bootstrap import python_command
from faultline.native import file_hash
from faultline.storage import atomic_json, read_json

CELLS = (
    ("resnet50", "auto"),
    ("bert", "auto"),
    ("gpt2", "auto"),
    ("smollm2", "auto"),
    ("bert", "1"),
    ("gpt2", "1"),
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable", required=True)
    parser.add_argument("--nightly", required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.output.resolve()
    records = []
    for model, jobs in CELLS:
        bundle = args.prepared / model / "bundle"
        destination = root / f"{model}-jobs-{jobs}"
        existing = destination / "verdict.json"
        completed = existing.exists() and read_json(existing).get("execution_state") == "complete"
        observed = None
        if not completed:
            arguments = [
                "check",
                str(bundle),
                "--baseline",
                args.stable,
                "--candidate",
                args.nightly,
                "--level",
                "standard",
                "--jobs",
                jobs,
                "--output",
                str(destination),
            ]
            if (destination / "check.json").exists() or (
                destination / "memory-budget.json"
            ).exists():
                arguments.append("--resume")
            attempt = root / "workers" / f"{model}-{jobs}-{time.time_ns()}"
            observed = supervise(
                python_command(
                    args.stable,
                    "from faultline.cli import main;raise SystemExit(main())",
                    arguments,
                    watch_parent=True,
                ),
                attempt,
                timeout=14400,
                max_rss_mb=131072,
            )
        verdict = read_json(existing) if existing.exists() else {}
        record = {
            "model": model,
            "jobs": jobs,
            "bundle_hash": file_hash(bundle / "batches.json"),
            "verdict": verdict.get("verdict", "INCONCLUSIVE"),
            "complete": verdict.get("execution_state") == "complete",
            "supervision": observed,
            "metrics": verdict.get("metrics", {}),
        }
        records.append(record)
        atomic_json(root / "progress.json", records)
        print(json.dumps(record), flush=True)
    atomic_json(root / "complete.json", records)


if __name__ == "__main__":
    main()
