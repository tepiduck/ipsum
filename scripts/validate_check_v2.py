"""Fresh 0.17 held-out checks and fixed-budget costs; completed cells are never overwritten."""

import argparse
import json
import time
from pathlib import Path

from faultline.bisect_env import supervise
from faultline.bootstrap import python_command
from faultline.native import file_hash
from faultline.storage import atomic_json, read_json

MODELS = (
    ("regnet_x_400mf", "torchvision", "regnet_x_400mf", None),
    ("shufflenet_v2_x0_5", "torchvision", "shufflenet_v2_x0_5", None),
    ("squeezenet1_0", "torchvision", "squeezenet1_0", None),
    (
        "bert_tiny",
        "hf",
        "google/bert_uncased_L-2_H-128_A-2",
        "30b0a37ccaaa32f332884b96992754e246e48c5f",
    ),
    (
        "distilroberta",
        "hf",
        "distilbert/distilroberta-base",
        "fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b",
    ),
    ("pythia70m", "hf", "EleutherAI/pythia-70m", "a39f36b100fe8a5377810d56c3f4789b9c53ac42"),
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable", required=True)
    parser.add_argument("--nightly", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepared-016", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "held-out", "costs"), required=True)
    args = parser.parse_args(argv)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = []
    if args.phase == "prepare":
        for name, provider, model, revision in MODELS:
            destination = root / "prepared" / name
            if (destination / "prepared.json").exists():
                records.append({"model": name, "status": "complete", "prepared": str(destination)})
                continue
            specification = {
                "factory": None,
                "torchvision": model if provider == "torchvision" else None,
                "hf": model if provider == "hf" else None,
                "revision": revision,
                "batches": 10,
                "explicit_batches": True,
                "include": [],
                "timeout": 600,
                "max_rss_mb": 12288,
            }
            spec_path = root / "specifications" / f"{name}.json"
            atomic_json(spec_path, specification)
            attempt = root / "preparation-workers" / name / f"attempt-{time.time_ns()}"
            result = supervise(
                python_command(
                    args.stable,
                    "from faultline.check_prepare import main;main()",
                    ["--specification", str(spec_path), "--output", str(destination)],
                    watch_parent=True,
                ),
                attempt,
                timeout=1800,
                max_rss_mb=12288,
            )
            records.append(
                {
                    "model": name,
                    "status": result["status"],
                    "prepared": str(destination),
                    "supervision": str(attempt),
                    "specification_sha256": file_hash(spec_path),
                }
            )
            atomic_json(root / "prepare-progress.json", records)
    else:
        cells = (
            [
                (
                    cohort,
                    name,
                    root / "prepared" / name / "bundle",
                    args.stable if cohort == "same_version" else args.nightly,
                    1,
                )
                for name, _, _, _ in MODELS
                for cohort in ("same_version", "upgrade")
            ]
            if args.phase == "held-out"
            else [
                ("cost", name, args.prepared_016 / name / "bundle", args.stable, 4)
                for name in ("resnet50", "bert", "gpt2")
            ]
        )
        for cohort, name, bundle, candidate, jobs in cells:
            destination = root / args.phase / cohort / name
            existing = destination / "verdict.json"
            if existing.exists() and read_json(existing).get("execution_state") == "complete":
                verdict = read_json(existing)
                records.append(
                    {
                        "cohort": cohort,
                        "model": name,
                        "evidence": str(destination),
                        "verdict": verdict["verdict"],
                        "metrics": verdict.get("metrics", {}),
                    }
                )
                continue
            command_args = [
                "check",
                str(bundle),
                "--baseline",
                args.stable,
                "--candidate",
                candidate,
                "--level",
                "standard",
                "--jobs",
                str(jobs),
                "--max-rss-mb",
                "12288",
                "--output",
                str(destination),
            ]
            if (destination / "check.json").exists():
                command_args.append("--resume")
            attempt = root / "check-workers" / cohort / name / f"attempt-{time.time_ns()}"
            result = supervise(
                python_command(
                    args.stable,
                    "from faultline.cli import main;raise SystemExit(main())",
                    command_args,
                    watch_parent=True,
                ),
                attempt,
                timeout=14400,
                max_rss_mb=18000,
            )
            verdict = read_json(existing) if existing.exists() else {}
            records.append(
                {
                    "cohort": cohort,
                    "model": name,
                    "evidence": str(destination),
                    "verdict": verdict.get("verdict", "INCONCLUSIVE"),
                    "metrics": verdict.get("metrics", {}),
                    "supervision": result,
                    "worker": str(attempt),
                }
            )
            atomic_json(root / f"{args.phase}-progress.json", records)
    atomic_json(root / f"{args.phase}-complete.json", records)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
