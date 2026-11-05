"""Reproducible native workload validation. Large model checks are opt-in."""

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from faultline.evaluate import Settings
from faultline.native import capture_workload, verify_native
from faultline.native_artifacts import compare_native_environments
from faultline.storage import atomic_json, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-python", default=sys.executable)
    parser.add_argument("--historical-python", help="optional PyTorch 2.12.0 CPU interpreter")
    parser.add_argument("--compiler", action="store_true")
    parser.add_argument(
        "--large", action="store_true", help="also run ResNet-18 and 48.9M-parameter transformer"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1] / "examples" / "native_models.py"
    settings = Settings(
        backend="inductor" if args.compiler else "eager", repeats=2, timeout_s=600, max_rss_mb=12288
    )
    records = []

    def compare(name, baseline, candidate, *, candidate_python=sys.executable, reduce=False):
        directory = args.output / name
        bundle = capture_workload(f"{source}:{name}", directory / "bundle", settings)
        report = compare_native_environments(
            bundle,
            directory / "comparison",
            args.baseline_python,
            candidate_python,
            baseline,
            candidate,
            reduce=reduce,
            max_evaluations=16,
        )
        manifest = verify_native(bundle)
        record = {
            "name": name,
            "classification": report["classification"],
            "baseline_status": report["baseline"]["status"],
            "candidate_status": report["candidate"]["status"],
            "parameter_count": manifest["parameter_count"],
            "snapshot_bytes": manifest["snapshot_bytes"],
            "candidate_graphs": report["candidate"].get("graph_count"),
            "baseline_torch": report["baseline"]
            .get("worker_environment", {})
            .get("packages", {})
            .get("torch"),
            "candidate_torch": report["candidate"]
            .get("worker_environment", {})
            .get("packages", {})
            .get("torch"),
            "injected_fault": report["injected_fault"],
        }
        if "reduction" in report:
            reduction = report["reduction"]
            record["reduction"] = {
                key: reduction.get(key)
                for key in (
                    "confirmed",
                    "native_graph",
                    "original_complexity",
                    "reduced_complexity",
                    "evaluations",
                )
            }
            if reduction["confirmed"]:
                script = directory / "comparison" / "reduction" / "reproducer" / "repro.py"
                with (directory / "standalone.json").open("w", encoding="utf-8") as stream:
                    process = subprocess.run(
                        [str(candidate_python), str(script.resolve())],
                        stdout=stream,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=600,
                        check=False,
                    )
                result = json.loads((directory / "standalone.json").read_text(encoding="utf-8"))
                record["standalone_reproduces"] = process.returncode == 1 and result.get(
                    "reproduces_expected"
                )
        records.append(record)
        print(json.dumps(record), flush=True)
        atomic_json(args.output / "summary.json", {"source": provenance(), "experiments": records})
        return record

    demo = compare(
        "reduction_demo",
        replace(settings, backend="eager"),
        replace(settings, backend="mutant:relu-as-neg"),
        reduce=True,
    )
    checks = [
        demo["classification"] == "regression",
        demo.get("standalone_reproduces") is True,
        demo["reduction"]["reduced_complexity"] < demo["reduction"]["original_complexity"],
    ]
    if args.compiler:
        current = compare("expanded_index_add", settings, settings)
        checks.append(current["classification"] == "agreement")
    if args.large:
        for name in ("resnet18", "transformer"):
            record = compare(name, settings, settings)
            checks.append(record["classification"] == "agreement")
    # Historical reproduction runs the old release as candidate deliberately, not as an upgrade claim.
    if args.historical_python:
        directory = args.output / "historical"
        bundle = capture_workload(f"{source}:expanded_index_add", directory / "bundle", settings)
        report = compare_native_environments(
            bundle,
            directory / "comparison",
            sys.executable,
            args.historical_python,
            replace(settings, backend="inductor"),
            reduce=True,
            max_evaluations=12,
        )
        records.append(
            {
                "name": "historical_issue_183986",
                "classification": report["classification"],
                "baseline_status": report["baseline"]["status"],
                "candidate_status": report["candidate"]["status"],
                "injected_fault": False,
                "reduction_confirmed": report.get("reduction", {}).get("confirmed"),
            }
        )
        checks.append(report["classification"] == "regression" and report["reduction"]["confirmed"])
    atomic_json(
        args.output / "summary.json",
        {"source": provenance(), "experiments": records, "checks_passed": all(checks)},
    )
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
