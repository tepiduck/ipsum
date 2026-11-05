"""Validate AOT graph capture, guarded backward reduction, and standalone replay."""

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from faultline.evaluate import Settings
from faultline.native import NativeJob, capture_workload
from faultline.native_artifacts import compare_native_environments, export_native
from faultline.runner import run_case
from faultline.storage import atomic_json, provenance, read_json


def replay(script, python, output):
    with output.open("w", encoding="utf-8") as stream:
        process = subprocess.run(
            [str(python), str(script.resolve())],
            stdout=stream,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300,
            check=False,
        )
    if not output.stat().st_size:
        raise RuntimeError(f"Standalone produced no result: {process.stderr[-1000:]}")
    result = read_json(output)
    return {
        "exit_code": process.returncode,
        "status": result["status"],
        "reproduces_expected": result.get("reproduces_expected", False),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-python", default=sys.executable)
    parser.add_argument("--historical-python", help="optional PyTorch 2.12.0 CPU environment")
    parser.add_argument("--compiler", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1] / "examples" / "native_models.py"
    settings = Settings(
        backend="aot_eager", gradients=True, repeats=2, timeout_s=300, max_rss_mb=8192
    )
    bundle = capture_workload(f"{source}:reduction_demo", args.output / "bundle", settings)
    checks, records = [], []

    def compare(
        name, workload, baseline, candidate, baseline_python, candidate_python, reduce=False
    ):
        directory = args.output / name
        result = compare_native_environments(
            workload,
            directory,
            baseline_python,
            candidate_python,
            baseline,
            candidate,
            reduce=reduce,
            max_evaluations=20,
            aot=True,
        )
        record = {
            "name": name,
            "classification": result["classification"],
            "baseline_status": result["baseline"]["status"],
            "candidate_status": result["candidate"]["status"],
            "baseline_torch": result["baseline"]
            .get("worker_environment", {})
            .get("packages", {})
            .get("torch"),
            "candidate_torch": result["candidate"]
            .get("worker_environment", {})
            .get("packages", {})
            .get("torch"),
            "forward_unchanged": all(
                x["matches"]
                for x in result["candidate"].get("comparison", {}).get("outputs", [])
                if ":gradient:" not in x["target"]
            ),
            "stages": sorted({x["stage"] for x in result["candidate"].get("aot_graphs", [])}),
            "injected_fault": result["injected_fault"],
        }
        if "reduction" in result:
            reduced = result["reduction"]
            record["reduction"] = {
                key: reduced.get(key)
                for key in (
                    "confirmed",
                    "native_graph",
                    "graph_stage",
                    "original_complexity",
                    "reduced_complexity",
                    "evaluations",
                )
            }
            if reduced["confirmed"]:
                for key, python in (
                    ("reproducer", candidate_python),
                    ("baseline-reproducer", baseline_python),
                ):
                    record[key] = replay(
                        directory / "reduction" / key / "repro.py",
                        python,
                        directory / f"{key}.json",
                    )
                checks.append(
                    record["reproducer"]["reproduces_expected"]
                    and record["baseline-reproducer"]["reproduces_expected"]
                )
        records.append(record)
        atomic_json(args.output / "summary.json", {"source": provenance(), "experiments": records})
        print(json.dumps(record), flush=True)
        return result

    controlled = compare(
        "backward-mutant",
        bundle,
        settings,
        replace(settings, backend="mutant:aot-backward-scale"),
        args.baseline_python,
        sys.executable,
        reduce=True,
    )
    reduction = controlled["reduction"]
    checks.extend(
        [
            controlled["classification"] == "regression",
            reduction["confirmed"],
            reduction["graph_stage"] == "backward",
            reduction["reduced_complexity"] < reduction["original_complexity"],
        ]
    )
    if args.compiler:
        compiled = replace(settings, backend="inductor")
        actual = compare(
            "inductor", bundle, compiled, compiled, args.baseline_python, sys.executable
        )
        checks.append(actual["classification"] == "agreement")
        backward = next(x for x in actual["candidate"]["aot_graphs"] if x["stage"] == "backward")
        graph = NativeJob.create(
            args.output / "inductor" / "candidate" / "graphs" / backward["directory"]
        )
        outcome = run_case(graph, compiled, args.output / "backward-only")
        script = export_native(graph, compiled, args.output / "backward-export", outcome)
        independent = replay(script, sys.executable, args.output / "backward-standalone.json")
        checks.append(
            outcome["status"] == "pass"
            and independent["exit_code"] == 0
            and independent["reproduces_expected"]
        )
        records.append({"name": "independent-backward-inductor", **independent})
    if args.historical_python:
        historic = capture_workload(
            f"{source}:scatter_backward", args.output / "historical-bundle", settings
        )
        actual = compare(
            "historical-182012",
            historic,
            replace(settings, backend="inductor"),
            replace(settings, backend="inductor"),
            sys.executable,
            args.historical_python,
            reduce=True,
        )
        checks.extend(
            [
                actual["classification"] == "regression",
                actual["reduction"]["confirmed"],
                actual["reduction"]["native_graph"],
                actual["reduction"]["graph_stage"] == "backward",
            ]
        )
    atomic_json(
        args.output / "summary.json",
        {"source": provenance(), "experiments": records, "checks_passed": all(checks)},
    )
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
