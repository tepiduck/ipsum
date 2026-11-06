"""Versioned historical-issue calibration set with explicit denominators and controls."""

import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

from .compare_envs import interpreter
from .diagnose import diagnose_native
from .evaluate import Settings
from .native import NativeJob, capture_workload, verify_native
from .native_artifacts import compare_native_environments, write_native_report
from .native_reduce import reduce_native
from .standalone import verify_standalone
from .storage import atomic_json, provenance

CASES = {
    "183986": {
        "factory": "expanded_index",
        "expected_target": "step:0:output",
        "family": "expanded_storage",
        "gradients": False,
        "reported_device": "CUDA",
        "expected_status": "numerical_mismatch",
    },
    "182012": {
        "factory": "diagonal_backward",
        "expected_target": "step:0:gradient:input[0][0]:value",
        "family": "expanded_storage",
        "gradients": True,
        "reported_device": "CUDA",
        "expected_status": "gradient_mismatch",
    },
    "180164": {
        "factory": "slice_backward",
        "expected_target": "step:0:gradient:input[0][0]:value",
        "family": "expanded_storage",
        "gradients": True,
        "reported_device": "CUDA",
        "expected_status": "gradient_mismatch",
    },
    "185337": {
        "factory": "cast_roundtrip",
        "expected_target": "step:0:output[0]",
        "family": "explicit_rounding",
        "gradients": False,
        "reported_device": "CPU",
        "expected_status": "numerical_mismatch",
    },
}


def case_catalog():
    return [
        {
            "id": key,
            "issue": f"https://github.com/pytorch/pytorch/issues/{key}",
            "upstream_state": "closed",
            "state_checked": "2026-11-05",
            **value,
        }
        for key, value in CASES.items()
    ]


def score_case(expected_status, baseline, candidate, expected_target=None):
    failed_targets = {
        x["target"] for x in candidate.get("comparison", {}).get("outputs", []) if not x["matches"]
    }
    detected = candidate["status"] == expected_status and (
        expected_target is None or expected_target in failed_targets
    )
    paired = detected and baseline["status"] == "pass"
    if paired:
        status = "paired_reproduction"
    elif detected:
        status = "candidate_detected_without_passing_control"
    elif candidate["status"] == "pass":
        status = "not_observed"
    else:
        status = "inconclusive"
    if baseline["status"] not in {
        "pass",
        "numerical_mismatch",
        "gradient_mismatch",
        "structural_mismatch",
    }:
        status = "inconclusive"
    return {"status": status, "candidate_detected": detected, "paired_reproduction": paired}


def historical_benchmark(
    output,
    good_python,
    bad_python,
    settings=None,
    *,
    cases=None,
    reduce=True,
    diagnose=False,
    max_evaluations=16,
    require_paired=False,
    progress=print,
):
    settings = settings or Settings(repeats=2, timeout_s=300)
    settings.validate()
    if settings.backend != "inductor" or settings.compile_mode != "default" or settings.autocast:
        raise ValueError("Historical fixtures use default Inductor without autocast")
    selected = list(dict.fromkeys(CASES if cases is None else cases))
    if not selected or any(key not in CASES for key in selected):
        raise ValueError("Select at least one known historical issue ID")
    if max_evaluations < 4:
        raise ValueError("Historical reduction needs at least four evaluations")
    good_python, bad_python = interpreter(good_python), interpreter(bad_python)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name("historical_workloads.py")
    summary = {
        "schema_version": 1,
        "kind": "historical_benchmark",
        "corpus_version": 1,
        "source": provenance(),
        "settings": settings.to_dict(),
        "selected_count": len(selected),
        "family_count": len({CASES[k]["family"] for k in selected}),
        "cases": [],
        "status": "in_progress",
        "exit_code": 2,
        "note": "Small curated calibration set, not a representative bug population or false-negative estimate. CPU adaptations of CUDA reports are labeled. Closed issue state does not establish that a release is fixed. A paired reproduction requires the expected candidate symptom AND a passing control. Only paired reproductions are automatically reduced.",
    }
    atomic_json(output / "catalog.json", case_catalog())
    for key in selected:
        spec = CASES[key]
        policy = replace(settings, gradients=spec["gradients"])
        directory = output / "cases" / key
        started = time.monotonic()
        record = {
            "id": key,
            "issue": f"https://github.com/pytorch/pytorch/issues/{key}",
            **spec,
            "device": settings.device,
            "aot_capture": spec["gradients"],
        }
        try:
            # Factories construct CPU data. Both compiler workers load the same binary snapshot
            # on the explicitly selected device; this is not a fallback execution of a GPU check.
            bundle = capture_workload(
                f"{source}:{spec['factory']}", directory / "bundle", replace(policy, device="cpu")
            )
            comparison = compare_native_environments(
                bundle,
                directory / "comparison",
                good_python,
                bad_python,
                policy,
                aot=spec["gradients"],
            )
            record.update(
                score_case(
                    spec["expected_status"],
                    comparison["baseline"],
                    comparison["candidate"],
                    spec["expected_target"],
                )
            )
            record.update(
                classification=comparison["classification"],
                baseline_status=comparison["baseline"]["status"],
                candidate_status=comparison["candidate"]["status"],
                baseline_environment=comparison["baseline"].get("worker_environment"),
                candidate_environment=comparison["candidate"].get("worker_environment"),
                original_snapshot_bytes=verify_native(bundle)["snapshot_bytes"],
                report=f"cases/{key}/comparison/report.html",
            )
            scripts = {
                "candidate": directory / "comparison" / "candidate-reproducer" / "repro.py",
                "baseline": directory / "comparison" / "baseline-reproducer" / "repro.py",
            }
            if record["paired_reproduction"] and reduce:
                reduce_started = time.monotonic()
                result = reduce_native(
                    NativeJob.create(bundle, aot=spec["gradients"]),
                    policy,
                    comparison["candidate"],
                    directory / "reduction",
                    candidate_python=bad_python,
                    baseline_settings=policy,
                    baseline_python=good_python,
                    max_evaluations=max_evaluations,
                    progress=progress,
                )
                record["reduction_wall_s"] = time.monotonic() - reduce_started
                record["reduction"] = {
                    name: result.get(name)
                    for name in (
                        "confirmed",
                        "native_graph",
                        "graph_stage",
                        "original_complexity",
                        "reduced_complexity",
                        "evaluations",
                        "budget_exhausted",
                        "predicate",
                    )
                }
                if result["confirmed"]:
                    scripts = {
                        "candidate": directory / "reduction" / "reproducer" / "repro.py",
                        "baseline": directory / "reduction" / "baseline-reproducer" / "repro.py",
                    }
                else:
                    record["reduction_error"] = result.get("reason", "Final confirmation failed")
            elif reduce:
                record["reduction_skipped"] = "No paired reproduction with a passing baseline"
            if record["paired_reproduction"]:
                record["standalone"] = {}
                for label, python in (("candidate", bad_python), ("baseline", good_python)):
                    checked = verify_standalone(
                        scripts[label], python, directory / f"standalone-{label}", policy
                    )
                    record["standalone"][label] = {
                        k: checked[k]
                        for k in ("verified", "status", "returncode", "wall_s", "peak_tree_rss_mb")
                    }
            if diagnose and record["candidate_detected"]:
                ladder = diagnose_native(
                    bundle, directory / "diagnosis", policy, python=bad_python, progress=progress
                )
                record["diagnosis"] = {
                    "status": ladder["status"],
                    "boundary": ladder.get("boundary"),
                    "rungs": [{"backend": x["id"], "status": x["status"]} for x in ladder["cases"]],
                }
        except (ValueError, RuntimeError, OSError) as exc:
            record.update(status="inconclusive", error=f"{type(exc).__name__}: {exc}")
        record["wall_s"] = time.monotonic() - started
        summary["cases"].append(record)
        atomic_json(output / "summary.json", summary)
        if progress:
            progress(f"issue {key}: {record['status']}", flush=True)
    summary["counts"] = dict(Counter(x["status"] for x in summary["cases"]))
    summary["candidate_detections"] = sum(
        x.get("candidate_detected", False) for x in summary["cases"]
    )
    summary["paired_reproductions"] = sum(
        x.get("paired_reproduction", False) for x in summary["cases"]
    )
    summary["detection_fraction_of_selected"] = summary["candidate_detections"] / len(selected)
    incomplete = any(
        x["status"] == "inconclusive"
        or x.get("reduction_error")
        or any(not y["verified"] for y in x.get("standalone", {}).values())
        for x in summary["cases"]
    )
    summary.update(
        status="inconclusive" if incomplete else "complete",
        exit_code=2
        if incomplete
        else 1
        if require_paired and summary["paired_reproductions"] != len(selected)
        else 0,
    )
    write_native_report(output, summary)
    return summary
