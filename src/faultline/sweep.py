"""Pretrained environment sweeps with explicit denominators and resumable evidence."""

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from .compare_envs import gate_status, interpreter
from .evaluate import Settings
from .native import capture_workload, file_hash, verify_native
from .native_artifacts import compare_native_environments, write_native_report
from .storage import atomic_json, campaign_lock, now, provenance, read_json
from .sweep_models import HELD_OUT_MODELS, MODELS, ORIGINAL_MODELS, prepare_model, real_inputs


def selected_models(names):
    selected = list(names or ORIGINAL_MODELS)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(MODELS):
        raise ValueError("Select distinct models from the pretrained catalog")
    return selected


def prepare_sweep(output, names=None, *, python=None, progress=print):
    output = Path(output).resolve()
    names = selected_models(names)
    output.mkdir(parents=True, exist_ok=False)
    inputs = real_inputs(output / "inputs")
    manifest = {
        "schema_version": 1,
        "created_utc": now(),
        "planned": names,
        "inputs": inputs,
        "models": [],
    }
    atomic_json(output / "prepared.json", manifest)
    for name in names:
        directory = output / "models" / name
        try:
            spec = prepare_model(name, directory / "source", output / "inputs")
            capture_settings = Settings(backend="eager", repeats=2, timeout_s=600, max_rss_mb=12288)
            bundle = capture_workload(
                f"{directory / 'source/factory.py'}:factory",
                directory / "bundle",
                capture_settings,
                include=("model.json", "weights.pt", "inputs.pt"),
                python=python,
            )
            row = {
                "id": name,
                "status": "ready",
                "bundle": str(bundle.relative_to(output)),
                "bundle_hash": file_hash(bundle / "workload.json"),
                "model": spec,
            }
        except Exception as exc:  # noqa: BLE001 -- each download/provider/capture failure stays in denominator
            row = {
                "id": name,
                "status": "preparation_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        manifest["models"].append(row)
        atomic_json(output / "prepared.json", manifest)
        if progress:
            progress(f"prepare {name}: {row['status']}", flush=True)
    return manifest


def runtime_environment(python):
    environment = os.environ.copy()
    source = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
    program = """import json, importlib.metadata
from faultline.environment import describe
result = describe()
result['model_packages'] = {k: importlib.metadata.version(k) for k in ['torchvision', 'transformers', 'safetensors', 'pillow']}
print(json.dumps(result))
"""
    run = subprocess.run(
        [python, "-c", program],
        capture_output=True,
        text=True,
        timeout=60,
        env=environment,
        check=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    result = json.loads(run.stdout)
    return {k: v for k, v in result.items() if k not in {"created_utc", "git_sha"}}


def finding_context(result):
    context = result.get("numerical_context", {})
    labels = context.get("labels", [])
    triage = context.get("triage", {}).get("classification") or (
        "agreement" if result["status"] == "pass" else "unexplained_missing_evidence"
    )
    return {
        "status": result["status"],
        "labels": labels,
        "label_coverage": {
            k: {name: value for name, value in row.items() if name != "targets"}
            for k, row in context.get("label_coverage", {}).items()
        },
        "control_policy": context.get("control_policy"),
        "masks_status": context.get("masks", {}).get("status", "not_run"),
        "triage": triage,
        "triage_evidence": context.get("triage", {}),
        "has_full_diagnostic_label": triage == "explained",
        "unexplained": triage.startswith("unexplained_"),
        "primary_observation_status": result.get("primary_observation", {}).get("status"),
        "fp64_status": result.get("fp64_advisory", {}).get("status", "not_run"),
    }


def sweep_counts(rows, planned=None):
    counts = Counter(
        {
            "planned": len(rows) if planned is None else planned,
            "pending": 0 if planned is None else planned - len(rows),
            "fully_agree": 0,
            "finding": 0,
            "inconclusive": 0,
            "unexplained_findings": 0,
            "accuracy_regression": 0,
            "accuracy_shift": 0,
            "explained": 0,
            "unexplained_contrary_evidence": 0,
            "unexplained_missing_evidence": 0,
        }
    )
    labels = Counter()
    for row in rows:
        if row.get("accuracy_comparison", {}).get("accuracy_regression"):
            counts["accuracy_regression"] += 1
        if row.get("accuracy_shift", {}).get("accuracy_shift"):
            counts["accuracy_shift"] += 1
        if row["classification"] == "inconclusive":
            counts["inconclusive"] += 1
        elif row["classification"] == "agreement":
            counts["fully_agree"] += 1
        else:
            counts["finding"] += 1
            states = {
                side.get("triage", "unexplained_missing_evidence")
                for side in (row["baseline"], row["candidate"])
            }
            category = (
                "unexplained_contrary_evidence"
                if "unexplained_contrary_evidence" in states
                else "unexplained_missing_evidence"
                if "unexplained_missing_evidence" in states
                else "explained"
            )
            counts[category] += 1
            if category.startswith("unexplained_"):
                counts["unexplained_findings"] += 1
        for label in set(row["baseline"].get("labels", []) + row["candidate"].get("labels", [])):
            labels[label] += 1
    return {
        **dict(counts),
        "labels": dict(labels),
        "note": "Labels can overlap. Counts are model/environment-pair outcomes, not a measured false-positive rate.",
    }


def verify_prepared(prepared, names):
    entries = {row["id"]: row for row in read_json(prepared / "prepared.json")["models"]}
    if not set(names) <= set(entries):
        raise ValueError("Requested model was not prepared")
    for name in names:
        row = entries[name]
        if row["status"] == "ready":
            bundle = (prepared / row["bundle"]).resolve()
            if (
                not bundle.is_relative_to(prepared)
                or file_hash(bundle / "workload.json") != row["bundle_hash"]
            ):
                raise ValueError("Prepared workload changed or escaped its directory")
            verify_native(bundle)
    return entries


def run_sweep(
    prepared,
    output,
    baseline_python,
    candidate_python,
    settings,
    *,
    names=None,
    resume=False,
    progress=print,
):
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    names = selected_models(names)
    settings.validate()
    if not settings.gradients or not settings.numerics or settings.backend != "inductor":
        raise ValueError("Pretrained sweeps require Inductor, gradients, and numerical diagnostics")
    baseline_python, candidate_python = interpreter(baseline_python), interpreter(candidate_python)
    entries = verify_prepared(prepared, names)
    config = {
        "schema_version": 1,
        "prepared_hash": file_hash(prepared / "prepared.json"),
        "models": names,
        "settings": settings.to_dict(),
        "baseline_python": baseline_python,
        "candidate_python": candidate_python,
        "baseline_environment": runtime_environment(baseline_python),
        "candidate_environment": runtime_environment(candidate_python),
    }
    with campaign_lock(output):
        manifest = output / "sweep.json"
        if resume:
            if read_json(manifest) != config:
                raise ValueError("Sweep settings, source, inputs, or environments changed")
        elif manifest.exists():
            raise ValueError("Sweep exists; use --resume or a new output directory")
        else:
            atomic_json(manifest, config)
        rows = []
        source_hash = provenance()["faultline_source_sha256"]
        for name in names:
            if provenance()["faultline_source_sha256"] != source_hash:
                raise ValueError(
                    "Source changed during the sweep; completed artifacts are preserved"
                )
            case_root = output / "cases" / name
            committed = sorted(case_root.glob("attempt-*/commit.json"))
            if committed:
                marker = read_json(committed[-1])
                root = committed[-1].parent
                for path, expected in marker["files"].items():
                    target = (root / path).resolve()
                    if not target.is_relative_to(root) or file_hash(target) != expected:
                        raise ValueError("Committed sweep evidence changed")
                row = read_json(root / "row.json")
            else:
                attempt = len(list(case_root.glob("attempt-*"))) + 1
                root = case_root / f"attempt-{attempt:04d}"
                root.mkdir(parents=True, exist_ok=False)
                entry = entries[name]
                files = ["row.json"]
                if entry["status"] != "ready":
                    side = {"status": "preparation_error", "labels": [], "unexplained": True}
                    row = {
                        "id": name,
                        "classification": "inconclusive",
                        "baseline": side,
                        "candidate": side,
                        "error": entry.get("error"),
                    }
                else:
                    result = compare_native_environments(
                        prepared / entry["bundle"],
                        root / "comparison",
                        baseline_python,
                        candidate_python,
                        settings,
                        aot=True,
                    )
                    row = {
                        "id": name,
                        "classification": result["classification"],
                        "baseline": finding_context(result["baseline"]),
                        "candidate": finding_context(result["candidate"]),
                        "accuracy_comparison": {
                            k: v
                            for k, v in result.get("accuracy_comparison", {}).items()
                            if k != "targets"
                        },
                        "accuracy_shift": {
                            k: v
                            for k, v in result.get("accuracy_shift", {}).items()
                            if k != "targets"
                        },
                        "report": str((root / "comparison/report.html").relative_to(output)),
                        "model": {
                            k: entry["model"][k]
                            for k in (
                                "provider",
                                "parameter_count",
                                "weights_sha256",
                                "inputs_sha256",
                            )
                        },
                    }
                    files += [
                        "comparison/summary.json",
                        "comparison/baseline/outcome.json",
                        "comparison/candidate/outcome.json",
                        "comparison/workload/workload.json",
                    ]
                atomic_json(root / "row.json", row)
                atomic_json(
                    root / "commit.json", {"files": {p: file_hash(root / p) for p in files}}
                )
            rows.append(row)
            summary = {
                "schema_version": 1,
                "device": settings.device,
                "planned": len(names),
                "completed": len(rows),
                "cases": rows,
                "counts": sweep_counts(rows, len(names)),
                "exit_code": gate_status(Counter(r["classification"] for r in rows)),
                "scope": "Fixed real inputs and pretrained weights; diagnostic coverage, not adjudicated bug labels.",
            }
            # The general native report expects a status column for each case.
            rendered = {**summary, "cases": [{**r, "status": r["classification"]} for r in rows]}
            write_native_report(output, rendered)
            if progress:
                progress(
                    f"{settings.device} {name}: {row['classification']} "
                    f"({row['baseline']['status']} -> {row['candidate']['status']})",
                    flush=True,
                )
        return summary


def add_sweep_commands(commands, execution_options):
    parser = commands.add_parser("sweep", help="prepare and compare a pretrained model catalog")
    sub = parser.add_subparsers(dest="sweep_command", required=True)
    sub.add_parser("list", help="list the built-in pretrained models")
    prepare = sub.add_parser("prepare", help="download pretrained weights and freeze real inputs")
    prepare.add_argument("--model", action="append", choices=MODELS)
    prepare.add_argument("--set", choices=("original", "held-out", "all"), dest="catalog_set")
    prepare.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run", help="compare prepared models with stable/nightly interpreters")
    run.add_argument("prepared", type=Path)
    run.add_argument("--model", action="append", choices=MODELS)
    run.add_argument("--set", choices=("original", "held-out", "all"), dest="catalog_set")
    run.add_argument("--baseline-python", required=True)
    run.add_argument("--candidate-python", default=sys.executable)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument(
        "--deterministic",
        action="store_true",
        help="require deterministic algorithms in both environments",
    )
    execution_options(run)
    run.set_defaults(gradients=True, numerics=True, repeats=2, timeout=600, max_rss_mb=12288)


def sweep_command(args, settings_from):
    if args.sweep_command == "list":
        print("\n".join(MODELS))
        return 0
    if args.model and args.catalog_set:
        raise ValueError("Choose --model or --set, not both")
    names = args.model or {
        "original": ORIGINAL_MODELS,
        "held-out": HELD_OUT_MODELS,
        "all": MODELS,
    }.get(args.catalog_set, ORIGINAL_MODELS)
    if args.sweep_command == "prepare":
        result = prepare_sweep(args.output, names)
        return 0 if all(row["status"] == "ready" for row in result["models"]) else 2
    result = run_sweep(
        args.prepared,
        args.output,
        args.baseline_python,
        args.candidate_python,
        settings_from(args),
        names=names,
        resume=args.resume,
    )
    print(json.dumps(result["counts"], indent=2))
    return result["exit_code"]
