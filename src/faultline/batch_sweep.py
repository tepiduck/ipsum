"""Multi-input sweep orchestration, retaining per-batch primary outcomes."""

from collections import Counter
from pathlib import Path

from .batches import compare_batches
from .compare_envs import gate_status
from .native import file_hash
from .native_artifacts import write_native_report
from .replication import accuracy_shift_replicated
from .storage import atomic_json, campaign_lock, provenance, read_json
from .sweep import runtime_environment


def run_batch_sweep(
    prepared,
    output,
    baseline_python,
    candidate_python,
    settings,
    *,
    names,
    entries,
    resume=False,
    progress=print,
):
    prepared, output = Path(prepared), Path(output)
    count = read_json(prepared / "prepared.json")["input_batches"]
    config = {
        "prepared_hash": file_hash(prepared / "prepared.json"),
        "models": names,
        "input_batches": count,
        "settings": settings.to_dict(),
        "source_sha256": provenance()["faultline_source_sha256"],
        "baseline_python": baseline_python,
        "candidate_python": candidate_python,
        "baseline_environment": runtime_environment(baseline_python),
        "candidate_environment": runtime_environment(candidate_python),
    }
    with campaign_lock(output):
        path = output / "sweep.json"
        if path.exists():
            if not resume or read_json(path) != config:
                raise ValueError("Sweep exists or source/settings/input/environment changed")
        else:
            atomic_json(path, config)
        rows = []
        for name in names:
            entry = entries[name]
            if entry["status"] == "ready":
                destination = output / "cases" / name
                result = compare_batches(
                    prepared / entry["bundle"],
                    destination,
                    baseline_python,
                    candidate_python,
                    settings,
                    aot=True,
                    resume=resume and (destination / "batch-comparison.json").exists(),
                )
            else:
                result = {
                    "batches": [],
                    "exit_code": 2,
                    "accuracy_shift_replicated": accuracy_shift_replicated([], count),
                    "preparation_error": entry.get("error"),
                }
            rows.append({"id": name, **result})
            classifications = Counter(
                batch["classification"] for row in rows for batch in row["batches"]
            )
            missing = len(names) * count - sum(len(row["batches"]) for row in rows)
            if missing:
                classifications["inconclusive"] += missing
            summary = {
                "schema_version": 2,
                "models": rows,
                "planned_models": len(names),
                "planned_batches": len(names) * count,
                "completed_models": len(rows),
                "counts": {
                    "planned": len(names) * count,
                    "missing": missing,
                    "primary_classifications": dict(classifications),
                    "accuracy_shift_replicated": sum(
                        row["accuracy_shift_replicated"]["accuracy_shift_replicated"]
                        for row in rows
                    ),
                    "single_input_accuracy_shift": sum(
                        batch.get("accuracy_shift", {}).get("accuracy_shift", False)
                        for row in rows
                        for batch in row["batches"]
                    ),
                },
                "exit_code": gate_status(classifications),
            }
            write_native_report(output, summary)
            if progress:
                evidence = result["accuracy_shift_replicated"]
                progress(
                    f"{name}: {evidence['completed']}/{count} accuracy batches; replicated={evidence['accuracy_shift_replicated']}",
                    flush=True,
                )
        return summary
