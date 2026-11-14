"""Summarize worker profiles and verify numerical evidence against retained checks."""

import argparse
from collections import defaultdict
from pathlib import Path

from summarize_check_memory import compact

from faultline.native import file_hash
from faultline.storage import atomic_json, read_json

MODELS = ("resnet50", "bert", "gpt2", "smollm2")
# Execution resources and paths necessarily change. Every other observation field
# must match, including target-level values, signatures and advisory evidence.
RESOURCE_FIELDS = {
    "cold_execution_s",
    "warm_execution_s",
    "worker_wall_s",
    "peak_tree_rss_mb",
    "worker_environment",
    "supervision",
}


def numerical(value):
    if isinstance(value, dict):
        return {key: numerical(item) for key, item in value.items() if key not in RESOURCE_FIELDS}
    if isinstance(value, list):
        return [numerical(item) for item in value]
    return value


def parity(before, after):
    old, new = [read_json(root / "evidence.json") for root in (before, after)]
    left, right = [read_json(root / "verdict.json") for root in (before, after)]
    checks = {
        "primary_and_diagnostic_evidence": [
            {
                key: numerical(row[key])
                for key in ("batch", "baseline", "candidate", "diagnostics", "accuracy_evaluated")
            }
            for row in old["batches"]
        ]
        == [
            {
                key: numerical(row[key])
                for key in ("batch", "baseline", "candidate", "diagnostics", "accuracy_evaluated")
            }
            for row in new["batches"]
        ],
    }
    for key in (
        "verdict",
        "policy_version",
        "exit_code",
        "scope",
        "reasons",
        "notes",
        "pre_existing_differences",
        "accuracy_shift_replicated",
    ):
        checks[key] = left[key] == right[key]
    return checks


def profile(root):
    rows = []
    totals, repeats = defaultdict(float), defaultdict(float)
    phases = {name: defaultdict(float) for name in ("primary", "diagnostic")}
    maxima = defaultdict(float)
    for path in sorted(root.rglob("profile.json")):
        row = read_json(path)
        phase = "primary" if "primary-attempt-" in str(path.relative_to(root)) else "diagnostic"
        for key, seconds in row["seconds"].items():
            totals[key] += seconds
            phases[phase][key] += seconds
        for key, seconds in row["repeat_seconds"].items():
            repeats[key] += seconds
        for boundary in [row["peak"], *row["boundaries"]]:
            for key, size in boundary["known_storage_mb"].items():
                maxima[key] = max(maxima[key], size)
            maxima["compiler_subprocess_rss_mb"] = max(
                maxima["compiler_subprocess_rss_mb"], boundary["compiler_subprocess_rss_mb"]
            )
        rows.append(
            {
                "worker": path.parent.relative_to(root).as_posix(),
                "profile_sha256": file_hash(path),
                "phase": phase,
                "seconds": row["seconds"],
                "repeat_seconds": row["repeat_seconds"],
                "peak": row["peak"],
            }
        )
    if not rows:
        raise ValueError("No worker profiles found")
    return {
        "workers": rows,
        "worker_count": len(rows),
        "command_metrics": read_json(root / "verdict.json")["metrics"],
        "exclusive_worker_seconds": dict(totals),
        "exclusive_seconds_by_worker_phase": {key: dict(value) for key, value in phases.items()},
        "repeat_seconds_subset": dict(repeats),
        "largest_peak": max(rows, key=lambda row: row["peak"]["tree_rss_mb"])["peak"],
        "independent_boundary_maxima_mb": dict(maxima),
        "note": "Instrumented worker wall time, summed across workers. Repeat times are subsets. Known tensor storage is not an allocator census. Independent memory maxima must not be added together.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--baseline-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Preserve completed summaries; choose a new output")
    cells = []
    for model in MODELS:
        before, after = [root / f"{model}-jobs-auto" for root in (args.before, args.after)]
        cells.append(
            {
                "model": model,
                "before": compact(before, model, "auto"),
                "after": compact(after, model, "auto"),
                "identical_evidence": parity(before, after),
            }
        )
    atomic_json(
        args.output,
        {
            "version": "0.19.0",
            "policy_version": 2,
            "measurements": cells,
            "baseline_profile": profile(args.baseline_profile),
            "baseline_profile_parity": parity(
                args.before / "gpt2-jobs-auto", args.baseline_profile
            ),
        },
    )
    if not all(all(cell["identical_evidence"].values()) for cell in cells):
        raise SystemExit("Numerical evidence changed; inspect the retained results")


if __name__ == "__main__":
    main()
