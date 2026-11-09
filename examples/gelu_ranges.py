"""Describe saved GELU inputs and reconstructed tanh arguments; PyTorch only.

The reconstructed argument is evaluated in fp64, not read from a fused kernel.
The -2*argument coordinate is hypothetical decomposed-tanh input to exp. These
distributions do not establish which math implementation a model actually calls.
"""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import torch

EDGES = [
    -104,
    -103.97,
    -90,
    -87.3,
    -80,
    -40,
    -20,
    -10,
    -5,
    -3,
    -1,
    0,
    1,
    3,
    5,
    10,
    20,
    40,
    80,
    89,
    104,
]


def tanh_argument(value):
    value = value.detach().double()
    return math.sqrt(2 / math.pi) * (value + 0.044715 * value**3)


def describe(value, edges=EDGES):
    if value.device.type != "cpu" or not value.is_floating_point() or not value.numel():
        raise ValueError("Expected nonempty CPU floating-point data")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Nonfinite values require a separate investigation")
    if any(not math.isfinite(x) for x in edges) or list(edges) != sorted(set(edges)):
        raise ValueError("Histogram edges must be finite and strictly increasing")
    flat = value.detach().double().reshape(-1)
    bins = torch.bucketize(flat, torch.tensor(edges, dtype=torch.float64), right=True)
    return {
        "count": flat.numel(),
        "minimum": float(flat.min()),
        "maximum": float(flat.max()),
        "abs_p95": float(torch.quantile(flat.abs(), 0.95)),
        "abs_p99": float(torch.quantile(flat.abs(), 0.99)),
        "edges": list(edges),
        "counts": torch.bincount(bins, minlength=len(edges) + 1).tolist(),
        "below_minus87_3": int((flat < -87.3).sum()),
        "below_minus103_97": int((flat < -103.97).sum()),
        "outside_v013_grid": int(((flat < -110) | (flat > 89)).sum()),
    }


def exp_disagreement(parent, child):
    if parent.shape != child.shape or not parent.numel():
        raise ValueError("Require equal nonempty output shapes")
    if parent.dtype != torch.float32 or child.dtype != torch.float32:
        raise ValueError("Require fp32 exponential outputs")
    equal = (parent == child) | (torch.isnan(parent) & torch.isnan(child))
    finite = torch.isfinite(parent) & torch.isfinite(child)
    changed = int((~equal).sum())
    return {
        "count": parent.numel(),
        "changed": changed,
        "changed_fraction": changed / parent.numel(),
        "both_finite": int(finite.sum()),
        "parent_nonfinite": int((~torch.isfinite(parent)).sum()),
        "child_nonfinite": int((~torch.isfinite(child)).sum()),
        "max_abs_difference_finite": float((parent[finite].double() - child[finite]).abs().max())
        if bool(finite.any())
        else None,
    }


def block12_comparison(rows):
    """Apply the fixed descriptive criteria; rows contain blocks 1 through 12."""
    if len(rows) != 12 or {r["block"] for r in rows} != set(range(1, 13)):
        raise ValueError("Require exactly one row for every block")
    rows = sorted(rows, key=lambda r: r["block"])
    previous, last = rows[:-1], rows[-1]
    baseline_p99 = statistics.median(r["abs_p99"] for r in previous)
    baseline_fraction = sum(r["changed"] for r in previous) / sum(r["count"] for r in previous)
    fraction = last["changed"] / last["count"]
    return {
        "block12_abs_p99": last["abs_p99"],
        "previous_median_abs_p99": baseline_p99,
        "p99_over_twice_previous_median": last["abs_p99"] > 2 * baseline_p99,
        "block12_changed_fraction": fraction,
        "previous_pooled_changed_fraction": baseline_fraction,
        "changed_fraction_difference": fraction - baseline_fraction,
        "changed_fraction_at_least_10pp_higher": fraction - baseline_fraction >= 0.10,
        "abs_p99_rank_descending": 1 + sum(r["abs_p99"] > last["abs_p99"] for r in previous),
        "changed_fraction_rank_descending": 1
        + sum(r["changed"] / r["count"] > fraction for r in previous),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file to preserve earlier evidence")

    def cpu_storage(storage, location):
        if location != "cpu":
            raise ValueError("Expected a CPU capture")
        return storage

    value = torch.load(args.input, weights_only=True, map_location=cpu_storage)["value"]
    if value.dtype != torch.float32:
        raise ValueError("Expected frozen fp32 GELU inputs")
    argument = tanh_argument(value)
    result = {
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "layout": {
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "storage_offset": value.storage_offset(),
        },
        "input": describe(value),
        "reconstructed_tanh_argument_fp64": describe(argument),
        "hypothetical_exp_argument_fp64": describe(-2 * argument),
        "note": "Advisory distributions. Reconstructed intermediates and hypothetical exp coordinates do not establish actual exp_u20 exposure.",
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
