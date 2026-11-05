"""Numerical comparisons and an optional higher-precision advisory."""

import math

import torch

from .portable import compare, reference_health, tolerances

__all__ = ["compare", "fp64_advisory", "reference_health", "tolerances"]


def fp64_advisory(reference, candidate, high_precision):
    """An advisory comparison for this operator subset, not a universal ground truth."""

    def error(values):
        results = []
        for a, b in zip(values, high_precision, strict=True):
            if a.shape != b.shape or not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
                results.append(None)
            else:
                delta = float((a.detach().cpu().double() - b.detach().cpu().double()).abs().max())
                results.append(delta if math.isfinite(delta) else None)
        return results

    return {
        "reference_max_abs_error": error(reference),
        "candidate_max_abs_error": error(candidate),
        "note": "CPU float64 eager advisory; not proof of which implementation is correct.",
    }
