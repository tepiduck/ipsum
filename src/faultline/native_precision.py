"""Higher-precision numerical evidence; shared with standalone native reproducers."""

import math

import torch

NOISE_FLOOR_C = 32.0
FP32_UNIT_ROUNDOFF = 2.0**-24


def reference_rms(value):
    value = value.double().abs()
    maximum = float(value.max()) if value.numel() else 0.0
    return maximum * float((value / maximum).square().mean().sqrt()) if maximum else 0.0


def noise_floor(eager_error, control_error, rms):
    """A declared engineering floor, not a fitted error bound or primary tolerance."""
    eager = eager_error.get("rmse")
    control = (control_error or {}).get("rmse")
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in (eager, rms)):
        return {"status": "unavailable"}
    control_valid = isinstance(control, (int, float)) and math.isfinite(control) and control >= 0
    rounding = NOISE_FLOOR_C * FP32_UNIT_ROUNDOFF * rms
    return {
        "status": "complete",
        "value": max(eager, control if control_valid else 0.0, rounding),
        "eager_rmse": eager,
        "eager_control_rmse": control if control_valid else None,
        "reference_rms": rms,
        "rounding_term": rounding,
        "c": NOISE_FLOOR_C,
        "unit_roundoff": FP32_UNIT_ROUNDOFF,
        "control_available": control_valid,
    }


def promote_tree(value):
    """Promote floating storage once, retaining aliases, offsets, strides and integer inputs."""
    tensors, storages = {}, {}
    storage_types = {}

    def inspect(item):
        if isinstance(item, torch.Tensor):
            if item.layout != torch.strided or item.is_complex():
                raise ValueError("fp64 promotion requires real strided tensors")
            key = (str(item.device), item.untyped_storage()._cdata)
            if key in storage_types and storage_types[key] != item.dtype:
                raise ValueError("fp64 promotion cannot preserve mixed-dtype storage aliases")
            storage_types[key] = item.dtype
        elif isinstance(item, dict):
            for child in item.values():
                inspect(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                inspect(child)

    inspect(value)

    def convert(item):
        if isinstance(item, torch.Tensor):
            if id(item) in tensors:
                return tensors[id(item)]
            if item.layout != torch.strided or item.is_complex():
                raise ValueError("fp64 promotion requires real strided tensors")
            if not item.is_floating_point():
                return item
            storage = item.untyped_storage()
            key = (str(item.device), storage._cdata)
            if key in storages and storages[key][0] != item.dtype:
                raise ValueError("fp64 promotion cannot preserve mixed-dtype storage aliases")
            if key not in storages:
                flat = torch.empty(0, dtype=item.dtype, device=item.device).set_(
                    storage, 0, (storage.nbytes() // item.element_size(),), (1,)
                )
                storages[key] = (item.dtype, flat.to(torch.float64))
            result = storages[key][1].as_strided(item.shape, item.stride(), item.storage_offset())
            result.requires_grad_(item.requires_grad)
            tensors[id(item)] = result
            return result
        if isinstance(item, dict):
            return {key: convert(v) for key, v in item.items()}
        if isinstance(item, tuple):
            return tuple(convert(v) for v in item)
        if isinstance(item, list):
            return [convert(v) for v in item]
        return item

    return convert(value)


def promote_model(model):
    parameters = dict(model.named_parameters(remove_duplicate=False))
    buffers = dict(model.named_buffers(remove_duplicate=False))
    converted = promote_tree({"parameters": parameters, "buffers": buffers})
    for name, parameter in parameters.items():
        parameter.data = converted["parameters"][name]
    for name in buffers:
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent), leaf, converted["buffers"][name])


def error_stats(actual, reference):
    error = (actual.double() - reference.double()).abs().reshape(-1)
    if not error.numel():
        return {"elements": 0, "max_abs": 0.0, "rmse": 0.0}
    maximum = float(error.max())
    if not math.isfinite(maximum):
        return {"elements": error.numel(), "nonfinite_error": True}
    rmse = maximum * float((error / maximum).square().mean().sqrt()) if maximum else 0.0
    stride = max(1, math.ceil(error.numel() / 32768))
    sample = error[::stride]
    quantiles = torch.quantile(sample, torch.tensor([0.5, 0.95, 0.99], dtype=torch.float64))
    return {
        "elements": error.numel(),
        "max_abs": maximum,
        "rmse": rmse,
        "sampled_elements": sample.numel(),
        "sample_stride": stride,
        "p50_abs": float(quantiles[0]),
        "p95_abs": float(quantiles[1]),
        "p99_abs": float(quantiles[2]),
    }


def accuracy_label(eager_error, candidate_error):
    """Descriptive RMSE bands, never an acceptance tolerance or correctness decision."""
    a, b = eager_error.get("rmse"), candidate_error.get("rmse")
    if a is None or b is None:
        return "unavailable"
    if a == b == 0:
        return "both_exact"
    if a == 0:
        return "eager_exact_candidate_differs"
    ratio = b / a
    if ratio < 0.5:
        return "candidate_closer_to_fp64"
    if ratio <= 2.0:
        return "comparable_to_eager_against_fp64"
    return "candidate_farther_from_fp64"


def assess_fp64(
    reference,
    eager,
    candidate,
    compare,
    policy,
    repeats,
    *,
    reference_is_factory=False,
    control=None,
    control_name=None,
):
    """Report errors without changing the eager-versus-compiled verdict."""
    note = (
        "Advisory only: promoting storage and disabling autocast changes numerical semantics. "
        "Explicit casts remain in the program. Closer to fp64 does not establish correctness. "
        "The same captured VJP probes are used when present; forward intermediates may still differ."
    )
    try:
        if reference_is_factory:
            reference = reference()
        high = reference()
        if not policy.reference_health(tuple(high[0].values())):
            return {"status": "reference_nonfinite", "note": note}
        for _ in range(repeats - 1):
            other = reference()
            if not compare(high, other, policy)["matches"]:
                return {"status": "unstable_reference", "note": note}
            del other
        records = []
        for name, expected in high[0].items():
            if (
                not expected.is_floating_point()
                or name.startswith("state:")
                or ":input_after:" in name
            ):
                continue
            left, right = eager[0].get(name), candidate[0].get(name)
            if (
                left is None
                or right is None
                or left.shape != expected.shape
                or right.shape != expected.shape
            ):
                records.append({"target": name, "status": "structure_mismatch"})
                continue
            a, b = error_stats(left, expected), error_stats(right, expected)
            alternate = control[0].get(name) if control is not None else None
            control_error = (
                error_stats(alternate, expected)
                if alternate is not None and alternate.shape == expected.shape
                else None
            )
            floor = noise_floor(a, control_error, reference_rms(expected))
            farther = None
            if expected.dtype == torch.float64 and floor.get("status") == "complete":
                if b.get("rmse") is not None:
                    farther = b["rmse"] > 2.0 * floor["value"]
                elif b.get("nonfinite_error"):
                    farther = True
            ratio = b["rmse"] / a["rmse"] if a.get("rmse", 0) and "rmse" in b else None
            records.append(
                {
                    "target": name,
                    "reference_dtype": str(expected.dtype),
                    "eager": a,
                    "candidate": b,
                    "candidate_to_eager_rmse": ratio
                    if ratio is None or math.isfinite(ratio)
                    else None,
                    "eager_exact": a.get("rmse") == 0,
                    "candidate_exact": b.get("rmse") == 0,
                    "legacy_accuracy_label": accuracy_label(a, b),
                    "eager_control": control_error,
                    "eager_control_policy": control_name,
                    "noise_floor": floor,
                    "farther_than_noise_floor": farther,
                    "accuracy_label": (
                        "unavailable"
                        if farther is None
                        else "candidate_farther_from_fp64"
                        if farther
                        else "comparable_to_eager_against_fp64"
                    ),
                }
            )
        return {
            "status": "complete",
            "outputs": records,
            "ratio_band": [0.5, 2.0],
            "ratio_band_scope": "legacy labels only; comparability now uses compiled_rmse <= 2 * noise_floor",
            "noise_floor_policy": {
                "c": NOISE_FLOOR_C,
                "unit_roundoff": FP32_UNIT_ROUNDOFF,
                "criterion": "compiled_err > 2 * max(eager_err, eager_control_err, c*u*RMS(reference))",
                "eager_control": control_name,
                "control_status": "available" if control is not None else "unavailable",
            },
            "note": note,
        }
    except Exception as exc:  # noqa: BLE001 -- arbitrary user model's optional fp64 execution
        return {"status": "reference_error", "error": f"{type(exc).__name__}: {exc}", "note": note}


def assess_backward_fp64(reference, eager, candidate, compare, policy, repeats):
    """Judge same-input backward error relative to eager, retaining the raw evidence."""
    evidence = assess_fp64(reference, eager, candidate, compare, policy, repeats)
    judgments = []
    if evidence["status"] == "complete":
        for row in evidence["outputs"]:
            agrees = None
            if row.get("reference_dtype") == "torch.float64":
                farther = row.get("farther_than_noise_floor")
                agrees = not farther if farther is not None else None
            judgments.append({"target": row["target"], "agrees": agrees})
    total = sum(value.is_floating_point() for value in eager[0].values())
    available = sum(row["agrees"] is not None for row in judgments)
    matched = sum(row["agrees"] is True for row in judgments)
    agrees = (
        False
        if any(row["agrees"] is False for row in judgments)
        else True
        if total and available == total and matched == total
        else None
    )
    return {
        "agrees": agrees,
        "matched": matched,
        "available": available,
        "total": total,
        "outputs": judgments,
        "fp64": evidence,
        "criterion": "Every floating backward output has compiled RMSE <= 2 * noise floor "
        "against repeated fp64 backward on identical compiled saved inputs. "
        "The floor uses max(eager RMSE, 32 * 2^-24 * RMS(reference)); an alternate-control "
        "backward is unavailable here. Old eager ratios are retained. Missing, unstable, "
        "unsupported, or explicitly downcast references are unknown, never agreement.",
    }
