"""Higher-precision numerical evidence; shared with standalone native reproducers."""

import math

import torch


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
    reference, eager, candidate, compare, policy, repeats, *, reference_is_factory=False
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
                    "accuracy_label": accuracy_label(a, b),
                }
            )
        return {
            "status": "complete",
            "outputs": records,
            "ratio_band": [0.5, 2.0],
            "note": note,
        }
    except Exception as exc:  # noqa: BLE001 -- arbitrary user model's optional fp64 execution
        return {"status": "reference_error", "error": f"{type(exc).__name__}: {exc}", "note": note}
