"""Measure saved CPU flash-attention calls against explicit fp64 math.

Requires only PyTorch. Inputs are torch.save records of actual forward/backward
arguments, loaded with weights_only=True. This produces advisory measurements;
it does not decide compiler correctness or change Faultline's numerical policy.
"""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import torch

FWD = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default
BWD = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu_backward.default
UNIT_ROUNDOFF = 2**-24


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def layout(value):
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "storage_offset": value.storage_offset(),
        "dtype": str(value.dtype),
        "contiguous": value.is_contiguous(),
    }


def bind_record(record, backward=False):
    names = (
        ["grad_out", "query", "key", "value", "out", "logsumexp", "dropout_p", "is_causal"]
        if backward
        else ["query", "key", "value", "dropout_p", "is_causal"]
    )
    arguments, keywords = record["args"], record["kwargs"]
    if len(arguments) > len(names) or set(keywords) - {*names, "attn_mask", "scale"}:
        raise ValueError("Unsupported operator argument contract")
    values = dict(zip(names, arguments))
    if set(values) & set(keywords):
        raise ValueError("Duplicate operator argument")
    values.update(keywords)
    required = names if backward else names[:3]
    if any(name not in values for name in required):
        raise ValueError("Missing required operator argument")
    for key, default in (
        ("dropout_p", 0.0),
        ("is_causal", False),
        ("attn_mask", None),
        ("scale", None),
    ):
        values.setdefault(key, default)
    return values


def load_records(forward_path, backward_path):
    def cpu_storage(storage, location):
        if location != "cpu":
            raise ValueError("This measurement requires records captured on CPU")
        return storage

    forward = torch.load(forward_path, map_location=cpu_storage, weights_only=True)
    backward = torch.load(backward_path, map_location=cpu_storage, weights_only=True)
    f, b = bind_record(forward), bind_record(backward, True)
    if f["dropout_p"] != 0 or b["dropout_p"] != 0:
        raise ValueError("This measurement supports dropout=0 only")
    for name in ("query", "key", "value"):
        a, z = f[name], b[name]
        if not all(
            isinstance(x, torch.Tensor) and x.dtype == torch.float32 and x.ndim == 4 for x in (a, z)
        ):
            raise ValueError("Expected four-dimensional CPU fp32 query/key/value tensors")
        if not torch.equal(a, z):
            raise ValueError(f"Forward/backward saved values differ: {name}")
    for name in ("is_causal", "scale"):
        if f[name] != b[name]:
            raise ValueError(f"Forward/backward setting differs: {name}")
    a, z = f["attn_mask"], b["attn_mask"]
    if (a is None) != (z is None) or (a is not None and not torch.equal(a, z)):
        raise ValueError("Forward/backward attention masks differ")
    q, k, v = (f[name] for name in ("query", "key", "value"))
    if (
        q.shape[:2] != k.shape[:2]
        or q.shape[:2] != v.shape[:2]
        or k.shape[-2] != v.shape[-2]
        or q.shape[-1] != k.shape[-1]
    ):
        raise ValueError("Unsupported query/key/value shapes; grouped-query attention is excluded")
    expected = (*q.shape[:-1], v.shape[-1])
    if b["grad_out"].shape != expected or b["grad_out"].dtype != torch.float32:
        raise ValueError("Upstream gradient contract differs")
    for name, saved in zip(("out", "logsumexp"), forward["outputs"], strict=True):
        if not torch.equal(saved, b[name]):
            raise ValueError(f"Saved forward result differs in backward: {name}")
    for value in (*[f[n] for n in ("query", "key", "value")], b["grad_out"]):
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Nonfinite attention data is outside this measurement contract")
    return f, b


def attention_scores(q, k, settings):
    scale = settings["scale"] if settings["scale"] is not None else 1 / math.sqrt(q.shape[-1])
    scores = (q @ k.transpose(-1, -2)) * scale
    mask = settings["attn_mask"]
    if mask is not None:
        scores = (
            scores.masked_fill(~mask, -math.inf)
            if mask.dtype == torch.bool
            else scores + mask.to(scores.dtype)
        )
    if settings["is_causal"]:
        permitted = torch.ones(scores.shape[-2:], dtype=torch.bool).tril()
        scores = scores.masked_fill(~permitted, -math.inf)
    if not bool(torch.isfinite(scores).any(-1).all()):
        raise ValueError("All-masked rows are outside this reference contract")
    return scores


def reference_math(f, b):
    with torch.enable_grad():
        q, k, v = [f[n].detach().double().requires_grad_(True) for n in ("query", "key", "value")]
        scores = attention_scores(q, k, f)
        probability = scores.softmax(-1)
        out = probability @ v
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), b["grad_out"].double())
        return {
            "out": out.detach(),
            "logsumexp": scores.logsumexp(-1).detach(),
            "dq": dq.detach(),
            "dk": dk.detach(),
            "dv": dv.detach(),
        }


def copy_layout(value, template):
    """Give a newly computed result the recorded backward input layout explicitly."""
    if tuple(value.shape) != tuple(template.shape) or value.dtype != template.dtype:
        raise ValueError("Fresh forward output differs from the saved backward contract")
    if layout(value) == layout(template):
        return value
    if any(stride < 0 for stride in template.stride()):
        raise ValueError("Negative strides are unsupported")
    size = (
        template.storage_offset()
        + 1
        + sum((n - 1) * s for n, s in zip(template.shape, template.stride(), strict=True))
    )
    storage = torch.empty(size, dtype=template.dtype)
    target = storage.as_strided(template.shape, template.stride(), template.storage_offset())
    target.copy_(value)
    return target


def flash_call(f, b, frozen_forward=False):
    with torch.no_grad():
        out, lse = FWD(
            f["query"],
            f["key"],
            f["value"],
            f["dropout_p"],
            f["is_causal"],
            attn_mask=f["attn_mask"],
            scale=f["scale"],
        )
        saved_out = b["out"] if frozen_forward else copy_layout(out, b["out"])
        saved_lse = b["logsumexp"] if frozen_forward else copy_layout(lse, b["logsumexp"])
        dq, dk, dv = BWD(
            b["grad_out"],
            b["query"],
            b["key"],
            b["value"],
            saved_out,
            saved_lse,
            b["dropout_p"],
            b["is_causal"],
            attn_mask=b["attn_mask"],
            scale=b["scale"],
        )
    return {"out": out, "logsumexp": lse, "dq": dq, "dk": dk, "dv": dv}


def rms(value):
    return float(value.double().square().mean().sqrt())


def measurements(actual, reference):
    rows = {}
    for name, expected in reference.items():
        difference = actual[name].double() - expected
        magnitude, error = rms(expected), rms(difference)
        rows[name] = {
            "rmse": error,
            "max_abs": float(difference.abs().max()),
            "reference_rms": magnitude,
            "rmse_over_u_reference_rms": error / (UNIT_ROUNDOFF * magnitude) if magnitude else None,
            "per_row_rmse": difference.square().mean(-1).sqrt().tolist()
            if name in ("dq", "dk", "dv")
            else None,
            "per_head_rmse": difference.flatten(2).square().mean(-1).sqrt().tolist(),
        }
    return rows


def measure(forward_path, backward_path, reference_path=None):
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    f, b = load_records(forward_path, backward_path)
    reference = reference_math(f, b)
    actual = flash_call(f, b)
    repeat = flash_call(f, b)
    if any(not torch.equal(actual[n], repeat[n]) for n in actual):
        raise ValueError("Kernel changed across the two internal repeats")
    result = {
        "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "inputs": {
            "forward_sha256": file_hash(forward_path),
            "backward_sha256": file_hash(backward_path),
        },
        "forward_layouts": {n: layout(f[n]) for n in ("query", "key", "value")},
        "backward_layouts": {
            n: layout(b[n]) for n in ("query", "key", "value", "grad_out", "out", "logsumexp")
        },
        "settings": {
            "threads": 1,
            "internal_repeats": 2,
            "dropout_p": f["dropout_p"],
            "is_causal": f["is_causal"],
            "scale": f["scale"],
            "effective_scale": f["scale"]
            if f["scale"] is not None
            else 1 / math.sqrt(f["query"].shape[-1]),
            "attn_mask": layout(f["attn_mask"]) if f["attn_mask"] is not None else None,
        },
        "targets": measurements(actual, reference),
        "note": "Advisory CPU flash measurement on fixed tensors. No compiler-correctness verdict. Newly computed out/logsumexp use the recorded backward layouts explicitly; q/k/v and grad_out retain their saved layouts.",
    }
    if reference_path:
        baseline = json.loads(Path(reference_path).read_text())
        if baseline["inputs"] != result["inputs"] or baseline["settings"] != result["settings"]:
            raise ValueError("Reference inputs or settings differ")
        growth, missing = {}, []
        for name in ("dq", "dk", "dv"):
            before, after = baseline["targets"][name]["rmse"], result["targets"][name]["rmse"]
            if not all(math.isfinite(v) and v > 0 for v in (before, after)):
                missing.append(name)
            else:
                growth[name] = math.log2(after) - math.log2(before)
        result["comparison"] = {
            "gradient_log2_growth": growth,
            "missing": missing,
            "median_log2_growth": statistics.median(growth.values()) if not missing else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward", type=Path, required=True)
    parser.add_argument("--backward", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file to preserve earlier evidence")
    result = measure(args.forward, args.backward, args.reference)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
