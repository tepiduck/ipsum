"""Shared comparison predicate, embedded verbatim in standalone reproducers.

Keep this module independent of Faultline imports: exported scripts need only
PyTorch and the Python standard library. Exception handling is the compiler
boundary, where arbitrary backend exception classes are expected.
"""

import math
import re
import time
import traceback

import torch

TOLERANCES = {
    "float64": (1e-8, 1e-10),
    "float32": (1e-4, 1e-5),
    "float16": (1e-2, 1e-3),
    "bfloat16": (5e-2, 5e-3),
}

REDUCIBLE = {"numerical_mismatch", "structural_mismatch", "compile_error", "runtime_error"}


def same_failure(baseline, result):
    return (
        baseline["status"] in REDUCIBLE
        and result["status"] == baseline["status"]
        and result["fingerprint"] == baseline["fingerprint"]
        and result.get("injected_fault") == baseline.get("injected_fault")
        and result.get("backend") == baseline.get("backend")
        and result.get("device") == baseline.get("device")
    )


def tolerances(dtype, rtol=None, atol=None):
    default_rtol, default_atol = TOLERANCES[dtype]
    rtol = default_rtol if rtol is None else rtol
    atol = default_atol if atol is None else atol
    if any(not math.isfinite(v) or v < 0 for v in (rtol, atol)):
        raise ValueError("tolerances must be finite and nonnegative")
    return rtol, atol


def compare(reference, candidate, rtol, atol):
    if len(reference) != len(candidate):
        return {"matches": False, "kind": "structure", "outputs": []}
    reports = []
    for index, (ref, actual) in enumerate(zip(reference, candidate, strict=True)):
        base = {
            "index": index,
            "reference_shape": list(ref.shape),
            "candidate_shape": list(actual.shape),
        }
        if ref.shape != actual.shape or ref.dtype != actual.dtype:
            reports.append({**base, "matches": False, "kind": "structure"})
            continue
        ref, actual = ref.detach().cpu().double(), actual.detach().cpu().double()
        ref_finite, actual_finite = torch.isfinite(ref), torch.isfinite(actual)
        finite = ref_finite & actual_finite
        # Matched nonfinites are recorded separately; they do not establish correctness.
        nonfinite_equal = (torch.isnan(ref) & torch.isnan(actual)) | (ref == actual)
        delta = torch.where(finite, (ref - actual).abs(), 0.0)
        allowance = atol + rtol * ref.abs()
        mismatches = (finite & (delta > allowance)) | (~finite & ~nonfinite_equal)
        denominator = torch.where(finite, allowance.clamp_min(torch.finfo(torch.float64).tiny), 1)
        scaled = float((delta / denominator).max())
        max_abs = float(delta.max())
        reports.append(
            {
                **base,
                "matches": not bool(mismatches.any()),
                "kind": "numeric",
                "elements": ref.numel(),
                "mismatched_elements": int(mismatches.sum()),
                "reference_nonfinite": int((~ref_finite).sum()),
                "candidate_nonfinite": int((~actual_finite).sum()),
                "max_abs_error": max_abs if math.isfinite(max_abs) else None,
                "max_scaled_error": scaled if math.isfinite(scaled) else None,
                "bitwise_equal": bool(torch.equal(ref, actual)),
            }
        )
    return {
        "matches": all(x["matches"] for x in reports),
        "kind": "structure" if any(x["kind"] == "structure" for x in reports) else "numeric",
        "outputs": reports,
    }


def reference_health(outputs):
    return all(bool(torch.isfinite(t).all()) for t in outputs)


def fingerprint(result):
    """A failure-family key, not a claim that deduplicated cases share a root cause."""
    status = result["status"]
    if status in {"numerical_mismatch", "structural_mismatch"}:
        failed = [
            (x["index"], "nonfinite" if x.get("candidate_nonfinite", 0) else x["kind"])
            for x in result["comparison"]["outputs"]
            if not x["matches"]
        ]
        return f"{status}:outputs={failed}:backend={result['backend']}"
    if status in {"compile_error", "runtime_error"}:
        return f"{status}:{result.get('exception_type')}:{result.get('error_signature')}"
    return status


def exception_record(exc, status):
    message = str(exc)
    # Do not let paths, addresses or generated graph names dominate deduplication.
    signature = re.sub(r"0x[0-9a-fA-F]+", "<address>", message.splitlines()[0] if message else "")
    signature = re.sub(r"(?:[A-Za-z]:)?[/\\][^\s:]+", "<path>", signature)[:300]
    return {
        "status": status,
        "exception_type": type(exc).__name__,
        "error": message[:6000],
        "error_signature": signature,
        "traceback": traceback.format_exc()[-16000:],
    }


def confirm_exception(exc, status, repeat, repeats):
    original = exception_record(exc, status)
    key = (original["exception_type"], original["error_signature"])
    for _ in range(repeats - 1):
        try:
            repeat()
        except Exception as other:
            observed = exception_record(other, status)
            if (observed["exception_type"], observed["error_signature"]) != key:
                return {
                    "status": "unstable_candidate",
                    "first_error": original,
                    "different_error": observed,
                }
        else:
            return {
                "status": "unstable_candidate",
                "first_error": original,
                "error": "Exception did not recur",
            }
    return {**original, "exception_confirmations": repeats}


def configure(device, threads, seed):
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            return {"status": "unavailable", "error": "No CUDA/ROCm device is available"}
        try:
            torch.cuda.set_device(device)
        except Exception as exc:
            return exception_record(exc, "unavailable")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return None


def comparison_key(comparison):
    """Require the same affected outputs and failure kinds on every invocation."""
    return (
        comparison["matches"],
        comparison["kind"],
        tuple(
            (x["index"], "nonfinite" if x.get("candidate_nonfinite", 0) else x["kind"])
            for x in comparison["outputs"]
            if not x["matches"]
        ),
    )


def check_execution(reference, create_candidate, args, *, device, repeats, rtol, atol, mutant=None):
    """Return (record, first reference, first candidate) using one repeated predicate.

    Factories and inputs are supplied by either the IR runner or standalone code.
    First-call compilation errors are retried with fresh candidate instances;
    warm execution errors are retried on the same compiled callable.
    """

    def execute(fn):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        values = fn(*args)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        return tuple(t.detach().clone() for t in values), time.perf_counter() - start

    try:
        references = [execute(reference)[0] for _ in range(repeats)]
    except Exception as exc:
        return exception_record(exc, "reference_error"), None, None
    if not all(reference_health(r) for r in references):
        return {"status": "reference_nonfinite", "error": "Nonfinite eager output"}, None, None
    if any(not compare(references[0], r, rtol, atol)["matches"] for r in references[1:]):
        return {"status": "unstable_reference"}, None, None

    def cold_start():
        candidate = create_candidate()
        return candidate, execute(candidate)

    try:
        candidate, (cold, cold_s) = cold_start()
    except Exception as exc:
        return confirm_exception(exc, "compile_error", cold_start, repeats), None, None
    candidates, times = [], []
    try:
        for iteration in range(repeats):
            output, duration = execute(candidate)
            if mutant == "nan":
                output = tuple(torch.full_like(t, float("nan")) for t in output)
            elif mutant == "flaky" and iteration % 2 == 0:
                output = tuple(t + 1 for t in output)
            candidates.append(output)
            times.append(duration)
    except Exception as exc:
        record = confirm_exception(exc, "runtime_error", lambda: execute(candidate), repeats)
        return record, None, None
    if any(not compare(candidates[0], r, rtol, atol)["matches"] for r in candidates[1:]):
        return {"status": "unstable_candidate", "cold_execution_s": cold_s}, None, None

    comparisons = [
        compare(ref, out, rtol, atol) for ref, out in zip(references, candidates, strict=True)
    ]
    comparison = comparisons[0]
    cold_comparison = compare(references[0], cold, rtol, atol)
    target = comparison_key(comparison)
    if any(comparison_key(c) != target for c in comparisons[1:]) or (
        mutant is None and comparison_key(cold_comparison) != target
    ):
        return (
            {
                "status": "unstable_candidate",
                "reason": "Failure classification changed between invocations",
                "cold_comparison": cold_comparison,
                "comparisons": comparisons,
            },
            None,
            None,
        )
    status = (
        "pass"
        if comparison["matches"]
        else "structural_mismatch"
        if comparison["kind"] == "structure"
        else "numerical_mismatch"
    )
    return (
        {
            "status": status,
            "comparison": comparison,
            "cold_execution_s": cold_s,
            "warm_execution_s": times,
            "timing_note": "First call includes compilation; diagnostic timings, not a benchmark.",
        },
        references[0],
        candidates[0],
    )
