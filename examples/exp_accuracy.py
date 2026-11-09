"""Measure an existing cpu_exp_probe.cpp adapter; requires PyTorch and an AVX512 CPU.

This is a sampled numerical and timing experiment, not a compiler-correctness test.
The adapter must be built against the same PyTorch environment that loads it.
"""

import argparse
import ctypes
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch


class Adapter:
    def __init__(self, path):
        if torch.backends.cpu.get_cpu_capability() != "AVX512":
            raise ValueError("This adapter requires AVX512 CPU dispatch")
        self.library = ctypes.CDLL(str(Path(path).resolve()))
        self.library.measured_mxcsr.restype = ctypes.c_uint32
        self.library.measured_vector_width.restype = ctypes.c_int
        self.library.measured_exp.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int,
        ]
        self.library.measured_exp.restype = None
        self.library.measured_scores.argtypes = (
            [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 5 + [ctypes.c_float, ctypes.c_int]
        )
        self.library.measured_scores.restype = None
        self.library.measured_forward_probabilities.argtypes = [ctypes.c_void_p] * 3 + [
            ctypes.c_int64
        ] * 2
        self.library.measured_forward_probabilities.restype = None
        if self.library.measured_vector_width() != 16:
            raise ValueError("Expected the AVX512 float32 adapter")

    @staticmethod
    def check(value, contiguous=True):
        if (
            value.device.type != "cpu"
            or value.dtype != torch.float32
            or (contiguous and not value.is_contiguous())
        ):
            raise ValueError(
                "Adapter requires CPU fp32 data; vector-array inputs must be contiguous"
            )

    def exp(self, value, kind, output=None):
        self.check(value)
        if kind not in (0, 1):
            raise ValueError("Choose exp_u20 (0) or exp (1)")
        output = torch.empty_like(value) if output is None else output
        self.check(output)
        if output.shape != value.shape:
            raise ValueError("Output shape differs")
        self.library.measured_exp(value.data_ptr(), output.data_ptr(), value.numel(), kind)
        return output

    def scores(self, q, k, scale, backward):
        self.check(q, contiguous=False)
        self.check(k, contiguous=False)
        if (
            q.ndim != 4
            or k.ndim != 4
            or q.shape[:2] != k.shape[:2]
            or q.shape[-1] != k.shape[-1]
            or q.stride(-1) != 1
            or k.stride(-1) != 1
        ):
            raise ValueError("Unsupported score layout")
        if not 0 < q.shape[-2] < 192 or not 0 < k.shape[-2] <= 512:
            raise ValueError("Adapter supports the 32-query, single-key-block source path only")
        if not math.isfinite(scale) or any(not x.shape[-1] <= x.stride(-2) < 2**31 for x in (q, k)):
            raise ValueError("Unsupported BLAS leading dimension or scale")
        output = torch.empty((*q.shape[:-1], k.shape[-2]), dtype=torch.float32)
        for batch in range(q.shape[0]):
            for head in range(q.shape[1]):
                self.library.measured_scores(
                    q[batch, head].data_ptr(),
                    k[batch, head].data_ptr(),
                    output[batch, head].data_ptr(),
                    q.shape[-2],
                    k.shape[-2],
                    q.shape[-1],
                    q.stride(-2),
                    k.stride(-2),
                    scale,
                    int(backward),
                )
        return output

    def probabilities(self, scores):
        self.check(scores)
        if (
            scores.ndim < 2
            or bool((torch.isnan(scores) | torch.isposinf(scores)).any())
            or not bool(torch.isfinite(scores).any(-1).all())
        ):
            raise ValueError("Require a finite maximum in every score row")
        probability = torch.empty_like(scores)
        lse = torch.empty(scores.shape[:-1], dtype=torch.float32)
        self.library.measured_forward_probabilities(
            scores.data_ptr(), probability.data_ptr(), lse.data_ptr(), lse.numel(), scores.shape[-1]
        )
        return probability, lse


def summarize_errors(inputs, output):
    """Separate representable ranges; real-valued ULP error differs from bit distance."""
    reference = inputs.double().exp()
    rounded = reference.float()
    tiny = torch.finfo(torch.float32).tiny
    groups = {
        "normal": (rounded >= tiny) & torch.isfinite(rounded),
        "subnormal": (rounded > 0) & (rounded < tiny),
        "underflow_to_zero": rounded == 0,
        "overflow_to_infinity": torch.isinf(rounded),
    }
    rows = {}
    for name, mask in groups.items():
        actual, exact, nearest = output[mask], reference[mask], rounded[mask]
        finite = torch.isfinite(actual) & torch.isfinite(nearest)
        record = {
            "count": int(mask.sum()),
            "finite_output_count": int(torch.isfinite(actual).sum()),
            "correctly_rounded_count": int((actual == nearest).sum()),
        }
        if bool(finite.any()):
            a, r, rounded_r = actual[finite], exact[finite], nearest[finite]
            exponent = torch.frexp(rounded_r.double())[1]
            spacing = torch.where(
                rounded_r < tiny,
                torch.full_like(r, 2**-149),
                torch.ldexp(torch.ones_like(r), exponent - 24),
            )
            error = (a.double() - r).abs() / spacing
            bits = (
                a.contiguous().view(torch.int32).long()
                - rounded_r.contiguous().view(torch.int32).long()
            ).abs()
            record.update(
                {
                    "max_ulp_error_vs_fp64": float(error.max()),
                    "median_ulp_error_vs_fp64": float(error.median()),
                    "p99_ulp_error_vs_fp64": float(torch.quantile(error, 0.99)),
                    "max_distance_to_rounded_fp32": int(bits.max()),
                    "over_two_ulp_count": int((error > 2).sum()),
                    "max_abs_error": float((a.double() - r).abs().max()),
                }
            )
        rows[name] = record
    return rows


def json_number(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def run_measurement(library):
    torch.set_num_threads(1)
    adapter = Adapter(library)
    control_before = adapter.library.measured_mxcsr()
    main = torch.linspace(-104, 0, 2**20 + 1, dtype=torch.float64).float()
    outputs = [torch.empty_like(main), torch.empty_like(main)]
    for _ in range(5):
        for kind in (0, 1):
            adapter.exp(main, kind, outputs[kind])
    times = {0: [], 1: []}
    for repeat in range(11):
        for kind in (0, 1) if repeat % 2 == 0 else (1, 0):
            started = time.perf_counter_ns()
            adapter.exp(main, kind, outputs[kind])
            times[kind].append((time.perf_counter_ns() - started) / 1e6)
    samples = {
        "minus104_to_zero": main,
        "subnormal_boundary": torch.linspace(-110, -86, 2**18 + 1, dtype=torch.float64).float(),
        "positive_to_overflow": torch.linspace(0, 89, 2**18 + 1, dtype=torch.float64).float(),
    }
    records = {}
    for name, values in samples.items():
        records[name] = {
            "count": values.numel(),
            "input_sha256": hashlib.sha256(values.numpy().tobytes()).hexdigest(),
            "implementations": {
                label: summarize_errors(values, adapter.exp(values, kind))
                for kind, label in ((0, "exp_u20"), (1, "exp"))
            },
        }
    exceptional = torch.tensor(
        [
            -math.inf,
            -110,
            -104,
            -103.97,
            -100,
            -90,
            -87.5,
            -87.3,
            -87.2,
            -87.0,
            -86.99,
            0,
            1,
            20,
            80,
            88.7,
            88.72,
            89,
            math.inf,
            math.nan,
        ],
        dtype=torch.float32,
    )
    results = [adapter.exp(exceptional, kind) for kind in (0, 1)]
    specials = [
        {
            "input": json_number(x),
            "reference_fp64": json_number(x.double().exp()),
            "exp_u20": json_number(results[0][i]),
            "exp": json_number(results[1][i]),
        }
        for i, x in enumerate(exceptional)
    ]
    timing = {
        label: {
            "samples_ms": times[kind],
            "median_ms": statistics.median(times[kind]),
            "min_ms": min(times[kind]),
            "max_ms": max(times[kind]),
            "ns_per_element": statistics.median(times[kind]) * 1e6 / main.numel(),
        }
        for kind, label in ((0, "exp_u20"), (1, "exp"))
    }
    return {
        "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "threads": 1,
        "vector_width": 16,
        "mxcsr_before": control_before,
        "mxcsr_after": adapter.library.measured_mxcsr(),
        "sampled_accuracy": records,
        "exceptional": specials,
        "timing": timing,
        "note": "Sampled coverage, not an exhaustive ULP bound. Timings include the ctypes call on a fixed 1,048,577-element array; five warmups and eleven alternating timed calls per implementation. Repeat in three fresh workers. No compiler-correctness verdict.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file")
    result = run_measurement(args.library)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
