"""Worker-local execution. The coordinator always invokes this in a subprocess."""

import os
import re
import time
from dataclasses import asdict, dataclass

import torch

from .lowering import build, inputs_for
from .oracle import fp64_advisory
from .portable import (
    check_workload,
    configure,
    confirm_exception,
    exception_record,
    fingerprint,
    tolerances,
)

__all__ = ["Settings", "confirm_exception", "evaluate", "fingerprint"]

BACKENDS = {"eager", "aot_eager", "aot_eager_decomp_partition", "inductor"}
MUTANTS = {
    "sum-as-mean",
    "relu-as-neg",
    "softmax-axis",
    "nan",
    "flaky",
    "crash",
    "hang",
    "worker-exit",
    "gradient-scale",
    "aot-backward-scale",
    "sequence-cache",
}


@dataclass(frozen=True)
class Settings:
    backend: str = "inductor"
    device: str = "cpu"
    repeats: int = 3
    rtol: float | None = None
    atol: float | None = None
    threads: int = 1
    timeout_s: float = 120.0
    max_rss_mb: int = 4096
    gradients: bool = False
    dynamic: bool | None = False
    compile_mode: str = "default"
    autocast: str | None = None
    fp64_advisory: bool = False
    numerics: bool = False

    def validate(self):
        valid = self.backend in BACKENDS or (
            self.backend.startswith("mutant:") and self.backend.split(":", 1)[1] in MUTANTS
        )
        if not valid:
            raise ValueError("unsupported backend")
        if self.device != "cpu" and not re.fullmatch(r"cuda(?::[0-9]+)?", self.device):
            raise ValueError("device must be cpu or cuda[:index] (also used by ROCm)")
        if not 2 <= self.repeats <= 20 or not 1 <= self.threads <= 64:
            raise ValueError("repeats must be 2..20 and threads 1..64")
        if not 0 < self.timeout_s <= 3600 or not 128 <= self.max_rss_mb <= 131072:
            raise ValueError("invalid execution limits")
        tolerances("float32", self.rtol, self.atol)
        if type(self.gradients) is not bool or (
            self.dynamic is not None and type(self.dynamic) is not bool
        ):
            raise ValueError(
                "gradients and dynamic must be boolean (dynamic also accepts auto/None)"
            )
        if self.backend == "mutant:gradient-scale" and not self.gradients:
            raise ValueError("gradient-scale requires gradient checking")
        if self.compile_mode not in {
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        }:
            raise ValueError("Unsupported compile mode")
        if self.compile_mode != "default" and self.backend != "inductor":
            raise ValueError("Compile modes require the inductor backend")
        if self.autocast not in {None, "float16", "bfloat16"}:
            raise ValueError("Autocast must be float16 or bfloat16")
        if type(self.fp64_advisory) is not bool:
            raise ValueError("fp64_advisory must be boolean")
        if type(self.numerics) is not bool:
            raise ValueError("numerics must be boolean")

    def to_dict(self):
        data = asdict(self)
        # Preserve hashes and serialized settings of version-1 findings.
        if not self.gradients:
            data.pop("gradients")
        if self.dynamic is False:
            data.pop("dynamic")
        if self.compile_mode == "default":
            data.pop("compile_mode")
        if self.autocast is None:
            data.pop("autocast")
        if not self.fp64_advisory:
            data.pop("fp64_advisory")
        if not self.numerics:
            data.pop("numerics")
        return data


def evaluate(case, settings):
    from .trace import Trace

    settings.validate()
    if (
        settings.compile_mode != "default"
        or settings.autocast
        or settings.fp64_advisory
        or settings.numerics
    ):
        raise ValueError("Compile modes and autocast currently require native model workloads")
    if settings.backend == "mutant:aot-backward-scale":
        raise ValueError("aot-backward-scale is only supported by native AOT workloads")
    case.validate()
    is_trace = isinstance(case, Trace)
    steps = case.steps if is_trace else (case,)
    graph = steps[0]
    if settings.backend == "mutant:sequence-cache" and not is_trace:
        raise ValueError("sequence-cache requires a trace")
    base = {
        "schema_version": 1,
        "case_hash": case.digest(),
        "backend": settings.backend,
        "device": settings.device,
        "injected_fault": settings.backend.startswith("mutant:"),
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "cuda_version": torch.version.cuda,
        "gradients": settings.gradients,
        "dynamic": settings.dynamic,
        "trace": is_trace,
    }

    def finish(record):
        result = {**base, **record}
        result["fingerprint"] = fingerprint(result)
        return result

    unavailable = configure(settings.device, settings.threads, case.seed)
    if unavailable:
        return finish(unavailable)
    if settings.device.startswith("cuda"):
        base["device_name"] = torch.cuda.get_device_name()
    rtol, atol = tolerances(case.inputs[0].dtype, settings.rtol, settings.atol)
    base.update(rtol=rtol, atol=atol, repeats=settings.repeats)
    mutant = settings.backend.split(":", 1)[1] if base["injected_fault"] else None

    def create_candidate():
        module = build(graph, mutant)
        if mutant == "worker-exit":
            os._exit(86)
        if mutant == "crash":
            raise RuntimeError("Injected compiler failure: faultline validation fixture")
        if mutant == "hang":
            time.sleep(settings.timeout_s + 30)
        return (
            module
            if settings.backend == "eager" or mutant
            else torch.compile(
                module, backend=settings.backend, fullgraph=True, dynamic=settings.dynamic
            )
        )

    with torch.enable_grad() if settings.gradients else torch.inference_mode():
        try:
            requests = [inputs_for(step, settings.device) for step in steps]
        except Exception as exc:
            return finish(exception_record(exc, "reference_error"))
        record, reference, candidate = check_workload(
            lambda: build(graph),
            create_candidate,
            requests,
            graph.to_dict()["inputs"],
            graph.outputs,
            gradients=settings.gradients,
            trace=is_trace,
            device=settings.device,
            repeats=settings.repeats,
            rtol=rtol,
            atol=atol,
            mutant=mutant,
        )
        if record["status"] == "numerical_mismatch" and not is_trace and not settings.gradients:
            try:
                high = build(case)(*inputs_for(case, "cpu", "float64"))
                record["fp64_advisory"] = fp64_advisory(reference, candidate, high)
            except Exception as exc:
                record["fp64_advisory"] = {"unavailable": str(exc)[:1000]}
        if reference is not None:
            record["input_strides"] = (
                [[list(t.stride()) for t in args] for args in requests]
                if is_trace
                else [list(t.stride()) for t in requests[0]]
            )
            record["fx_code"] = build(graph, mutant).code
        return finish(record)
