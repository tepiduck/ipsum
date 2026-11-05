"""Worker-local execution. The coordinator always invokes this in a subprocess."""

import os
import re
import time
from dataclasses import asdict, dataclass

import torch

from .lowering import build, inputs_for
from .oracle import fp64_advisory
from .portable import (
    check_execution,
    configure,
    confirm_exception,
    exception_record,
    fingerprint,
    tolerances,
)

__all__ = ["Settings", "confirm_exception", "evaluate", "fingerprint"]

BACKENDS = {"eager", "aot_eager", "inductor"}
MUTANTS = {
    "sum-as-mean",
    "relu-as-neg",
    "softmax-axis",
    "nan",
    "flaky",
    "crash",
    "hang",
    "worker-exit",
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

    def to_dict(self):
        return asdict(self)


def evaluate(case, settings):
    settings.validate()
    case.validate()
    base = {
        "schema_version": 1,
        "case_hash": case.digest(),
        "backend": settings.backend,
        "device": settings.device,
        "injected_fault": settings.backend.startswith("mutant:"),
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "cuda_version": torch.version.cuda,
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
        module = build(case, mutant)
        if mutant == "worker-exit":
            os._exit(86)
        if mutant == "crash":
            raise RuntimeError("Injected compiler failure: faultline validation fixture")
        if mutant == "hang":
            time.sleep(settings.timeout_s + 30)
        return (
            module
            if settings.backend == "eager" or mutant
            else torch.compile(module, backend=settings.backend, fullgraph=True, dynamic=False)
        )

    with torch.inference_mode():
        try:
            args = inputs_for(case, settings.device)
            reference_module = build(case)
        except Exception as exc:
            return finish(exception_record(exc, "reference_error"))
        record, reference, candidate = check_execution(
            reference_module,
            create_candidate,
            args,
            device=settings.device,
            repeats=settings.repeats,
            rtol=rtol,
            atol=atol,
            mutant=mutant,
        )
        if record["status"] == "numerical_mismatch":
            try:
                high = build(case)(*inputs_for(case, "cpu", "float64"))
                record["fp64_advisory"] = fp64_advisory(reference, candidate, high)
            except Exception as exc:
                record["fp64_advisory"] = {"unavailable": str(exc)[:1000]}
        if reference is not None:
            record["input_strides"] = [list(t.stride()) for t in args]
            record["fx_code"] = build(case, mutant).code
        return finish(record)
