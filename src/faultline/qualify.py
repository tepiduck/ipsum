"""Execute a reproducible device/compiler matrix without silently falling back to CPU."""

from pathlib import Path

from .environment import describe
from .native import capture_workload
from .native_artifacts import run_native, write_native_report
from .storage import atomic_json


def qualify_device(
    output,
    settings,
    *,
    dtypes=("float32", "float16", "bfloat16"),
    workloads=("mlp", "convolution", "attention"),
    require_rocm=False,
    require_cuda=False,
    aot=True,
    progress=print,
):
    settings.validate()
    if settings.backend not in {"inductor", "aot_eager"} or not settings.gradients:
        raise ValueError("Device qualification requires inductor/aot_eager and gradient checking")
    if not dtypes or not set(dtypes) <= {"float32", "float16", "bfloat16"}:
        raise ValueError("Qualification dtypes must be float32, float16, or bfloat16")
    if require_rocm and require_cuda:
        raise ValueError("Select one required GPU runtime")
    if not workloads or not set(workloads) <= {
        "mlp",
        "convolution",
        "attention",
        "buffer_update",
        "shape_sequence",
    }:
        raise ValueError("Unknown qualification workload")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    environment = describe()
    summary = {
        "schema_version": 1,
        "device": settings.device,
        "backend": settings.backend,
        "aot_capture": aot,
        "requested_gpu_runtime": "rocm" if require_rocm else "cuda" if require_cuda else None,
        "cudagraph_note": "A requested compile mode does not prove CUDA graph use. Inspect worker logs for compiler skips; GPU execution requires a matching runtime and visible device.",
        "settings": settings.to_dict(),
        "environment": environment,
        "cases": [],
        "status": "in_progress",
        "exit_code": 2,
        "scope": "Fixed synthetic kernel matrix; not a performance benchmark or arbitrary-model certification.",
    }
    reason = None
    if require_cuda and (
        not settings.device.startswith("cuda")
        or not environment["cuda_version"]
        or environment["hip_version"]
    ):
        reason = (
            "NVIDIA CUDA was required, but this interpreter has no selected CUDA device/runtime"
        )
    elif require_rocm and (
        not settings.device.startswith("cuda") or not environment["hip_version"]
    ):
        reason = "ROCm was required, but this interpreter has no selected ROCm device/runtime"
    elif settings.device.startswith("cuda") and not environment["gpu_available"]:
        reason = (
            "Installed PyTorch has no CUDA/ROCm runtime"
            if not environment["hip_version"] and not environment["cuda_version"]
            else "PyTorch has a GPU runtime, but no usable GPU is visible"
        )
    if reason:
        summary.update(status="unavailable", reason=reason)
        write_native_report(output, summary)
        return summary
    source = Path(__file__).with_name("device_workloads.py").read_text(encoding="utf-8")
    for dtype in dict.fromkeys(dtypes):
        for name in dict.fromkeys(workloads):
            key = f"{name}-{dtype}"
            directory = output / "cases" / key
            directory.mkdir(parents=True)
            factory = directory / "source.py"
            factory.write_text(source + f"\nDTYPE = torch.{dtype}\n", encoding="utf-8")
            try:
                bundle = capture_workload(
                    f"{factory}:{name}",
                    directory / "bundle",
                    settings,
                    trace=name in {"buffer_update", "shape_sequence"},
                )
                result = run_native(bundle, directory / "check", settings, aot=aot)
                record = {
                    "id": key,
                    "status": result["status"],
                    "exit_code": result["exit_code"],
                    "report": f"cases/{key}/check/report.html",
                }
            except (ValueError, OSError) as exc:
                record = {"id": key, "status": "capture_error", "exit_code": 2, "error": str(exc)}
            summary["cases"].append(record)
            if progress:
                progress(f"{key}: {record['status']}")
            atomic_json(output / "summary.json", summary)
    codes = [item["exit_code"] for item in summary["cases"]]
    code = 2 if 2 in codes else 1 if 1 in codes else 0
    summary.update(exit_code=code, status={0: "pass", 1: "discrepancy", 2: "inconclusive"}[code])
    write_native_report(output, summary)
    return summary
