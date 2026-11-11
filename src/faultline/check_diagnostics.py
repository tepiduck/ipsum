"""Separate, supervised diagnostics for check; the original primary result is immutable."""

import argparse
import gc
from pathlib import Path

import torch

from . import native_numerics, native_precision, portable
from .evaluate import Settings
from .native import GraphRecorder, NativeJob, load_native_graph, verify_native
from .native_runtime import compare_native, make_native_execution, native_failure_key
from .storage import atomic_json, read_json


def release():
    torch._dynamo.reset()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def value_observation(observation):
    # State/input-after values participate in primary replay verification, then are
    # released: the existing fp64 advisory explicitly excludes these targets.
    return (
        {
            k: v
            for k, v in observation[0].items()
            if not k.startswith("state:") and ":input_after:" not in k
        },
        observation[1],
    )


def memory_diagnostic(result):
    """Preserve inner fp64/control errors while exposing an allocator OOM as a skip."""
    evidence = [
        result.get("fp64_advisory", {}),
        result.get("numerical_context", {}).get("eager_control", {}),
    ]
    if any(
        str(row.get("error", "")).startswith(("OutOfMemoryError:", "MemoryError:"))
        or "DefaultCPUAllocator: can't allocate memory" in str(row.get("error", ""))
        for row in evidence
    ):
        return {
            **result,
            "status": "diagnostic_skipped_memory",
            "message": "Diagnostic skipped (memory); primary result retained.",
        }
    return result


def diagnose(bundle, settings, primary, directory, *, control=False):
    manifest = verify_native(bundle)
    unavailable = portable.configure(
        settings.device, settings.threads, manifest["seed"], settings.deterministic
    )
    if unavailable:
        return {"status": "unavailable", "detail": unavailable}
    if manifest["kind"] != "workload":
        return {
            "status": "unsupported",
            "message": "Compact diagnostics require a whole-model workload.",
        }
    raw = settings.to_dict()

    def execution(*, compiler=None, precision=None, transform=None):
        return make_native_execution(
            bundle,
            manifest,
            raw,
            load_native_graph,
            compiler,
            before_run=(lambda n: setattr(recorder, "enabled", n == 0)) if compiler else None,
            precision=precision,
            request_transform=transform,
        )

    eager_fn = execution()
    eager = eager_fn()
    if not portable.reference_health(tuple(eager[0].values())):
        return {"status": "reference_nonfinite"}
    for _ in range(settings.repeats - 1):
        other = eager_fn()
        if (
            not portable.reference_health(tuple(other[0].values()))
            or not compare_native(eager, other, portable, settings.rtol, settings.atol)["matches"]
        ):
            return {"status": "unstable_reference"}
        del other
    del eager_fn
    release()
    recorder = GraphRecorder(Path(directory) / "graphs", settings)
    candidate_fn = execution(
        compiler=lambda model, is_graph: torch.compile(
            model, backend=recorder.backend, dynamic=settings.dynamic
        )
    )
    candidate = candidate_fn()
    comparison = compare_native(eager, candidate, portable, settings.rtol, settings.atol)
    if native_failure_key(comparison) != native_failure_key(primary.get("comparison", {})):
        return {"status": "replay_mismatch", "comparison": comparison}
    for _ in range(settings.repeats):
        other = candidate_fn()
        if not compare_native(candidate, other, portable, settings.rtol, settings.atol)[
            "matches"
        ] or native_failure_key(
            compare_native(eager, other, portable, settings.rtol, settings.atol)
        ) != native_failure_key(comparison):
            return {"status": "unstable_candidate"}
        del other
    del candidate_fn
    recorder = None
    release()
    eager, candidate = value_observation(eager), value_observation(candidate)
    control_result, sink = {}, {}
    if control:

        def control_factory(transform):
            fn = execution(transform=transform)
            return lambda: value_observation(fn())

        control_result = native_numerics.assess_control(
            control_factory,
            eager,
            candidate,
            raw,
            compare_native,
            portable,
            native_precision.error_stats,
            observation_sink=sink,
        )
        release()

    def reference_factory():
        fn = execution(precision=native_precision)
        return lambda: value_observation(fn())

    fp64 = native_precision.assess_fp64(
        reference_factory,
        eager,
        candidate,
        compare_native,
        portable,
        settings.repeats,
        reference_is_factory=True,
        control=sink.pop("observation", None),
        control_name=control_result.get("control"),
    )
    masks = {"status": "unavailable", "reason": "Lazy mask diagnostics not requested in this phase"}
    context = native_numerics.annotate(primary, fp64, control_result, masks)
    return {
        "status": "complete",
        "replay_matches_primary": True,
        "fp64_advisory": fp64,
        "numerical_context": context,
        "control_requested": control,
        "note": "Separate advisory replay. Existing primary observation and tolerances are unchanged.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    data = read_json(args.job)
    job = NativeJob.create(data["bundle"])
    if job.digest() != data["case_hash"]:
        raise ValueError("Diagnostic workload identity changed")
    try:
        result = diagnose(
            job.bundle,
            Settings(**data["settings"]),
            data["primary"],
            args.output.parent,
            control=data.get("control", False),
        )
    except (torch.OutOfMemoryError, MemoryError) as exc:
        result = {
            "status": "diagnostic_skipped_memory",
            "message": "Diagnostic skipped (memory); primary result retained.",
            "error": f"{type(exc).__name__}: {exc}",
        }
    atomic_json(args.output, memory_diagnostic(result))


if __name__ == "__main__":
    main()
