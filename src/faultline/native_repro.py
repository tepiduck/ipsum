"""Standalone entry point, copied together with the exact worker comparison modules."""

import hashlib
import json
from pathlib import Path

import native_aot
import native_compile
import native_graph
import native_masks
import native_numerics
import native_precision
import native_runtime
import portable
import torch


def main():
    root = Path(__file__).resolve().parent
    bundle = root / "workload"
    config = json.loads((root / "reproducer.json").read_text(encoding="utf-8"))
    for name, digest in config["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Artifact path escapes reproducer")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != digest:
            raise ValueError(f"Modified artifact: {name}")
    manifest = json.loads((bundle / "workload.json").read_text(encoding="utf-8"))
    settings = config["settings"]
    unavailable = portable.configure(settings["device"], settings["threads"], manifest["seed"])
    if unavailable:
        print(json.dumps(unavailable))
        return 2
    torch._dynamo.reset()
    compiled_graphs = []
    aot_graphs = []

    def compile_model(model, is_graph):
        compiled_graphs.clear()
        aot_graphs.clear()

        def backend(module, inputs):
            index = len(compiled_graphs)
            compiled_graphs.append(index)
            selected = config.get("only_graph") in (None, index)
            name = settings["backend"]
            if not selected or name == "eager":
                return module.forward
            if config.get("aot"):

                def stage_compiler(graph, args, stage, options):
                    ordinal = sum(x["parent"] == index and x["stage"] == stage for x in aot_graphs)
                    key = f"{index}:{stage}:{ordinal}"
                    aot_graphs.append({"parent": index, "stage": stage, "id": key})
                    return native_aot.compile_stage(
                        graph,
                        args,
                        name,
                        stage,
                        options=options,
                        selected=config.get("only_aot") in (None, key),
                    )

                return native_aot.compile_pipeline(
                    module, inputs, name, stage_compiler, native_compile.mode_options(settings)
                )
            if name.startswith("mutant:"):
                return native_graph.mutate_native(module, name.split(":", 1)[1]).forward
            return native_compile.compile_graph(module, inputs, settings)

        if is_graph and manifest.get("aot_stage"):
            return native_runtime.compile_native_graph(
                model,
                lambda gm, args: native_aot.replay_stage_backend(
                    gm, args, settings["backend"], manifest, native_compile.mode_options(settings)
                ),
            )
        if not is_graph:
            return torch.compile(model, backend=backend, dynamic=settings.get("dynamic", False))
        return native_runtime.compile_native_graph(model, backend)

    def execution(candidate=False, precision=None, request_transform=None, probe=None):
        def diagnostic_model(model, is_graph):
            return torch.compile(
                model, backend=probe.backend, dynamic=settings.get("dynamic", False)
            )

        return native_runtime.make_native_execution(
            bundle,
            manifest,
            settings,
            native_graph.load_native_graph,
            diagnostic_model if probe else compile_model if candidate else None,
            requests_selection=config.get("requests"),
            precision=precision,
            request_transform=request_transform,
            gradient_hook=probe.observe_gradients if probe else None,
            step_hook=(lambda n: setattr(probe, "step", n)) if probe else None,
        )

    def advisory(eager, candidate, primary):
        fp64 = native_precision.assess_fp64(
            lambda: execution(precision=native_precision),
            eager,
            candidate,
            native_runtime.compare_native,
            portable,
            settings["repeats"],
            reference_is_factory=True,
        )
        if settings.get("numerics"):
            compare = native_runtime.compare_native
            control = native_numerics.assess_control(
                lambda transform: execution(request_transform=transform),
                eager,
                candidate,
                settings,
                compare,
                portable,
                native_precision.error_stats,
            )
            masks = {
                "status": "unavailable",
                "reason": "Requires a whole-model --aot --gradients run",
            }
            if (
                config.get("aot")
                and manifest["kind"] == "workload"
                and settings["backend"] != "eager"
            ):
                torch._dynamo.reset()
                probe = native_masks.MaskProbe(
                    native_aot, native_compile, settings, compare, portable, config
                )
                masks = native_numerics.assess_masks(
                    probe,
                    lambda p: execution(probe=p),
                    eager,
                    candidate,
                    settings,
                    compare,
                    portable,
                )
            primary["numerical_context"] = native_numerics.annotate(primary, fp64, control, masks)
        return fp64

    try:
        with native_aot.capture_context(config.get("aot", False)):
            result = native_runtime.check_native(
                execution(),
                lambda: execution(True),
                settings,
                portable,
                advisory if settings.get("fp64_advisory") or settings.get("numerics") else None,
            )
    except native_aot.AOTUnsupported as exc:
        result = {"status": "unsupported_aot", "error": str(exc)}
    if (
        not compiled_graphs
        and not manifest.get("aot_stage")
        and settings["backend"] in {"inductor", "aot_eager", "aot_eager_decomp_partition"}
        and result["status"] == "pass"
    ):
        result["status"] = "no_compiled_graph"
    native_aot.annotate_aot_result(
        result,
        manifest,
        config.get("aot", False),
        aot_graphs if settings["backend"] != "eager" else None,
    )
    native_compile.annotate_compiler_result(result, settings)
    result.update(
        backend=settings["backend"],
        device=settings["device"],
        gradients=settings.get("gradients", False),
        dynamic=settings.get("dynamic", False),
        trace=True,
        native=True,
        backward_probe=manifest.get("backward_probe", "loss")
        if not manifest.get("aot_stage")
        else "aot_outputs",
        injected_fault=settings["backend"].startswith("mutant:"),
        repeats=settings["repeats"],
        rtol=settings.get("rtol"),
        atol=settings.get("atol"),
    )
    result["fingerprint"] = portable.fingerprint(result)
    expected = config.get("expected")
    if expected:
        matched = (
            result["status"] == "pass"
            if expected["status"] == "pass"
            else portable.same_failure(expected, result)
        )
        result["reproduces_expected"] = matched
        code = (0 if result["status"] == "pass" else 1) if matched else 2
    else:
        code = (
            0 if result["status"] == "pass" else 1 if result["status"] in portable.REDUCIBLE else 2
        )
    print(json.dumps(result, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    try:
        code = main()
    except Exception as exc:  # noqa: BLE001 -- standalone artifact/dependency boundary
        print(json.dumps({"status": "harness_error", "error": f"{type(exc).__name__}: {exc}"}))
        code = 2
    raise SystemExit(code)
