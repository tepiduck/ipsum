"""AOT compiler adapters shared by workers and standalone native reproducers.

Inductor retains its own decomposition and partitioning pipeline. Only its final
boxed callable is observed. AOTAutograd cache reuse is disabled in diagnostic mode
so compiler callbacks cannot disappear on a cache hit; generated-kernel caching stays on.
"""

import copy
from contextlib import nullcontext

import torch


class AOTUnsupported(RuntimeError):
    pass


def capture_context(enabled):
    if not enabled:
        return nullcontext()
    from torch._functorch import config

    if not hasattr(config, "enable_autograd_cache"):
        raise AOTUnsupported("AOT diagnostic capture needs the enable_autograd_cache control")
    return config.patch(enable_autograd_cache=False)


def scale_backward(module):
    """Controlled compiler fault: multiply returned derivatives, leaving forward unchanged."""
    module = copy.deepcopy(module)
    output = next(node for node in module.graph.nodes if node.op == "output")
    with module.graph.inserting_before(output):

        def scale(node):
            scaled = module.graph.call_function(torch.ops.aten.mul.Scalar, (node, 2))
            scaled.meta = dict(node.meta)
            return scaled

        output.args = torch.fx.map_arg(output.args, scale)
    module.graph.lint()
    module.recompile()
    return module


def compile_stage(module, inputs, backend, stage, *, options=None, selected=True, observe=None):
    """Compile one already differentiated graph; return the backend's boxed ABI."""
    from functorch.compile import make_boxed_func

    module = (
        scale_backward(module)
        if selected and backend == "mutant:aot-backward-scale" and stage == "backward"
        else module
    )
    if not selected or backend in {"eager", "aot_eager", "mutant:aot-backward-scale"}:
        compiled = make_boxed_func(module.forward)
        if observe is None:
            return compiled

        def boxed(values):
            observe(values)
            return compiled(values)

        boxed._boxed_call = True
        return boxed
    if backend != "inductor":
        raise AOTUnsupported(f"Unsupported AOT backend: {backend}")
    from torch._inductor.compile_fx import compile_fx_inner

    compiled = compile_fx_inner(module, inputs, **(options or {"is_backward": stage == "backward"}))
    if observe is not None:
        if not callable(getattr(compiled, "current_callable", None)):
            raise AOTUnsupported("Inductor OutputCode has no observable current_callable")
        original = compiled.current_callable

        def recorded(values):
            observe(values)
            return original(values)

        # Keep OutputCode's type, cache metadata, runtime hooks, and CUDA graph flags intact.
        compiled.current_callable = recorded
    return compiled


def compile_pipeline(module, inputs, backend, stage_compiler):
    """Use the selected backend's normal AOT partitioner and compiler options."""
    if backend == "inductor":
        from torch._inductor.compile_fx import compile_fx

        def inner(graph, args, **options):
            stage = "backward" if options.get("is_backward") else "forward"
            return stage_compiler(graph, args, stage, options)

        return compile_fx(module, inputs, inner_compile=inner)
    if backend in {"aot_eager", "mutant:aot-backward-scale"}:
        from torch._dynamo.backends.debugging import aot_eager

        return aot_eager(
            module,
            inputs,
            fw_compiler=lambda graph, args: stage_compiler(graph, args, "forward", {}),
            bw_compiler=lambda graph, args: stage_compiler(graph, args, "backward", {}),
        )
    raise AOTUnsupported(f"Unsupported AOT pipeline: {backend}")


def replay_stage_backend(module, inputs, backend, manifest):
    stage = manifest["aot_stage"]
    placeholders = [node.name for node in module.graph.nodes if node.op == "placeholder"]
    static = set(manifest.get("aot_static_inputs", []))
    options = {
        "is_backward": stage == "backward",
        "is_inference": manifest.get("aot_inference", False),
        "static_input_idxs": [i for i, name in enumerate(placeholders) if name in static],
    }
    compiled = compile_stage(module, inputs, backend, stage, options=options)
    return lambda *args: compiled(list(args))


def annotate_aot_result(result, manifest, enabled, entries=None):
    """Shared worker/export status policy for AOT graph outputs and API failures."""
    if (
        "AOTUnsupported" in result.get("error", "")
        or result.get("exception_type") == "AOTUnsupported"
    ):
        result["status"] = "unsupported_aot"
    stage = manifest.get("aot_stage")
    if stage:
        result["graph_stage"] = stage
        result["observation_mode"] = f"aot_{stage}_outputs"
        if stage == "backward" and result["status"] == "numerical_mismatch":
            result["status"] = "gradient_mismatch"
    if enabled:
        result["aot"] = True
        result["aot_autograd_cache"] = "disabled for diagnostic capture"
        if entries is not None:
            result["aot_graphs"] = entries
            if result["status"] == "pass" and not any(x["stage"] == "backward" for x in entries):
                result["status"] = "aot_capture_missing"
    return result
