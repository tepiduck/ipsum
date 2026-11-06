"""Explicit compiler policy shared by native workers and exported reproducers."""

import torch


class CompilerUnsupported(RuntimeError):
    pass


def backend_for(name):
    from torch._dynamo.backends.registry import lookup_backend
    from torch._dynamo.exc import InvalidBackend

    try:
        return lookup_backend(name)
    except (InvalidBackend, ImportError) as exc:
        raise CompilerUnsupported(f"Backend unavailable in this interpreter: {name}") from exc


def mode_options(settings):
    mode = settings.get("compile_mode", "default")
    if mode == "default":
        return {}
    try:
        return dict(torch._inductor.list_mode_options(mode, dynamic=settings.get("dynamic", False)))
    except (AttributeError, KeyError, RuntimeError) as exc:
        raise CompilerUnsupported(f"Inductor compile mode unavailable: {mode}") from exc


def compile_graph(module, inputs, settings):
    compiler = backend_for(settings["backend"])
    if settings["backend"] == "inductor":
        return compiler(module, inputs, config_patches=mode_options(settings))
    return compiler(module, inputs)


def annotate_compiler_result(result, settings):
    if (
        "CompilerUnsupported" in result.get("error", "")
        or result.get("exception_type") == "CompilerUnsupported"
    ):
        result["status"] = "unsupported_backend"
    result["compile_mode"] = settings.get("compile_mode", "default")
    result["autocast"] = settings.get("autocast")
    result["fp64_requested"] = settings.get("fp64_advisory", False) or settings.get(
        "numerics", False
    )
    result["numerics_requested"] = settings.get("numerics", False)
    result["deterministic"] = settings.get("deterministic", False)
    return result
