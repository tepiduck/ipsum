"""Standalone entry point, copied together with the exact worker comparison modules."""

import hashlib
import json
from pathlib import Path

import native_graph
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

    def compile_model(model, is_graph):
        from torch._dynamo.backends.registry import lookup_backend

        compiled_graphs.clear()

        def backend(module, inputs):
            index = len(compiled_graphs)
            compiled_graphs.append(index)
            selected = config.get("only_graph") in (None, index)
            name = settings["backend"]
            if not selected or name == "eager":
                return module.forward
            if name.startswith("mutant:"):
                return native_graph.mutate_native(module, name.split(":", 1)[1]).forward
            return lookup_backend(name)(module, inputs)

        if not is_graph:
            return torch.compile(model, backend=backend, dynamic=settings.get("dynamic", False))
        return native_runtime.compile_native_graph(model, backend)

    def execution(candidate=False):
        return native_runtime.make_native_execution(
            bundle,
            manifest,
            settings,
            native_graph.load_native_graph,
            compile_model if candidate else None,
            requests_selection=config.get("requests"),
        )

    result = native_runtime.check_native(execution(), lambda: execution(True), settings, portable)
    if (
        not compiled_graphs
        and settings["backend"] in {"inductor", "aot_eager"}
        and result["status"] == "pass"
    ):
        result["status"] = "no_compiled_graph"
    result.update(
        backend=settings["backend"],
        device=settings["device"],
        gradients=settings.get("gradients", False),
        dynamic=settings.get("dynamic", False),
        trace=True,
        native=True,
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
