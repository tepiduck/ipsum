"""Capture smaller forward graphs at actual tensor boundaries in a supervised worker."""

import copy
import io
from pathlib import Path

import torch

from .native import seal_native, verify_native
from .native_graph import load_native_graph, save_native_graph
from .native_reduce import graph_complexity
from .storage import atomic_json


def cut_graph(module, name):
    proposal = copy.deepcopy(module)
    cut = next(node for node in proposal.graph.nodes if node.name == name)
    inputs = [node for node in proposal.graph.nodes if node.op == "placeholder"]
    with proposal.graph.inserting_before(
        next(node for node in proposal.graph.nodes if node.op != "placeholder")
    ):
        replacement = proposal.graph.placeholder("faultline_boundary")
    replacement.meta = copy.deepcopy(cut.meta)
    cut.replace_all_uses_with(replacement)
    proposal.graph.eliminate_dead_code()
    retained = [i for i, node in enumerate(inputs) if node.users]
    for node in inputs:
        if not node.users:
            proposal.graph.erase_node(node)
    proposal.graph.lint()
    proposal.recompile()
    return proposal, retained


def capture_boundaries(job, settings, directory):
    manifest = verify_native(job.bundle)
    if manifest["kind"] != "native-graph" or manifest.get("aot_stage") or settings.gradients:
        raise ValueError("Boundary capture requires a forward-only Dynamo graph")
    module, requests, _ = load_native_graph(job.bundle, settings.device)
    if job.requests is not None:
        requests = [requests[i] for i in job.requests]
    calls = [node for node in module.graph.nodes if node.op.startswith("call_")]
    # Bound disk and snapshot cost. Final output producers cannot retain their own fault when frozen.
    selected = [
        node for node in reversed(calls) if any(user.op != "output" for user in node.users)
    ][:12]
    candidates = {}
    for node in selected:
        proposal, retained = cut_graph(module, node.name)
        candidates[node.name] = {"module": proposal, "retained": retained, "requests": []}
    directory = Path(directory)

    class Recorder(torch.fx.Interpreter):
        def run_node(self, node):
            result = super().run_node(node)
            if node.name in candidates:
                item = candidates[node.name]
                if not isinstance(result, torch.Tensor) or result.is_complex():
                    item["unsupported"] = True
                else:
                    # One serialization preserves views and aliases between every retained input
                    # and the boundary value, before subsequent graph operations can mutate them.
                    frozen = tuple(self.arguments[i] for i in item["retained"]) + (result,)
                    stream = io.BytesIO()
                    torch.save(frozen, stream)
                    stream.seek(0)
                    item["requests"].append(
                        torch.load(stream, map_location="cpu", weights_only=True)
                    )
            return result

    recorder = Recorder(module)
    with torch.no_grad():
        for arguments in requests:
            recorder.arguments = arguments
            recorder.run(*arguments)
    records = []
    original = graph_complexity(module, requests)
    for name, item in candidates.items():
        if item.get("unsupported") or len(item["requests"]) != len(requests):
            continue
        complexity = graph_complexity(item["module"], item["requests"])
        if complexity >= original:
            continue
        target = directory / "boundaries" / name
        save_native_graph(target, item["module"], item["requests"])
        seal_native(
            target,
            {
                "schema_version": 1,
                "kind": "native-graph",
                "sources": [],
                "seed": job.seed,
                "trace": len(requests) > 1,
                "request_count": len(requests),
                "captured": True,
                "frozen_boundary": name,
            },
        )
        records.append({"node": name, "bundle": f"boundaries/{name}", "complexity": complexity})
    records.sort(key=lambda item: item["complexity"])
    atomic_json(directory / "boundaries.json", records)
    return {"status": "pass", "boundaries": records, "original_complexity": original}
