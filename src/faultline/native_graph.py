"""Native FX graph snapshots. Tensor storage stays binary and retains aliases/strides.

This module is also embedded in standalone reproducers. Artifacts are trusted local
programs, not a sandbox; only standard torch/operator/math targets are portable.
"""

import builtins
import copy
import importlib
import json
import math
import operator
from pathlib import Path

import torch


def native_target(target):
    if isinstance(target, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)):
        return {"module": "torch.ops", "name": str(target)}
    module, name = getattr(target, "__module__", None), getattr(target, "__name__", None)
    if module == "_operator":
        module = "operator"
    if module is None or name is None:
        raise ValueError(f"Cannot serialize native target {target}")
    record = {"module": module, "name": name}
    if resolve_native_target(record) is not target:
        raise ValueError(f"Native target has no stable import: {module}.{name}")
    return record


def resolve_native_target(record):
    module, name = record["module"], record["name"]
    allowed = {
        "torch",
        "torch.nn.functional",
        "torch._C._nn",
        "torch.ops",
        "operator",
        "math",
        "builtins",
    }
    if module not in allowed or any(part.startswith("__") for part in name.split(".")):
        raise ValueError(f"Native target is not portable: {module}.{name}")
    value = {"operator": operator, "math": math, "builtins": builtins}.get(module)
    if value is None:
        value = importlib.import_module(module) if module != "torch.ops" else torch.ops
    for part in name.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise ValueError("Native function target is not callable")  # noqa: TRY004
    return value


def encode_native(value, tensors):
    if isinstance(value, torch.fx.Node):
        return {"node": value.name}
    if isinstance(value, torch.Tensor):
        key = f"constant_{len(tensors)}"
        tensors[key] = value.detach()
        return {"tensor": key}
    if isinstance(value, tuple):
        return {"tuple": [encode_native(v, tensors) for v in value]}
    if isinstance(value, list):
        return {"list": [encode_native(v, tensors) for v in value]}
    if isinstance(value, dict):
        return {
            "dict": [
                [encode_native(k, tensors), encode_native(v, tensors)] for k, v in value.items()
            ]
        }
    if isinstance(value, slice):
        return {"slice": [encode_native(v, tensors) for v in (value.start, value.stop, value.step)]}
    if isinstance(value, torch.dtype):
        return {"dtype": str(value).removeprefix("torch.")}
    if isinstance(value, torch.device):
        return {"device": str(value)}
    if isinstance(value, torch.layout):
        return {"layout": str(value).removeprefix("torch.")}
    if isinstance(value, torch.memory_format):
        return {"memory_format": str(value).removeprefix("torch.")}
    if isinstance(value, torch.SymInt):
        return int(value)
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {"float": str(value)}
    raise ValueError(f"Unsupported native graph constant: {type(value).__name__}")


def decode_native(value, nodes, tensors, device=None):
    if not isinstance(value, dict):
        return value
    if len(value) != 1:
        raise ValueError("Invalid native graph value")
    kind, data = next(iter(value.items()))
    if kind == "node":
        return nodes[data]
    if kind == "tensor":
        return tensors[data]
    if kind in {"dtype", "layout", "memory_format"}:
        return getattr(torch, data)
    if kind == "device":
        return torch.device(device or data)
    if kind == "float":
        return float(data)
    values = [decode_native(v, nodes, tensors, device) for v in data] if kind != "dict" else None
    if kind == "tuple":
        return tuple(values)
    if kind == "list":
        return values
    if kind == "slice":
        return slice(*values)
    if kind == "dict":
        return {
            decode_native(k, nodes, tensors, device): decode_native(v, nodes, tensors, device)
            for k, v in data
        }
    raise ValueError(f"Unknown native graph value: {kind}")


def graph_spec(module):
    tensors, nodes, attributes = {}, [], {}
    for node in module.graph.nodes:
        if node.op == "call_module":
            raise ValueError(
                "Opaque submodules cannot be exported; full workload remains available"
            )
        target = native_target(node.target) if node.op == "call_function" else node.target
        if node.op == "get_attr":
            value = module
            for part in node.target.split("."):
                value = getattr(value, part)
            attributes[node.target] = encode_native(value, tensors)
        nodes.append(
            {
                "name": node.name,
                "op": node.op,
                "target": target,
                "args": encode_native(node.args, tensors),
                "kwargs": encode_native(node.kwargs, tensors),
                "tensor_meta": tensor_metadata(node),
            }
        )
    return {
        "schema_version": 1,
        "kind": "native-graph",
        "nodes": nodes,
        "attributes": attributes,
    }, tensors


def tensor_metadata(node):
    if "faultline_tensor" in node.meta:
        return node.meta["faultline_tensor"]
    value = node.meta.get("example_value", node.meta.get("val"))
    if isinstance(value, torch.Tensor) and all(type(n) is int for n in value.shape):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    return None


def build_native_graph(spec, tensors, device=None):
    if spec.get("schema_version") != 1 or spec.get("kind") != "native-graph":
        raise ValueError("Unsupported native graph schema")
    graph, root, nodes = torch.fx.Graph(), torch.nn.Module(), {}
    for path, encoded in spec["attributes"].items():
        obj = root
        parts = path.split(".")
        for part in parts[:-1]:
            if not hasattr(obj, part):
                obj.add_module(part, torch.nn.Module())
            obj = getattr(obj, part)
        value = decode_native(encoded, {}, tensors, device)
        if isinstance(value, torch.Tensor):
            obj.register_buffer(parts[-1], value)
        else:
            setattr(obj, parts[-1], value)
    for node in spec["nodes"]:
        op = node["op"]
        if op not in {"placeholder", "get_attr", "call_function", "call_method", "output"}:
            raise ValueError(f"Unsupported native node: {op}")
        target = resolve_native_target(node["target"]) if op == "call_function" else node["target"]
        args = decode_native(node["args"], nodes, tensors, device)
        kwargs = decode_native(node["kwargs"], nodes, tensors, device)
        nodes[node["name"]] = graph.create_node(op, target, args, kwargs, name=node["name"])
        nodes[node["name"]].meta["faultline_tensor"] = node.get("tensor_meta")
    graph.lint()
    return torch.fx.GraphModule(root, graph)


def mutate_native(module, mutant):
    module = copy.deepcopy(module)
    for node in module.graph.nodes:
        if mutant == "relu-as-neg" and node.op == "call_function":
            name = str(node.target)
            if node.target in {torch.relu, torch.nn.functional.relu} or "aten.relu" in name:
                node.target, node.kwargs = torch.neg, {}
        elif mutant == "sum-as-mean" and node.op == "call_method" and node.target == "sum":
            node.target = "mean"
    module.recompile()
    return module


def save_native_graph(directory, module, requests):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    spec, tensors = graph_spec(module)
    # torch.save preserves shared storage, offsets, strides, and repeated references.
    torch.save({"constants": tensors, "requests": requests}, directory / "tensors.pt")
    (directory / "graph.json").write_text(
        json.dumps(spec, indent=2, allow_nan=False), encoding="utf-8"
    )
    (directory / "forward.py").write_text(module.code, encoding="utf-8")
    return spec


def load_native_graph(directory, device="cpu"):
    directory = Path(directory)
    spec = json.loads((directory / "graph.json").read_text(encoding="utf-8"))
    payload = torch.load(directory / "tensors.pt", map_location=device, weights_only=True)
    return build_native_graph(spec, payload["constants"], device), payload["requests"], spec
