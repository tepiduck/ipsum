"""Capture a supported, trusted local PyTorch model into the portable tensor IR.

Unsupported operations fail explicitly. There is no silent approximation or
arbitrary Python embedded in the serialized case. Capture is inference-only.
"""

import operator

import torch
import torch.nn.functional as F
from torch import fx, nn

from .ir import Case, Input, Node
from .lowering import inputs_for


def capture(model, examples, *, tag="captured-model", seed=0, frontend="fx"):
    if frontend == "export":
        from .importer import capture_export

        return capture_export(model, examples, tag=tag, seed=seed)
    if frontend != "fx":
        raise ValueError("capture frontend must be fx or export")
    if model.training:
        raise ValueError("capture requires model.eval(); training graphs are not supported")
    if not examples or any(not isinstance(value, torch.Tensor) for value in examples):
        raise ValueError("capture requires one or more example tensors")
    if isinstance(
        model, (nn.Linear, nn.LayerNorm, nn.ReLU, nn.Tanh, nn.Sigmoid, nn.Softmax, nn.Identity)
    ):
        model = nn.Sequential(model).eval()
    traced = model if isinstance(model, fx.GraphModule) else fx.symbolic_trace(model)
    inputs, nodes, refs = [], [], {}
    dtype = str(examples[0].dtype).removeprefix("torch.") if examples else None
    counter = 0

    def fresh():
        nonlocal counter
        counter += 1
        return f"value{counter}"

    def tensor(value):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=getattr(torch, dtype))
        value = value.detach().cpu()
        name = fresh()
        spec = Input(
            name,
            tuple(value.shape),
            str(value.dtype).removeprefix("torch."),
            values=tuple(value.flatten().tolist()),
        )
        for layout in ("contiguous", "transposed", "strided", "expanded"):
            from dataclasses import replace

            candidate = replace(spec, layout=layout)
            test = Case((candidate,), (), (name,))
            try:
                test.validate()
                rebuilt = inputs_for(test)[0]
            except ValueError:
                continue
            if rebuilt.stride() == value.stride():
                inputs.append(candidate)
                return name
        raise ValueError(f"Unsupported input stride {value.stride()}; no implicit contiguous copy")

    def ref(value):
        return refs[value] if isinstance(value, fx.Node) else tensor(value)

    def emit(op, args, params=None):
        name = fresh()
        nodes.append(Node(name, op, tuple(args), params or {}))
        return name

    def linear(x, weight, bias):
        transposed = emit("transpose", (weight,), {"dim0": -1, "dim1": -2})
        result = emit("matmul", (x, transposed))
        return emit("add", (result, bias)) if bias is not None else result

    unary = {
        torch.relu: "relu",
        F.relu: "relu",
        torch.tanh: "tanh",
        torch.sigmoid: "sigmoid",
        torch.sin: "sin",
        torch.neg: "neg",
        operator.neg: "neg",
        torch.clone: "clone",
    }
    binary = {
        torch.add: "add",
        operator.add: "add",
        torch.sub: "sub",
        operator.sub: "sub",
        torch.mul: "mul",
        operator.mul: "mul",
        torch.matmul: "matmul",
        operator.matmul: "matmul",
    }
    position = 0
    output = None
    for node in traced.graph.nodes:
        if node.op == "placeholder":
            if position >= len(examples):
                raise ValueError("example input count does not match graph")
            refs[node] = tensor(examples[position])
            position += 1
        elif node.op == "get_attr":
            value = traced
            for attr in node.target.split("."):
                value = getattr(value, attr)
            refs[node] = tensor(value)
        elif node.op == "call_module":
            module = traced.get_submodule(node.target)
            if len(node.args) != 1 or node.kwargs:
                raise ValueError("module call must have one positional tensor")
            x = ref(node.args[0])
            if isinstance(module, nn.Linear):
                refs[node] = linear(
                    x,
                    tensor(module.weight),
                    tensor(module.bias) if module.bias is not None else None,
                )
            elif isinstance(module, nn.LayerNorm):
                if len(module.normalized_shape) != 1:
                    raise ValueError("only last-dimension LayerNorm is supported")
                shape = Case(tuple(inputs), tuple(nodes), (x,)).validate()[x]
                if not shape or shape[-1] != module.normalized_shape[0]:
                    raise ValueError("LayerNorm normalized shape does not match its input")
                result = emit("layer_norm", (x,), {"eps": module.eps})
                if module.weight is not None:
                    result = emit("mul", (result, tensor(module.weight)))
                if module.bias is not None:
                    result = emit("add", (result, tensor(module.bias)))
                refs[node] = result
            elif isinstance(module, (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.Softmax)):
                if getattr(module, "inplace", False):
                    raise ValueError("in-place mutation is not supported")
                params = {"dim": module.dim} if isinstance(module, nn.Softmax) else {}
                refs[node] = emit(type(module).__name__.lower(), (x,), params)
            elif isinstance(module, nn.Identity):
                refs[node] = x
            else:
                raise ValueError(f"Unsupported module: {type(module).__name__}")
        elif node.op in {"call_function", "call_method"}:
            target, args, kwargs = node.target, node.args, dict(node.kwargs)
            if target in unary:
                if len(args) != 1 or kwargs:
                    raise ValueError("unsupported unary arguments")
                refs[node] = emit(unary[target], (ref(args[0]),))
            elif target in binary:
                if len(args) != 2 or kwargs:
                    raise ValueError("unsupported binary arguments")
                refs[node] = emit(binary[target], (ref(args[0]), ref(args[1])))
            elif target is F.linear:
                if kwargs or len(args) not in {2, 3}:
                    raise ValueError("unsupported linear arguments")
                refs[node] = linear(
                    ref(args[0]),
                    ref(args[1]),
                    ref(args[2]) if len(args) == 3 and args[2] is not None else None,
                )
            else:
                names = {
                    torch.sum: "sum",
                    torch.mean: "mean",
                    torch.softmax: "softmax",
                    torch.transpose: "transpose",
                    torch.reshape: "reshape",
                }
                op = names.get(target, target if isinstance(target, str) else None)
                if op in {"relu", "tanh", "sigmoid", "sin", "neg", "clone", "contiguous"}:
                    if len(args) != 1 or kwargs:
                        raise ValueError("unsupported method arguments")
                    params = {}
                elif op in {"sum", "mean", "softmax"}:
                    dim = args[1] if len(args) > 1 else kwargs.pop("dim", None)
                    params = {"dim": dim}
                    if op != "softmax":
                        params["keepdim"] = (
                            args[2] if len(args) > 2 else kwargs.pop("keepdim", False)
                        )
                    if kwargs or len(args) > (2 if op == "softmax" else 3):
                        raise ValueError("unsupported reduction arguments")
                elif op == "transpose":
                    if len(args) != 3 or kwargs:
                        raise ValueError("transpose requires two positional dimensions")
                    params = {"dim0": args[1], "dim1": args[2]}
                elif op == "reshape":
                    if kwargs or len(args) < 2:
                        raise ValueError("reshape requires a concrete shape")
                    shape = args[1] if isinstance(args[1], (list, tuple)) else args[1:]
                    params = {"shape": list(shape)}
                else:
                    raise ValueError(f"Unsupported operation: {target}")
                refs[node] = emit(op, (ref(args[0]),), params)
        elif node.op == "output":
            values = node.args[0]
            values = (values,) if isinstance(values, fx.Node) else values
            if not isinstance(values, (list, tuple)) or any(
                not isinstance(v, fx.Node) for v in values
            ):
                raise ValueError("outputs must be a tensor or flat tuple/list of tensors")
            output = tuple(refs[v] for v in values)
        else:
            raise ValueError(f"Unsupported FX node: {node.op}")
    if position != len(examples) or output is None:
        raise ValueError("input count mismatch or missing output")
    case = Case(tuple(inputs), tuple(nodes), output, seed=seed, tag=tag).prune()
    case.validate()
    return case
