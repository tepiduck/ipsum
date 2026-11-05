"""Lower the validated IR to inspectable FX using public graph APIs."""

from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import fx


def inputs_for(case, device="cpu", dtype=None):
    result = []
    for index, spec in enumerate(case.inputs):
        target_dtype = getattr(torch, dtype or spec.dtype)
        if spec.values is None:
            generator = torch.Generator(device="cpu").manual_seed(case.seed + index * 104729)
            base_shape = spec.shape[:-1] + (1,) if spec.layout == "expanded" else spec.shape
            logical = torch.randn(base_shape, generator=generator, dtype=torch.float64)
            logical = logical.expand(spec.shape) if spec.layout == "expanded" else logical
        else:
            logical = torch.tensor(spec.values, dtype=torch.float64).reshape(spec.shape)
        logical = logical.to(device=device, dtype=target_dtype)
        if spec.layout == "transposed":
            value = logical.transpose(-1, -2).contiguous().transpose(-1, -2)
        elif spec.layout == "strided":
            backing = torch.empty(
                (*spec.shape[:-1], spec.shape[-1] * 2), dtype=target_dtype, device=device
            )
            value = backing[..., ::2]
            value.copy_(logical)
        elif spec.layout == "expanded":
            value = logical[..., :1].contiguous().expand(spec.shape)
        else:
            value = logical.contiguous()
        result.append(value)
    return tuple(result)


def freeze(case):
    from .trace import Trace

    if isinstance(case, Trace):
        result = replace(case, steps=tuple(freeze(step) for step in case.steps))
        result.validate()
        return result
    case.validate()
    values = inputs_for(case)
    return replace(
        case,
        inputs=tuple(
            replace(spec, values=tuple(tensor.flatten().tolist()))
            for spec, tensor in zip(case.inputs, values, strict=True)
        ),
    )


def build(case, mutant=None):
    shapes = case.validate()
    graph = fx.Graph()
    refs = {i.name: graph.placeholder(i.name) for i in case.inputs}
    for node in case.nodes:
        args = tuple(refs[a] for a in node.args)
        op, params = node.op, dict(node.params)
        # Controlled faults change lowering, not the ground-truth graph.
        if mutant == "sum-as-mean" and op == "sum":
            op = "mean"
        elif mutant == "relu-as-neg" and op == "relu":
            op = "neg"
        elif mutant == "softmax-axis" and op == "softmax":
            params["dim"] = 0
        if op in {"contiguous", "clone"}:
            value = graph.call_method(op, args)
        elif op == "layer_norm":
            value = graph.call_function(F.layer_norm, (*args, (shapes[node.args[0]][-1],)), params)
        elif op == "reshape":
            value = graph.call_function(torch.reshape, (*args, tuple(params["shape"])))
        elif op == "permute":
            value = graph.call_method("permute", (args[0], tuple(params["dims"])))
        elif op == "attention":
            value = graph.call_function(F.scaled_dot_product_attention, args, params)
        elif op in {"gelu", "conv2d", "adaptive_avg_pool2d"}:
            value = graph.call_function(getattr(F, op), args, params)
        else:
            value = graph.call_function(getattr(torch, op), args, params)
        value.name = node.name
        refs[node.name] = value
    graph.output(tuple(refs[o] for o in case.outputs))
    graph.lint()
    return fx.GraphModule(torch.nn.Module(), graph)
