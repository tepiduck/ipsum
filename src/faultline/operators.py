"""Explicit shape and parameter contracts for model-facing operators."""

import math


def pair(value, name, minimum=1):
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(type(v) is not int or v < minimum or v > 1024 for v in value)
    ):
        raise ValueError(f"{name} must contain two bounded integers >= {minimum}")
    return tuple(value)


def infer_extended(node, shapes):
    from .ir import MAX_ELEMENTS, broadcast, numel

    p, op = node.params, node.op
    allowed = {
        "gelu": {"approximate"},
        "conv2d": {"stride", "padding", "dilation", "groups"},
        "adaptive_avg_pool2d": {"output_size"},
        "attention": {"is_causal", "scale"},
        "permute": {"dims"},
    }
    arities = {"conv2d": {2, 3}, "attention": {3, 4}}
    if set(p) != allowed[op] or len(node.args) not in arities.get(op, {1}):
        raise ValueError(f"invalid parameters or arity for {op}")
    shape = shapes[node.args[0]]
    if op == "gelu":
        if p["approximate"] not in {"none", "tanh"}:
            raise ValueError("invalid GELU approximation")
        return shape
    if op == "permute":
        dims = p["dims"]
        if not isinstance(dims, (tuple, list)) or sorted(dims) != list(range(len(shape))):
            raise ValueError("permute must contain every axis exactly once")
        if any(type(v) is not int for v in dims):
            raise ValueError("permutation axes must be integers")
        return tuple(shape[i] for i in dims)
    if op == "adaptive_avg_pool2d":
        if len(shape) != 4:
            raise ValueError("pooling requires NCHW rank 4")
        return shape[:2] + pair(p["output_size"], "output_size")
    if op == "conv2d":
        weight = shapes[node.args[1]]
        groups = p["groups"]
        if len(shape) != 4 or len(weight) != 4 or type(groups) is not int or groups < 1:
            raise ValueError("conv2d requires NCHW/OIHW and positive integer groups")
        if shape[1] != weight[1] * groups or weight[0] % groups:
            raise ValueError("convolution channel/group mismatch")
        if len(node.args) == 3 and shapes[node.args[2]] != (weight[0],):
            raise ValueError("convolution bias shape mismatch")
        stride = pair(p["stride"], "stride")
        padding = pair(p["padding"], "padding", 0)
        dilation = pair(p["dilation"], "dilation")
        spatial = tuple(
            (size + 2 * pad - dil * (kernel - 1) - 1) // step + 1
            for size, kernel, pad, dil, step in zip(
                shape[2:], weight[2:], padding, dilation, stride, strict=True
            )
        )
        return (shape[0], weight[0], *spatial)
    query, key, value = (shapes[a] for a in node.args[:3])
    if any(len(s) != 4 for s in (query, key, value)):
        raise ValueError("attention requires rank-4 batch/head/sequence/feature tensors")
    if query[-1] != key[-1] or key[-2] != value[-2]:
        raise ValueError("attention key/query/value dimension mismatch")
    batch = broadcast(broadcast(query[:2], key[:2]), value[:2])
    if numel((*batch, query[-2], key[-2])) > MAX_ELEMENTS:
        raise ValueError("attention scores exceed tensor budget")
    if type(p["is_causal"]) is not bool:
        raise ValueError("attention is_causal must be bool")
    scale = p["scale"]
    if scale is not None and (type(scale) not in (int, float) or not math.isfinite(scale)):
        raise ValueError("attention scale must be finite")
    if len(node.args) == 4:
        if p["is_causal"]:
            raise ValueError("combined causal and explicit masks are not supported")
        scores = (*batch, query[-2], key[-2])
        if broadcast(shapes[node.args[3]], scores) != scores:
            raise ValueError("attention mask must broadcast to attention scores")
    return (*batch, query[-2], value[-1])
