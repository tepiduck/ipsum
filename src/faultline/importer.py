"""Convert trusted torch.export programs through an explicit ATen operator registry."""

from dataclasses import replace

import torch

from .ir import Case, Input, Node
from .lowering import build, inputs_for


def capture_export(model, examples, *, tag="captured-model", seed=0):
    if any(module.training for module in model.modules()):
        raise ValueError(
            "capture requires model.eval(); stochastic training/state mutation is unsupported"
        )
    if not examples or any(not isinstance(x, torch.Tensor) for x in examples):
        raise ValueError("capture requires flat positional example tensors")
    # Model execution is trusted local code. The returned IR contains data only.
    exported = torch.export.export(model, tuple(examples), strict=True)
    return import_exported(exported, examples, tag=tag, seed=seed)


def import_exported(exported, examples, *, tag="exported-model", seed=0):
    specs = exported.graph_signature.input_specs
    if any(s.kind.name != "USER_OUTPUT" for s in exported.graph_signature.output_specs):
        raise ValueError("exported buffer/input mutations and non-user outputs are unsupported")
    inputs, nodes, refs = [], [], {}
    position = 0
    dtype = str(examples[0].dtype).removeprefix("torch.")

    def tensor(value, name, role="constant", requires_grad=False):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=getattr(torch, dtype))
        logical = value.detach().cpu()
        spec = Input(
            name,
            tuple(logical.shape),
            str(logical.dtype).removeprefix("torch."),
            values=tuple(logical.flatten().tolist()),
            role=role,
            requires_grad=requires_grad,
        )
        for layout in ("contiguous", "transposed", "strided", "expanded"):
            candidate = replace(spec, layout=layout)
            probe = Case((candidate,), (), (name,), schema_version=2)
            try:
                probe.validate()
                rebuilt = inputs_for(probe)[0]
            except ValueError:
                continue
            if rebuilt.stride() == logical.stride():
                inputs.append(candidate)
                return name
        raise ValueError(f"Unsupported tensor dtype, shape, values or stride for {name}")

    def ref(value):
        if isinstance(value, torch.fx.Node):
            return refs[value]
        return tensor(value, f"constant{len(inputs)}")

    def emit(name, op, args, params=None):
        nodes.append(Node(name, op, tuple(args), params or {}))
        return name

    for node in exported.graph.nodes:
        if node.op == "placeholder":
            spec = specs[position]
            kind = spec.kind.name
            if kind == "USER_INPUT":
                user_index = sum(s.kind.name == "USER_INPUT" for s in specs[:position])
                value, role = examples[user_index], "input"
            elif kind in {"PARAMETER", "BUFFER"}:
                value = exported.state_dict[spec.target]
                role = kind.lower()
            elif kind == "CONSTANT_TENSOR":
                value, role = exported.constants[spec.target], "constant"
            else:
                raise ValueError(f"Unsupported export input kind: {kind}")
            refs[node] = tensor(value, node.name, role, bool(value.requires_grad))
            position += 1
        elif node.op == "call_function":
            target = str(node.target)
            args, kw = list(node.args), dict(node.kwargs)

            def arg(index, key, default=None, args=args, kw=kw):
                return args[index] if len(args) > index else kw.pop(key, default)

            unary = {
                f"aten.{op}.default": op
                for op in ("relu", "tanh", "sigmoid", "sin", "neg", "clone")
            }
            binary = {f"aten.{op}.Tensor": op for op in ("add", "sub", "mul")}
            binary.update({f"aten.{op}.default": "matmul" for op in ("matmul", "mm", "bmm")})
            if target in unary:
                if len(args) != 1 or kw:
                    raise ValueError(f"Unsupported arguments for {target}")
                value = emit(node.name, unary[target], (ref(args[0]),))
            elif target in binary:
                alpha = arg(2, "alpha", 1) if target.startswith(("aten.add.", "aten.sub.")) else 1
                if alpha != 1 or kw or len(args) > (3 if "Tensor" in target else 2):
                    raise ValueError(f"Unsupported binary arguments for {target}")
                value = emit(node.name, binary[target], (ref(args[0]), ref(args[1])))
            elif target == "aten.linear.default":
                transposed = emit(
                    node.name + "_weight_t", "transpose", (ref(args[1]),), {"dim0": -1, "dim1": -2}
                )
                bias = arg(2, "bias")
                value = emit(
                    node.name + "_mm" if bias is not None else node.name,
                    "matmul",
                    (ref(args[0]), transposed),
                )
                if bias is not None:
                    value = emit(node.name, "add", (value, ref(bias)))
            elif target == "aten.gelu.default":
                value = emit(
                    node.name,
                    "gelu",
                    (ref(args[0]),),
                    {"approximate": arg(1, "approximate", "none")},
                )
            elif target == "aten.conv2d.default":
                bias = arg(2, "bias")
                operands = [ref(args[0]), ref(args[1])] + ([ref(bias)] if bias is not None else [])
                value = emit(
                    node.name,
                    "conv2d",
                    operands,
                    {
                        "stride": list(arg(3, "stride", [1, 1])),
                        "padding": list(arg(4, "padding", [0, 0])),
                        "dilation": list(arg(5, "dilation", [1, 1])),
                        "groups": arg(6, "groups", 1),
                    },
                )
            elif target == "aten.adaptive_avg_pool2d.default":
                value = emit(
                    node.name,
                    "adaptive_avg_pool2d",
                    (ref(args[0]),),
                    {"output_size": list(args[1])},
                )
            elif target == "aten.scaled_dot_product_attention.default":
                mask = arg(3, "attn_mask")
                if arg(4, "dropout_p", 0.0) != 0 or arg(7, "enable_gqa", False):
                    raise ValueError(
                        "attention dropout and grouped-query attention are unsupported"
                    )
                operands = [ref(x) for x in args[:3]] + ([ref(mask)] if mask is not None else [])
                value = emit(
                    node.name,
                    "attention",
                    operands,
                    {
                        "is_causal": arg(5, "is_causal", False),
                        "scale": arg(6, "scale"),
                    },
                )
            elif target == "aten.layer_norm.default":
                x = ref(args[0])
                normalized = tuple(args[1])
                shape = Case(tuple(inputs), tuple(nodes), (x,), schema_version=2).validate()[x]
                if normalized != shape[-1:]:
                    raise ValueError("only last-dimension normalized shape is supported")
                weight, bias = arg(2, "weight"), arg(3, "bias")
                eps = arg(4, "eps", 1e-5)
                arg(5, "cudnn_enable", True)
                value = emit(node.name, "layer_norm", (x,), {"eps": eps})
                if weight is not None:
                    value = emit(node.name + "_weight", "mul", (value, ref(weight)))
                if bias is not None:
                    value = emit(node.name + "_bias", "add", (value, ref(bias)))
            elif target in {
                "aten.view.default",
                "aten.reshape.default",
                "aten._unsafe_view.default",
            }:
                shape = [int(d) for d in node.meta["val"].shape]
                value = emit(node.name, "reshape", (ref(args[0]),), {"shape": shape})
            elif target in {"aten.transpose.int", "aten.permute.default", "aten.t.default"}:
                if target == "aten.permute.default":
                    value = emit(node.name, "permute", (ref(args[0]),), {"dims": list(args[1])})
                else:
                    a, b = (args[1], args[2]) if target == "aten.transpose.int" else (0, 1)
                    value = emit(node.name, "transpose", (ref(args[0]),), {"dim0": a, "dim1": b})
            elif target in {"aten.sum.dim_IntList", "aten.mean.dim"}:
                dims = arg(1, "dim")
                keep = arg(2, "keepdim", False)
                if not isinstance(dims, (list, tuple)) or len(dims) != 1:
                    raise ValueError("capture supports one reduction dimension at a time")
                value = emit(
                    node.name,
                    target.split(".")[1],
                    (ref(args[0]),),
                    {"dim": dims[0], "keepdim": keep},
                )
            elif target in {"aten.softmax.int", "aten._softmax.default"}:
                if target == "aten._softmax.default" and args[2]:
                    raise ValueError("softmax dtype promotion is unsupported")
                value = emit(node.name, "softmax", (ref(args[0]),), {"dim": args[1]})
            else:
                raise ValueError(f"Unsupported exported operator: {target} at {node.name}")
            if kw:
                raise ValueError(f"Unsupported keyword arguments for {target}: {sorted(kw)}")
            refs[node] = value
        elif node.op == "output":
            outputs = tuple(refs[v] for v in node.args[0])
        else:
            raise ValueError(f"Unsupported exported graph node: {node.op}")
    case = Case(tuple(inputs), tuple(nodes), outputs, seed=seed, tag=tag, schema_version=2).prune()
    case.validate()
    # Independently validate the translation, before treating it as compiler evidence.
    with torch.no_grad():
        expected = exported.module()(*examples)
        expected = (expected,) if isinstance(expected, torch.Tensor) else tuple(expected)
        if any(not isinstance(value, torch.Tensor) for value in expected):
            raise ValueError("capture requires a tensor or flat sequence of tensor outputs")
        actual = build(case)(*inputs_for(case, str(examples[0].device)))
        try:
            torch.testing.assert_close(actual, expected)
        except AssertionError as exc:
            raise ValueError(f"Export translation did not preserve model outputs: {exc}") from exc
    return case
