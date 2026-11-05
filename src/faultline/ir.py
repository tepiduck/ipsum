"""A small, data-only tensor IR. Validation never executes user-provided Python.

Inputs are immutable logical values plus a reproducible physical layout. Graphs
are acyclic, single dtype, and bounded before any tensors are allocated.
"""

import hashlib
import itertools
import json
import keyword
import math
from dataclasses import asdict, dataclass, replace

MAX_ELEMENTS = 262_144
MAX_NODES = 128
DTYPES = {"float32", "float64", "float16", "bfloat16"}
LAYOUTS = {"contiguous", "transposed", "strided", "expanded"}
UNARY = {"relu", "tanh", "sigmoid", "sin", "neg", "contiguous", "clone"}
BINARY = {"add", "sub", "mul", "matmul"}
OPS = UNARY | BINARY | {"sum", "mean", "transpose", "reshape", "softmax", "layer_norm"}
EXTENDED_OPS = {"gelu", "conv2d", "adaptive_avg_pool2d", "attention", "permute"}


def numel(shape):
    return math.prod(shape)


def checked_shape(shape):
    if not isinstance(shape, (list, tuple)) or len(shape) > 4:
        raise ValueError("shape must have at most four dimensions")
    if any(type(d) is not int or d < 1 or d > 1024 for d in shape):
        raise ValueError("dimensions must be integers in [1, 1024]")
    if numel(shape) > MAX_ELEMENTS:
        raise ValueError("tensor exceeds element budget")
    return tuple(shape)


def broadcast(a, b):
    out = []
    for x, y in itertools.zip_longest(reversed(a), reversed(b), fillvalue=1):
        if x != y and x != 1 and y != 1:
            raise ValueError(f"incompatible broadcast: {a}, {b}")
        out.append(max(x, y))
    return checked_shape(tuple(reversed(out)))


def axis(value, rank):
    if type(value) is not int or not -rank <= value < rank:
        raise ValueError("axis outside tensor rank")
    return value % rank


@dataclass(frozen=True)
class Input:
    name: str
    shape: tuple[int, ...]
    dtype: str = "float32"
    layout: str = "contiguous"
    values: tuple[float, ...] | None = None
    role: str = "input"
    requires_grad: bool = False


@dataclass(frozen=True)
class Node:
    name: str
    op: str
    args: tuple[str, ...]
    params: dict


@dataclass(frozen=True)
class Case:
    inputs: tuple[Input, ...]
    nodes: tuple[Node, ...]
    outputs: tuple[str, ...]
    seed: int = 0
    tag: str = "generated"
    schema_version: int = 1

    def to_dict(self):
        data = asdict(self)
        if self.schema_version == 1:
            for spec in data["inputs"]:
                spec.pop("role")
                spec.pop("requires_grad")
        return data

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("case must be a JSON object")  # noqa: TRY004 - JSON validation boundary
        if set(data) != {"inputs", "nodes", "outputs", "seed", "tag", "schema_version"}:
            raise ValueError("unexpected or missing case fields")
        try:
            result = cls(
                inputs=tuple(
                    Input(
                        **{
                            **x,
                            "shape": tuple(x["shape"]),
                            "values": tuple(x["values"]) if x.get("values") is not None else None,
                        }
                    )
                    for x in data["inputs"]
                ),
                nodes=tuple(Node(**{**n, "args": tuple(n["args"])}) for n in data["nodes"]),
                outputs=tuple(data["outputs"]),
                seed=data["seed"],
                tag=data["tag"],
                schema_version=data["schema_version"],
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"invalid case structure: {exc}") from exc
        try:
            result.validate()
        except (TypeError, AttributeError, KeyError) as exc:
            raise ValueError(f"invalid case field types: {exc}") from exc
        return result

    def digest(self):
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    def validate(self):
        if type(self.schema_version) is not int or self.schema_version not in {1, 2}:
            raise ValueError("unsupported case schema")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be uint32")
        if not isinstance(self.tag, str) or len(self.tag) > 256:
            raise ValueError("invalid tag")
        if (
            not 1 <= len(self.inputs) <= (16 if self.schema_version == 1 else 64)
            or len(self.nodes) > MAX_NODES
        ):
            raise ValueError("input/node budget exceeded")
        shapes = {}

        def name_ok(name):
            if (
                not isinstance(name, str)
                or not name.isidentifier()
                or name.startswith("_")
                or keyword.iskeyword(name)
                or name in {"self", "torch"}
            ):
                raise ValueError("invalid node name")
            if name in shapes:
                raise ValueError("duplicate name")

        for item in self.inputs:
            name_ok(item.name)
            if (
                item.role not in {"input", "parameter", "buffer", "constant"}
                or type(item.requires_grad) is not bool
            ):
                raise ValueError("invalid input role or gradient flag")
            if self.schema_version == 1 and (item.role != "input" or item.requires_grad):
                raise ValueError("input roles and gradient flags require case schema 2")
            shape = checked_shape(item.shape)
            if item.dtype not in DTYPES or item.layout not in LAYOUTS:
                raise ValueError("unsupported dtype/layout")
            if item.layout == "transposed" and len(shape) < 2:
                raise ValueError("transposed layout requires rank >= 2")
            if item.layout in {"strided", "expanded"} and not shape:
                raise ValueError("layout requires a dimension")
            if item.values is not None:
                if len(item.values) != numel(shape) or any(
                    type(v) not in (int, float) or not math.isfinite(v) for v in item.values
                ):
                    raise ValueError("input values must be finite and match shape")
                if item.layout == "expanded":
                    width = shape[-1]
                    for pos in range(0, len(item.values), width):
                        if len(set(item.values[pos : pos + width])) != 1:
                            raise ValueError("expanded input values must respect aliases")
            shapes[item.name] = shape
        if len({x.dtype for x in self.inputs}) != 1:
            raise ValueError("mixed input dtypes are not supported")
        for node in self.nodes:
            name_ok(node.name)
            supported = OPS | EXTENDED_OPS if self.schema_version == 2 else OPS
            if node.op not in supported or not isinstance(node.params, dict):
                raise ValueError("unsupported operator")
            if node.op in EXTENDED_OPS:
                from .operators import infer_extended

                if any(a not in shapes for a in node.args):
                    raise ValueError("invalid forward reference")
                shapes[node.name] = checked_shape(infer_extended(node, shapes))
                continue
            arity = 2 if node.op in BINARY else 1
            if len(node.args) != arity or any(a not in shapes for a in node.args):
                raise ValueError("invalid arity or forward reference")
            shape = shapes[node.args[0]]
            p = node.params
            allowed = {
                "sum": {"dim", "keepdim"},
                "mean": {"dim", "keepdim"},
                "transpose": {"dim0", "dim1"},
                "reshape": {"shape"},
                "softmax": {"dim"},
                "layer_norm": {"eps"},
            }.get(node.op, set())
            if set(p) != allowed:
                raise ValueError(f"invalid parameters for {node.op}")
            if node.op in {"add", "sub", "mul"}:
                shape = broadcast(shape, shapes[node.args[1]])
            elif node.op == "matmul":
                other = shapes[node.args[1]]
                if len(shape) < 2 or len(other) < 2 or shape[-1] != other[-2]:
                    raise ValueError("matmul requires compatible rank >= 2 operands")
                shape = broadcast(shape[:-2], other[:-2]) + (shape[-2], other[-1])
            elif node.op in {"sum", "mean"}:
                dim = axis(p["dim"], len(shape))
                if type(p["keepdim"]) is not bool:
                    raise ValueError("keepdim must be bool")
                shape = shape[:dim] + ((1,) if p["keepdim"] else ()) + shape[dim + 1 :]
            elif node.op == "transpose":
                a, b = axis(p["dim0"], len(shape)), axis(p["dim1"], len(shape))
                temp = list(shape)
                temp[a], temp[b] = temp[b], temp[a]
                shape = tuple(temp)
            elif node.op == "reshape":
                target = checked_shape(p["shape"])
                if numel(target) != numel(shape):
                    raise ValueError("reshape changes element count")
                shape = target
            elif node.op == "softmax":
                axis(p["dim"], len(shape))
            elif node.op == "layer_norm" and (
                not shape or type(p["eps"]) not in (float, int) or not 0 < p["eps"] <= 1
            ):
                raise ValueError("layer_norm requires rank >= 1 and finite positive epsilon")
            shapes[node.name] = checked_shape(shape)
        if not 1 <= len(self.outputs) <= 8 or any(x not in shapes for x in self.outputs):
            raise ValueError("invalid outputs")
        if sum(numel(s) for s in shapes.values()) > 8 * MAX_ELEMENTS:
            raise ValueError("aggregate graph element budget exceeded")
        return shapes

    def prune(self):
        live = set(self.outputs)
        nodes = []
        for node in reversed(self.nodes):
            if node.name in live:
                nodes.append(node)
                live.update(node.args)
        return replace(
            self,
            nodes=tuple(reversed(nodes)),
            inputs=tuple(i for i in self.inputs if i.name in live),
        )

    def complexity(self):
        values = [v for i in self.inputs for v in (i.values or ())]
        return (
            len(self.nodes),
            sum(numel(i.shape) for i in self.inputs),
            sum(i.layout != "contiguous" for i in self.inputs),
            sum(v != 0 for v in values),
        )
