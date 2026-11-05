"""Seeded, shape-aware graph generation and a targeted boundary corpus."""

import random
from dataclasses import replace

from .ir import Case, Input, Node


def corpus(dtype="float32"):
    cases = [
        Case(
            (Input("x", (7, 17), dtype, "transposed"), Input("bias", (1, 17), dtype)),
            (
                Node("add", "add", ("x", "bias"), {}),
                Node("act", "relu", ("add",), {}),
                Node("total", "sum", ("act",), {"dim": -1, "keepdim": True}),
            ),
            ("total",),
            tag="strided-broadcast-reduction",
        ),
        Case(
            (Input("x", (3, 17), dtype, "strided"),),
            (
                Node("norm", "layer_norm", ("x",), {"eps": 1e-5}),
                Node("prob", "softmax", ("norm",), {"dim": -1}),
            ),
            ("prob",),
            tag="normalization-softmax",
        ),
        Case(
            (Input("a", (2, 7, 17), dtype, "transposed"), Input("b", (1, 17, 5), dtype, "strided")),
            (Node("mm", "matmul", ("a", "b"), {}), Node("out", "tanh", ("mm",), {})),
            ("out",),
            tag="batched-matmul",
        ),
        Case(
            (Input("x", (5, 1), dtype), Input("y", (1, 17), dtype, "expanded")),
            (
                Node("prod", "mul", ("x", "y"), {}),
                Node("out", "mean", ("prod",), {"dim": 0, "keepdim": False}),
            ),
            ("out",),
            tag="zero-stride-broadcast",
        ),
        Case(
            (Input("x", (3, 5, 7), dtype),),
            (
                Node("t", "transpose", ("x",), {"dim0": 0, "dim1": 2}),
                Node("r", "reshape", ("t",), {"shape": [7, 15]}),
                Node("s", "sin", ("r",), {}),
            ),
            ("s",),
            tag="transpose-reshape",
        ),
        Case(
            (Input("x", (1, 31), dtype),),
            (
                Node("a", "sigmoid", ("x",), {}),
                Node("b", "sum", ("a",), {"dim": 1, "keepdim": False}),
                Node("c", "sum", ("b",), {"dim": 0, "keepdim": False}),
            ),
            ("a", "c"),
            tag="multiple-outputs-scalar",
        ),
    ]
    return [replace(c, seed=i) for i, c in enumerate(cases)]


def generate(seed, depth=10, dtype="float32"):
    if not 1 <= depth <= 64:
        raise ValueError("depth must be in [1, 64]")
    rng = random.Random(seed)
    dims = [1, 2, 3, 7, 15, 16, 17, 31, 32, 33]
    m, n = rng.choice(dims), rng.choice(dims)
    inputs = [
        Input("x", (m, n), dtype, rng.choice(["contiguous", "transposed", "strided"])),
        Input("bias", (1, n), dtype, rng.choice(["contiguous", "expanded"])),
    ]
    nodes = [Node("v0", "add", ("x", "bias"), {})]
    current = "v0"
    for index in range(1, depth + 1):
        case = Case(tuple(inputs), tuple(nodes), (current,), seed)
        shape = case.validate()[current]
        choices = ["relu", "tanh", "sigmoid", "sin", "neg", "contiguous", "clone", "add", "mul"]
        if shape:
            choices += ["sum", "mean", "softmax", "layer_norm"]
        if len(shape) >= 2:
            choices += ["transpose"]
            if len(inputs) < 16:
                choices += ["matmul"]
        op = rng.choice(choices)
        params, args = {}, (current,)
        if op in {"sum", "mean"}:
            params = {"dim": rng.randrange(len(shape)), "keepdim": rng.choice([True, False])}
        elif op == "softmax":
            params = {"dim": -1}
        elif op == "layer_norm":
            params = {"eps": rng.choice([1e-5, 1e-3])}
        elif op == "transpose":
            params = {"dim0": -1, "dim1": -2}
        elif op in {"add", "mul"}:
            args = (current, current)
        elif op == "matmul":
            weight = Input(f"w{index}", (shape[-1], rng.choice(dims)), dtype)
            inputs.append(weight)
            args = (current, weight.name)
        name = f"v{index}"
        nodes.append(Node(name, op, args, params))
        current = name
    case = Case(tuple(inputs), tuple(nodes), (current,), seed).prune()
    case.validate()
    return case


def reduction_fixture():
    """A noisy graph with one reduction targeted by a controlled compiler mutant."""
    return Case(
        (Input("x", (8, 16), layout="strided"), Input("bias", (1, 16))),
        (
            Node("a", "add", ("x", "bias"), {}),
            Node("b", "tanh", ("a",), {}),
            Node("c", "clone", ("b",), {}),
            Node("d", "relu", ("c",), {}),
            Node("e", "sum", ("d",), {"dim": -1, "keepdim": True}),
            Node("f", "clone", ("e",), {}),
            Node("g", "neg", ("f",), {}),
        ),
        ("g",),
        seed=101,
        tag="injected-fault-reduction-fixture",
    )
