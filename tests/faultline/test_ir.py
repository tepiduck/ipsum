from dataclasses import replace

import pytest
import torch

from faultline.generate import corpus, generate
from faultline.ir import Case, Input, Node
from faultline.lowering import build, freeze, inputs_for


@pytest.mark.parametrize("seed", range(40))
def test_generated_shape_inference_matches_real_execution(seed):
    case = generate(seed, depth=12)
    shapes = case.validate()
    output = build(case)(*inputs_for(case))
    assert [tuple(t.shape) for t in output] == [shapes[o] for o in case.outputs]
    assert generate(seed, depth=12).digest() == case.digest()
    assert Case.from_dict(case.to_dict()).digest() == case.digest()


@pytest.mark.parametrize("case", corpus(), ids=lambda c: c.tag)
def test_frozen_inputs_round_trip_values_and_strides(case):
    original = inputs_for(case)
    frozen = Case.from_dict(freeze(case).to_dict())
    restored = inputs_for(frozen)
    for a, b in zip(original, restored, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert a.stride() == b.stride()


@pytest.mark.parametrize("layout", ["contiguous", "transposed", "strided", "expanded"])
def test_layouts_are_actually_materialized(layout):
    case = Case((Input("x", (3, 5), layout=layout),), (), ("x",))
    tensor = inputs_for(case)[0]
    assert tensor.is_contiguous() == (layout == "contiguous")
    if layout == "expanded":
        assert tensor.stride()[-1] == 0
    elif layout == "strided":
        assert tensor.stride()[-1] == 2


@pytest.mark.parametrize(
    "bad",
    [
        Case((Input("x", (2, 3)),), (Node("bad", "eval", ("x",), {}),), ("bad",)),
        Case((Input("x", (2, 3)),), (Node("a", "add", ("x", "future"), {}),), ("a",)),
        Case((Input("x", (2, 3)),), (Node("a", "reshape", ("x",), {"shape": [7]}),), ("a",)),
        Case(
            (Input("x", (2, 3)),), (Node("a", "sum", ("x",), {"dim": 3, "keepdim": True}),), ("a",)
        ),
        Case(
            (Input("x", (2, 3)), Input("y", (7, 5))), (Node("a", "matmul", ("x", "y"), {}),), ("a",)
        ),
        Case((Input("x", (True, 3)),), (), ("x",)),
        Case((Input("x", (1024, 1024)),), (), ("x",)),
        Case((Input("for", (2, 3)),), (), ("for",)),
        Case((Input("x", (2,), values=(float("nan"), 0.0)),), (), ("x",)),
        Case((Input("x", (2,), layout="expanded", values=(1.0, 2.0)),), (), ("x",)),
    ],
)
def test_invalid_graphs_are_rejected_before_execution(bad):
    with pytest.raises(ValueError):
        bad.validate()


def test_unknown_schema_and_fields_rejected():
    data = corpus()[0].to_dict()
    with pytest.raises(ValueError):
        Case.from_dict({**data, "arbitrary_python": "print(1)"})
    with pytest.raises(ValueError):
        replace(corpus()[0], schema_version=2).validate()
    for reserved in ("self", "torch"):
        with pytest.raises(ValueError):
            Case((Input(reserved, (2,)),), (), (reserved,)).validate()


def test_pruning_preserves_execution():
    case = corpus()[0]
    redundant = replace(case, nodes=case.nodes + (Node("unused", "sin", ("x",), {}),))
    assert redundant.prune() == case
