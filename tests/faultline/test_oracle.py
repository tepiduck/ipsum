import json

import pytest
import torch

from faultline.evaluate import Settings, confirm_exception, evaluate
from faultline.generate import corpus
from faultline.lowering import freeze
from faultline.oracle import compare, tolerances


def test_tolerances_and_nan_semantics():
    a = (torch.tensor([1.0, 0.0]),)
    assert compare(a, (a[0] + 1e-6,), 1e-4, 1e-5)["matches"]
    assert not compare(a, (torch.tensor([1.0, float("nan")]),), 1e-4, 1e-5)["matches"]
    nan = (torch.tensor([float("nan")]),)
    assert compare(nan, nan, 0, 0)["matches"]  # repeat stability only; reference health is separate
    assert not compare((torch.tensor([float("inf")]),), (torch.tensor([float("-inf")]),), 0, 0)[
        "matches"
    ]
    with pytest.raises(ValueError):
        tolerances("float32", -1, 0)


def test_structural_mismatch_is_not_broadcast_compared():
    result = compare((torch.ones(2, 3),), (torch.ones(1, 3),), 1e-4, 1e-5)
    assert not result["matches"]
    assert result["kind"] == "structure"


def test_extreme_finite_differences_serialize_without_nan():
    result = compare(
        (torch.tensor([1e308], dtype=torch.float64),),
        (torch.tensor([-1e308], dtype=torch.float64),),
        0,
        0,
    )
    assert not result["matches"]
    json.dumps(result, allow_nan=False)


def test_fp64_advisory_overflow_does_not_destroy_the_finding():
    from faultline.ir import Case, Input, Node

    case = Case(
        (Input("x", (1,), dtype="float64", values=(1e308,)),),
        (Node("a", "relu", ("x",), {}),),
        ("a",),
    )
    result = evaluate(case, Settings(backend="mutant:relu-as-neg", repeats=2))
    assert result["status"] == "numerical_mismatch"
    assert result["fp64_advisory"]["candidate_max_abs_error"] == [None]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "backend,index,status",
    [
        ("eager", 0, "pass"),
        ("mutant:sum-as-mean", 0, "numerical_mismatch"),
        ("mutant:relu-as-neg", 0, "numerical_mismatch"),
        ("mutant:softmax-axis", 1, "numerical_mismatch"),
        ("mutant:nan", 0, "numerical_mismatch"),
        ("mutant:flaky", 0, "unstable_candidate"),
        ("mutant:crash", 0, "compile_error"),
    ],
)
def test_controlled_faults_are_detected_and_labeled(backend, index, status):
    result = evaluate(freeze(corpus()[index]), Settings(backend=backend))
    assert result["status"] == status
    assert result["injected_fault"] == backend.startswith("mutant:")
    if backend == "mutant:crash":
        assert result["exception_confirmations"] == 3
    json.dumps(result, allow_nan=False)


def test_exception_that_disappears_is_not_reducible():
    result = confirm_exception(RuntimeError("first"), "compile_error", lambda: None, 3)
    assert result["status"] == "unstable_candidate"


def test_reference_nonfinite_is_inconclusive():
    from dataclasses import replace

    from faultline.ir import Case, Input, Node

    case = Case((Input("x", (1,), values=(1e30,)),), (Node("a", "mul", ("x", "x"), {}),), ("a",))
    assert evaluate(case, Settings(backend="eager"))["status"] == "reference_nonfinite"
    with pytest.raises(ValueError):
        replace(Settings(), repeats=1).validate()


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64"])
def test_corpus_eager_control_for_each_advertised_dtype(dtype):
    for case in corpus(dtype):
        result = evaluate(freeze(case), Settings(backend="eager", repeats=2))
        assert result["status"] == "pass", result
