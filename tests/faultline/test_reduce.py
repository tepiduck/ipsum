import pytest

from faultline.evaluate import Settings, evaluate
from faultline.generate import reduction_fixture
from faultline.lowering import freeze
from faultline.reduce import minimize, same_failure


def test_reduces_graph_shapes_and_layout_while_preserving_real_fault():
    case = freeze(reduction_fixture())
    settings = Settings(backend="mutant:sum-as-mean", repeats=2)
    calls = []

    def oracle(candidate):
        candidate.validate()
        calls.append(candidate.digest())
        return evaluate(candidate, settings)

    reduced, evidence = minimize(case, oracle, max_evaluations=64)
    assert len(reduced.nodes) == 1
    assert reduced.nodes[0].op == "sum"
    assert reduced.complexity() < case.complexity()
    assert reduced.inputs[0].layout == "contiguous"
    assert reduced.complexity()[1] == 2
    assert evidence["evaluations"] == len(calls)
    assert calls[-1] == reduced.digest()
    assert same_failure(evidence["baseline"], evidence["confirmation"])
    accepted = [event["complexity"] for event in evidence["history"] if event["accepted"]]
    assert all(b < a for a, b in zip([case.complexity(), *accepted], accepted))


@pytest.mark.parametrize(
    "replacement", ["reference_error", "compile_error", "unstable_candidate", "pass"]
)
def test_unrelated_failure_cannot_replace_numerical_discrepancy(replacement):
    case = freeze(reduction_fixture())
    baseline = {"status": "numerical_mismatch", "fingerprint": "numeric", "injected_fault": True}

    def oracle(candidate):
        if candidate.digest() == case.digest():
            return baseline
        return {"status": replacement, "fingerprint": "numeric", "injected_fault": True}

    reduced, _ = minimize(case, oracle)
    assert reduced.digest() == case.digest()


def test_final_confirmation_is_not_cached_and_budget_is_honored():
    case = freeze(reduction_fixture())
    calls = []

    def oracle(candidate):
        calls.append(candidate)
        return {"status": "compile_error", "fingerprint": "same", "injected_fault": True}

    _, result = minimize(case, oracle, max_evaluations=3)
    assert len(calls) == 3
    assert result["budget_exhausted"]
    assert calls[-1] == calls[-2]


def test_failure_disappearing_at_final_confirmation_is_rejected():
    case = freeze(reduction_fixture())
    calls = 0

    def oracle(candidate):
        nonlocal calls
        calls += 1
        status = "compile_error" if calls < 3 else "pass"
        return {"status": status, "fingerprint": status, "injected_fault": True}

    with pytest.raises(RuntimeError, match="independent final confirmation"):
        minimize(case, oracle, max_evaluations=3)
