import json
import subprocess
import sys
from dataclasses import replace

import pytest
import torch

from faultline.evaluate import Settings, evaluate
from faultline.export import export_case
from faultline.finding import load_finding, replay_finding
from faultline.ir import Case, Input, Node
from faultline.lowering import freeze
from faultline.portable import check_workload, fingerprint, same_failure
from faultline.reduce import minimize
from faultline.trace import Trace, program_from_dict


def request(n):
    return freeze(
        Case(
            (Input("x", (n, 2), values=(1.0,) * (n * 2)),), (Node("y", "sin", ("x",), {}),), ("y",)
        )
    )


def stream():
    return Trace(tuple(request(n) for n in (5, 2, 7, 3, 6, 2, 8)), tag="sequence-fixture")


def test_whole_trace_reduces_to_aba_but_isolated_requests_pass():
    trace = stream()
    settings = Settings(backend="mutant:sequence-cache", repeats=2)
    for step in trace.steps:
        assert evaluate(Trace((step,)), settings)["status"] == "pass"
    original = evaluate(trace, settings)
    assert original["failing_steps"] == [5]
    calls = []

    def oracle(case):
        calls.append(case.digest())
        return evaluate(case, settings)

    reduced, evidence = minimize(trace, oracle, 64, expected=original)
    assert [step.inputs[0].shape[0] for step in reduced.steps] == [2, 3, 2]
    assert evidence["confirmation"]["failing_steps"] == [2]
    assert same_failure(original, evidence["confirmation"])
    assert calls[-1] == reduced.digest() and calls.count(reduced.digest()) >= 2
    assert len(calls) <= 64
    assert program_from_dict(reduced.to_dict()) == reduced
    assert replace(trace, steps=tuple(reversed(trace.steps))).digest() != trace.digest()


def test_gradient_mismatch_has_matching_forward_and_keeps_target_during_reduction():
    case = request(3)
    settings = Settings(backend="mutant:gradient-scale", gradients=True, repeats=2)
    original = evaluate(case, settings)
    assert original["status"] == "gradient_mismatch"
    assert all(
        x["matches"]
        for x in original["comparison"]["outputs"]
        if not x["target"].startswith("gradient:")
    )
    reduced, evidence = minimize(case, lambda c: evaluate(c, settings), 20, expected=original)
    assert same_failure(original, evidence["confirmation"])
    assert reduced.complexity() < case.complexity()


def test_absent_gradient_is_distinct_from_a_zero_gradient():
    def reference(x, unused):
        return (x.sin(),)

    def candidate(x, unused):
        return (x.sin() + 0 * unused,)

    result, _, _ = check_workload(
        lambda: reference,
        lambda: candidate,
        [(torch.ones(2), torch.ones(2))],
        [{"name": "x"}, {"name": "unused"}],
        ["y"],
        gradients=True,
        device="cpu",
        repeats=2,
        rtol=1e-4,
        atol=1e-5,
    )
    assert result["status"] == "gradient_mismatch"
    failed = [x["target"] for x in result["comparison"]["outputs"] if not x["matches"]]
    assert failed == ["gradient:unused:present"]


def test_mixed_forward_backward_signature_retains_gradient_identity():
    outputs = [
        {"index": 0, "target": "output:y", "matches": False, "kind": "numeric"},
        {"index": 3, "target": "gradient:weight:value", "matches": False, "kind": "numeric"},
    ]
    result = {
        "status": "numerical_mismatch",
        "gradients": True,
        "backend": "inductor",
        "comparison": {"outputs": outputs},
    }
    original = fingerprint(result)
    outputs[1]["target"] = "gradient:bias:value"
    assert fingerprint(result) != original


@pytest.mark.parametrize(
    "trace,gradients,backend",
    [
        (True, False, "mutant:sequence-cache"),
        (False, True, "mutant:gradient-scale"),
        (True, True, "eager"),
    ],
)
def test_extended_standalone_matches_worker_predicate(tmp_path, trace, gradients, backend):
    case = stream() if trace else request(3)
    settings = Settings(backend=backend, gradients=gradients, repeats=2, dynamic=None)
    expected = evaluate(case, settings)
    path = export_case(case, settings, tmp_path, expected)
    result = subprocess.run(
        [sys.executable, "-I", str(path)], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == (0 if backend == "eager" else 1), result.stderr
    observed = json.loads(result.stdout.splitlines()[-1])
    for key in ("status", "fingerprint", "comparison", "failing_steps", "gradient_targets"):
        assert observed.get(key) == expected.get(key), key


def test_trace_reduction_does_not_publish_a_disappearing_failure():
    trace = stream()
    original = evaluate(trace, Settings(backend="mutant:sequence-cache", repeats=2))
    results = iter([original, {"status": "pass"}])
    with pytest.raises(RuntimeError, match="final confirmation"):
        minimize(trace, lambda case: next(results), 2)


def test_saved_gradient_trace_replays_with_original_policy(tmp_path):
    case = stream()
    settings = Settings(backend="mutant:gradient-scale", gradients=True, dynamic=True, repeats=2)
    outcome = evaluate(case, settings)
    path = export_case(case, settings, tmp_path / "original", outcome)
    finding = load_finding(path.parent)
    assert finding.case == case and finding.settings == settings
    replay = replay_finding(finding, tmp_path / "replay")
    assert replay["reproduces_expected"]
    # Raw attempts must enforce mode identity even without bundle or campaign hashes.
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "case.json").write_text(json.dumps(case.to_dict()))
    (raw / "settings.json").write_text(json.dumps(settings.to_dict()))
    (raw / "outcome.json").write_text(json.dumps({**outcome, "gradients": False}))
    with pytest.raises(ValueError, match="execution settings"):
        load_finding(raw)
