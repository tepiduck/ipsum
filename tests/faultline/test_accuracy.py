import copy
import json
import sys

import pytest
import torch

from faultline import portable
from faultline.accuracy import accuracy_comparison
from faultline.evaluate import Settings
from faultline.native_artifacts import write_native_report
from faultline.native_precision import assess_fp64
from faultline.native_runtime import compare_native, observe_native


def planted_observation(gradient_offset, dtype=torch.float32):
    class Cancellation(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones((), dtype=dtype))

        def forward(self, x):
            return (x * self.weight).sum()

    model = Cancellation().eval()
    # Explicit fixture fault, not a real compiler result. Forward outputs are unchanged.
    if gradient_offset:
        model.weight.register_hook(lambda grad: grad + gradient_offset)
    request = {
        "args": (torch.tensor([1e8, 1, -1e8], dtype=dtype),),
        "kwargs": {},
        "cotangents": {"output": torch.ones((), dtype=dtype)},
    }
    return observe_native(model, model, [request], True)


def planted_result(offset):
    eager, candidate = planted_observation(0), planted_observation(offset)
    return {
        "status": "gradient_mismatch",
        "backend": "controlled-fixture",
        "injected_fault": True,
        "comparison": compare_native(eager, candidate, portable),
        "fp64_advisory": assess_fp64(
            lambda: planted_observation(0, torch.float64),
            eager,
            candidate,
            compare_native,
            portable,
            2,
        ),
    }


def test_candidate_only_accuracy_degradation_when_both_versions_fail():
    baseline, candidate = planted_result(0.1), planted_result(4)
    before = copy.deepcopy((baseline, candidate))
    assert not baseline["comparison"]["matches"] and not candidate["comparison"]["matches"]
    result = accuracy_comparison(baseline, candidate)
    assert result["accuracy_regression"] and result["matched"] == result["total"] == 1
    assert result["targets"][0]["baseline"]["value"] == pytest.approx(0.9)
    assert result["targets"][0]["candidate"]["value"] == pytest.approx(3)
    assert (baseline, candidate) == before
    assert not accuracy_comparison(baseline, baseline)["accuracy_regression"]
    assert portable.fingerprint(baseline) == portable.fingerprint(before[0])


def outcome(errors):
    return {
        "comparison": {"outputs": [{"target": k, "matches": False} for k in errors]},
        "fp64_advisory": {
            "status": "complete",
            "outputs": [
                {
                    "target": k,
                    "reference_dtype": "torch.float64",
                    "eager": {"rmse": e},
                    "candidate": {"rmse": c},
                }
                for k, (e, c) in errors.items()
            ],
        },
    }


@pytest.mark.parametrize(
    "baseline,candidate,flag",
    [
        ((1, 1), (1, 2), False),  # Candidate must strictly exceed two.
        ((1, 1.5), (1, 3), True),
        ((1, 2), (1, 3.9), False),
        ((0, 0), (1, 3), True),
        ((0, 0), (0, 1), True),
        ((0, 1), (0, 4), False),  # Infinity/infinity establishes no ratio growth.
        ((1, 0), (1, 3), True),
        ((1, 1), (0, 0), False),
        ((1, 1), (1, float("nan")), False),
        ((1, 1), (1e-308, 1e308), False),  # Ratio overflow is missing evidence, not infinity.
    ],
)
def test_zero_denominators_and_threshold_boundaries(baseline, candidate, flag):
    result = accuracy_comparison(outcome({"gradient": baseline}), outcome({"gradient": candidate}))
    assert result["accuracy_regression"] is flag
    json.dumps(result, allow_nan=False)


def test_missing_evidence_stays_in_union_denominator_and_html(tmp_path):
    baseline = outcome({str(i): (1, 1) for i in range(6)})
    candidate = outcome({"0": (1, 3)})
    result = accuracy_comparison(baseline, candidate)
    assert (result["matched"], result["total"], result["available"], result["missing"]) == (
        1,
        6,
        1,
        5,
    )
    assert not result["accuracy_regression"] and result["status"] == "partial"
    baseline["comparison"]["outputs"].pop()
    result = accuracy_comparison(baseline, candidate)
    assert result["accuracy_regression"] and result["fraction"] == 0.2
    write_native_report(tmp_path, {"accuracy_comparison": result})
    report = (tmp_path / "report.html").read_text()
    assert "accuracy_regression: 1/5 targets" in report
    candidate["fp64_advisory"]["status"] = "reference_error"
    assert accuracy_comparison(baseline, candidate)["available"] == 0
    candidate["fp64_advisory"]["status"] = "complete"
    candidate["fp64_advisory"]["outputs"][0]["reference_dtype"] = "torch.float32"
    assert accuracy_comparison(baseline, candidate)["available"] == 0


def test_model_compare_keeps_recurring_gate_and_surfaces_accuracy_advisory(tmp_path, monkeypatch):
    from faultline import native_artifacts

    baseline, candidate = planted_result(0.1), planted_result(4)
    results = iter([baseline, candidate])
    for result in (baseline, candidate):
        result["fingerprint"] = portable.fingerprint(result)
    monkeypatch.setattr(native_artifacts, "copy_workload", lambda *_: tmp_path)
    monkeypatch.setattr(native_artifacts.NativeJob, "create", lambda *a, **kw: object())
    monkeypatch.setattr(native_artifacts, "run_case", lambda *a, **kw: next(results))
    monkeypatch.setattr(native_artifacts, "export_native", lambda *a, **kw: None)
    output = tmp_path / "comparison"
    summary = native_artifacts.compare_native_environments(
        tmp_path,
        output,
        sys.executable,
        sys.executable,
        Settings(gradients=True, numerics=True),
    )
    assert summary["classification"] == "recurring"
    assert summary["accuracy_comparison"]["accuracy_regression"]
    assert summary["exit_code"] == 0
    assert "accuracy_regression: 1/1 targets" in (output / "report.html").read_text()


def test_inconclusive_worker_retains_checkpointed_targets_in_denominator():
    baseline = outcome({"a": (1, 1)})
    candidate = {"status": "resource_limit", "primary_observation": outcome({"b": (1, 3)})}
    result = accuracy_comparison(baseline, candidate)
    assert result["total"] == result["missing"] == 2
    assert result["status"] == "unavailable" and not result["accuracy_regression"]
