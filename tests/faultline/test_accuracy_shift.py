import copy
import json
import math

import pytest
import torch

from faultline import portable
from faultline.accuracy import accuracy_shift, sign_test
from faultline.native_numerics import annotate, assess_control
from faultline.native_precision import assess_backward_fp64, assess_fp64, error_stats, noise_floor
from faultline.native_runtime import compare_native


def outcome(errors):
    return {
        "status": "gradient_mismatch",
        "injected_fault": True,
        "comparison": {
            "outputs": [
                {"target": f"step:0:gradient:{i}:value", "matches": False}
                for i in range(len(errors))
            ]
        },
        "fp64_advisory": {
            "status": "complete",
            "outputs": [
                {
                    "target": f"step:0:gradient:{i}:value",
                    "reference_dtype": "torch.float64",
                    "candidate": {"rmse": value},
                }
                for i, value in enumerate(errors)
            ],
        },
    }


def test_uniform_planted_degradation_flags_but_same_version_does_not():
    baseline, candidate = outcome([1e-7] * 20), outcome([1.5e-7] * 20)
    before = copy.deepcopy((baseline, candidate))
    shifted = accuracy_shift(baseline, candidate)
    assert shifted["accuracy_shift"]
    assert shifted["median_log2_growth"] == pytest.approx(math.log2(1.5))
    assert shifted["sign_test_p"] == 2**-20
    baseline["comparison"]["outputs"].append(
        {"target": "step:0:gradient:0:present", "matches": True}
    )
    assert accuracy_shift(baseline, candidate)["total"] == 20
    baseline["comparison"]["outputs"].pop()
    same = accuracy_shift(baseline, baseline)
    assert not same["accuracy_shift"] and same["ties"] == 20 and same["sign_test_p"] == 1
    assert (baseline, candidate) == before


def test_zeros_ties_missing_coverage_and_passing_targets():
    baseline, candidate = outcome([0, 1, 0] * 4), outcome([1, 0, 0] * 4)
    for row in baseline["comparison"]["outputs"]:
        row["matches"] = True  # Distributional advisory does not select only failing tensors.
    result = accuracy_shift(baseline, candidate)
    assert result["paired"] == 12 and result["ties"] == 4
    assert result["median_log2_growth"] == 0
    json.dumps(result, allow_nan=False)
    candidate["fp64_advisory"]["outputs"] = candidate["fp64_advisory"]["outputs"][:4]
    result = accuracy_shift(baseline, candidate)
    assert result["total"] == 12 and result["coverage"] == pytest.approx(1 / 3)
    assert not result["sufficient_evidence"] and not result["accuracy_shift"]
    assert sign_test(3, 1) == 5 / 16


def observe(values):
    return ({str(i): torch.tensor([v], dtype=torch.float64) for i, v in enumerate(values)}, [])


def test_floor_uses_control_against_reference_and_retains_old_ratios():
    high, eager, candidate, control = [
        observe(x) for x in ([1], [1 + 1e-9], [1 + 5e-6], [1 + 4e-6])
    ]
    result = assess_fp64(
        lambda: high,
        eager,
        candidate,
        compare_native,
        portable,
        2,
        control=control,
        control_name="onednn_disabled",
    )
    row = result["outputs"][0]
    assert row["candidate_to_eager_rmse"] > 4000
    assert row["legacy_accuracy_label"] == "candidate_farther_from_fp64"
    assert row["noise_floor"]["value"] == pytest.approx(4e-6)
    assert row["accuracy_label"] == "comparable_to_eager_against_fp64"
    assert row["eager_control"]["rmse"] == pytest.approx(4e-6)
    floor = noise_floor({"rmse": 1e-9}, None, 1.0)
    assert floor["value"] == 32 * 2**-24 and not floor["control_available"]


def test_backward_noise_floor_separates_roundoff_from_planted_fault():
    high, eager = observe([1]), observe([1 + 1e-9])
    benign = assess_backward_fp64(
        lambda: high, eager, observe([1 + 1e-6]), compare_native, portable, 2
    )
    fault = assess_backward_fp64(lambda: high, eager, observe([1.1]), compare_native, portable, 2)
    assert benign["agrees"] is True and fault["agrees"] is False


def test_control_retains_only_fp64_values_without_copying_or_changing_layout():
    value = torch.arange(12, dtype=torch.float32)[1::3]
    observation = (
        {
            "step:0:output": value,
            "step:0:gradient:weight:value": value,
            "step:0:gradient:weight:present": torch.tensor(True),
            "state:parameter:weight": torch.ones(100),
            "step:0:input_after:0": torch.ones(100),
        },
        [],
    )
    sink = {}
    result = assess_control(
        lambda _: lambda: observation,
        observation,
        observation,
        {"device": "cpu", "repeats": 2},
        compare_native,
        portable,
        error_stats,
        observation_sink=sink,
    )
    assert result["status"] == "complete"
    retained = sink["observation"][0]
    assert set(retained) == {"step:0:output", "step:0:gradient:weight:value"}
    assert all(item is value and item.stride() == (3,) for item in retained.values())
    assert len(observation[0]) == 5  # The original observation still supports primary checks.
    full = assess_fp64(
        lambda: observation,
        observation,
        observation,
        compare_native,
        portable,
        2,
        control=observation,
    )
    compact = assess_fp64(
        lambda: observation,
        observation,
        observation,
        compare_native,
        portable,
        2,
        control=sink["observation"],
    )
    assert compact == full


def test_partial_explanations_are_missing_not_contrary():
    primary = outcome([1, 1])
    names = [r["target"] for r in primary["comparison"]["outputs"]]
    fp64 = {
        "status": "complete",
        "outputs": [
            {
                "target": names[0],
                "reference_dtype": "torch.float64",
                "accuracy_label": "comparable_to_eager_against_fp64",
                "farther_than_noise_floor": False,
            }
        ],
    }
    context = annotate(primary, fp64, {}, {})
    assert context["triage"]["classification"] == "unexplained_missing_evidence"
    assert not context["triage"]["contrary_evidence"]
    assert context["label_coverage"]["comparable_to_eager_against_fp64"]["matched"] == 1


@pytest.mark.parametrize("kind", ["fp64", "backward", "mask"])
def test_positive_contrary_checks_take_precedence(kind):
    primary = outcome([1])
    target = primary["comparison"]["outputs"][0]["target"]
    fp64 = {
        "status": "complete",
        "outputs": [
            {
                "target": target,
                "reference_dtype": "torch.float64",
                "accuracy_label": "comparable_to_eager_against_fp64",
                "farther_than_noise_floor": kind == "fp64",
            }
        ],
    }
    masks = {
        "status": "complete",
        "replay_matches_primary": True,
        "backward_same_inputs": [{"fp64_relative": {"agrees": kind != "backward"}}],
    }
    if kind == "mask":
        masks.update(
            flips=1,
            gradient_mapping={
                target: {"status": "mapped", "origins": [{"step": 0, "stage": "b", "output": 2}]}
            },
            masks=[
                {
                    "status": "complete",
                    "flips": 1,
                    "step": 0,
                    "stage": "b",
                    "affected_backward_outputs": [0],
                }
            ],
        )
    context = annotate(primary, fp64, {}, masks)
    assert context["triage"]["classification"] == "unexplained_contrary_evidence"
    assert context["triage"]["contrary_evidence"]
