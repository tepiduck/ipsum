import copy

import pytest
import torch

from faultline import native_aot, native_compile, portable
from faultline.native_masks import MaskProbe
from faultline.native_numerics import annotate
from faultline.native_runtime import compare_native
from faultline.sweep import finding_context, sweep_counts


@pytest.mark.parametrize("wrong", [False, True])
def test_backward_reordering_uses_fp64_while_elementwise_result_is_retained(wrong):
    class Backward(torch.nn.Module):
        def forward(self, saved):
            return (saved.sum(),)

    graph = torch.fx.symbolic_trace(Backward())
    saved = torch.tensor([1e8, 1, -1e8, 1])
    probe = MaskProbe(native_aot, native_compile, {"repeats": 2}, compare_native, portable, {})
    result = probe.around(
        graph,
        "backward",
        "0:backward:0",
        lambda args: (args[0].flip(0).sum() + (10 if wrong else 0),),
        [saved],
    )
    assert result[0].dtype == torch.float32
    check = probe.summary()["backward_same_inputs"][0]
    assert check["matches"] is False
    assert check["fp64_relative"]["agrees"] is (not wrong)
    assert check["fp64_relative"]["available"] == 1
    assert torch.equal(saved, torch.tensor([1e8, 1, -1e8, 1]))


def primary():
    return {
        "status": "gradient_mismatch",
        "comparison": {
            "outputs": [{"target": "step:0:gradient:parameter:weight:value", "matches": False}]
        },
    }


def explained_control():
    return {
        "status": "complete",
        "control": "onednn_disabled",
        "outputs": [
            {
                "target": "step:0:gradient:parameter:weight:value",
                "exceeds_primary_tolerance": True,
                "control_to_candidate_rmse": 1.0,
            }
        ],
    }


def test_missing_mask_diagnostics_do_not_overrule_complete_eager_control():
    masks = {
        "status": "partial",
        "flips": 3,
        "replay_matches_primary": True,
        "backward_same_inputs": [],
        "errors": [{"error": "RuntimeError: view layout"}],
    }
    result = primary()
    context = annotate(result, {}, explained_control(), masks)
    assert "discontinuity_present" in context["labels"]
    assert context["triage"]["classification"] == "explained"
    assert (
        context["label_coverage"]["discontinuity_downstream"]["backward_same_inputs_agree"] is None
    )
    assert context["triage"]["missing_evidence"] and not context["triage"]["contrary_evidence"]
    masks["backward_same_inputs"] = [{"matches": False, "fp64_relative": {"agrees": False}}]
    context = annotate(result, {}, explained_control(), masks)
    assert context["triage"]["classification"] == "unexplained_contrary_evidence"
    masks["backward_same_inputs"][0]["fp64_relative"]["agrees"] = None
    assert (
        annotate(result, {}, explained_control(), masks)["triage"]["classification"] == "explained"
    )


def test_incomplete_mapping_is_not_proof_that_a_flip_cannot_explain_gradient():
    target = primary()["comparison"]["outputs"][0]["target"]
    masks = {
        "status": "complete",
        "flips": 1,
        "replay_matches_primary": True,
        "gradient_mapping": {
            target: {
                "status": "mapped",
                "origins": [{"stage": "0:backward:0", "step": 0, "output": 1}],
            }
        },
        "masks": [
            {
                "status": "complete",
                "flips": 1,
                "step": 0,
                "stage": "0:backward:0",
                "affected_backward_outputs": [0],
            }
        ],
    }
    assert (
        annotate(primary(), {}, explained_control(), masks)["triage"]["classification"]
        == "unexplained_contrary_evidence"
    )
    masks["status"] = "partial"
    assert (
        annotate(primary(), {}, explained_control(), masks)["triage"]["classification"]
        == "explained"
    )
    masks["status"] = "complete"
    masks["gradient_mapping"][target]["status"] = "unmapped"
    assert (
        annotate(primary(), {}, explained_control(), masks)["triage"]["classification"]
        == "explained"
    )


def test_unexplained_categories_and_counts_keep_inconclusive_separate():
    result = primary()
    missing = annotate(result, {"status": "reference_error"}, {}, {})
    assert missing["triage"]["classification"] == "unexplained_missing_evidence"
    contrary = annotate(
        result,
        {},
        {},
        {
            "replay_matches_primary": True,
            "backward_same_inputs": [{"matches": False, "fp64_relative": {"agrees": False}}],
        },
    )
    explained = annotate(result, {}, explained_control(), {})
    rows = []
    for context in (missing, contrary, explained):
        side = finding_context({**result, "numerical_context": context})
        rows.append(
            {"classification": "recurring", "baseline": side, "candidate": copy.deepcopy(side)}
        )
    rows.append(
        {"classification": "inconclusive", "baseline": {"labels": []}, "candidate": {"labels": []}}
    )
    counts = sweep_counts(rows, planned=20)
    assert (
        counts["explained"]
        == counts["unexplained_contrary_evidence"]
        == counts["unexplained_missing_evidence"]
        == 1
    )
    assert (
        counts["unexplained_findings"] == 2
        and counts["inconclusive"] == 1
        and counts["pending"] == 16
    )
