"""Controlled batch effects are harness tests, not compiler findings."""

import copy
import json
import math

import pytest

from faultline.batch_bisect import predicate
from faultline.replication import POLICY, accuracy_shift_replicated, batch_score


def outcome(values):
    return {
        "status": "gradient_mismatch",
        "injected_fault": True,
        "fp64_advisory": {
            "status": "complete",
            "outputs": [
                {
                    "target": f"step:0:gradient:parameter:p{i}:value",
                    "reference_dtype": "torch.float64",
                    "candidate": {"rmse": value},
                }
                for i, value in enumerate(values)
            ],
        },
    }


def summarize(factors, planned=None):
    baseline = outcome([1e-7] * 30)
    return accuracy_shift_replicated(
        [
            {"batch": i, **batch_score(baseline, outcome([1e-7 * factor] * 30))}
            for i, factor in enumerate(factors)
        ],
        planned or len(factors),
    )


def test_uniform_planted_degradation_flags():
    result = summarize([1.5] * 10)
    assert result["accuracy_shift_replicated"]
    assert result["worsened"] == 10 and result["sign_test_p"] == 2**-10
    assert result["median_log2_growth"] == pytest.approx(math.log2(1.5))


def test_one_planted_batch_is_not_thirty_independent_gradient_targets():
    result = summarize([2] + [1] * 9)
    assert not result["accuracy_shift_replicated"]
    assert result["worsened"] == 1 and result["ties"] == 9


def test_same_version_errors_do_not_flag_and_primary_evidence_is_unchanged():
    baseline = outcome([0, 1e-6, 2e-6])
    saved = copy.deepcopy(baseline)
    result = accuracy_shift_replicated(
        [{"batch": i, **batch_score(baseline, baseline)} for i in range(10)], 10
    )
    assert result["ties"] == 10 and not result["accuracy_shift_replicated"]
    assert baseline == saved


def test_predeclared_policy_and_boundaries():
    from pathlib import Path

    policy = json.loads((Path(__file__).parents[2] / "docs/v016-policy.json").read_text())
    assert all(policy[key] == value for key, value in POLICY.items())
    assert summarize([1.25] * 8 + [1] * 2)["accuracy_shift_replicated"]
    assert summarize([1.25] * 16 + [1] * 4)["accuracy_shift_replicated"]
    assert not summarize([1.25] * 7 + [1] * 3)["accuracy_shift_replicated"]
    assert not summarize([1.24] * 10)["accuracy_shift_replicated"]
    assert not summarize([2] * 9, 10)["accuracy_shift_replicated"]


def test_missing_batches_stay_planned_and_binomial_excludes_ties():
    result = summarize([2] * 8 + [1] * 2, 20)
    assert result["planned"] == 20 and result["completed"] == 10
    assert result["missing"] == 10 and result["coverage"] == 0.5
    assert result["sign_test_p"] == 2**-8
    assert result["accuracy_shift_replicated"]
    assert predicate(result) == "shifted"
    assert predicate(summarize([1] * 10, 20)) == "inconclusive"
    assert predicate(summarize([1] * 10)) == "not_shifted"


def test_zero_errors_and_missing_gradients():
    score = batch_score(outcome([0]), outcome([1]))
    result = accuracy_shift_replicated([{"batch": i, **score} for i in range(10)], 10)
    assert result["accuracy_shift_replicated"] and result["median_status"] == "positive_infinity"
    json.dumps(result, allow_nan=False)
    assert batch_score(outcome([0, 1]), outcome([1, 0]))["status"] == "incomplete"
    assert batch_score(outcome([1, 2]), outcome([1]))["status"] == "incomplete"
    assert batch_score(outcome([1]), {"status": "timeout"})["status"] == "incomplete"


def test_duplicate_and_unplanned_batches_rejected():
    score = {"batch": 0, **batch_score(outcome([1]), outcome([1]))}
    with pytest.raises(ValueError, match="Duplicate"):
        accuracy_shift_replicated([score, score], 10)
    with pytest.raises(ValueError, match="unplanned"):
        accuracy_shift_replicated([{**score, "batch": 10}], 10)


def test_replicated_nightly_search_and_fresh_confirmation(tmp_path, monkeypatch):
    import sys

    from faultline import batch_bisect
    from faultline.evaluate import Settings

    bundle = tmp_path / "batches"
    bundle.mkdir()
    (bundle / "batches.json").write_text('{"planned": 10}')
    monkeypatch.setattr(batch_bisect, "verify_batches", lambda _: {})
    monkeypatch.setattr(batch_bisect, "runtime_environment", lambda _: {"fixed": True})
    wheels = [{"date": f"2026-11-0{i}"} for i in range(1, 5)]
    monkeypatch.setattr(batch_bisect, "nightly_catalog", lambda *args: {"wheels": wheels})
    monkeypatch.setattr(
        batch_bisect,
        "ensure_environment",
        lambda root, wheel, *args, **kwargs: {
            "status": "ready",
            "python": wheel["date"],
            "identity": {"torch_git_version": wheel["date"]},
        },
    )
    calls = []

    def compare(bundle, output, baseline, candidate, settings, **kwargs):
        calls.append((str(output), candidate))
        return {
            "accuracy_shift_replicated": summarize([1.5 if candidate >= "2026-11-03" else 1] * 10)
        }

    monkeypatch.setattr(batch_bisect, "compare_batches", compare)
    result = batch_bisect.run_replicated_bisect(
        bundle,
        tmp_path / "bisect",
        sys.executable,
        "2026-11-01",
        "2026-11-04",
        settings=Settings(),
        progress=None,
    )
    assert result["status"] == "localized_accuracy_shift" and result["exit_code"] == 0
    assert result["boundary"]["last_observed_not_shifted"] == "2026-11-02"
    assert result["boundary"]["first_observed_shifted"] == "2026-11-03"
    assert len([path for path, _ in calls if "confirmation" in path]) == 2
