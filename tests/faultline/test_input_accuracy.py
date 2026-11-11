import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest


def analysis():
    path = Path(__file__).resolve().parents[2] / "scripts/summarize_input_accuracy.py"
    spec = importlib.util.spec_from_file_location("input_accuracy_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measurement(error=1.0, targets=149):
    return {
        "schema": 1,
        "identity_sha256": "model",
        "inputs_sha256": "input",
        "worker_sha256": "worker",
        "threads": 1,
        "input_layouts": {},
        "observations_per_path": 1,
        "absent_gradients": [],
        "gradients": [
            {
                "parameter": str(i),
                "compiled_rmse": error,
                "eager_rmse": 1.0,
                "reference_sha256": "fp64",
            }
            for i in range(targets)
        ],
    }


def batches(worse=20, growth=1.5):
    module = analysis()
    return [
        {
            "batch": i,
            **module.paired_batch(measurement(), measurement(growth if i < worse else 0.9)),
        }
        for i in range(20)
    ]


@pytest.mark.parametrize(
    "worse,growth,decision",
    [
        (20, 1.5, "systematic_across_inputs"),
        (16, 1.25, "systematic_across_inputs"),
        (16, 1.24, "mixed"),
        (15, 1.5, "mixed"),
        (14, 1.5, "mixed"),
        (13, 1.5, "not_systematic"),
    ],
)
def test_predeclared_decision_boundaries(worse, growth, decision):
    result = analysis().summarize_batches(batches(worse, growth))
    assert result["compiled"]["decision"] == decision
    assert result["compiled"]["worse"] == worse
    assert result["eager"]["decision"] == "not_systematic"
    assert result["eager"]["ties"] == 20


def test_correlated_targets_are_one_replicate_and_same_version_is_a_tie():
    module = analysis()
    result = module.summarize_batches(batches())
    assert result["compiled"]["sign_trials"] == 20
    assert result["compiled"]["descriptive_p_value"] == 2**-20
    assert result["eager"]["sign_trials"] == 0
    assert result["eager"]["descriptive_p_value"] == 1
    assert result["eager"]["median_growth"]["value"] == 1


def test_missing_batch_stays_in_denominator_without_a_complete_decision():
    rows = batches()[:19]
    result = analysis().summarize_batches(rows)["compiled"]
    assert result["conclusive_batches"] == 19
    assert result["decision"] == "inconclusive"
    assert result["inconclusive_batch_ids"] == [19]
    rows.append({"batch": 19, "status": "timeout"})
    assert analysis().summarize_batches(rows)["compiled"] == result
    with pytest.raises(ValueError, match="duplicate"):
        analysis().summarize_batches(rows + [rows[0]])


def test_extended_zero_errors_and_indeterminate_median_are_explicit():
    module = analysis()
    assert module.log_growth(0, 0) == 0
    assert module.log_growth(0, 1) == math.inf
    assert module.log_growth(1, 0) == -math.inf
    parent, child = measurement(targets=2), measurement(targets=2)
    parent["gradients"][0]["compiled_rmse"] = 0
    child["gradients"][1]["compiled_rmse"] = 0
    result = module.paired_batch(parent, child)
    assert result["status"] == "inconclusive"
    assert result["compiled"]["median_log2_growth"] == {"status": "indeterminate", "value": None}
    json.dumps(result, allow_nan=False)
    for value in (-1, math.nan, math.inf):
        with pytest.raises(ValueError, match="RMSE"):
            module.log_growth(value, 1)


def test_pair_identity_presence_and_reference_changes_are_visible():
    module = analysis()
    parent = measurement()
    for key in ("inputs_sha256", "identity_sha256", "worker_sha256"):
        child = copy.deepcopy(parent)
        child[key] = "changed"
        with pytest.raises(ValueError, match="identity"):
            module.paired_batch(parent, child)
    child = copy.deepcopy(parent)
    child["gradients"].pop()
    with pytest.raises(ValueError, match="sets differ"):
        module.paired_batch(parent, child)
    child = copy.deepcopy(parent)
    child["gradients"][0]["reference_sha256"] = "changed"
    result = module.paired_batch(parent, child)
    assert result["fp64_reference_hashes_equal"] is False
    assert result["status"] == "complete"


def test_exact_binomial_excludes_only_ties():
    module = analysis()
    rows = batches(16)
    rows[19] = {"batch": 19, **module.paired_batch(measurement(), measurement())}
    result = module.summarize_batches(rows)["compiled"]
    assert (result["worse"], result["better"], result["ties"]) == (16, 3, 1)
    assert result["descriptive_p_value"] == sum(math.comb(19, k) for k in range(16, 20)) / 2**19
