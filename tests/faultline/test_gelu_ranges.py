import importlib.util
from pathlib import Path

import pytest
import torch


def example():
    path = Path(__file__).resolve().parents[2] / "examples/gelu_ranges.py"
    spec = importlib.util.spec_from_file_location("gelu_ranges_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ranges_preserve_lower_edges_and_reject_nonfinite_inputs():
    module = example()
    result = module.describe(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]), [-1, 0, 1])
    assert result["counts"] == [1, 1, 1, 2]
    assert sum(result["counts"]) == result["count"] == 5
    with pytest.raises(ValueError, match="Nonfinite"):
        module.describe(torch.tensor([float("nan")]))
    with pytest.raises(ValueError, match="increasing"):
        module.describe(torch.ones(2), [0, 0])


def test_exp_disagreement_separates_nonfinite_results():
    module = example()
    a = torch.tensor([0.0, 1.0, float("inf"), float("nan")])
    b = torch.tensor([0.0, 2.0, float("inf"), float("inf")])
    row = module.exp_disagreement(a, b)
    assert row["changed"] == 2
    assert row["changed_fraction"] == 0.5
    assert row["both_finite"] == 2
    assert row["parent_nonfinite"] == row["child_nonfinite"] == 2
    assert row["max_abs_difference_finite"] == 1


def test_block12_criteria_pool_counts_and_preserve_null_results():
    module = example()
    rows = [{"block": i, "abs_p99": 1.0, "changed": 5, "count": 10} for i in range(1, 13)]
    null = module.block12_comparison(rows)
    assert not null["p99_over_twice_previous_median"]
    assert not null["changed_fraction_at_least_10pp_higher"]
    rows[-1].update(abs_p99=2.01, changed=7)
    planted = module.block12_comparison(rows)
    assert planted["p99_over_twice_previous_median"]
    assert planted["changed_fraction_at_least_10pp_higher"]
    assert planted["abs_p99_rank_descending"] == 1
    with pytest.raises(ValueError, match="every block"):
        module.block12_comparison(rows[:-1])
