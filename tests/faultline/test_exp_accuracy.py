import importlib.util
from pathlib import Path

import pytest
import torch


def example():
    path = Path(__file__).resolve().parents[2] / "examples/exp_accuracy.py"
    spec = importlib.util.spec_from_file_location("exp_accuracy_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sampled_ulp_stats_separate_zero_subnormal_and_overflow():
    module = example()
    inputs = torch.tensor([-110.0, -104.0, -100.0, -90.0, -1.0, 0.0, 89.0])
    reference = inputs.double().exp()
    output = reference.float()
    rows = module.summarize_errors(inputs, output)
    assert rows["normal"]["count"] == 2
    assert rows["subnormal"]["count"] == 2
    assert rows["underflow_to_zero"]["count"] == 2
    assert rows["overflow_to_infinity"]["correctly_rounded_count"] == 1
    expected = float(reference[:2].max() / 2**-149)
    assert rows["underflow_to_zero"]["max_ulp_error_vs_fp64"] == pytest.approx(expected)
    assert rows["underflow_to_zero"]["max_distance_to_rounded_fp32"] == 0
    assert rows["normal"]["max_ulp_error_vs_fp64"] <= 0.5
    broken = output.clone()
    broken[2:4] = 0
    failed = module.summarize_errors(inputs, broken)
    assert failed["subnormal"]["over_two_ulp_count"] == 2


def test_vector_adapter_rejects_implicit_layout_and_precision_changes():
    module = example()
    with pytest.raises(ValueError, match="CPU fp32"):
        module.Adapter.check(torch.zeros(4, dtype=torch.float64))
    with pytest.raises(ValueError, match="contiguous"):
        module.Adapter.check(torch.zeros(4, 4).t())
    module.Adapter.check(torch.zeros(4, 4).t(), contiguous=False)
