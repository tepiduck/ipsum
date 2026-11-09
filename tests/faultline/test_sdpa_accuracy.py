import importlib.util
from pathlib import Path

import pytest
import torch


def example():
    path = Path(__file__).resolve().parents[2] / "examples/sdpa_accuracy.py"
    spec = importlib.util.spec_from_file_location("sdpa_accuracy_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def records(tmp_path, module):
    generator = torch.Generator().manual_seed(37)
    # A shared packed buffer produces gapped q/k/v views with distinct offsets.
    packed = torch.randn(2, 4, 3, 2, 8, generator=generator)
    q, k, v = (packed[:, :, i].transpose(1, 2) for i in range(3))
    grad_out = torch.randn(2, 4, 2, 8, generator=generator).transpose(1, 2)
    out, lse = module.FWD(q, k, v, 0.0, True)
    forward = {"args": (q, k, v, 0.0, True), "kwargs": {}, "outputs": (out, lse)}
    backward = {
        "args": (grad_out, q, k, v, out, lse, 0.0, True),
        "kwargs": {},
    }
    paths = tmp_path / "forward.pt", tmp_path / "backward.pt"
    for path, value in zip(paths, (forward, backward), strict=True):
        torch.save(value, path)
    return paths, forward, backward


def test_saved_sdpa_measurement_preserves_packed_layouts_and_checks_reference(tmp_path):
    module = example()
    paths, forward, _ = records(tmp_path, module)
    f, b = module.load_records(*paths)
    for index, name in enumerate(("query", "key", "value")):
        assert module.layout(f[name]) == module.layout(forward["args"][index])
    assert f["value"].storage_offset() != 0
    assert not f["query"].is_contiguous()
    actual = module.flash_call(f, b)
    reference = module.reference_math(f, b)
    for name in ("out", "logsumexp", "dq", "dk", "dv"):
        torch.testing.assert_close(actual[name].double(), reference[name], rtol=1e-5, atol=1e-6)
    planted = {**actual, "dq": actual["dq"] + 0.1}
    assert module.measurements(planted, reference)["dq"]["rmse"] > 0.09
    result = module.measure(*paths)
    assert result["forward_layouts"]["value"]["storage_offset"] == f["value"].storage_offset()
    assert result["settings"]["is_causal"] is True


def test_sdpa_rejects_missing_backward_settings_and_changed_saved_values(tmp_path):
    module = example()
    paths, _, backward = records(tmp_path, module)
    truncated = {**backward, "args": backward["args"][:6]}
    with pytest.raises(ValueError, match="Missing required"):
        module.bind_record(truncated, backward=True)
    changed = list(backward["args"])
    changed[1] = changed[1] + 1
    torch.save({**backward, "args": tuple(changed)}, paths[1])
    with pytest.raises(ValueError, match="saved values differ"):
        module.load_records(*paths)


def test_fresh_forward_results_use_recorded_backward_layout_explicitly():
    module = example()
    template = torch.empty(100).as_strided((2, 3), (12, 2), 7)
    value = torch.arange(6, dtype=torch.float32).view(2, 3)
    result = module.copy_layout(value, template)
    assert module.layout(result) == module.layout(template)
    assert torch.equal(value, result)
    assert result.untyped_storage().data_ptr() != template.untyped_storage().data_ptr()
    with pytest.raises(ValueError, match="contract"):
        module.copy_layout(value.double(), template)
