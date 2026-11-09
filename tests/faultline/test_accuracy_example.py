"""The independent reproducer must reject a different workload before measuring it."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch


def example():
    path = Path(__file__).resolve().parents[2] / "examples/gpt2_accuracy.py"
    spec = importlib.util.spec_from_file_location("gpt2_accuracy_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reproducer_rejects_modified_assets_before_loading_model(tmp_path):
    module = example()
    metadata = {"repository": module.REPOSITORY, "revision": module.REVISION}
    for name, key in (("weights.pt", "weights_sha256"), ("inputs.pt", "inputs_sha256")):
        payload = name.encode()
        (tmp_path / name).write_bytes(payload)
        metadata[key] = hashlib.sha256(payload).hexdigest()
    (tmp_path / "model.json").write_text(json.dumps(metadata))
    assert module.validate_assets(tmp_path) == metadata
    for name in ("weights.pt", "inputs.pt"):
        original = (tmp_path / name).read_bytes()
        (tmp_path / name).write_bytes(original + b"changed")
        # Invalid payloads cannot reach optional model imports or deserialization.
        with pytest.raises(ValueError, match="checksum differs"):
            module.load_model(tmp_path, offline=True)
        (tmp_path / name).write_bytes(original)


def test_reproducer_rejects_equal_values_with_changed_dtype_or_layout():
    module = example()
    tokens = torch.tensor(module.TOKENS, dtype=torch.int64)
    inputs = {"input_ids": tokens, "attention_mask": torch.ones_like(tokens)}
    module.validate_inputs(inputs)
    offset = torch.cat((torch.zeros(1, dtype=torch.int64), tokens.flatten()))[1:].view_as(tokens)
    for changed in (tokens.int(), tokens.t().contiguous().t(), offset):
        assert torch.equal(tokens, changed)
        with pytest.raises(ValueError, match="dtype, or layout differ"):
            module.validate_inputs({**inputs, "input_ids": changed})
