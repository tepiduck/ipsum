import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from faultline import sweep, sweep_models
from faultline.evaluate import Settings
from faultline.storage import atomic_json


@pytest.mark.parametrize(
    "name", ["distilbert", "t5_small_encoder", "roberta_base", "opt_125m", "custom_hf"]
)
def test_pretrained_loader_serializes_key_sets_and_rejects_missing_weights(
    tmp_path, monkeypatch, name
):
    import importlib.metadata

    if name == "custom_hf":
        monkeypatch.setitem(sweep_models.TEXT_MODELS, name, "fixture/roberta")

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    atomic_json(inputs / "inputs.json", {"sources": [], "texts": ["first", "second"]})
    loading = {
        "missing_keys": set(),
        "unexpected_keys": {"head.bias", "head.weight"},
        "mismatched_keys": set(),
        "error_msgs": [],
    }
    inner = torch.nn.Linear(2, 2)
    inner.config = SimpleNamespace(use_cache=True, to_dict=lambda: {"model_type": "fixture"})

    class Tokenizer:
        pad_token_id = 0

        def __call__(self, texts, **kwargs):
            return {"input_ids": torch.ones(len(texts), kwargs["max_length"], dtype=torch.int64)}

    def load_model(*args, **kwargs):
        assert kwargs["output_loading_info"] and kwargs["use_safetensors"]
        assert kwargs["trust_remote_code"] is False
        assert kwargs["attn_implementation"] == "sdpa"
        assert kwargs.get("add_pooling_layer") is (
            False if name in {"roberta_base", "custom_hf"} else None
        )
        return (SimpleNamespace(base_model=inner) if name == "opt_125m" else inner), loading

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            model_info=lambda *a, **kw: SimpleNamespace(
                sha="fixed-revision", siblings=[SimpleNamespace(rfilename="model.safetensors")]
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoConfig=SimpleNamespace(
                from_pretrained=lambda *a, **kw: SimpleNamespace(model_type="roberta")
            ),
            AutoModel=SimpleNamespace(from_pretrained=load_model),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
            T5EncoderModel=SimpleNamespace(from_pretrained=load_model),
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer()),
        ),
    )
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "fixture-version")
    spec = sweep_models.prepare_model(name, tmp_path / "model", inputs)
    assert spec["unused_checkpoint_keys"] == ["head.bias", "head.weight"]
    saved = torch.load(tmp_path / "model/weights.pt", weights_only=True)
    assert set(saved) == {"inner.weight", "inner.bias"}
    assert torch.equal(saved["inner.weight"], inner.weight)
    if name == "opt_125m":
        assert spec["checkpoint_loader"] == "AutoModelForCausalLM"
        assert spec["loader"] == "AutoModel"
    assert json.loads((tmp_path / "model/model.json").read_text()) == spec
    loading["missing_keys"].add("encoder.weight")
    with pytest.raises(ValueError, match="did not fully initialize"):
        sweep_models.prepare_model(name, tmp_path / "incomplete", inputs)


def test_sweep_denominators_and_suspicious_mixed_evidence():
    suspicious = sweep.finding_context(
        {
            "status": "gradient_mismatch",
            "numerical_context": {
                "labels": ["discontinuity_present", "comparable_to_eager_against_fp64"],
                "masks": {"backward_same_inputs": [{"matches": False}]},
                "control_policy": "input_one_ulp_up",
            },
        }
    )
    assert suspicious["unexplained"] and not suspicious["has_full_diagnostic_label"]
    passed = sweep.finding_context({"status": "pass"})
    rows = [
        {"classification": "agreement", "baseline": passed, "candidate": passed},
        {"classification": "regression", "baseline": passed, "candidate": suspicious},
        {"classification": "inconclusive", "baseline": passed, "candidate": {"labels": []}},
    ]
    counts = sweep.sweep_counts(rows, planned=5)
    assert (
        counts["fully_agree"],
        counts["finding"],
        counts["inconclusive"],
        counts["pending"],
    ) == (1, 1, 1, 2)
    assert counts["unexplained_findings"] == 1
    assert counts["labels"]["discontinuity_present"] == 1


def test_sweep_resume_retains_failed_models_and_rejects_changed_evidence(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    atomic_json(prepared / "prepared.json", {"models": []})
    names = ["resnet18", "resnet34"]
    model = {
        "provider": "fixture",
        "parameter_count": 2,
        "weights_sha256": "fixed",
        "inputs_sha256": "fixed",
    }
    entries = {
        names[0]: {"status": "ready", "bundle": "fixture", "model": model},
        names[1]: {"status": "preparation_error", "error": "controlled download failure"},
    }
    monkeypatch.setattr(sweep, "verify_prepared", lambda *_: entries)
    monkeypatch.setattr(sweep, "runtime_environment", lambda _: {"packages": {"torch": "fixed"}})
    calls = []

    def compare(bundle, output, *_args, **_kwargs):
        calls.append(bundle)
        result = {
            "classification": "agreement",
            "baseline": {"status": "pass"},
            "candidate": {"status": "pass"},
        }
        for path in [
            "summary.json",
            "baseline/outcome.json",
            "candidate/outcome.json",
            "workload/workload.json",
        ]:
            atomic_json(Path(output) / path, result)
        return result

    monkeypatch.setattr(sweep, "compare_native_environments", compare)
    settings = Settings(gradients=True, numerics=True)
    output = tmp_path / "sweep"
    args = (prepared, output, sys.executable, sys.executable, settings)
    first = sweep.run_sweep(*args, names=names, progress=None)
    assert first["counts"]["planned"] == first["completed"] == 2
    assert first["counts"]["inconclusive"] == 1 and len(calls) == 1
    assert sweep.run_sweep(*args, names=names, resume=True, progress=None) == first
    assert len(calls) == 1
    row = output / "cases/resnet18/attempt-0001/row.json"
    content = json.loads(row.read_text())
    content["classification"] = "regression"
    row.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="evidence changed"):
        sweep.run_sweep(*args, names=names, resume=True, progress=None)
