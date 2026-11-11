import json
import sys
from pathlib import Path

import pytest
import torch

from faultline import batches
from faultline.evaluate import Settings
from faultline.native import capture_workload, file_hash, seal_native
from faultline.storage import atomic_json, read_json


def fake_capture(factory, output, settings, **kwargs):
    output = Path(output)
    output.mkdir(parents=True)
    (output / "factory.py").write_text("# controlled fixture\n")
    index = kwargs["batch_index"]
    payload = {
        "state": {"weight": torch.ones(1)},
        "requests": [
            {
                "args": (torch.tensor([[index * 2.0], [index * 2.0 + 1]]),),
                "kwargs": {},
                "cotangents": {"out": torch.tensor([[1.0], [-1.0]])},
            }
        ],
    }
    torch.save(payload, output / "snapshot.pt")
    seal_native(
        output,
        {
            "schema_version": 3,
            "kind": "workload",
            "backward_probe": "vjp",
            "sources": ["factory.py"],
            "captured": True,
            "seed": 0,
        },
    )
    return output


def prepared(tmp_path, monkeypatch, count=10):
    monkeypatch.setattr(batches, "capture_workload", fake_capture)
    return batches.capture_batches("unused:factory", tmp_path / "prepared", Settings(), count)


def test_capture_hashes_samples_probes_and_preserves_layout(tmp_path, monkeypatch):
    root = prepared(tmp_path, monkeypatch)
    manifest = batches.verify_batches(root)
    assert manifest["planned"] == len(manifest["batches"]) == 10
    assert len({row["inputs_sha256"] for row in manifest["batches"]}) == 10
    assert all(len(row["probe_sha256"]) == 64 for row in manifest["batches"])
    path = root / "batch-0000/snapshot.pt"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="modified"):
        batches.verify_batches(root)


def test_capture_repeated_samples_are_rejected(tmp_path, monkeypatch):
    def duplicate(factory, output, settings, **kwargs):
        return fake_capture(factory, output, settings, **{**kwargs, "batch_index": 0})

    monkeypatch.setattr(batches, "capture_workload", duplicate)
    with pytest.raises(ValueError, match="repeated sample"):
        batches.capture_batches("unused:factory", tmp_path / "prepared", Settings(), 2)


def test_failed_capture_keeps_planned_denominator(tmp_path, monkeypatch):
    def fail_one(factory, output, settings, **kwargs):
        if kwargs["batch_index"] == 1:
            raise ValueError("controlled capture timeout")
        return fake_capture(factory, output, settings, **kwargs)

    monkeypatch.setattr(batches, "capture_workload", fail_one)
    root = batches.capture_batches("unused:factory", tmp_path / "prepared", Settings(), 3)
    result = batches.verify_batches(root)
    assert result["planned"] == 3 and len(result["batches"]) == 2
    assert result["preparation_errors"][0]["batch"] == 1


def test_resume_retains_commits_and_retries_only_interrupted_batch(tmp_path, monkeypatch):
    from faultline import native_artifacts, sweep

    root = prepared(tmp_path, monkeypatch, 3)
    monkeypatch.setattr(sweep, "runtime_environment", lambda python: {"python": python})
    calls = []
    interrupt = [True]

    def compare(bundle, output, *args, **kwargs):
        index = int(Path(bundle).name.split("-")[1])
        calls.append(index)
        output.mkdir(parents=True)
        if index == 1 and interrupt[0]:
            raise RuntimeError("controlled interruption")
        side = {
            "status": "pass",
            "fp64_advisory": {
                "status": "complete",
                "outputs": [
                    {
                        "target": "step:0:gradient:parameter:w:value",
                        "reference_dtype": "torch.float64",
                        "candidate": {"rmse": 1e-7},
                    }
                ],
            },
        }
        result = {
            "classification": "agreement",
            "baseline": side,
            "candidate": side,
            "exit_code": 0,
        }
        for name in (
            "summary.json",
            "baseline/outcome.json",
            "candidate/outcome.json",
            "baseline/job.json",
            "candidate/job.json",
        ):
            atomic_json(output / name, result)
        return result

    monkeypatch.setattr(native_artifacts, "compare_native_environments", compare)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="interruption"):
        batches.compare_batches(root, output, sys.executable, sys.executable, Settings())
    commit = output / "batch-0000/attempt-0001/commit.json"
    digest = file_hash(commit)
    interrupt[0] = False
    result = batches.compare_batches(
        root, output, sys.executable, sys.executable, Settings(), resume=True
    )
    assert calls == [0, 1, 1, 2]
    assert result["accuracy_shift_replicated"]["completed"] == 3
    assert result["exit_code"] == 0 and file_hash(commit) == digest
    assert (output / "batch-0001/attempt-0001").exists()
    with pytest.raises(ValueError, match="changed"):
        batches.compare_batches(
            root, output, sys.executable, sys.executable, Settings(threads=2), resume=True
        )
    saved = output / "batch-0000/attempt-0001/summary.json"
    saved.write_text("{}")
    with pytest.raises(ValueError, match="evidence changed"):
        batches.compare_batches(
            root, output, sys.executable, sys.executable, Settings(), resume=True
        )


def test_real_capture_freezes_individual_batches(tmp_path):
    factory = tmp_path / "model.py"
    factory.write_text(
        "import torch, random, numpy as np\ndef factory():\n"
        "    torch.manual_seed(5)\n"
        "    inputs = [(torch.tensor([[random.random(), np.random.random()]]),) for _ in range(2)]\n"
        "    return torch.nn.Linear(2, 1).eval(), inputs\n"
    )
    root = capture_workload(
        f"{factory}:factory",
        tmp_path / "captured",
        Settings(backend="eager", repeats=2),
        input_batches=2,
    )
    manifest = batches.verify_batches(root)
    assert len(manifest["batches"]) == 2, json.dumps(manifest)
    assert len({row["weights_sha256"] for row in manifest["batches"]}) == 1
    assert read_json(root / "batch-0001/workload.json")["batch_index"] == 1
    import random

    import numpy as np

    python_rng, numpy_rng = random.Random(0), np.random.RandomState(0)
    for i in range(2):
        payload = torch.load(root / f"batch-{i:04d}/snapshot.pt", weights_only=True)
        expected = torch.tensor([[python_rng.random(), numpy_rng.random()]])
        assert torch.equal(payload["requests"][0]["args"][0], expected)


@pytest.mark.parametrize(
    "mode,flag,worse", [("uniform", True, 10), ("one", False, 1), ("same", False, 0)]
)
def test_paired_coordinator_planted_effects_do_not_change_primary_gate(
    tmp_path, monkeypatch, mode, flag, worse
):
    from faultline import native_artifacts, sweep

    root = prepared(tmp_path, monkeypatch)
    monkeypatch.setattr(sweep, "runtime_environment", lambda python: {"python": python})

    def compare(bundle, output, *args, **kwargs):
        output.mkdir(parents=True)
        index = int(Path(bundle).name.split("-")[1])
        factor = 1.5 if mode == "uniform" or (mode == "one" and index == 0) else 1

        def side(error):
            return {
                "status": "pass",
                "injected_fault": True,
                "fp64_advisory": {
                    "status": "complete",
                    "outputs": [
                        {
                            "target": "step:0:gradient:parameter:w:value",
                            "reference_dtype": "torch.float64",
                            "candidate": {"rmse": error},
                        }
                    ],
                },
            }

        result = {
            "classification": "agreement",
            "baseline": side(1e-7),
            "candidate": side(factor * 1e-7),
            "exit_code": 0,
        }
        for name in (
            "summary.json",
            "baseline/outcome.json",
            "candidate/outcome.json",
            "baseline/job.json",
            "candidate/job.json",
        ):
            atomic_json(output / name, result)
        return result

    monkeypatch.setattr(native_artifacts, "compare_native_environments", compare)
    result = batches.compare_batches(
        root, tmp_path / "run", sys.executable, sys.executable, Settings()
    )
    assert result["exit_code"] == 0
    assert result["accuracy_shift_replicated"]["accuracy_shift_replicated"] is flag
    assert result["accuracy_shift_replicated"]["worsened"] == worse
    assert all(row["classification"] == "agreement" for row in result["batches"])
