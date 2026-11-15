import json
import subprocess
import sys

import pytest

from faultline.batches import verify_batches
from faultline.evaluate import Settings
from faultline.native import capture_workload
from faultline.verdict_bisect import export_collection


def test_replicated_export_preserves_every_batch_and_rejects_tampering(tmp_path):
    source = tmp_path / "factory.py"
    source.write_text(
        """import torch
def factory(seed):
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 2).eval()
    inputs = torch.arange(6, dtype=torch.float32).reshape(3, 2).t() + seed * 10
    return model, (inputs,)
"""
    )
    settings = Settings(gradients=True, repeats=2, threads=1)
    bundle = capture_workload(
        f"{source}:factory", tmp_path / "batches", settings, input_batches=10, seeded_factory=True
    )
    script = export_collection(bundle, tmp_path / "export", settings)
    record = json.loads((script.parent / "reproducer.json").read_text())
    assert verify_batches(script.parent / record["bundle"]) == verify_batches(bundle)
    probe = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
    assert "--baseline" in probe.stdout and "--candidate" in probe.stdout
    assert export_collection(bundle, script.parent, settings) == script
    snapshot = next((script.parent / record["bundle"]).glob("batch-*/snapshot.pt"))
    snapshot.write_bytes(b"modified")
    with pytest.raises(ValueError, match="changed"):
        export_collection(bundle, script.parent, settings)
    probe = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert probe.returncode != 0 and "Modified artifact" in probe.stderr
