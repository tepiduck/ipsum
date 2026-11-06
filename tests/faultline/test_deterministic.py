import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest
import torch

from faultline import portable
from faultline.evaluate import Settings
from faultline.native import capture_workload
from faultline.native_artifacts import (
    compare_native_environments,
    export_native,
    load_native_run,
    run_native,
)


def test_deterministic_policy_is_explicit_and_validated(monkeypatch):
    assert "deterministic" not in Settings().to_dict()
    with pytest.raises(ValueError, match="deterministic must be boolean"):
        Settings(deterministic=1).validate()
    old = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    for name in portable.deterministic_environment():
        monkeypatch.setenv(name, "prior-value")
    try:
        portable.configure("cpu", 1, 0, True)
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert all(os.environ[k] == v for k, v in portable.deterministic_environment().items())
    finally:
        torch.use_deterministic_algorithms(old, warn_only=warn)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = cudnn


def test_deterministic_policy_survives_worker_replay_and_standalone(tmp_path):
    source = tmp_path / "model.py"
    source.write_text(
        "import torch\ndef factory():\n    return torch.nn.Linear(3, 2).eval(), (torch.ones(2, 3),)\n"
    )
    settings = Settings(backend="eager", gradients=True, deterministic=True, repeats=2)
    bundle = capture_workload(f"{source}:factory", tmp_path / "bundle", settings)
    first = run_native(bundle, tmp_path / "first", settings)
    assert first["status"] == "pass"
    job, saved, expected = load_native_run(tmp_path / "first")
    assert saved.deterministic and expected["deterministic"]
    environment = expected["worker_environment"]
    assert environment["deterministic_algorithms"] and not environment["deterministic_warn_only"]
    assert environment["blas_workspace"] == portable.deterministic_environment()
    replay = run_native(job.bundle, tmp_path / "replay", saved, expected=expected)
    assert replay["status"] == "pass"
    script = export_native(job, saved, tmp_path / "standalone", expected)
    run = subprocess.run(
        [sys.executable, str(script)], check=False, capture_output=True, text=True, timeout=60
    )
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["deterministic"]
    with pytest.raises(ValueError, match="policy"):
        export_native(job, replace(saved, deterministic=False), tmp_path / "wrong-policy", expected)


@pytest.mark.parametrize("limit", ["timeout", "resource_limit"])
def test_inconclusive_deterministic_comparison_still_exports(tmp_path, monkeypatch, limit):
    source = tmp_path / "model.py"
    source.write_text(
        "import torch\ndef factory():\n    return torch.nn.Linear(3, 2).eval(), (torch.ones(2, 3),)\n"
    )
    settings = Settings(backend="eager", deterministic=True)
    bundle = capture_workload(f"{source}:factory", tmp_path / "bundle", settings)
    limited = (
        replace(settings, timeout_s=0.001)
        if limit == "timeout"
        else replace(settings, max_rss_mb=128)
    )
    if limit == "resource_limit":
        # Force the supervisor branch without depending on the platform's import RSS.
        monkeypatch.setattr("faultline.runner.tree_rss", lambda *_: 129 * 1024**2)
    output = tmp_path / "comparison"
    summary = compare_native_environments(bundle, output, sys.executable, sys.executable, limited)
    assert summary["classification"] == "inconclusive"
    assert summary["exit_code"] == 2
    for side in ("baseline", "candidate"):
        result = summary[side]
        assert result["status"] == limit and result["fingerprint"] == limit
        assert result["deterministic"] is True
        exported = json.loads((output / f"{side}-reproducer/reproducer.json").read_text())
        assert exported["settings"]["deterministic"] and exported["expected"]["deterministic"]
