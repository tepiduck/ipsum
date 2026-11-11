import sys

import pytest

from faultline import environments
from faultline.storage import atomic_json


def test_requirement_validation(tmp_path):
    assert environments.torch_requirement("2.15.0") == "torch==2.15.0"
    assert environments.torch_requirement("nightly") == "torch"
    for bad in (
        "numpy==2.0",
        "torch>=2",
        "torch==2.*",
        "https://user:password@example.org/torch.whl",
    ):
        with pytest.raises(ValueError):
            environments.torch_requirement(bad)
    with pytest.raises(ValueError, match="--index-url"):
        environments.create_environment(tmp_path / "gpu", "2.15.0", device="cuda")


def test_mocked_install_and_identity_checked_resume(tmp_path, monkeypatch):
    calls = []
    identity = {"torch": "2.15.0+cpu", "packages": {"torch": "2.15.0+cpu"}}

    def supervise(command, directory, **kwargs):
        calls.append(command)
        directory.mkdir(parents=True)
        if directory.name == "device-probe":
            atomic_json(directory / "stdout.log", {"cuda": None, "hip": None})
        return {"status": "complete"}

    monkeypatch.setattr(environments, "supervise", supervise)
    monkeypatch.setattr(environments, "environment_identity", lambda *a: dict(identity))
    root = tmp_path / "environment"
    ready = environments.create_environment(root, "2.15.0", requirements=["einops==0.8.1"])
    assert ready["status"] == "ready"
    assert calls[0][:3] == [sys.executable, "-m", "venv"]
    assert "torch==2.15.0" in calls[1] and "torchvision" in calls[1]
    assert "-c" in calls[2] and "einops==0.8.1" in calls[2]
    count = len(calls)
    assert (
        environments.create_environment(root, "2.15.0", requirements=["einops==0.8.1"], resume=True)
        == ready
    )
    assert len(calls) == count
    identity["torch"] = "2.16.0"
    with pytest.raises(ValueError, match="environment changed"):
        environments.create_environment(root, "2.15.0", requirements=["einops==0.8.1"], resume=True)


def test_installation_failure_is_not_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(environments, "supervise", lambda *a, **kw: {"status": "timeout"})
    with pytest.raises(ValueError, match="timeout"):
        environments.create_environment(tmp_path / "env", "nightly", model_dependencies=False)
    assert not (tmp_path / "env/ready.json").exists()
