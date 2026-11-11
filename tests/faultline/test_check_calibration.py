import importlib.util
from pathlib import Path

import pytest

from faultline.native import file_hash
from faultline.storage import atomic_json


def script():
    path = Path(__file__).parents[2] / "scripts/summarize_check_calibration.py"
    spec = importlib.util.spec_from_file_location("calibration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibration_retains_completed_primary_after_diagnostic_limit(tmp_path):
    root = tmp_path / "batch-0000/attempt-0001"
    raw = {"status": "resource_limit", "primary_observation": {"status": "gradient_mismatch"}}
    files = {}
    for side in ("baseline", "candidate"):
        path = root / side / "outcome.json"
        atomic_json(path, raw)
        files[path.relative_to(root).as_posix()] = file_hash(path)
    atomic_json(root / "commit.json", {"files": files})
    module = script()
    rows, hashes = module.verified_attempts(tmp_path)
    assert rows[0]["candidate"]["status"] == "gradient_mismatch"
    assert rows[0]["diagnostics"]["candidate"][0]["status"] == "diagnostic_skipped_memory"
    result = module.compact("controlled", "fixture", rows, 1, hashes=hashes)
    assert result["verdict"] == "NEEDS_REVIEW"
    assert result["scope"]["completed_batches"] == 1
    atomic_json(root / "candidate/outcome.json", {"status": "pass"})
    with pytest.raises(ValueError, match="evidence changed"):
        module.verified_attempts(tmp_path)


def test_calibration_rejects_commit_path_escape(tmp_path):
    root = tmp_path / "batch-0000/attempt-0001"
    outside = root.parent / "outside.json"
    atomic_json(outside, {})
    atomic_json(root / "commit.json", {"files": {"../outside.json": file_hash(outside)}})
    with pytest.raises(ValueError, match="evidence changed"):
        script().verified_attempts(tmp_path)
