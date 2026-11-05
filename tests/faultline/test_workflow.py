import json
import subprocess
import sys
from dataclasses import replace

import pytest

from faultline.campaign import config_hash, run_campaign, validate_manifest
from faultline.evaluate import Settings
from faultline.export import export_case
from faultline.generate import corpus
from faultline.lowering import freeze
from faultline.report import write_report
from faultline.runner import run_case
from faultline.storage import atomic_json, campaign_lock, read_json


def test_worker_eager_and_timeout_leave_durable_outcomes(tmp_path):
    case = freeze(corpus()[0])
    result = run_case(case, Settings(backend="eager", repeats=2), tmp_path / "ok")
    assert result["status"] == "pass"
    assert result["peak_tree_rss_mb"] > 0
    timeout = run_case(case, Settings(backend="mutant:hang", timeout_s=0.2), tmp_path / "timeout")
    assert timeout["status"] == "timeout"
    assert read_json(tmp_path / "timeout" / "outcome.json")["status"] == "timeout"
    assert timeout["worker_wall_s"] < 15


def test_hard_worker_exit_is_not_misreported_as_compiler_exception(tmp_path):
    result = run_case(freeze(corpus()[0]), Settings(backend="mutant:worker-exit"), tmp_path)
    assert result["status"] == "crash"
    assert result["worker_returncode"] == 86
    assert result["injected_fault"]


def test_resume_does_not_rerun_completed_cases_and_checks_source(tmp_path, monkeypatch):
    from faultline import campaign

    settings = Settings(backend="eager", repeats=2)
    first = run_campaign(tmp_path, corpus()[:2], settings, stop_after=1, progress=None)
    assert first["completed"] == 1
    first_path = next((tmp_path / "cases").glob("*/outcome.json"))
    before = first_path.read_bytes()
    second = run_campaign(tmp_path, resume=True, progress=None)
    assert second["counts"] == {"pass": 2}
    assert first_path.read_bytes() == before
    assert read_json(tmp_path / "manifest.json")["state"] == "completed"
    original = campaign.describe
    monkeypatch.setattr(
        campaign, "describe", lambda: {**original(), "faultline_source_sha256": "changed"}
    )
    with pytest.raises(ValueError, match="source change"):
        run_campaign(tmp_path, resume=True, progress=None)


def test_campaign_lock_prevents_two_writers(tmp_path):
    with (
        campaign_lock(tmp_path),
        pytest.raises(RuntimeError, match="owns"),
        campaign_lock(tmp_path),
    ):
        pytest.fail("second writer acquired lock")


def test_resume_rejects_modified_input_even_for_completed_case(tmp_path):
    run_campaign(tmp_path, corpus()[:1], Settings(backend="eager"), progress=None)
    path = next((tmp_path / "cases").glob("*/case.json"))
    modified = read_json(path)
    modified["inputs"][0]["values"][0] += 1
    atomic_json(path, modified)
    with pytest.raises(ValueError, match="Input case changed"):
        run_campaign(tmp_path, resume=True, progress=None)


def test_manifest_detects_changed_settings():
    settings = Settings(backend="eager").to_dict()
    manifest = {
        "schema_version": 1,
        "settings": settings,
        "settings_hash": config_hash(settings),
        "cases": [],
    }
    settings["backend"] = "inductor"
    with pytest.raises(ValueError, match="settings changed"):
        validate_manifest(manifest)


@pytest.mark.parametrize("identifier", ["../escape", "/tmp/escape", "C:/escape"])
def test_manifest_cannot_redirect_case_writes(identifier):
    settings = Settings(backend="eager").to_dict()
    manifest = {
        "schema_version": 1,
        "settings": settings,
        "settings_hash": config_hash(settings),
        "cases": [{"id": identifier}],
    }
    with pytest.raises(ValueError, match="case identifiers"):
        validate_manifest(manifest)


def test_atomic_write_rejects_nonfinite_without_corrupting_existing_file(tmp_path):
    path = tmp_path / "data.json"
    atomic_json(path, {"original": 1})
    with pytest.raises(ValueError):
        atomic_json(path, {"bad": float("nan")})
    assert read_json(path) == {"original": 1}


@pytest.mark.parametrize(
    "backend,status",
    [
        ("eager", "pass"),
        ("mutant:sum-as-mean", "numerical_mismatch"),
        ("mutant:flaky", "unstable_candidate"),
        ("mutant:nan", "numerical_mismatch"),
    ],
)
def test_export_runs_without_importing_faultline(tmp_path, backend, status):
    path = export_case(corpus()[0], Settings(backend=backend, repeats=2), tmp_path)
    script = path.read_text()
    assert "from faultline" not in script and "import faultline" not in script
    run = subprocess.run(
        [sys.executable, "-I", str(path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert run.returncode == (
        0 if status == "pass" else 2 if status == "unstable_candidate" else 1
    ), run.stderr
    assert json.loads(run.stdout.splitlines()[-1])["status"] == status


def test_report_escapes_workload_tags(tmp_path):
    case = replace(corpus()[0], tag='<script>alert("x")</script>')
    run_campaign(tmp_path, [case], Settings(backend="eager", repeats=2), progress=None)
    document = write_report(tmp_path).read_text()
    assert "<script>" not in document
    assert "&lt;script&gt;" in document


@pytest.mark.compiler
def test_real_compiler_agrees_on_strided_reduction(tmp_path):
    result = run_case(freeze(corpus()[0]), Settings(backend="inductor", timeout_s=180), tmp_path)
    assert result["status"] == "pass", result
