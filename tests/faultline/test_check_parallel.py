import copy
import os
from pathlib import Path

import pytest

from faultline import check, check_execution, sweep
from faultline.evaluate import Settings
from faultline.replication import encoded
from faultline.storage import atomic_json, read_json


@pytest.fixture
def engine(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    atomic_json(source / "batches.json", {})
    monkeypatch.setattr(
        check_execution,
        "verify_batches",
        lambda p: {"planned": 10, "batches": [{"batch": i, "bundle": f"b{i}"} for i in range(10)]},
    )
    monkeypatch.setattr(check_execution.NativeJob, "create", lambda p: p)
    monkeypatch.setattr(sweep, "runtime_environment", lambda p: {"python": p})
    calls = []

    def run(job, settings, directory, **kwargs):
        calls.append((job.name, settings.max_rss_mb, kwargs["env"]))
        return {"status": "pass", "comparison": {"outputs": []}}

    def inspect(bundle, directory, settings, python, level, **kwargs):
        row = copy.deepcopy(kwargs["primary"])
        if kwargs["accuracy"]:
            row["fp64_advisory"] = {
                "status": "complete",
                "outputs": [
                    {
                        "target": "output:gradient:parameter:w",
                        "reference_dtype": "torch.float64",
                        "candidate": {"rmse": 1e-5},
                    }
                ],
            }
        return row, []

    monkeypatch.setattr(check, "run_case", run)
    monkeypatch.setattr(check, "inspect_side", inspect)
    return source, calls


def test_serial_parallel_have_identical_decisions_and_stopping_prefix(engine, tmp_path):
    source, calls = engine
    settings = Settings(gradients=True, max_rss_mb=12000)
    original_env = dict(os.environ)
    serial = check.run_check(
        source, tmp_path / "serial", baseline_python=os.sys.executable, settings=settings, jobs=1
    )
    parallel = check.run_check(
        source, tmp_path / "parallel", baseline_python=os.sys.executable, settings=settings, jobs=4
    )
    assert serial["verdict"] == parallel["verdict"] == "SAFE"
    assert serial["accuracy_shift_replicated"] == parallel["accuracy_shift_replicated"]
    assert serial["accuracy_shift_replicated"]["distribution"] == [
        {"batch": i, "status": "complete", "paired_targets": 1, **encoded(0)} for i in range(3)
    ]
    assert len(calls) == 40 and {row[1] for row in calls} == {12000, 3000}
    assert len({row[2]["TORCHINDUCTOR_CACHE_DIR"] for row in calls}) == 10
    assert dict(os.environ) == original_env
    assert read_json(tmp_path / "parallel/check.json")["allocation"]["effective_jobs"] == 4


def test_resume_preserves_primary_commits_after_diagnostic_interruption(
    engine, tmp_path, monkeypatch
):
    source, calls = engine
    root = tmp_path / "resume"
    inspect = check.inspect_side

    def interrupted(*args, **kwargs):
        if Path(args[0]).name == "b1":
            raise RuntimeError("interrupted diagnostics")
        return inspect(*args, **kwargs)

    monkeypatch.setattr(check, "inspect_side", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        check.run_check(source, root, baseline_python=os.sys.executable, jobs=2)
    assert len(calls) == 20
    monkeypatch.setattr(check, "inspect_side", inspect)
    result = check.run_check(source, root, baseline_python=os.sys.executable, jobs=2, resume=True)
    assert len(calls) == 20 and result["verdict"] == "SAFE"
    with pytest.raises(ValueError, match="settings changed"):
        check.run_check(source, root, baseline_python=os.sys.executable, jobs=1, resume=True)
    with pytest.raises(ValueError, match="settings changed"):
        check.run_check(
            source,
            root,
            baseline_python=os.sys.executable,
            jobs=2,
            verdict_policy="v1",
            resume=True,
        )
    atomic_json(root / "batch-0000/primary-attempt-0001/primary.json", {})
    with pytest.raises(ValueError, match="evidence changed"):
        check.run_check(source, root, baseline_python=os.sys.executable, jobs=2, resume=True)


def test_no_early_stop_and_v1_keep_all_diagnostics(engine, tmp_path):
    source, _ = engine
    for policy, stop in (("v2", False), ("v1", True)):
        result = check.run_check(
            source,
            tmp_path / policy,
            baseline_python=os.sys.executable,
            jobs=2,
            verdict_policy=policy,
            early_stop=stop,
        )
        assert result["accuracy_shift_replicated"]["completed"] == 10
        assert result["accuracy_shift_replicated"]["stopping"] is None


def test_late_primary_failure_is_not_hidden_by_accuracy_stopping(engine, tmp_path, monkeypatch):
    source, calls = engine
    run = check.run_case

    def late_failure(job, settings, directory, **kwargs):
        result = run(job, settings, directory, **kwargs)
        if job.name == "b9" and directory.parent.name == "candidate":
            result["status"] = "numerical_mismatch"
        return result

    def export(job, settings, directory, **kwargs):
        atomic_json(directory / "reproducer.json", kwargs["expected"])

    monkeypatch.setattr(check, "run_case", late_failure)
    monkeypatch.setattr(check_execution, "export_native", export)
    result = check.run_check(source, tmp_path / "late", baseline_python=os.sys.executable, jobs=4)
    assert result["verdict"] == "REGRESSION" and len(calls) == 20
    assert result["accuracy_shift_replicated"]["completed"] == 3
    assert any(
        reason["code"] == "candidate_only_failure" and reason["batch"] == 9
        for reason in result["reasons"]
    )


@pytest.mark.parametrize("backend", ["eager", pytest.param("inductor", marks=pytest.mark.compiler)])
def test_actual_workers_match_in_serial_and_parallel(tmp_path, backend):
    from faultline.native import capture_workload

    factory = tmp_path / "model.py"
    factory.write_text(
        "import torch\ndef factory():\n"
        "    torch.manual_seed(7)\n"
        "    return torch.nn.Linear(3, 2).eval(), [(torch.randn(2, 3),) for _ in range(3)]\n",
        encoding="utf-8",
    )
    settings = Settings(backend=backend, repeats=2, timeout_s=120, max_rss_mb=8192)
    bundle = capture_workload(f"{factory}:factory", tmp_path / "bundle", settings, input_batches=3)
    evidence = []
    for jobs in (1, 2):
        root = tmp_path / f"jobs-{jobs}"
        result = check.run_check(
            bundle,
            root,
            baseline_python=os.sys.executable,
            settings=settings,
            level="quick",
            jobs=jobs,
        )
        assert result["verdict"] == "SAFE" and result["scope"]["completed_batches"] == 3
        evidence.append(
            [
                {
                    side: {
                        key: row[side].get(key)
                        for key in ("status", "comparison", "injected_fault")
                    }
                    for side in ("baseline", "candidate")
                }
                for row in read_json(root / "evidence.json")["batches"]
            ]
        )
    assert evidence[0] == evidence[1]
