import copy
import os

import pytest
from test_check_parallel import engine as coordinator_engine

from faultline import check, check_execution, check_retry
from faultline.check_cli import terminal
from faultline.evaluate import Settings
from faultline.storage import read_json

engine = coordinator_engine


@pytest.mark.parametrize("unstable", ["unstable_reference", "unstable_candidate"])
def test_gpu_retry_pairs_both_environments_and_preserves_resume(
    engine, monkeypatch, tmp_path, unstable
):
    source, calls = engine
    original = check.run_case
    executions, diagnostics = [], []
    inspect = check.inspect_side

    def run(job, settings, directory, **kwargs):
        executions.append((job.name, directory.parent.name, settings, kwargs["env"]))
        result = original(job, settings, directory, **kwargs)
        if job.name == "b5" and not settings.deterministic and directory.parent.name == "candidate":
            result.update(status=unstable, peak_tree_rss_mb=4500)
        return result

    def diagnostic(bundle, directory, settings, *args, **kwargs):
        diagnostics.append((bundle.name, settings.deterministic))
        return inspect(bundle, directory, settings, *args, **kwargs)

    monkeypatch.setattr(check, "run_case", run)
    monkeypatch.setattr(check, "inspect_side", diagnostic)
    root = tmp_path / "retry"
    options = {
        "baseline_python": os.sys.executable,
        "settings": Settings(device="cuda:0", max_rss_mb=12000),
        "jobs": 4,
        "early_stop": False,
    }
    verdict = check.run_check(source, root, **options)
    assert verdict["verdict"] == "SAFE" and len(calls) == 22
    retried = [e for e in executions if e[2].deterministic]
    assert len(retried) == 2 and {e[1] for e in retried} == {"baseline", "candidate"}
    assert all(e[0] == "b5" and e[2].max_rss_mb == 12000 for e in retried)
    assert all("deterministic" in e[3]["TORCHINDUCTOR_CACHE_DIR"] for e in retried)
    assert all(flag == (name == "b5") for name, flag in diagnostics)
    row = read_json(root / "evidence.json")["batches"][5]
    assert row["deterministic_retry"]["original_results"]["candidate"]["status"] == unstable
    assert row["candidate"]["status"] == "pass"
    initial = root / "batch-0005/primary-attempt-0001/primary.json"
    preserved = initial.read_bytes()
    assert read_json(initial)["candidate"]["status"] == unstable
    assert verdict["metrics"]["peak_worker_rss_mb"] == 4500
    assert check_retry.MESSAGE in terminal(verdict, root)
    assert "GPU deterministic retries" in (root / "report.html").read_text(encoding="utf-8")
    resumed = check.run_check(source, root, resume=True, **options)
    assert resumed["verdict"] == "SAFE" and len(calls) == 22 and initial.read_bytes() == preserved
    with pytest.raises(ValueError, match="settings changed"):
        check.run_check(source, root, resume=True, deterministic_retry=False, **options)
    retry_file = root / "batch-0005/deterministic-retry-0001/primary.json"
    retry_file.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence changed"):
        check.run_check(source, root, resume=True, **options)


@pytest.mark.parametrize(
    "device,enabled,deterministic",
    [("cpu", True, False), ("cuda:0", False, False), ("cuda:0", True, True)],
)
def test_cpu_opt_out_and_explicit_deterministic_do_not_retry(
    engine, monkeypatch, tmp_path, device, enabled, deterministic
):
    source, calls = engine
    original = check.run_case
    monkeypatch.setattr(
        check, "run_case", lambda *a, **kw: {**original(*a, **kw), "status": "unstable_reference"}
    )
    result = check.run_check(
        source,
        tmp_path / "check",
        level="quick",
        settings=Settings(device=device, deterministic=deterministic),
        deterministic_retry=enabled,
        jobs=1,
    )
    assert result["verdict"] == "INCONCLUSIVE" and len(calls) == 3
    assert not list((tmp_path / "check").glob("batch-*/deterministic-retry-*"))


@pytest.mark.parametrize(
    "retry_status,error",
    [
        ("unstable_reference", ""),
        ("unstable_candidate", ""),
        ("compile_error", "aten::scatter_add_cuda does not have a deterministic implementation"),
        ("reference_error", "aten::cumsum_cuda does not have a deterministic implementation"),
    ],
)
def test_failed_retry_is_inconclusive_and_raw_status_and_operation_are_retained(
    engine, monkeypatch, tmp_path, retry_status, error
):
    source, calls = engine
    original = check.run_case

    def run(job, settings, directory, **kwargs):
        result = original(job, settings, directory, **kwargs)
        if directory.parent.name == "candidate":
            result["injected_fault"] = True  # Only the unavailable candidate is a controlled fault.
            result.update(
                status=retry_status if settings.deterministic else "unstable_reference", error=error
            )
            if settings.deterministic and retry_status == "unstable_candidate":
                result["comparison"]["outputs"] = [{"matches": False, "target": "output"}]
        return result

    monkeypatch.setattr(check, "run_case", run)
    root = tmp_path / "check"
    result = check.run_check(
        source,
        root,
        baseline_python=os.sys.executable,
        level="quick",
        settings=Settings(device="cuda:0"),
        jobs=1,
    )
    assert result["verdict"] == "INCONCLUSIVE" and len(calls) == 12
    assert result["injected_fault"]
    row = read_json(root / "evidence.json")["batches"][0]
    assert "candidate" not in row
    assert row["deterministic_retry"]["outcomes"]["candidate"]["status"] == retry_status
    if error:
        assert error in terminal(result, root)


def test_stable_failure_after_retry_still_regresses_and_exports_deterministic(
    engine, monkeypatch, tmp_path
):
    source, _ = engine
    original = check.run_case
    exports = []

    def run(job, settings, directory, **kwargs):
        result = original(job, settings, directory, **kwargs)
        if directory.parent.name == "candidate":
            result["status"] = (
                "numerical_mismatch" if settings.deterministic else "unstable_reference"
            )
        return result

    monkeypatch.setattr(check, "run_case", run)
    monkeypatch.setattr(
        check_execution, "export_native", lambda job, settings, *a, **k: exports.append(settings)
    )
    result = check.run_check(
        source,
        tmp_path / "check",
        baseline_python=os.sys.executable,
        level="quick",
        settings=Settings(device="cuda:0"),
        jobs=1,
    )
    assert result["verdict"] == "REGRESSION" and exports and all(s.deterministic for s in exports)


def test_retry_does_not_hide_unrelated_compiler_error():
    result = {"status": "compile_error", "error": "deterministic test: invalid generated code"}
    saved = copy.deepcopy(result)
    assert check_retry.unavailable(result) is None and result == saved


def test_gpu_retry_serial_parallel_decisions_agree(engine, monkeypatch, tmp_path):
    source, _ = engine
    run = check.run_case

    def unstable(job, settings, directory, **kwargs):
        result = run(job, settings, directory, **kwargs)
        if job.name in {"b0", "b5"} and not settings.deterministic:
            result["status"] = "unstable_reference"
        return result

    monkeypatch.setattr(check, "run_case", unstable)
    results = [
        check.run_check(
            source,
            tmp_path / str(jobs),
            baseline_python=os.sys.executable,
            settings=Settings(device="cuda:0", gradients=True, max_rss_mb=12000),
            jobs=jobs,
        )
        for jobs in (1, 4)
    ]
    assert results[0]["verdict"] == results[1]["verdict"] == "SAFE"
    assert results[0]["accuracy_shift_replicated"] == results[1]["accuracy_shift_replicated"]
    assert results[0]["notes"] == results[1]["notes"]


def test_completed_gpu_retry_is_reused_after_diagnostic_interruption(engine, monkeypatch, tmp_path):
    source, calls = engine
    run, inspect = check.run_case, check.inspect_side

    def unstable(job, settings, directory, **kwargs):
        result = run(job, settings, directory, **kwargs)
        if job.name == "b0" and not settings.deterministic:
            result["status"] = "unstable_reference"
        return result

    def interrupted(*args, **kwargs):
        raise RuntimeError("interrupted diagnostics")

    monkeypatch.setattr(check, "run_case", unstable)
    monkeypatch.setattr(check, "inspect_side", interrupted)
    options = {
        "baseline_python": os.sys.executable,
        "settings": Settings(device="cuda:0"),
        "jobs": 1,
    }
    root = tmp_path / "resume"
    with pytest.raises(RuntimeError, match="interrupted diagnostics"):
        check.run_check(source, root, **options)
    assert len(calls) == 22
    monkeypatch.setattr(check, "inspect_side", inspect)
    result = check.run_check(source, root, resume=True, **options)
    assert result["verdict"] == "SAFE" and len(calls) == 22


def test_retry_opt_out_cli_argument():
    import argparse

    from faultline.check_cli import add_check_command

    parser = argparse.ArgumentParser()
    add_check_command(parser.add_subparsers())
    assert parser.parse_args(["check", "--no-deterministic-retry"]).no_deterministic_retry
