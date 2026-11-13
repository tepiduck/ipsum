import copy
import os
import threading
import time

import pytest
from test_check_parallel import engine as coordinator_engine

from faultline import check, check_execution, check_resources
from faultline.evaluate import Settings
from faultline.storage import atomic_json, read_json

engine = coordinator_engine


def test_auto_budget_cap_and_resume_are_explicit(monkeypatch, tmp_path):
    monkeypatch.setattr(check_resources, "available_memory_mb", lambda: 20000)
    first = check_resources.memory_budget(tmp_path)
    assert first["chosen_mb"] == 15000
    monkeypatch.setattr(check_resources, "available_memory_mb", lambda: 100000)
    assert check_resources.memory_budget(tmp_path, resume=True) == first
    assert check_resources.memory_budget(tmp_path / "large")["chosen_mb"] == 32768
    assert check_resources.memory_budget(tmp_path / "explicit", 12345)["chosen_mb"] == 12345
    with pytest.raises(ValueError, match="changed"):
        check_resources.memory_budget(tmp_path, 12000, resume=True)


def test_auto_measures_full_budget_and_selects_fewer_jobs(engine, monkeypatch, tmp_path):
    source, calls = engine
    original = check.run_case
    monkeypatch.setattr(check_execution, "default_jobs", lambda: 4)

    def measured(*args, **kwargs):
        return {**original(*args, **kwargs), "peak_tree_rss_mb": 4000}

    monkeypatch.setattr(check, "run_case", measured)
    result = check.run_check(
        source,
        tmp_path / "auto",
        baseline_python=os.sys.executable,
        settings=Settings(max_rss_mb=12000),
        jobs="auto",
    )
    evidence = read_json(tmp_path / "auto/evidence.json")
    assert result["verdict"] == "SAFE"
    assert evidence["scheduling"]["effective_jobs"] == 2
    assert evidence["scheduling"]["measured_peak_rss_mb"] == 4000
    assert all(limit == (12000 if batch == "b0" else 6000) for batch, limit, _ in calls)
    before = len(calls)
    check.run_check(
        source,
        tmp_path / "auto",
        baseline_python=os.sys.executable,
        settings=Settings(max_rss_mb=12000),
        jobs="auto",
        resume=True,
    )
    assert len(calls) == before
    serial = check.run_check(
        source,
        tmp_path / "serial",
        baseline_python=os.sys.executable,
        settings=Settings(max_rss_mb=12000),
        jobs=1,
    )
    assert serial["accuracy_shift_replicated"] == result["accuracy_shift_replicated"]
    other = read_json(tmp_path / "serial/evidence.json")
    assert [{s: row[s] for s in ("baseline", "candidate")} for row in evidence["batches"]] == [
        {s: row[s] for s in ("baseline", "candidate")} for row in other["batches"]
    ]


@pytest.mark.parametrize("jobs", ["auto", 4])
def test_parallel_memory_limit_retries_after_pool_drains(engine, monkeypatch, tmp_path, jobs):
    source, calls = engine
    original = check.run_case
    active = 0
    lock = threading.Lock()
    monkeypatch.setattr(check_execution, "default_jobs", lambda: 4)

    def limited(job, settings, directory, **kwargs):
        nonlocal active
        with lock:
            active += 1
            if "serial-retry-" in str(directory):
                assert active == 1 and settings.max_rss_mb == 12000
        try:
            time.sleep(0.005)
            result = original(job, settings, directory, **kwargs)
            if job.name == "b5" and settings.max_rss_mb < 12000:
                result["status"] = "resource_limit"
            return {**result, "peak_tree_rss_mb": 2000}
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(check, "run_case", limited)
    root = tmp_path / "retry"
    options = {
        "baseline_python": os.sys.executable,
        "settings": Settings(max_rss_mb=12000),
        "jobs": jobs,
    }
    result = check.run_check(source, root, **options)
    assert result["verdict"] == "SAFE" and len(calls) == 22
    record = read_json(root / "evidence.json")
    assert record["scheduling"]["serial_retries"][0]["batch"] == 5
    assert (
        read_json(root / "batch-0005/primary-attempt-0001/primary.json")["candidate"]["status"]
        == "resource_limit"
    )
    check.run_check(source, root, resume=True, **options)
    assert len(calls) == 22


def test_full_budget_failure_stays_inconclusive(engine, monkeypatch, tmp_path):
    source, _ = engine
    original = check.run_case

    def limited(*args, **kwargs):
        return {**original(*args, **kwargs), "status": "resource_limit", "peak_tree_rss_mb": 12001}

    monkeypatch.setattr(check, "run_case", limited)
    result = check.run_check(
        source,
        tmp_path / "limit",
        baseline_python=os.sys.executable,
        settings=Settings(max_rss_mb=12000),
        jobs="auto",
    )
    assert result["verdict"] == "INCONCLUSIVE"
    assert read_json(tmp_path / "limit/evidence.json")["scheduling"]["effective_jobs"] == 1


def test_auto_diagnostics_get_full_budget_and_explicit_retries_memory(
    engine, monkeypatch, tmp_path
):
    source, _ = engine
    inspect = check.inspect_side
    limits = []

    def diagnostic(bundle, directory, settings, *args, **kwargs):
        limits.append(settings.max_rss_mb)
        if settings.max_rss_mb < 12000:
            return copy.deepcopy(kwargs["primary"]), [{"status": "diagnostic_skipped_memory"}]
        return inspect(bundle, directory, settings, *args, **kwargs)

    monkeypatch.setattr(check, "inspect_side", diagnostic)
    for jobs in ("auto", 4):
        result = check.run_check(
            source,
            tmp_path / str(jobs),
            baseline_python=os.sys.executable,
            settings=Settings(max_rss_mb=12000),
            jobs=jobs,
        )
        assert result["verdict"] == "SAFE"
        if jobs == "auto":
            assert set(limits) == {12000}
    assert set(limits) == {3000, 12000}
    assert any(
        p["status"] == "serial_memory_retry"
        for p in read_json(tmp_path / "4/evidence.json")["batches"][0]["diagnostics"]["candidate"]
    )


def test_generic_crash_is_not_retried_as_memory(engine, monkeypatch, tmp_path):
    source, calls = engine
    original = check.run_case
    monkeypatch.setattr(check, "run_case", lambda *a, **k: {**original(*a, **k), "status": "crash"})
    result = check.run_check(source, tmp_path / "crash", baseline_python=os.sys.executable, jobs=4)
    assert result["verdict"] == "INCONCLUSIVE" and len(calls) == 20


def test_schedule_tampering_is_rejected(engine, tmp_path):
    source, _ = engine
    root = tmp_path / "tamper"
    check.run_check(source, root, baseline_python=os.sys.executable)
    atomic_json(root / "schedule/schedule.json", {"effective_jobs": 4})
    with pytest.raises(ValueError, match="evidence changed"):
        check.run_check(source, root, baseline_python=os.sys.executable, resume=True)


def test_interrupted_memory_retry_keeps_all_primary_commits(engine, monkeypatch, tmp_path):
    source, calls = engine
    original = check.run_case
    interrupt = True

    def worker(job, settings, directory, **kwargs):
        if "serial-retry-" in str(directory) and interrupt:
            directory.mkdir(parents=True)
            atomic_json(directory / "job.json", {"partial": True})
            raise RuntimeError("interrupted serial retry")
        result = original(job, settings, directory, **kwargs)
        if job.name == "b4" and settings.max_rss_mb < 12000:
            result["status"] = "resource_limit"
        return result

    monkeypatch.setattr(check, "run_case", worker)
    root = tmp_path / "interrupted-retry"
    options = {
        "baseline_python": os.sys.executable,
        "settings": Settings(max_rss_mb=12000),
        "jobs": 4,
    }
    with pytest.raises(RuntimeError, match="interrupted serial retry"):
        check.run_check(source, root, **options)
    assert len(calls) == 20
    interrupt = False
    result = check.run_check(source, root, resume=True, **options)
    assert result["verdict"] == "SAFE" and len(calls) == 22
    assert (root / "batch-0004/serial-retry-0002/commit.json").is_file()
