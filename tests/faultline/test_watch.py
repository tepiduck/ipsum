import argparse

import pytest

from faultline import verdict_bisect, watch
from faultline.evaluate import Settings
from faultline.storage import atomic_json
from faultline.watch_support import (
    catalog,
    execute_check,
    render_history,
    validate_catalog,
    wheel_index,
)


def wheels():
    return [
        {
            "date": f"2026-11-{i:02d}",
            "version": f"2.16.0.dev202611{i:02d}+cpu",
            "url": f"https://download.pytorch.org/nightly{i}.whl",
        }
        for i in range(1, 5)
    ]


def checked(destination, verdict="SAFE"):
    result = {
        "complete": True,
        "verdict": verdict,
        "reasons": [],
        "notes": [],
        "scope": {},
        "metrics": {},
        "accuracy_shift_replicated": {"accuracy_shift_replicated": verdict == "REGRESSION"},
        "compilation_structure": {"labels": ["new_graph_breaks"]},
        "reproducers": [],
    }
    for name in ("check.json", "evidence.json", "verdict.json"):
        atomic_json(destination / name, result)
    (destination / "report.html").write_text(verdict)
    return result


@pytest.fixture
def engine(monkeypatch, tmp_path):
    bundle = tmp_path / "inputs"
    atomic_json(bundle / "batches.json", {"planned": 10, "batches": []})
    model = {"id": "small", "bundle": str(bundle)}
    calls = []
    monkeypatch.setattr(watch, "interpreter", str)
    monkeypatch.setattr(watch, "environment_identity", lambda *a: {"torch": "2.15.0+cpu"})
    monkeypatch.setattr(watch, "discover", lambda *a: {"wheels": wheels(), "missing_dates": []})
    monkeypatch.setattr(watch, "prepared_model", lambda *a: bundle)
    monkeypatch.setattr(
        watch,
        "ensure_wheel",
        lambda root, wheel, *a: {
            "python": wheel["date"],
            "identity": {"torch_git_version": wheel["date"]},
        },
    )

    def execute(bundle, destination, baseline, candidate, *args):
        calls.append((baseline, candidate))
        return checked(destination)

    monkeypatch.setattr(watch, "execute_check", execute)
    return (
        {"models": [model], "baseline_python": "baseline", "max_rss_mb": 1024, "progress": None},
        calls,
        bundle,
    )


def test_watch_two_rails_append_resume_and_tamper_detection(tmp_path, engine):
    options, calls, _ = engine
    root = tmp_path / "watch"
    result = watch.run_watch(root, seed_nightlies=2, **options)
    assert len(calls) == 4 and len(result["records"]) == 4
    assert calls == [
        ("baseline", "2026-11-03"),
        ("2026-11-02", "2026-11-03"),
        ("baseline", "2026-11-04"),
        ("2026-11-03", "2026-11-04"),
    ]
    assert all(r["verdict"] == "SAFE" for r in result["records"])
    assert (root / "timeline/small.html").is_file()
    watch.run_watch(root, seed_nightlies=2, **options)
    assert len(calls) == 4
    with pytest.raises(ValueError, match="changed"):
        watch.run_watch(root, seed_nightlies=2, jobs=1, **options)
    target = next((root / "checks").glob("*/*/*/verdict.json"))
    target.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        watch.run_watch(root, seed_nightlies=2, **options)


def test_watch_interruption_resumes_only_uncommitted_pair(tmp_path, engine, monkeypatch):
    options, calls, _ = engine
    original = watch.execute_check
    interrupted = [True]

    def execute(*args):
        if interrupted[0]:
            interrupted[0] = False
            return {"complete": False, "supervision": {"status": "timeout"}}
        return original(*args)

    monkeypatch.setattr(watch, "execute_check", execute)
    root = tmp_path / "watch"
    assert watch.run_watch(root, **options)["state"] == "check_interrupted_resume_required"
    monkeypatch.setattr(
        watch,
        "discover",
        lambda *a: pytest.fail("Interrupted plan must not discover a newer nightly"),
    )
    assert watch.run_watch(root, **options)["state"] == "complete"
    assert len(calls) == 2
    assert list((root / "interruptions").glob("*.json"))
    plan = next((root / "plans").glob("*.json"))
    plan.write_text("{}")
    with pytest.raises(ValueError, match="plan changed"):
        watch.run_watch(root, **options)


def test_watch_regression_uses_last_safe_same_rail(tmp_path, engine, monkeypatch):
    options, _calls, _bundle = engine
    bisections = []

    def execute(bundle, destination, baseline, candidate, *args):
        return checked(destination, "REGRESSION" if candidate == "2026-11-04" else "SAFE")

    def bisect(bundle, root, baseline, start, end, **kwargs):
        bisections.append((baseline, start, end))
        return {"status": "localized_upgrade_regression"}

    monkeypatch.setattr(watch, "execute_check", execute)
    monkeypatch.setattr(verdict_bisect, "run_verdict_bisect", bisect)
    monkeypatch.setattr(verdict_bisect, "export_collection", lambda b, p, s: p / "repro_check.py")
    result = watch.run_watch(tmp_path / "watch", seed_nightlies=2, **options)
    assert bisections == [
        ("baseline", "2026-11-03", "2026-11-04"),
        ("2026-11-03", "2026-11-03", "2026-11-04"),
    ]
    assert sum(r["verdict"] == "REGRESSION" for r in result["records"]) == 2


def test_catalog_validation_gpu_explicit_and_html_escaping(tmp_path):
    path = tmp_path / "models.json"
    atomic_json(path, [{"id": "../escape", "hf": "x"}])
    with pytest.raises(ValueError):
        catalog(path)
    with pytest.raises(ValueError):
        wheel_index("cuda:0")
    assert wheel_index("cuda:0", "https://download.pytorch.org/whl/nightly/rocm7.0")[2] == "rocm"
    with pytest.raises(ValueError):
        wheel_index("cpu", "https://download.pytorch.org/whl/nightly/cu130")
    render_history(
        tmp_path,
        {"catalog": [{"id": "small"}], "source_sha256": "test"},
        [
            {
                "date": "today",
                "model": "small",
                "reference": "stable",
                "verdict": "<script>x</script>",
            }
        ],
        "complete",
    )
    assert "<script>" not in (tmp_path / "timeline/small.html").read_text()


def test_watch_exit_considers_selected_dates_only(tmp_path, monkeypatch):
    parser = argparse.ArgumentParser()
    watch.add_watch_command(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["watch", "--output", str(tmp_path)])
    current = {"date": "2026-11-04", "model": "small", "reference": "stable", "verdict": "SAFE"}
    old = {**current, "date": "2026-11-01", "verdict": "INCONCLUSIVE"}
    monkeypatch.setattr(
        watch,
        "run_watch",
        lambda *a, **kw: {"state": "complete", "records": [old, current], "planned": [current]},
    )
    assert watch.watch_command(args) == 0
    current["verdict"] = "REGRESSION"
    assert watch.watch_command(args) == 1


def test_programmatic_catalog_and_check_policy_fail_explicitly(tmp_path):
    with pytest.raises(ValueError):
        validate_catalog([{"id": 7, "hf": "example"}])
    with pytest.raises(ValueError, match="standard-level"):
        execute_check(tmp_path, tmp_path, "a", "b", Settings(backend="eager"), "auto", 0)


def test_verdict_bisection_confirms_bounds_and_keeps_inconclusive(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    atomic_json(bundle / "batches.json", {"planned": 10})
    monkeypatch.setattr(
        verdict_bisect, "verify_batches", lambda _: {"planned": 10, "batches": list(range(10))}
    )
    monkeypatch.setattr(verdict_bisect, "environment_identity", lambda *a: {"torch": "2.15.0"})
    monkeypatch.setattr(
        verdict_bisect,
        "nightly_catalog",
        lambda *a, **kw: {"wheels": wheels(), "missing_dates": []},
    )
    monkeypatch.setattr(
        verdict_bisect,
        "ensure_wheel",
        lambda root, wheel, *a: {
            "python": wheel["date"],
            "identity": {"torch_git_version": wheel["date"]},
        },
    )
    exported = []
    monkeypatch.setattr(
        verdict_bisect,
        "export_collection",
        lambda b, p, s: exported.append(p) or p / "repro_check.py",
    )
    calls = []

    def execute(bundle, destination, baseline, candidate, *args):
        calls.append(candidate)
        return checked(
            destination,
            "REGRESSION"
            if candidate >= "2026-11-03"
            else "INCONCLUSIVE"
            if candidate == "2026-11-02"
            else "SAFE",
        )

    monkeypatch.setattr(verdict_bisect, "execute_check", execute)
    settings = Settings(gradients=True, repeats=2, threads=1)
    result = verdict_bisect.run_verdict_bisect(
        bundle,
        tmp_path / "bisect",
        "baseline",
        "2026-11-01",
        "2026-11-04",
        settings=settings,
        progress=None,
    )
    assert result["status"] == "inconclusive_boundary"
    assert result["boundary"]["last_SAFE"] == "2026-11-01"
    assert result["boundary"]["first_REGRESSION"] == "2026-11-03"
    assert result["boundary"]["independently_confirmed"]
    assert not result["boundary"]["adjacent_available_builds"]
    assert calls.count("2026-11-01") == calls.count("2026-11-03") == 2
    assert exported
    verdict_bisect.run_verdict_bisect(
        bundle,
        tmp_path / "bisect",
        "baseline",
        "2026-11-01",
        "2026-11-04",
        settings=settings,
        resume=True,
        progress=None,
    )
    assert len(calls) == 6
