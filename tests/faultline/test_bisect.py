import copy
import json
import sys

import pytest
from packaging.tags import Tag

from faultline import bisect as workflow
from faultline import runner
from faultline.bisect import (
    advisory_classification,
    baseline_errors,
    choose_next,
    commit_result,
    growth_score,
    verify_commit,
)
from faultline.bisect_env import dependency_lock, nightly_catalog, supervise
from faultline.evaluate import Settings
from faultline.storage import atomic_json


def observation(value=1.0):
    # Synthetic accuracy data, not an observed compiler finding.
    return {
        "status": "gradient_mismatch",
        "injected_fault": True,
        "fp64_advisory": {
            "status": "complete",
            "outputs": [
                {
                    "target": "step:0:gradient:weight",
                    "reference_dtype": "torch.float64",
                    "candidate": {"rmse": value},
                }
            ],
        },
    }


def test_score_fresh_repeat_consensus_and_missing():
    original = observation()
    before = copy.deepcopy(original)
    reference = baseline_errors([original, observation(2), observation(3)])
    assert reference == {"step:0:gradient:weight": 2}
    good, shifted = [growth_score(reference, observation(x)) for x in (2, 3)]
    assert advisory_classification([shifted] * 3, 1.4) == "shifted"
    assert advisory_classification([good] * 3, 1.4) == "not_shifted"
    assert advisory_classification([good, shifted], 1.4) == "inconclusive"
    missing = growth_score(reference, {"status": "timeout"})
    assert advisory_classification([missing] * 3, 1.4) == "inconclusive"
    assert original == before
    assert baseline_errors([original, {"status": "crash"}]) is None
    assert growth_score(reference, observation(0))["status"] == "incomplete"


def test_unknown_build_does_not_close_bracket():
    wheels = [{"date": str(i)} for i in range(7)]
    assert choose_next(wheels, {}, 0, 6) == 3
    assert choose_next(wheels, {"3": "inconclusive"}, 0, 6) == 2
    assert choose_next(wheels, {str(i): "inconclusive" for i in range(1, 6)}, 0, 6) is None


def test_catalog_filters_platform_and_records_unavailable_dates():
    prefix = "https://download-r2.pytorch.org/whl/nightly/cpu/"
    doc = "".join(
        f'<a href="{prefix}torch-2.16.0.dev2026110{day}%2Bcpu-{tag}.whl">x</a>'
        for day, tag in [
            (1, "cp312-cp312-manylinux_2_28_x86_64"),
            (2, "cp311-cp311-manylinux_2_28_x86_64"),
        ]
    )
    catalog = nightly_catalog(
        "2026-11-01",
        "2026-11-03",
        document=doc,
        tags=[Tag("cp312", "cp312", "manylinux_2_28_x86_64")],
    )
    assert len(catalog["wheels"]) == 1
    assert catalog["wheels"][0]["wheel_sha256"] is None
    assert catalog["missing_dates"] == ["2026-11-02", "2026-11-03"]


@pytest.mark.parametrize(
    "line",
    [
        "torch==2.15",
        "numpy>=2",
        "numpy==2.*",
        "--index-url x",
        "numpy @ https://example.org/a.whl",
        "numpy==2; os_name=='nt'",
    ],
)
def test_dependencies_must_be_pinned_and_cannot_replace_torch(line):
    with pytest.raises(ValueError):
        dependency_lock(line)


def test_committed_evidence_tampering_is_rejected(tmp_path):
    (tmp_path / "outcome.json").write_text(json.dumps(observation()))
    commit_result(tmp_path, {"classification": "shifted"}, ["outcome.json"])
    assert verify_commit(tmp_path)["classification"] == "shifted"
    (tmp_path / "outcome.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        verify_commit(tmp_path)


def test_supervised_installer_timeout_and_failure_are_distinct(tmp_path):
    timeout = supervise(
        [sys.executable, "-c", "import time; time.sleep(30)"], tmp_path / "slow", timeout=0.2
    )
    failed = supervise([sys.executable, "-c", "raise SystemExit(7)"], tmp_path / "failed")
    assert timeout["status"] == "timeout"
    assert failed["status"] == "command_failed" and failed["returncode"] == 7


def test_linux_parent_identity_survives_clock_correction_and_rejects_pid_reuse(monkeypatch):
    monkeypatch.setattr(runner.sys, "platform", "linux")
    ticks = [12345]
    monkeypatch.setattr(
        runner.Path,
        "read_text",
        lambda self: "50 (name with ) spaces) " + " ".join(["S"] + ["0"] * 18 + [str(ticks[0])]),
    )

    class Parent:
        def is_running(self):
            return True

        def status(self):
            return "running"

        def create_time(self):
            raise AssertionError("Linux identity must not depend on corrected wall-clock time")

    monkeypatch.setattr(runner.psutil, "Process", lambda pid: Parent())
    identity = runner.process_identity(50)
    assert runner.parent_alive(50, identity)
    ticks[0] += 1
    assert not runner.parent_alive(50, identity)


@pytest.mark.parametrize("installation_failure", [False, True])
@pytest.mark.parametrize("interrupted_copy", [False, True])
def test_bisection_confirms_neighbors_resumes_and_rejects_changed_evidence(
    tmp_path, monkeypatch, installation_failure, interrupted_copy
):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    atomic_json(bundle / "workload.json", {"fixture": "explicit synthetic accuracy shift"})
    wheels = [{"date": f"2026-11-{i + 1:02d}", "index": i} for i in range(7)]
    monkeypatch.setattr(
        workflow, "nightly_catalog", lambda *_: {"wheels": wheels, "missing_dates": []}
    )
    monkeypatch.setattr(workflow, "verify_native", lambda _: None)
    copy_attempts = []

    def copy_workload(a, b):
        copy_attempts.append(b)
        if interrupted_copy and len(copy_attempts) == 1:
            b.mkdir()
            (b / "partial").write_text("interrupted fixture")
            raise RuntimeError("controlled interruption while copying")
        return __import__("shutil").copytree(a, b)

    monkeypatch.setattr(workflow, "copy_workload", copy_workload)
    monkeypatch.setattr(workflow.NativeJob, "create", lambda _: "fixture-job")
    monkeypatch.setattr(workflow, "provenance", lambda: {"faultline_source_sha256": "fixed-source"})
    monkeypatch.setattr(workflow, "environment_identity", lambda *_: {"identity": "fixed"})

    def environment(root, wheel, requirements, **kwargs):
        if installation_failure:
            return {"status": "installation_inconclusive", "stage": "create"}
        return {
            "status": "ready",
            "python": str(wheel["index"]),
            "identity": {"torch_git_version": str(wheel["index"]) * 40},
        }

    monkeypatch.setattr(workflow, "ensure_environment", environment)
    calls = []

    def run(job, settings, directory, *, python):
        calls.append((directory, python))
        result = observation(2 if python.isdigit() and int(python) >= 3 else 1)
        atomic_json(directory / "outcome.json", result)
        return result

    monkeypatch.setattr(workflow, "run_case", run)
    output = tmp_path / "bisect"
    args = (bundle, output, sys.executable, "2026-11-01", "2026-11-07")
    if interrupted_copy:
        with pytest.raises(RuntimeError, match="controlled interruption"):
            workflow.run_bisect(*args, observations=2, progress=None)
        assert (copy_attempts[0] / "partial").exists()
    result = workflow.run_bisect(*args, observations=2, progress=None, resume=interrupted_copy)
    if installation_failure:
        assert result["status"] == "no_confirmed_bracket" and len(calls) == 2
        unchanged = workflow.run_bisect(*args, observations=2, progress=None, resume=True)
        assert unchanged["status"] == "no_confirmed_bracket" and len(calls) == 2
        installation_failure = False
        result = workflow.run_bisect(
            *args, observations=2, progress=None, resume=True, retry_installations=True
        )
        assert len(list((output / "search" / wheels[0]["date"]).glob("attempt-*"))) == 2
    assert result["status"] == "localized_accuracy_shift"
    assert result["boundary"]["last_observed_not_shifted"] == "2026-11-03"
    assert result["boundary"]["first_observed_shifted"] == "2026-11-04"
    assert result["boundary"]["independently_confirmed"]
    assert len([p for p, _ in calls if "confirmation" in p.parts]) == 4
    count = len(calls)
    resumed = workflow.run_bisect(*args, observations=2, progress=None, resume=True)
    assert len(calls) == count and resumed["boundary"] == result["boundary"]
    with pytest.raises(ValueError, match="changed"):
        workflow.run_bisect(
            *args,
            observations=2,
            resume=True,
            progress=None,
            settings=Settings(gradients=True, fp64_advisory=True, threads=2),
        )
    (calls[0][0] / "outcome.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        workflow.run_bisect(*args, observations=2, progress=None, resume=True)
