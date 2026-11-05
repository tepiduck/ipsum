import json
import subprocess
import sys
from dataclasses import replace

import pytest

from faultline import finding
from faultline.campaign import run_campaign
from faultline.cli import main
from faultline.evaluate import Settings, evaluate
from faultline.export import export_case
from faultline.generate import corpus
from faultline.lowering import freeze
from faultline.reduce import minimize
from faultline.storage import atomic_json, read_json


@pytest.fixture
def saved(tmp_path):
    case = freeze(corpus()[0])
    settings = Settings(backend="mutant:sum-as-mean", rtol=2e-4, atol=2e-5, threads=2)
    result = evaluate(case, settings)
    source = tmp_path / "source"
    export_case(case, settings, source, result)
    return source


def test_replay_uses_saved_policy_and_preserves_original_evidence(saved, tmp_path):
    before = {p.name: p.read_bytes() for p in saved.iterdir()}
    output = tmp_path / "replayed"
    assert main(["replay", str(saved), "--output", str(output)]) == 1
    replay = read_json(output / "replay.json")
    assert replay["classification"] == "reproduced"
    assert replay["original_environment_available"]
    assert replay["environment_changes"] == {}
    assert read_json(output / "job.json")["settings"] == read_json(saved / "settings.json")
    assert {p.name: p.read_bytes() for p in saved.iterdir()} == before
    assert finding.load_finding(output).case.digest() == finding.load_finding(saved).case.digest()
    assert main(["replay", str(saved), "--output", str(output)]) == 2


def test_directory_export_and_reduction_retain_the_failure_policy(saved, tmp_path):
    output = tmp_path / "export"
    assert main(["export", str(saved), "--output", str(output)]) == 0
    run = subprocess.run(
        [sys.executable, "-I", str(output / "repro.py")],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert run.returncode == 1, run.stderr
    result = json.loads(run.stdout.splitlines()[-1])
    assert result["reproduces_expected"]
    assert result["rtol"] == 2e-4 and result["atol"] == 2e-5
    assert result["fingerprint"] == read_json(saved / "outcome.json")["fingerprint"]
    reduced = tmp_path / "reduced"
    assert main(["reduce", str(saved), "--output", str(reduced), "--max-evaluations", "2"]) == 0
    assert read_json(reduced / "reproducer" / "settings.json") == read_json(saved / "settings.json")
    assert read_json(reduced / "source.json")["outcome"] == read_json(saved / "outcome.json")


@pytest.mark.parametrize("command", ["reduce", "export"])
def test_saved_settings_cannot_be_accidentally_overridden(saved, tmp_path, command):
    output = tmp_path / "unused"
    assert main([command, str(saved), "--output", str(output), "--backend", "eager"]) == 2
    assert not output.exists()


@pytest.mark.parametrize("name", ["case.json", "settings.json", "outcome.json", "environment.json"])
def test_bundle_edits_are_detected(saved, name):
    data = read_json(saved / name)
    if name == "case.json":
        data["inputs"][0]["values"][0] += 1
    elif name == "settings.json":
        data["threads"] += 1  # not otherwise represented in the historical outcome
    elif name == "outcome.json":
        data["fingerprint"] = "unrelated"
    else:
        data["python"] = "changed"
    atomic_json(saved / name, data)
    with pytest.raises(ValueError):
        finding.load_finding(saved)


def test_campaign_case_can_be_loaded_without_a_standalone_export(tmp_path):
    settings = Settings(backend="mutant:sum-as-mean", repeats=2)
    run_campaign(tmp_path, corpus()[:1], settings, progress=None)
    source = next((tmp_path / "cases").iterdir())
    loaded = finding.load_finding(source)
    assert loaded.settings == settings
    assert loaded.environment == read_json(tmp_path / "manifest.json")["environment"]
    job = read_json(source / "job.json")
    job["settings"]["threads"] = 2
    atomic_json(source / "job.json", job)
    with pytest.raises(ValueError, match="settings"):
        finding.load_finding(source)


@pytest.mark.parametrize(
    "status,classification",
    [
        ("pass", "not_observed"),
        ("compile_error", "changed_failure"),
        ("unstable_candidate", "inconclusive"),
    ],
)
def test_replay_distinguishes_absent_changed_and_inconclusive_failures(
    saved, tmp_path, monkeypatch, status, classification
):
    loaded = finding.load_finding(saved)

    def outcome(case, settings, output):
        result = {**loaded.outcome, "status": status, "fingerprint": status}
        atomic_json(output / "outcome.json", result)
        return result

    monkeypatch.setattr(finding, "run_case", outcome)
    previous = {**loaded.environment, "python": "previous Python"}
    replay = finding.replay_finding(replace(loaded, environment=previous), tmp_path / "new")
    assert replay["classification"] == classification
    assert not replay["reproduces_expected"]
    assert replay["environment_changes"]["python"]["before"] == "previous Python"


def test_reduction_rejects_a_new_failure_before_searching(saved):
    loaded = finding.load_finding(saved)
    calls = []

    def oracle(case):
        calls.append(case)
        return {**loaded.outcome, "fingerprint": "different"}

    with pytest.raises(ValueError, match="Saved failure did not reproduce"):
        minimize(loaded.case, oracle, expected=loaded.outcome)
    assert len(calls) == 1


def test_export_preserves_even_an_incomplete_evidence_directory(tmp_path):
    atomic_json(tmp_path / "outcome.json", {"historical": "evidence"})
    with pytest.raises(ValueError, match="overwrite"):
        export_case(corpus()[0], Settings(backend="eager"), tmp_path)
    assert read_json(tmp_path / "outcome.json") == {"historical": "evidence"}
    assert not (tmp_path / "repro.py").exists()


@pytest.mark.parametrize("historical_environment", [None, {"python": "old environment"}])
def test_reexport_does_not_replace_the_original_execution_environment(
    saved, tmp_path, historical_environment
):
    atomic_json(saved / "environment.json", historical_environment)
    finding.seal_bundle(saved)
    target = tmp_path / "reexport"
    assert main(["export", str(saved), "--output", str(target)]) == 0
    loaded = finding.load_finding(target)
    assert loaded.environment == historical_environment
    assert loaded.outcome == read_json(saved / "outcome.json")
