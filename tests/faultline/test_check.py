"""Summary-layer behavior; planted outcomes are explicitly controlled fixtures."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

from faultline import check
from faultline.check_cli import terminal
from faultline.evaluate import Settings
from faultline.native import capture_workload
from faultline.storage import atomic_json, read_json
from faultline.verdict import assessment, summarize


def test_environment_probe_error_keeps_actionable_stderr():
    from faultline.check_cli import setup_help

    error = subprocess.CalledProcessError(
        1,
        ["python", "-c", "a long bootstrap program"],
        stderr="Traceback ...\nModuleNotFoundError: No module named 'torch'\n",
    )
    message = setup_help(error)
    assert "ModuleNotFoundError: No module named 'torch'" in message
    assert "install PyTorch" in message
    assert "bootstrap program" not in message


def outcome(status="pass", explained=False, contrary=False):
    target = "step:0:gradient:parameter:weight:value"
    result = {"status": status, "injected_fault": True, "comparison": {"outputs": []}}
    if status != "pass":
        result["comparison"]["outputs"] = [{"target": target, "matches": False}]
    if explained or contrary:
        result["numerical_context"] = {
            "labels": ["comparable_to_eager_against_fp64"],
            "label_coverage": {"comparable_to_eager_against_fp64": {"targets": [target]}},
            "triage": {
                "classification": "explained" if not contrary else "unexplained_contrary_evidence",
                "contrary_evidence": ["controlled contrary evidence"] if contrary else [],
            },
        }
    return result


@pytest.mark.parametrize(
    "base,candidate,expected",
    [
        ("pass", "pass", "SAFE"),
        ("pass", "numerical_mismatch", "REGRESSION"),
        ("numerical_mismatch", "numerical_mismatch", "NEEDS_REVIEW"),
        ("timeout", "pass", "INCONCLUSIVE"),
        ("pass", "unstable", "INCONCLUSIVE"),
    ],
)
def test_verdict_paths(base, candidate, expected):
    result = summarize(
        [{"batch": 0, "baseline": outcome(base), "candidate": outcome(candidate)}],
        1,
        mode="environment_comparison",
        level="quick",
    )
    assert result["verdict"] == expected
    assert result["injected_fault"]
    schema = read_json(Path(__file__).parents[2] / "docs/verdict.schema.json")
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(result, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**result, "exit_code": (result["exit_code"] + 1) % 4}, schema)
    assert len(terminal(result, Path("report")).splitlines()) <= 15


def test_explanation_requires_full_coverage_and_no_contrary_evidence():
    candidate = outcome("gradient_mismatch", explained=True)
    saved = copy.deepcopy(candidate)
    result = summarize(
        [{"batch": 0, "candidate": candidate}], 1, mode="single_environment", level="standard"
    )
    assert result["verdict"] == "SAFE" and result["notes"]
    assert candidate == saved
    candidate["comparison"]["outputs"].append(
        {"target": "step:0:gradient:parameter:bias:value", "matches": False}
    )
    assert not assessment(candidate)["explained"]
    assert not assessment(outcome("gradient_mismatch", True, True))["acceptable"]
    assert not assessment(outcome("numerical_mismatch", True))["explained"]


def test_replication_required_for_paired_standard_but_not_single_environment():
    rows = [{"batch": 0, "baseline": outcome(), "candidate": outcome()}]
    result = summarize(rows, 1, mode="environment_comparison", level="standard")
    assert result["verdict"] == "NEEDS_REVIEW"
    assert result["reasons"][0]["code"] == "replication_unavailable"
    result = summarize(rows, 1, mode="single_environment", level="standard")
    assert result["verdict"] == "SAFE"
    result = summarize(
        rows,
        1,
        mode="environment_comparison",
        level="standard",
        replication={"accuracy_shift_replicated": True},
    )
    assert result["verdict"] == "REGRESSION"


def test_missing_batches_remain_inconclusive_even_if_present_rows_pass():
    result = summarize(
        [{"batch": 0, "candidate": outcome()}], 3, mode="single_environment", level="quick"
    )
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["scope"]["planned_batches"] == 3


@pytest.mark.parametrize(
    "args,code",
    [
        (["check", "--invalid"], 3),
        (["check", "--level", "invalid"], 3),
        (["model", "--invalid"], 2),
    ],
)
def test_argument_errors_use_new_codes_only_for_check(args, code):
    from faultline.cli import main

    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == code


@pytest.mark.parametrize("count", [-1, 0, 101])
def test_invalid_batch_count_keeps_inconclusive_exit_and_valid_report(tmp_path, count):
    from faultline.cli import main

    output = tmp_path / "report"
    assert (
        main(["check", "model.py:factory", "--batches", str(count), "--output", str(output)]) == 3
    )
    verdict = read_json(output / "verdict.json")
    assert verdict["verdict"] == "INCONCLUSIVE"
    assert verdict["reasons"][0]["code"] == "setup_error"
    jsonschema.validate(verdict, read_json(Path(__file__).parents[2] / "docs/verdict.schema.json"))


def test_unwritable_report_keeps_inconclusive_exit_and_existing_file(tmp_path, capsys):
    from faultline.cli import main

    output = tmp_path / "existing-file"
    output.write_text("retain this file", encoding="utf-8")
    assert main(["check", "--output", str(output)]) == 3
    assert output.read_text(encoding="utf-8") == "retain this file"
    assert "writable output directory" in capsys.readouterr().err


def test_diagnostic_memory_failure_preserves_primary(monkeypatch, tmp_path):
    primary = outcome("gradient_mismatch")
    saved = copy.deepcopy(primary)
    monkeypatch.setattr(check.NativeJob, "create", lambda *a, **kw: object())
    monkeypatch.setattr(
        check,
        "run_case",
        lambda *a, **kw: primary if a[2].name == "primary" else {"status": "resource_limit"},
    )
    monkeypatch.setattr(
        check, "diagnostic", lambda *a, **kw: {"status": "diagnostic_skipped_memory"}
    )
    observed, phases = check.inspect_side(
        tmp_path, tmp_path / "run", Settings(), sys.executable, "standard", paired=False
    )
    assert observed == saved == primary
    assert all(p["status"] == "diagnostic_skipped_memory" for p in phases)
    result = summarize(
        [{"batch": 0, "candidate": observed, "diagnostics": {"candidate": phases}}],
        1,
        mode="single_environment",
        level="standard",
    )
    assert result["verdict"] == "NEEDS_REVIEW"
    assert all(n["code"] == "diagnostic_skipped_memory" for n in result["notes"])


def test_fp64_allocator_failure_is_memory_skip_not_an_explanation():
    from faultline.check_diagnostics import memory_diagnostic

    raw = {
        "status": "complete",
        "replay_matches_primary": True,
        "fp64_advisory": {
            "status": "reference_error",
            "error": "OutOfMemoryError: controlled GPU allocation failure",
        },
    }
    result = memory_diagnostic(raw)
    assert result["status"] == "diagnostic_skipped_memory"
    assert result["fp64_advisory"] == raw["fp64_advisory"]
    primary = outcome("gradient_mismatch")
    assert check.enriched(primary, result) == primary
    raw["fp64_advisory"]["error"] = "RuntimeError: unsupported fp64 operation"
    assert memory_diagnostic(raw)["status"] == "complete"


def test_quick_never_runs_diagnostics(monkeypatch, tmp_path):
    monkeypatch.setattr(check.NativeJob, "create", lambda *a, **kw: object())
    monkeypatch.setattr(check, "run_case", lambda *a, **kw: outcome("gradient_mismatch"))
    monkeypatch.setattr(check, "diagnostic", lambda *a, **kw: pytest.fail("quick diagnostic"))
    _, phases = check.inspect_side(
        tmp_path, tmp_path / "run", Settings(), sys.executable, "quick", paired=True
    )
    assert phases == []
    assert check.PRESETS == {
        "quick": {"batches": 3, "gradients": False, "repeats": 2},
        "standard": {"batches": 10, "gradients": True, "repeats": 2},
        "thorough": {"batches": 10, "gradients": True, "repeats": 3},
    }


def test_completed_contrary_control_does_not_trigger_unneeded_standard_masks(monkeypatch, tmp_path):
    primary = outcome("gradient_mismatch")
    contrary = outcome("gradient_mismatch", contrary=True)
    calls = []
    monkeypatch.setattr(check.NativeJob, "create", lambda *a, **kw: object())

    def run(*args, **kwargs):
        assert args[2].name == "primary", (
            "resolved contrary evidence does not need mask explanation"
        )
        return primary

    def diagnostic(*args, **kwargs):
        calls.append(kwargs.get("control"))
        return {
            "status": "complete",
            "replay_matches_primary": True,
            "numerical_context": contrary["numerical_context"],
        }

    monkeypatch.setattr(check, "run_case", run)
    monkeypatch.setattr(check, "diagnostic", diagnostic)
    result, phases = check.inspect_side(
        tmp_path, tmp_path / "run", Settings(), sys.executable, "standard", paired=False
    )
    assert calls == [False, True]
    assert assessment(result)["contrary"] and len(phases) == 2


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("check-workload")
    factory = root / "model.py"
    factory.write_text(
        "import torch\ndef factory():\n    return torch.nn.ReLU().eval(), (torch.tensor([[-2., 3.], [4., -1.]]),)\n",
        encoding="utf-8",
    )
    return capture_workload(
        f"{factory}:factory", root / "bundle", Settings(backend="eager", repeats=2)
    )


def test_compact_diagnostics_verify_replay_without_duplicate_graph_tensors(
    bundle, tmp_path, monkeypatch
):
    from faultline import native
    from faultline.check_diagnostics import diagnose

    settings = Settings(backend="mutant:gradient-scale", repeats=2, gradients=True)
    primary = check.run_case(native.NativeJob.create(bundle), settings, tmp_path / "primary")
    assert primary["status"] == "gradient_mismatch" and primary["injected_fault"]
    retained = copy.deepcopy(primary)
    saves = []
    original = native.save_native_graph

    def save(*args, **kwargs):
        saves.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(native, "save_native_graph", save)
    result = diagnose(bundle, settings, primary, tmp_path / "diagnostic", control=True)
    assert result["status"] == "complete" and result["replay_matches_primary"]
    assert result["fp64_advisory"]["status"] == "complete"
    assert result["fp64_advisory"]["outputs"] and result["control_requested"]
    assert not saves and not list((tmp_path / "diagnostic").rglob("tensors.pt"))
    assert primary == retained


@pytest.mark.parametrize(
    "baseline,backend,expected,code",
    [
        (False, "eager", "SAFE", 0),
        (True, "mutant:relu-as-neg", "REGRESSION", 1),
        (False, "mutant:relu-as-neg", "NEEDS_REVIEW", 2),
        (False, "eager", "INCONCLUSIVE", 3),
    ],
)
def test_check_cli_end_to_end(bundle, tmp_path, baseline, backend, expected, code, capsys):
    from faultline.cli import main

    output = tmp_path / "check"
    args = ["check", str(bundle), "--level", "quick", "--backend", backend, "--output", str(output)]
    if baseline:
        args += ["--baseline", sys.executable, "--baseline-backend", "eager"]
    if expected == "INCONCLUSIVE":
        args += ["--candidate", str(tmp_path / "missing-python")]
    assert main(args) == code
    assert read_json(output / "verdict.json")["verdict"] == expected
    assert len(capsys.readouterr().out.splitlines()) <= 15
    assert (output / "report.html").is_file()
    if backend.startswith("mutant:"):
        record = read_json(output / "reproducer/reproducer.json")
        assert record["expected"]["injected_fault"]
    json.dumps(read_json(output / "verdict.json"), allow_nan=False)


def test_resume_checks_artifact_integrity(bundle, tmp_path, monkeypatch):
    from faultline import sweep

    monkeypatch.setattr(sweep, "runtime_environment", lambda p: {"python": p})
    monkeypatch.setattr(check, "inspect_side", lambda *a, **kw: (outcome(), []))
    root = tmp_path / "run"
    result = check.run_check(bundle, root, level="quick")
    assert result["verdict"] == "SAFE"
    monkeypatch.setattr(
        check, "inspect_side", lambda *a, **kw: pytest.fail("reran committed worker")
    )
    assert check.run_check(bundle, root, level="quick", resume=True)["verdict"] == "SAFE"
    atomic_json(root / "batch-0000/attempt-0001/evidence.json", {})
    with pytest.raises(ValueError, match="evidence changed"):
        check.run_check(bundle, root, level="quick", resume=True)


def test_seeded_factory_captures_distinct_inputs_with_fixed_weights(tmp_path):
    from faultline.batches import verify_batches
    from faultline.check_prepare import prepare

    factory = tmp_path / "seeded.py"
    factory.write_text(
        "import torch\ndef factory(seed=0):\n"
        "    torch.manual_seed(42)\n"
        "    model = torch.nn.Linear(2, 2).eval()\n"
        "    inputs = torch.randn(2, 2, generator=torch.Generator().manual_seed(seed))\n"
        "    return model, (inputs,)\n",
        encoding="utf-8",
    )
    prepare(
        {"factory": f"{factory}:factory", "batches": 2, "timeout": 60, "max_rss_mb": 2048},
        tmp_path / "prepared",
    )
    manifest = verify_batches(tmp_path / "prepared/bundle")
    assert len(manifest["batches"]) == 2
    assert len({row["inputs_sha256"] for row in manifest["batches"]}) == 2
    assert len({row["weights_sha256"] for row in manifest["batches"]}) == 1
    assert read_json(tmp_path / "prepared/bundle/batch-0001/workload.json")["factory_seed"] == 1


def test_one_batch_collection_keeps_argument_structure(tmp_path):
    from faultline.check_prepare import prepare

    factory = tmp_path / "one.py"
    factory.write_text(
        "import torch\ndef factory():\n    return torch.nn.ReLU().eval(), [(torch.tensor([[1.],[-1.]]),)]\n",
        encoding="utf-8",
    )
    prepare(
        {"factory": f"{factory}:factory", "batches": 10, "timeout": 60, "max_rss_mb": 2048},
        tmp_path / "prepared",
    )
    assert read_json(tmp_path / "prepared/prepared.json")["input_batches"] == 1
    result = check.run_check(
        tmp_path / "prepared/bundle",
        tmp_path / "check",
        level="quick",
        settings=Settings(backend="eager", repeats=2),
    )
    assert result["verdict"] == "SAFE"
