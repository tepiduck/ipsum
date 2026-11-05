import json
import subprocess
import sys
from dataclasses import replace

import pytest

from faultline import compare_envs
from faultline.cli import main
from faultline.evaluate import Settings, evaluate
from faultline.generate import corpus
from faultline.lowering import freeze


@pytest.mark.parametrize(
    "a,b,expected,code",
    [
        ("pass", "pass", "agreement", 0),
        ("pass", "numerical_mismatch", "regression", 1),
        ("numerical_mismatch", "pass", "not_observed", 0),
        ("timeout", "pass", "inconclusive", 2),
        ("pass", "unsupported_device", "inconclusive", 2),
        ("runtime_error", "runtime_error", "recurring", 0),
        ("runtime_error", "compile_error", "changed_failure", 1),
    ],
)
def test_upgrade_gate_classifications(a, b, expected, code):
    label = compare_envs.classification(
        {"status": a, "fingerprint": "x"}, {"status": b, "fingerprint": "x"}
    )
    assert label == expected
    assert compare_envs.gate_status({label: 1}) == code


def test_reduction_rejects_candidates_that_also_break_the_baseline(tmp_path, monkeypatch):
    case = freeze(corpus()[0])
    baseline = Settings(backend="eager", repeats=2)
    candidate = replace(baseline, backend="mutant:sum-as-mean")
    expected = evaluate(case, candidate)
    calls = []

    def worker(proposal, settings, directory, **kwargs):
        calls.append((proposal.digest(), settings.backend))
        if settings.backend == "eager" and proposal.digest() != case.digest():
            return {"status": "runtime_error", "fingerprint": "baseline-broke"}
        return evaluate(proposal, settings)

    monkeypatch.setattr(compare_envs, "run_case", worker)
    evidence = compare_envs.reduce_regression(
        case,
        baseline,
        candidate,
        sys.executable,
        sys.executable,
        tmp_path / "reduction",
        expected,
        12,
        progress=None,
    )
    assert evidence["original_complexity"] == evidence["reduced_complexity"]
    assert evidence["baseline_confirmation"]["status"] == "pass"
    assert all(digest == case.digest() for digest, backend in calls if backend != "eager")
    assert any(item["status"] == "baseline_rejected" for item in evidence["history"])


def test_upgrade_gate_runs_real_workers_and_exports_reproducers(tmp_path):
    settings = Settings(backend="eager", repeats=2)
    result = compare_envs.compare_environments(
        corpus()[:1],
        tmp_path / "comparison",
        sys.executable,
        sys.executable,
        settings,
        replace(settings, backend="mutant:sum-as-mean"),
        reduce=True,
        max_evaluations=4,
        progress=None,
    )
    assert result["counts"] == {"regression": 1}
    assert result["exit_code"] == 1
    item = result["cases"][0]
    assert item["reduction"]["baseline_confirmed"]
    assert item["baseline"]["worker_environment"]["packages"]["torch"]
    for side, code in (("baseline", 0), ("candidate", 1)):
        path = (
            tmp_path
            / "comparison"
            / "cases"
            / item["id"]
            / "reduced"
            / f"{side}-reproducer"
            / "repro.py"
        )
        run = subprocess.run(
            [sys.executable, "-I", str(path)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert run.returncode == code, run.stderr


def test_capture_trace_and_gradient_run_cli(tmp_path):
    model = tmp_path / "model.py"
    model.write_text(
        "import torch\nfrom torch import nn\nfrom dataclasses import dataclass\n"
        "from faultline_test_helper import WIDTH\n"
        "@dataclass\nclass Config:\n    width: int = WIDTH\n"
        "def factory():\n"
        "    model = nn.Sequential(nn.Linear(Config().width, 4), nn.GELU()).eval()\n"
        "    return model, [(torch.ones(n, 4),) for n in (2, 3, 2)]\n"
    )
    (tmp_path / "faultline_test_helper.py").write_text("WIDTH = 4\n")
    captured = tmp_path / "trace.json"
    assert main(["capture", f"{model}:factory", "--trace", "--output", str(captured)]) == 0
    assert json.loads(captured.read_text())["kind"] == "trace"
    output = tmp_path / "run"
    assert (
        main(
            [
                "run",
                "--case",
                str(captured),
                "--backend",
                "eager",
                "--gradients",
                "--dynamic",
                "auto",
                "--repeats",
                "2",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    outcomes = list((output / "cases").glob("*/outcome.json"))
    assert len(outcomes) == 1
    assert json.loads(outcomes[0].read_text())["gradients"]
    assert (
        main(
            [
                "compare",
                "--case",
                str(captured),
                "--backend",
                "eager",
                "--gradients",
                "--repeats",
                "2",
                "--baseline-python",
                sys.executable,
                "--output",
                str(tmp_path / "gate"),
            ]
        )
        == 0
    )
