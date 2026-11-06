import json
import sys
from dataclasses import replace

import pytest
import torch

from faultline.cli import main
from faultline.diagnose import LADDER, diagnose_native, interpret_ladder
from faultline.evaluate import Settings
from faultline.historical import score_case
from faultline.native import NativeJob, capture_workload
from faultline.native_artifacts import export_native, run_native
from faultline.native_graph import load_native_graph
from faultline.native_runtime import make_native_execution
from faultline.portable import same_failure
from faultline.runner import run_case
from faultline.standalone import verify_standalone


def outcomes(statuses):
    return [
        {
            "status": status,
            "backend": backend,
            "native": True,
            "trace": True,
            "comparison": {
                "outputs": [{"target": "step:0:output", "matches": False, "kind": "numeric"}]
            },
        }
        for backend, status in zip(LADDER, statuses, strict=True)
    ]


@pytest.mark.parametrize(
    "statuses,expected",
    [
        (["pass"] * 4, "agreement"),
        (["pass"] * 3 + ["numerical_mismatch"], "observed_boundary"),
        (
            ["pass", "numerical_mismatch", "numerical_mismatch", "numerical_mismatch"],
            "observed_boundary",
        ),
        (["pass", "numerical_mismatch", "pass", "numerical_mismatch"], "non_monotonic"),
        (["pass", "unsupported_backend", "pass", "numerical_mismatch"], "inconclusive"),
    ],
)
def test_ladder_requires_a_complete_monotonic_pattern(statuses, expected):
    result = interpret_ladder(outcomes(statuses))
    assert result["status"] == expected
    assert bool(result["boundary"]) == (expected == "observed_boundary")


def test_ladder_rejects_different_tensor_failures():
    results = outcomes(["pass", "numerical_mismatch", "numerical_mismatch", "numerical_mismatch"])
    results[-1]["comparison"]["outputs"][0]["target"] = "step:0:other_output"
    assert interpret_ladder(results)["status"] == "different_failures"


def test_historical_scoring_keeps_missing_controls_and_not_observed_visible():
    expected = "numerical_mismatch"
    assert score_case(expected, {"status": "pass"}, {"status": expected})["paired_reproduction"]
    repeated = score_case(expected, {"status": expected}, {"status": expected})
    assert repeated["candidate_detected"] and not repeated["paired_reproduction"]
    assert score_case(expected, {"status": "pass"}, {"status": "pass"})["status"] == "not_observed"
    assert (
        score_case(expected, {"status": "unavailable"}, {"status": expected})["status"]
        == "inconclusive"
    )
    wrong_target = {
        "status": expected,
        "comparison": {"outputs": [{"target": "other", "matches": False}]},
    }
    assert not score_case(expected, {"status": "pass"}, wrong_target, "expected")[
        "candidate_detected"
    ]


@pytest.fixture
def bundle(tmp_path):
    source = tmp_path / "linear.py"
    source.write_text(
        """import torch
def factory():
    torch.manual_seed(21)
    return torch.nn.Linear(8,4).eval(), (torch.randn(3,8),)
""",
        encoding="utf-8",
    )
    return capture_workload(f"{source}:factory", tmp_path / "bundle", Settings())


def test_amp_changes_forward_dtype_and_preserves_saved_policy_and_export(tmp_path, bundle):
    from faultline.native import verify_native

    settings = Settings(backend="aot_eager", autocast="bfloat16", gradients=True, repeats=2)
    observed, _ = make_native_execution(
        bundle, verify_native(bundle), settings.to_dict(), load_native_graph
    )()
    assert observed["step:0:output"].dtype == torch.bfloat16
    assert observed["step:0:gradient:parameter:weight:value"].dtype == torch.float32
    job = NativeJob.create(bundle)
    result = run_case(job, settings, tmp_path / "check")
    assert result["status"] == "pass", result
    assert result["autocast"] == "bfloat16"
    script = export_native(job, settings, tmp_path / "export", result)
    replay = verify_standalone(script, sys.executable, tmp_path / "standalone", settings)
    assert replay["verified"], replay
    with pytest.raises(ValueError, match="policy"):
        export_native(job, replace(settings, autocast=None), tmp_path / "wrong", result)


def test_default_settings_remain_serialization_compatible_and_modes_affect_signatures():
    assert "compile_mode" not in Settings().to_dict()
    assert "autocast" not in Settings().to_dict()
    old = {"status": "numerical_mismatch", "fingerprint": "example"}
    assert same_failure(old, {**old, "compile_mode": "default", "autocast": None})
    assert not same_failure(old, {**old, "compile_mode": "reduce-overhead"})
    assert not same_failure(old, {**old, "autocast": "bfloat16"})


def test_diagnose_saved_run_rejects_policy_overrides(tmp_path, bundle):
    settings = Settings(backend="eager", repeats=2)
    run_native(bundle, tmp_path / "saved", settings)
    with pytest.raises(ValueError, match="overrides"):
        diagnose_native(
            tmp_path / "saved", tmp_path / "diagnosis", replace(settings, autocast="bfloat16")
        )


def test_standalone_supervisor_requires_matching_exit_and_cleans_timeout(tmp_path):
    script = tmp_path / "fake.py"
    script.write_text(
        'import json, os\nprint(json.dumps({"status":"pass","reproduces_expected":True,"pythonpath":os.environ.get("PYTHONPATH")}))\n',
        encoding="utf-8",
    )
    settings = Settings(timeout_s=10)
    result = verify_standalone(script, sys.executable, tmp_path / "good", settings)
    assert result["verified"] and result["result"]["pythonpath"] is None
    script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    result = verify_standalone(
        script, sys.executable, tmp_path / "timeout", replace(settings, timeout_s=0.2)
    )
    assert not result["verified"] and result["status"] == "timeout"


@pytest.mark.compiler
def test_full_backend_ladder_on_real_compilers_and_saved_run(tmp_path, bundle):
    settings = Settings(backend="inductor", gradients=True, repeats=2)
    run_native(bundle, tmp_path / "saved", settings)
    assert (
        main(
            ["model", "diagnose", str(tmp_path / "saved"), "--output", str(tmp_path / "diagnosis")]
        )
        == 0
    )
    summary = json.loads((tmp_path / "diagnosis" / "summary.json").read_text())
    assert summary["source_reproduced"]
    assert [case["id"] for case in summary["cases"]] == list(LADDER)
    assert all(case["result"]["graph_count"] > 0 for case in summary["cases"])


@pytest.mark.compiler
def test_inductor_mode_and_aot_autocast_export(tmp_path):
    # Binary fractions and a power-of-two loss divisor isolate policy plumbing from
    # legitimate differences in low-precision bias-gradient accumulation.
    source = tmp_path / "exact.py"
    source.write_text(
        """import torch
def factory():
    model = torch.nn.Linear(8,4).eval()
    with torch.no_grad():
        model.weight.fill_(0.125)
        model.bias.fill_(0.125)
    return model, (torch.ones(4,8),)
""",
        encoding="utf-8",
    )
    bundle = capture_workload(f"{source}:factory", tmp_path / "bundle", Settings())
    settings = Settings(
        backend="inductor",
        compile_mode="reduce-overhead",
        autocast="bfloat16",
        gradients=True,
        repeats=2,
        timeout_s=240,
    )
    job = NativeJob.create(bundle, aot=True)
    result = run_case(job, settings, tmp_path / "run")
    assert result["status"] == "pass", result
    assert result["compile_mode"] == "reduce-overhead" and result["autocast"] == "bfloat16"
    assert {entry["stage"] for entry in result["aot_graphs"]} == {"forward", "backward"}
    script = export_native(job, settings, tmp_path / "reproducer", result)
    assert verify_standalone(script, sys.executable, tmp_path / "standalone", settings)["verified"]
