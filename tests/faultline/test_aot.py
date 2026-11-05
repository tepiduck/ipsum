import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from faultline import portable
from faultline.cli import main
from faultline.evaluate import Settings
from faultline.native import NativeJob, capture_workload, verify_native
from faultline.native_artifacts import export_native
from faultline.native_reduce import reduce_native
from faultline.runner import run_case


@pytest.fixture
def workload(tmp_path):
    source = tmp_path / "model.py"
    source.write_text(
        """import torch
def factory():
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Linear(4,8), torch.nn.GELU(), torch.nn.Linear(8,2)).eval()
    return model, (torch.randn(3,4),)
""",
        encoding="utf-8",
    )
    settings = Settings(backend="aot_eager", gradients=True, repeats=2, timeout_s=240)
    return capture_workload(f"{source}:factory", tmp_path / "bundle", settings)


def standalone(script, directory):
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=directory,
        env={**os.environ, "PYTHONPATH": ""},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.stdout, completed.stderr
    return completed.returncode, json.loads(completed.stdout)


def test_backward_capture_isolation_reduction_and_exports(tmp_path, workload):
    settings = Settings(backend="mutant:aot-backward-scale", gradients=True, repeats=2)
    job = NativeJob.create(workload, aot=True)
    result = run_case(job, settings, tmp_path / "run")
    assert result["status"] == "gradient_mismatch", result
    assert all(
        item["matches"]
        for item in result["comparison"]["outputs"]
        if ":gradient:" not in item["target"]
    )
    assert {entry["stage"] for entry in result["aot_graphs"]} == {"forward", "backward"}
    assert all(entry["portable"] for entry in result["aot_graphs"])
    assert result["aot_autograd_cache"].startswith("disabled")
    forward = next(entry for entry in result["aot_graphs"] if entry["stage"] == "forward")
    unchanged = run_case(
        replace(job, only_graph=forward["parent"], only_aot=forward["id"]),
        settings,
        tmp_path / "forward-only",
    )
    assert unchanged["status"] == "pass", unchanged
    script = export_native(job, settings, tmp_path / "full-reproducer", result)
    code, replay = standalone(script, tmp_path)
    assert code == 1 and replay["reproduces_expected"]
    reduction = reduce_native(
        job,
        settings,
        result,
        tmp_path / "reduction",
        max_evaluations=14,
        baseline_settings=replace(settings, backend="aot_eager"),
        baseline_python=sys.executable,
        progress=None,
    )
    assert reduction["confirmed"] and reduction["graph_stage"] == "backward"
    assert reduction["baseline_confirmation"]["status"] == "pass"
    assert reduction["reduced_complexity"] < reduction["original_complexity"]
    manifest = verify_native(tmp_path / "reduction" / "reproducer" / "workload")
    assert manifest["aot_stage"] == "backward"
    assert reduction["localization"][0]["whole_model_failure_retained"]
    code, replay = standalone(tmp_path / "reduction" / "reproducer" / "repro.py", tmp_path)
    assert (
        code == 1
        and replay["reproduces_expected"]
        and replay["observation_mode"] == "aot_backward_outputs"
    )


def test_aot_selection_integrity_and_failure_context(workload):
    legacy = NativeJob.create(workload)
    assert "aot" not in legacy.to_dict()
    selected = replace(legacy, aot=True, only_aot="0:backward:0")
    selected.validate()
    assert selected.digest() != legacy.digest()
    with pytest.raises(ValueError, match="AOT"):
        replace(legacy, only_aot="0:backward:0").validate()
    with pytest.raises(ValueError, match="AOT"):
        replace(legacy, aot=True, only_aot=7).validate()
    result = {"status": "gradient_mismatch", "fingerprint": "same", "aot": True}
    assert not portable.same_failure(result, {**result, "aot": False})


def test_compiler_metadata_and_static_input_roles_survive_argument_pruning(monkeypatch):
    import torch

    from faultline import native_aot
    from faultline.native_graph import build_native_graph, graph_spec

    graph = torch.fx.Graph()
    unused = graph.placeholder("unused")
    saved = graph.placeholder("saved_activation")
    tangent = graph.placeholder("tangent")
    saved.meta["is_static_input"] = True
    out = graph.output((graph.call_function(torch.ops.aten.mul.Tensor, (saved, tangent)),))
    out.meta["user_visible_output_idxs"] = [0]
    # The saved activation moves from index 1 to index 0 as a reducer removes an unused input.
    graph.erase_node(unused)
    spec, tensors = graph_spec(torch.fx.GraphModule(torch.nn.Module(), graph))
    assert spec["schema_version"] == 2
    rebuilt = build_native_graph(json.loads(json.dumps(spec)), tensors)
    assert next(n for n in rebuilt.graph.nodes if n.op == "output").meta[
        "user_visible_output_idxs"
    ] == [0]
    assert next(iter(rebuilt.graph.nodes)).meta["is_static_input"]
    seen = {}

    def compile_stage(module, inputs, backend, stage, **kwargs):
        seen.update(kwargs["options"])
        return lambda values: module(*values)

    monkeypatch.setattr(native_aot, "compile_stage", compile_stage)
    args = [torch.ones(2), torch.ones(2)]
    run = native_aot.replay_stage_backend(
        rebuilt,
        args,
        "inductor",
        {"aot_stage": "backward", "aot_static_inputs": ["saved_activation"]},
    )
    assert torch.equal(run(*args)[0], torch.ones(2))
    assert seen == {"is_backward": True, "is_inference": False, "static_input_idxs": [0]}


def test_unsupported_adapter_and_missing_backward_are_inconclusive():
    from faultline.native_aot import annotate_aot_result

    error = {"status": "compile_error", "error": "BackendCompilerFailed: AOTUnsupported: ABI"}
    assert annotate_aot_result(error, {}, True)["status"] == "unsupported_aot"
    missing = annotate_aot_result({"status": "pass"}, {}, True, [{"stage": "forward"}])
    assert missing["status"] == "aot_capture_missing"
    assert error["status"] not in portable.REDUCIBLE
    assert missing["status"] not in portable.REDUCIBLE


def test_aot_requires_gradient_workload(tmp_path, workload):
    result = run_case(
        NativeJob.create(workload, aot=True),
        Settings(backend="aot_eager", repeats=2),
        tmp_path / "run",
    )
    assert result["status"] == "harness_error" and "requires" in result["error"]


def test_cli_aot_saved_run_and_replay(tmp_path, workload):
    assert (
        main(
            [
                "model",
                "run",
                str(workload),
                "--aot",
                "--gradients",
                "--backend",
                "aot_eager",
                "--repeats",
                "2",
                "--output",
                str(tmp_path / "run"),
            ]
        )
        == 0
    )
    assert (
        main(["model", "replay", str(tmp_path / "run"), "--output", str(tmp_path / "replay")]) == 0
    )
    assert json.loads((tmp_path / "replay" / "selection.json").read_text())["aot"]


@pytest.mark.compiler
def test_inductor_aot_backward_graph_replays_without_second_derivatives(tmp_path, workload):
    settings = Settings(backend="inductor", gradients=True, repeats=2, timeout_s=240)
    result = run_case(NativeJob.create(workload, aot=True), settings, tmp_path / "run")
    assert result["status"] == "pass", result
    entry = next(item for item in result["aot_graphs"] if item["stage"] == "backward")
    graph = NativeJob.create(tmp_path / "run" / "graphs" / entry["directory"])
    independent = run_case(graph, settings, tmp_path / "graph")
    assert independent["status"] == "pass", independent
    assert independent["observation_mode"] == "aot_backward_outputs"
    assert not any(":gradient:" in item["target"] for item in independent["comparison"]["outputs"])
    script = export_native(graph, settings, tmp_path / "export", independent)
    code, replay = standalone(script, tmp_path)
    assert code == 0 and replay["reproduces_expected"]
