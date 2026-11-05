import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest
import torch

from faultline import portable
from faultline.cli import main
from faultline.evaluate import Settings
from faultline.native import NativeJob, capture_workload, verify_native
from faultline.native_artifacts import export_native, load_native_run, run_native
from faultline.native_graph import load_native_graph, save_native_graph
from faultline.native_reduce import reduce_native
from faultline.native_runtime import check_native, compare_native, observe_native
from faultline.runner import run_case


@pytest.fixture
def factory(tmp_path):
    path = tmp_path / "user_model.py"
    path.write_text(
        """import torch
from torch import nn
def factory():
    torch.manual_seed(13)
    model = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Sigmoid(), nn.Tanh()).eval()
    return model, (torch.randn(2, 768),)
""",
        encoding="utf-8",
    )
    return f"{path}:factory"


def test_native_model_end_to_end_large_weights_reduction_and_standalone(tmp_path, factory):
    settings = Settings(backend="mutant:relu-as-neg", repeats=2)
    bundle = capture_workload(factory, tmp_path / "bundle", settings)
    manifest = verify_native(bundle)
    assert manifest["parameter_count"] > 590000
    assert manifest["snapshot_bytes"] > 2_000_000
    summary = run_native(bundle, tmp_path / "run", settings)
    assert summary["status"] == "numerical_mismatch"
    job, saved, outcome = load_native_run(tmp_path / "run")
    reduced = reduce_native(
        job,
        saved,
        outcome,
        tmp_path / "reduction",
        max_evaluations=12,
        baseline_settings=replace(settings, backend="eager"),
        baseline_python=sys.executable,
        progress=None,
    )
    assert reduced["confirmed"] and reduced["native_graph"]
    assert reduced["reduced_complexity"] < reduced["original_complexity"]
    assert reduced["baseline_confirmation"]["status"] == "pass"
    assert reduced["localization"][0]["whole_model_failure_retained"]
    script = tmp_path / "reduction" / "reproducer" / "repro.py"
    environment = {**os.environ, "PYTHONPATH": ""}
    process = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert process.returncode == 1, process.stderr
    assert json.loads(process.stdout)["reproduces_expected"]
    config = script.parent / "reproducer.json"
    data = json.loads(config.read_text())
    data["files"]["workload/tensors.pt"] = "0" * 64
    config.write_text(json.dumps(data))
    changed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert changed.returncode == 2 and json.loads(changed.stdout)["status"] == "harness_error"
    assert all(
        "from ." not in (script.parent / name).read_text()
        for name in ("repro.py", "native_runtime.py", "native_graph.py", "portable.py")
    )


def test_nested_integer_boolean_and_alias_inputs(tmp_path):
    source = tmp_path / "nested.py"
    source.write_text(
        """import torch
from torch import nn
class Model(nn.Module):
    def forward(self, tokens, mask, views):
        return {"values": torch.where(mask, tokens, -tokens),
                "views": [views[0] + views[1]], "empty": tokens[:0], "flag": True}
def factory():
    storage = torch.arange(24, dtype=torch.float64).reshape(4, 6)
    return Model().eval(), {"tokens": torch.tensor([2**60, 2**60+1]),
        "mask": torch.tensor([True, False]), "views": [storage[:, ::2], storage[:, ::2]]}
""",
        encoding="utf-8",
    )
    settings = Settings(backend="eager", repeats=2)
    bundle = capture_workload(f"{source}:factory", tmp_path / "bundle", settings)
    payload = torch.load(bundle / "snapshot.pt", weights_only=True)
    a, b = payload["requests"][0]["kwargs"]["views"]
    assert a.stride() == (6, 2)
    assert a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
    result = run_case(NativeJob.create(bundle), settings, tmp_path / "run")
    assert result["status"] == "pass", result


def test_integer_comparison_is_exact_beyond_float_precision():
    a, b = torch.tensor([2**60]), torch.tensor([2**60 + 1])
    result = compare_native(({"integer": a}, []), ({"integer": b}, []), portable)
    assert not result["matches"]


def test_dictionary_keys_preserve_types_and_scalar_outputs():
    from faultline.native_runtime import structure

    left = {"value": torch.ones(1), 1: "left", "1": "same"}
    right = {"value": torch.ones(1), 1: "changed", "1": "same"}
    result = compare_native(
        ({"value": left["value"]}, [structure(left)]),
        ({"value": right["value"]}, [structure(right)]),
        portable,
    )
    assert not result["matches"] and result["kind"] == "structure"


def test_native_graph_retains_batchnorm_buffer_autograd_roles(tmp_path):
    source = tmp_path / "batchnorm.py"
    source.write_text(
        """import torch
def factory():
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(4), torch.nn.ReLU()).eval()
    return model, (torch.tensor([[-2., 1., 3., 4.], [1., -1., 2., 3.]]),)
""",
        encoding="utf-8",
    )
    settings = Settings(backend="mutant:gradient-scale", gradients=True, repeats=2)
    bundle = capture_workload(f"{source}:factory", tmp_path / "bundle", settings)
    result = run_case(NativeJob.create(bundle), settings, tmp_path / "run")
    assert result["status"] == "gradient_mismatch", result
    graph = NativeJob.create(tmp_path / "run" / "graphs" / "graph-0000")
    graph_result = run_case(graph, settings, tmp_path / "graph-run")
    assert graph_result["status"] == "gradient_mismatch", graph_result
    targets = [
        item["target"]
        for item in graph_result["comparison"]["outputs"]
        if ":gradient:" in item["target"]
    ]
    assert not any("running_mean" in target or "running_var" in target for target in targets)


def test_gradient_mismatch_with_identical_forward():
    model = torch.nn.Linear(4, 3).eval()
    x = torch.ones(2, 4)

    def execute(wrong=False):
        return observe_native(model, model, [{"args": (x.clone(),), "kwargs": {}}], True, wrong)

    result = check_native(
        execute,
        lambda: lambda: execute(True),
        Settings(backend="mutant:gradient-scale", gradients=True, repeats=2).to_dict(),
        portable,
    )
    assert result["status"] == "gradient_mismatch"
    assert all(
        item["matches"]
        for item in result["comparison"]["outputs"]
        if ":gradient:" not in item["target"]
    )


def test_native_graph_binary_storage_roundtrip(tmp_path):
    module = torch.fx.symbolic_trace(lambda a, b: (a + b, a.transpose(0, 1)))
    storage = torch.arange(30, dtype=torch.float32).reshape(5, 6)
    a, b = storage[:, ::2], storage[:, 1::2]
    save_native_graph(tmp_path, module, [(a, b)])
    rebuilt, inputs, _ = load_native_graph(tmp_path)
    assert inputs[0][0].stride() == a.stride()
    assert inputs[0][1].storage_offset() == b.storage_offset()
    assert inputs[0][0].untyped_storage().data_ptr() == inputs[0][1].untyped_storage().data_ptr()
    assert torch.equal(rebuilt(*inputs[0])[0], a + b)


def test_modified_snapshot_and_settings_rejected(tmp_path, factory):
    settings = Settings(backend="eager", repeats=2)
    bundle = capture_workload(factory, tmp_path / "bundle", settings)
    run_native(bundle, tmp_path / "run", settings)
    changed = tmp_path / "run" / "settings.json"
    changed.write_text(json.dumps(replace(settings, rtol=1.0).to_dict()))
    with pytest.raises(ValueError, match="settings.*changed"):
        load_native_run(tmp_path / "run")
    with (bundle / "snapshot.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="modified"):
        verify_native(bundle)


def test_cli_native_capture_and_run(tmp_path, factory):
    assert main(["model", "capture", factory, "--output", str(tmp_path / "bundle")]) == 0
    assert (
        main(
            [
                "model",
                "run",
                str(tmp_path / "bundle"),
                "--backend",
                "eager",
                "--repeats",
                "2",
                "--output",
                str(tmp_path / "run"),
            ]
        )
        == 0
    )
    assert (tmp_path / "run" / "report.html").is_file()


def test_mutated_input_detected_when_return_values_match():
    model = torch.nn.Identity().eval()

    def execute(mutate=False):
        def call(value):
            if mutate:
                value.add_(1)
            return torch.zeros_like(value)

        return observe_native(model, call, [{"args": (torch.ones(3),), "kwargs": {}}])

    compared = compare_native(execute(), execute(True), portable)
    assert not compared["matches"]
    failures = [item for item in compared["outputs"] if not item["matches"]]
    assert all("input_after" in item["target"] for item in failures)


def test_native_repetition_and_reference_errors_are_inconclusive():
    count = 0

    def unstable():
        nonlocal count
        count += 1
        return {"x": torch.tensor(float(count))}, []

    settings = Settings(backend="eager", repeats=2).to_dict()
    assert (
        check_native(unstable, lambda: unstable, settings, portable)["status"]
        == "unstable_reference"
    )

    def broken():
        raise RuntimeError("invalid model inputs")

    assert check_native(broken, lambda: broken, settings, portable)["status"] == "reference_error"


def test_graph_backend_flat_list_abi_is_adapted():
    from faultline.native_runtime import compile_native_graph

    module = torch.fx.symbolic_trace(lambda a: (a.sin(),))
    fn = compile_native_graph(module, lambda model, args: lambda x: list(model(x)))
    result = fn(torch.ones(3))
    assert isinstance(result, tuple) and torch.equal(result[0], torch.ones(3).sin())


def test_native_reducer_does_not_accept_baseline_failures(tmp_path, monkeypatch):
    from faultline import native_reduce
    from faultline.native import native_worker, seal_native

    module = torch.fx.symbolic_trace(lambda x: torch.sigmoid(torch.relu(x)))
    for node in module.graph.nodes:
        node.meta["faultline_tensor"] = {"shape": [3], "dtype": "torch.float32"}
    save_native_graph(tmp_path / "bundle", module, [(torch.tensor([-2.0, 1.0, 3.0]),)])
    seal_native(
        tmp_path / "bundle",
        {
            "schema_version": 1,
            "kind": "native-graph",
            "sources": [],
            "seed": 0,
            "trace": False,
            "request_count": 1,
            "captured": True,
        },
    )
    job = NativeJob.create(tmp_path / "bundle")
    settings = Settings(backend="mutant:relu-as-neg", repeats=2)
    expected = native_worker(job.to_dict(), settings, tmp_path / "initial")
    calls = []

    def worker(proposal, policy, directory, **kwargs):
        calls.append((proposal.digest(), policy.backend))
        if policy.backend == "eager" and proposal.digest() != job.digest():
            return {"status": "runtime_error", "fingerprint": "baseline-broke"}
        return native_worker(proposal.to_dict(), policy, directory)

    monkeypatch.setattr(native_reduce, "run_case", worker)
    evidence = reduce_native(
        job,
        settings,
        expected,
        tmp_path / "reduction",
        max_evaluations=8,
        baseline_settings=replace(settings, backend="eager"),
        baseline_python=sys.executable,
        progress=None,
    )
    assert evidence["confirmed"]
    assert evidence["original_complexity"] == evidence["reduced_complexity"]
    assert all(key == job.digest() for key, backend in calls if backend != "eager")
    assert any(key != job.digest() for key, backend in calls if backend == "eager")


def test_native_trace_resets_registered_state_and_reduces_requests(tmp_path):
    source = tmp_path / "trace.py"
    source.write_text(
        """import torch
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("count", torch.zeros(()))
    def forward(self, x):
        self.count.add_(1)
        return torch.relu(x) + self.count
def factory():
    return Model().eval(), [(torch.tensor([-2., 1.]),), (torch.tensor([-4., 3.]),)]
""",
        encoding="utf-8",
    )
    settings = Settings(backend="mutant:relu-as-neg", repeats=2)
    bundle = capture_workload(f"{source}:factory", tmp_path / "bundle", settings, trace=True)
    job = NativeJob.create(bundle)
    outcome = run_case(job, settings, tmp_path / "run")
    assert outcome["status"] == "numerical_mismatch"
    evidence = reduce_native(
        job, settings, outcome, tmp_path / "reduced", max_evaluations=4, progress=None
    )
    assert evidence["confirmed"]
    assert evidence["retained_requests"] is not None and len(evidence["retained_requests"]) == 1


def test_included_checkpoint_and_lazy_import_survive_relocation(tmp_path, monkeypatch):
    source = tmp_path / "model.py"
    source.write_text(
        """from pathlib import Path
import torch
from torch import nn
class Model(nn.Linear):
    def forward(self, x):
        from helper import offset
        return super().forward(x) + offset
def factory():
    model = Model(3, 2)
    model.load_state_dict(torch.load(Path(__file__).parent / "weights.pt", weights_only=True))
    return model.eval(), (torch.ones(1, 3),)
""",
        encoding="utf-8",
    )
    (tmp_path / "helper.py").write_text("offset = 3.0\n", encoding="utf-8")
    torch.save(torch.nn.Linear(3, 2).state_dict(), tmp_path / "weights.pt")
    settings = Settings(backend="eager", repeats=2)
    monkeypatch.chdir(os.path.dirname(sys.executable))
    bundle = capture_workload(
        f"{source}:factory",
        tmp_path / "bundle",
        settings,
        include=("helper.py", "weights.pt"),
        python="./" + os.path.basename(sys.executable),
    )
    from faultline.native_artifacts import copy_workload

    relocated = copy_workload(bundle, tmp_path / "elsewhere")
    # Original files are deliberately unavailable; only explicitly bundled dependencies remain.
    source.unlink()
    (tmp_path / "weights.pt").unlink()
    (tmp_path / "helper.py").unlink()
    result = run_case(NativeJob.create(relocated), settings, tmp_path / "run")
    assert result["status"] == "pass", result


@pytest.mark.compiler
def test_native_real_inductor_forward_backward(tmp_path, factory):
    settings = Settings(backend="inductor", gradients=True, repeats=2, timeout_s=240)
    bundle = capture_workload(factory, tmp_path / "bundle", settings)
    result = run_case(NativeJob.create(bundle), settings, tmp_path / "run")
    assert result["status"] == "pass", result
    assert result["graphs"][0]["portable"]
    export_native(NativeJob.create(bundle), settings, tmp_path / "export", result)
