import json
from dataclasses import replace

import torch

from faultline.evaluate import Settings
from faultline.native import NativeJob, capture_workload, verify_native
from faultline.native_graph import load_native_graph
from faultline.native_reduce import reduce_native
from faultline.runner import run_case


def test_supervised_boundary_reduction_removes_a_different_shape_prefix(tmp_path):
    source = tmp_path / "model.py"
    source.write_text(
        """import torch
class Model(torch.nn.Module):
    def forward(self, x, w):
        return torch.relu(x @ w)
def factory():
    return Model().eval(), (torch.ones(3,7), torch.arange(35).reshape(7,5).float())
""",
        encoding="utf-8",
    )
    settings = Settings(backend="mutant:relu-as-neg", repeats=2)
    bundle = capture_workload(f"{source}:factory", tmp_path / "bundle", settings)
    job = NativeJob.create(bundle)
    result = run_case(job, settings, tmp_path / "run")
    assert result["status"] == "numerical_mismatch"
    reduced = reduce_native(job, settings, result, tmp_path / "reduction", max_evaluations=12)
    assert reduced["confirmed"] and reduced["injected_fault"]
    assert reduced["original_complexity"][1] == 2
    assert reduced["reduced_complexity"] == (1, 1, 15)
    attempts = json.loads((tmp_path / "reduction" / "attempts.json").read_text())
    assert any(attempt["stage"] == "freeze-boundary" for attempt in attempts)
    graph_bundle = tmp_path / "reduction" / "reproducer" / "workload"
    assert verify_native(graph_bundle)["frozen_boundary"]
    _, requests, _ = load_native_graph(graph_bundle)
    assert torch.equal(requests[0][0], torch.ones(3, 7) @ torch.arange(35).reshape(7, 5).float())
    graph = NativeJob.create(graph_bundle, operation="boundaries")
    invalid = run_case(graph, replace(settings, gradients=True), tmp_path / "invalid")
    assert invalid["status"] == "harness_error"
