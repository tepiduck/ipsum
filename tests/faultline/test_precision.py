import sys

import pytest
import torch

from faultline import portable
from faultline.evaluate import Settings
from faultline.native import NativeJob, capture_workload, verify_native
from faultline.native_artifacts import export_native
from faultline.native_precision import assess_fp64, promote_model, promote_tree
from faultline.native_runtime import compare_native, observe_native
from faultline.runner import run_case
from faultline.standalone import verify_standalone


def test_fixed_probe_separates_output_offset_from_derivative_and_sees_zero_output_fault():
    model = torch.nn.Identity()

    def observe(offset=0.0, wrong=False, vjp=True):
        request = {"args": (torch.zeros(4),), "kwargs": {}}
        if vjp:
            request["cotangents"] = {"output": torch.tensor([1.0, -1.0, 1.0, -1.0])}
        return observe_native(model, lambda x: x + offset, [request], True, wrong)

    base, shifted = observe(), observe(0.1)
    name = "step:0:gradient:input[0][0]:value"
    assert torch.equal(base[0][name], shifted[0][name])
    assert not torch.equal(base[0]["step:0:output"], shifted[0]["step:0:output"])
    assert torch.equal(observe(wrong=True)[0][name], base[0][name] * 2)
    # The old square loss misses this derivative fault at an exactly zero output.
    assert torch.equal(observe(vjp=False)[0][name], observe(vjp=False, wrong=True)[0][name])


def test_fp64_promotion_preserves_storage_roles_and_views():
    base = torch.arange(20, dtype=torch.float32)
    first, second = base[2:12:2], base[4:14:2]
    integers = torch.tensor([2**60], dtype=torch.int64)
    a, b, again, ids = promote_tree((first, second, first, integers))
    assert a.dtype == torch.float64 and a.stride() == first.stride() and a.storage_offset() == 2
    assert a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr() and a is again
    assert ids is integers and ids.item() == 2**60
    a[1] = 99
    assert b[0].item() == 99 and base[4].item() == 4
    with pytest.raises(ValueError, match="mixed-dtype"):
        promote_tree((base, base.view(torch.int32)))
    model = torch.nn.Module()
    model.register_parameter("p", torch.nn.Parameter(first))
    model.register_buffer("b", second)
    promote_model(model)
    assert model.p.dtype == model.b.dtype == torch.float64
    assert model.p.untyped_storage().data_ptr() == model.b.untyped_storage().data_ptr()


def test_fp64_error_ratio_and_reference_failure_leave_primary_policy_separate():
    high = ({"output": torch.zeros(4, dtype=torch.float64)}, [])
    eager = ({"output": torch.full((4,), 0.001)}, [])
    candidate = ({"output": torch.full((4,), 0.04)}, [])
    result = assess_fp64(lambda: high, eager, candidate, compare_native, portable, 2)
    assert result["status"] == "complete"
    assert result["outputs"][0]["candidate_to_eager_rmse"] == pytest.approx(40)

    def unsupported():
        raise RuntimeError("operation has no double implementation")

    assert (
        assess_fp64(unsupported, eager, candidate, compare_native, portable, 2)["status"]
        == "reference_error"
    )
    assert (
        assess_fp64(
            unsupported, eager, candidate, compare_native, portable, 2, reference_is_factory=True
        )["status"]
        == "reference_error"
    )


def test_captured_vjp_freezes_mutating_trace_and_export_matches_fp64(tmp_path):
    source = tmp_path / "model.py"
    source.write_text(
        """import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("count", torch.zeros(()))
    def forward(self, x):
        self.count.add_(1)
        return {"values": x.sin() + self.count, "ids": torch.tensor([3], device=x.device)}
def factory():
    return Model().eval(), [(torch.zeros(3),), (torch.ones(5),)]
""",
        encoding="utf-8",
    )
    settings = Settings(backend="aot_eager", gradients=True, fp64_advisory=True, repeats=2)
    bundle = capture_workload(f"{source}:factory", tmp_path / "bundle", settings, trace=True)
    assert verify_native(bundle)["schema_version"] == 3
    payload = torch.load(bundle / "snapshot.pt", weights_only=True)
    assert payload["state"]["count"].item() == 0
    assert [r["cotangents"]["output['values']"].shape[0] for r in payload["requests"]] == [3, 5]
    job = NativeJob.create(bundle, requests=(1,))
    result = run_case(job, settings, tmp_path / "run")
    assert result["status"] == "pass", result
    assert result["backward_probe"] == "vjp" and result["fp64_advisory"]["status"] == "complete"
    script = export_native(job, settings, tmp_path / "export", result)
    replay = verify_standalone(script, sys.executable, tmp_path / "standalone", settings)
    assert replay["verified"] and replay["result"]["fp64_advisory"] == result["fp64_advisory"]


def test_legacy_loss_capture_remains_explicit_and_serialization_compatible(tmp_path):
    source = tmp_path / "model.py"
    source.write_text(
        "import torch\ndef factory():\n return torch.nn.Identity(), (torch.ones(3),)\n"
    )
    # Identity must still obey the public eval-model contract.
    source.write_text(source.read_text().replace("Identity()", "Identity().eval()"))
    settings = Settings(backend="eager", gradients=True, repeats=2)
    bundle = capture_workload(
        f"{source}:factory", tmp_path / "bundle", settings, backward_probe="loss"
    )
    assert verify_native(bundle)["schema_version"] == 1
    result = run_case(NativeJob.create(bundle), settings, tmp_path / "run")
    assert result["status"] == "pass" and result["backward_probe"] == "loss"
    assert "fp64_advisory" not in Settings().to_dict()
    old = {"status": "gradient_mismatch", "fingerprint": "known", "graph_stage": "backward"}
    assert portable.same_failure(old, {**old, "backward_probe": "aot_outputs"})
    assert not portable.same_failure(old, {**old, "backward_probe": "vjp"})
