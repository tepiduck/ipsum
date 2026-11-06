import sys

import pytest
import torch

from faultline import native_aot, native_compile, portable
from faultline.evaluate import Settings
from faultline.native import NativeJob, capture_workload
from faultline.native_artifacts import export_native
from faultline.native_masks import MaskProbe, clone_arguments
from faultline.native_numerics import annotate, assess_control, perturb_inputs
from faultline.native_precision import accuracy_label, error_stats
from faultline.native_runtime import compare_native
from faultline.runner import run_case
from faultline.standalone import verify_standalone


def test_accuracy_bands_and_sensitivity_labels_do_not_change_failure():
    assert accuracy_label({"rmse": 1}, {"rmse": 0.988}) == "comparable_to_eager_against_fp64"
    assert accuracy_label({"rmse": 1}, {"rmse": 0.61}) == "comparable_to_eager_against_fp64"
    assert accuracy_label({"rmse": 0}, {"rmse": 0}) == "both_exact"
    assert accuracy_label({"rmse": 0}, {"rmse": 1}) == "eager_exact_candidate_differs"
    assert accuracy_label({"rmse": 1}, {"rmse": 0.49}) == "candidate_closer_to_fp64"
    assert accuracy_label({"rmse": 1}, {"rmse": 2.01}) == "candidate_farther_from_fp64"
    assert accuracy_label({}, {"rmse": 1}) == "unavailable"
    primary = {
        "status": "gradient_mismatch",
        "fingerprint": "fixed",
        "comparison": {"outputs": [{"target": "gradient", "matches": False}]},
    }
    context = annotate(
        primary,
        {"outputs": [{"target": "gradient", "accuracy_label": "comparable_to_eager_against_fp64"}]},
        {
            "outputs": [
                {
                    "target": "gradient",
                    "exceeds_primary_tolerance": True,
                    "control_to_candidate_rmse": 0.98,
                }
            ]
        },
        {"flips": 2, "replay_matches_primary": True},
    )
    assert set(context["labels"]) == {
        "comparable_to_eager_against_fp64",
        "eager_control_sensitive",
        "discontinuity_sensitive",
    }
    assert portable.same_failure(primary, {**primary, "numerical_context": context})
    assert (
        "discontinuity_sensitive"
        not in annotate(primary, {}, {}, {"flips": 2, "status": "diagnostic_replay_changed"})[
            "labels"
        ]
    )


def test_ulp_control_preserves_views_integer_roles_and_cotangents():
    base = torch.arange(1, 9, dtype=torch.float32)
    old = base.clone()
    a, b = base[1::2], base[3::2]
    ids, mask, tangent = torch.tensor([2**60]), torch.tensor([True]), torch.ones(4)
    request = {"args": (a, b, ids, mask), "kwargs": {}, "cotangents": {"output": tangent}}
    perturb_inputs([request, request])
    assert torch.equal(base, torch.nextafter(old, torch.full_like(old, float("inf"))))
    assert a.storage_offset() == 1 and a.stride() == (2,)
    assert b.untyped_storage().data_ptr() == a.untyped_storage().data_ptr()
    assert ids.item() == 2**60 and mask.item() and torch.equal(tangent, torch.ones(4))
    with pytest.raises(ValueError, match="mixed-dtype"):
        perturb_inputs([{"args": (base, base.view(torch.int32)), "kwargs": {}}])


def test_eager_control_restores_backend_after_failure(monkeypatch):
    monkeypatch.setattr(torch.backends.mkldnn, "is_available", lambda: True)
    previous = torch.backends.mkldnn.enabled
    torch.backends.mkldnn.enabled = True
    try:

        def failed(transform):
            assert not torch.backends.mkldnn.enabled
            assert transform is None
            raise RuntimeError("unsupported control")

        result = assess_control(
            failed, ({}, []), ({}, []), Settings().to_dict(), compare_native, portable, error_stats
        )
        assert result["status"] == "control_error" and torch.backends.mkldnn.enabled
    finally:
        torch.backends.mkldnn.enabled = previous


def test_control_crossing_tolerance_between_repeats_is_unstable():
    values = iter([9e-6, 11e-6])
    eager = ({"output": torch.zeros(1)}, [])
    candidate = ({"output": torch.full((1,), 2e-5)}, [])
    result = assess_control(
        lambda _: lambda: ({"output": torch.tensor([next(values)])}, []),
        eager,
        candidate,
        Settings(repeats=2).to_dict(),
        compare_native,
        portable,
        error_stats,
    )
    assert result["status"] == "unstable_control"


def backward_graph(kind="relu"):
    graph = torch.fx.Graph()
    saved, tangent = graph.placeholder("saved"), graph.placeholder("tangent")
    if kind == "relu":
        result = graph.call_function(torch.ops.aten.threshold_backward.default, (tangent, saved, 0))
    else:
        # Clamp's derivative is represented by two comparisons and a where selector.
        low = graph.call_function(torch.ops.aten.ge.Scalar, (saved, 0))
        high = graph.call_function(torch.ops.aten.le.Scalar, (saved, 1))
        mask = graph.call_function(torch.ops.aten.bitwise_and.Tensor, (low, high))
        result = graph.call_function(torch.ops.aten.where.self, (mask, tangent, saved))
    graph.output((result,))
    return torch.fx.GraphModule(torch.nn.Module(), graph)


@pytest.mark.parametrize("kind", ["relu", "clamp"])
def test_saved_mask_pairing_finds_flips_and_rejects_unmatched_values(kind):
    probe = MaskProbe(
        native_aot, native_compile, Settings().to_dict(), compare_native, portable, {}
    )
    eager, compiled = torch.tensor([-1e-8, 0.5]), torch.tensor([1e-8, 0.5])
    probe.remember((eager,), (compiled,), "0:forward:0")
    probe.inspect_backward(backward_graph(kind), [compiled, torch.ones(2)], "0:backward:0")
    result = probe.summary()
    assert result["status"] == "complete" and result["flips"] == 1
    assert result["masks"][0]["affected_backward_outputs"] == [0]
    assert result["masks"][0]["changed_flat_indices"] == [0]
    compiled.add_(1)  # A wrapper mutation invalidates the saved-value correspondence.
    probe.inspect_backward(backward_graph(kind), [compiled, torch.ones(2)], "0:backward:0")
    assert probe.summary()["masks"][-1]["status"] == "unsupported"
    probe.reset()
    probe.inspect_backward(backward_graph(kind), [compiled, torch.ones(2)], "0:backward:0")
    assert probe.summary()["status"] == "partial"
    assert probe.summary()["masks"][0]["status"] == "unsupported"


def test_shadow_preserves_rng_storage_aliases_and_original_mutations():
    probe = MaskProbe(native_aot, native_compile, {}, compare_native, portable, {})

    class Model(torch.nn.Module):
        def forward(self, x, y):
            x.add_(1)
            return (y + torch.rand_like(y),)

    base = torch.ones(8)
    args = [base[::2], base[2::2]]
    torch.manual_seed(9)
    rng = torch.get_rng_state().clone()
    shadow = probe.shadow(Model(), args)
    assert torch.equal(torch.get_rng_state(), rng) and torch.equal(base, torch.ones(8))
    actual = Model()(*args)
    assert torch.equal(shadow[0], actual[0])
    with pytest.raises(ValueError, match="snapshot limit"):
        clone_arguments(args, limit_bytes=1)


def test_same_input_backward_check_does_not_hide_missing_gradient():
    probe = MaskProbe(
        native_aot, native_compile, Settings().to_dict(), compare_native, portable, {}
    )
    values = [torch.ones(2), torch.ones(2)]
    actual = probe.around(backward_graph(), "backward", "0:backward:0", lambda _: (None,), values)
    assert actual == (None,)
    assert not probe.summary()["backward_same_inputs"][0]["matches"]


@pytest.mark.parametrize(
    "backend",
    [
        "aot_eager",
        "mutant:aot-backward-scale",
        pytest.param("inductor", marks=pytest.mark.compiler),
    ],
)
def test_numerics_worker_and_export_keep_primary_and_detect_wrong_derivatives(tmp_path, backend):
    source = tmp_path / "model.py"
    source.write_text("""import torch
def factory():
    torch.manual_seed(7)
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU()).eval(), (torch.ones(2, 4),)
""")
    settings = Settings(backend=backend, gradients=True, numerics=True, repeats=2, timeout_s=240)
    bundle = capture_workload(f"{source}:factory", tmp_path / "bundle", settings)
    job = NativeJob.create(bundle, aot=True)
    result = run_case(job, settings, tmp_path / "run")
    assert result["status"] == ("gradient_mismatch" if backend.startswith("mutant") else "pass"), (
        result
    )
    context = result["numerical_context"]
    assert context["masks"]["status"] == "complete", context
    assert context["masks"]["flips"] == 0 and "discontinuity_sensitive" not in context["labels"]
    assert context["eager_control"]["status"] == "complete"
    if backend.startswith("mutant"):
        assert not all(c["matches"] for c in context["masks"]["backward_same_inputs"])
    script = export_native(job, settings, tmp_path / "export", result)
    replay = verify_standalone(script, sys.executable, tmp_path / "standalone", settings)
    assert replay["verified"], replay
    assert replay["result"]["numerical_context"] == context
    assert "numerics" not in Settings().to_dict()
