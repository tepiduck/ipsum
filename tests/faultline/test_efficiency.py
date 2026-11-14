import gc
import weakref

import pytest
import torch

from faultline import portable
from faultline.evaluate import Settings
from faultline.native import GraphRecorder, file_hash, verify_native
from faultline.native_runtime import snapshot_state


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "expanded"])
def test_blocked_comparison_matches_unblocked_fields(monkeypatch, dtype, layout):
    generator = torch.Generator().manual_seed(19)
    reference = torch.randn(9, 11, generator=generator).to(dtype)
    if layout == "transpose":
        reference = reference.T
    elif layout == "expanded":
        reference = reference[:1].expand(9, 11)
    for variant in ("equal", "rounding", "different", "nonfinite", "signed_zero"):
        left, right = reference.clone(), reference.clone()
        if variant == "rounding":
            right = torch.nextafter(right, torch.full_like(right, float("inf")))
        elif variant == "different":
            right[-1, -1] += 4
        elif variant == "nonfinite":
            left[0, :4] = torch.tensor([float("nan"), float("inf"), -float("inf"), 1])
            right[0, :4] = torch.tensor([float("nan"), float("inf"), float("inf"), float("nan")])
        elif variant == "signed_zero":
            left[0, :4], right[0, :4] = 0.0, -0.0
        if layout == "expanded":
            left = left.unsqueeze(0).expand(2, *left.shape)
            right = right.unsqueeze(0).expand(2, *right.shape)
        # Include a second target so the outer index and structural fields are checked.
        references = (left, torch.ones(2))
        candidates = (right, torch.ones(3))
        for rtol, atol in ((0.0, 0.0), (1e-4, 1e-5), (0.3, 0.01)):
            monkeypatch.setattr(portable, "COMPARE_BLOCK_ELEMENTS", 1000000)
            unblocked = portable.compare(references, candidates, rtol, atol)
            monkeypatch.setattr(portable, "COMPARE_BLOCK_ELEMENTS", 17)
            assert portable.compare(references, candidates, rtol, atol) == unblocked


def test_blocked_comparison_preserves_overflow_and_nonfinite_maxima(monkeypatch):
    largest = torch.finfo(torch.float64).max
    left = torch.tensor([largest, -largest, 0, 1e-320] * 20, dtype=torch.float64)
    right = -left
    for rtol, atol in ((0.0, 0.0), (1e308, 1e308), (0.5, 0.0)):
        monkeypatch.setattr(portable, "COMPARE_BLOCK_ELEMENTS", 1000000)
        original = portable.compare((left,), (right,), rtol, atol)
        monkeypatch.setattr(portable, "COMPARE_BLOCK_ELEMENTS", 7)
        assert portable.compare((left,), (right,), rtol, atol) == original


def test_state_copies_share_only_identical_bytes_and_do_not_keep_models_alive():
    source = torch.zeros(20000)
    first = snapshot_state("sharing-test", source)
    assert first.data_ptr() != source.data_ptr()
    assert snapshot_state("sharing-test", source) is first
    source[0] = -0.0
    second = snapshot_state("sharing-test", source)
    assert second is not first and torch.signbit(second[0]) and not torch.signbit(first[0])
    source[1] = 4
    third = snapshot_state("sharing-test", source)
    assert third is not second and first[1] == second[1] == 0
    refs = [weakref.ref(value) for value in (first, second, third)]
    del first, second, third
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_state_snapshot_retains_noncontiguous_layout():
    source = torch.arange(40000.0).reshape(200, 200).T
    first, second = snapshot_state("strided-test", source), snapshot_state("strided-test", source)
    assert first.stride() == second.stride() == source.stride()
    assert first is not second and torch.equal(first, source)


def test_quantized_state_is_copied_without_reinterpreting_storage():
    source = torch.quantize_per_tensor(torch.ones(70000), 0.1, 0, torch.qint8)
    first, second = (
        snapshot_state("quantized-test", source),
        snapshot_state("quantized-test", source),
    )
    assert first is not second and torch.equal(first, source)


def test_releasing_graph_snapshot_keeps_complete_frozen_trace(tmp_path):
    model = torch.fx.symbolic_trace(torch.nn.Linear(3, 2).eval())
    recorder = GraphRecorder(tmp_path / "graphs", Settings(backend="eager"))
    inputs = torch.arange(6.0).reshape(2, 3).T.contiguous().T
    fn = recorder.backend(model, [inputs])
    fn(inputs)
    recorder.current_step = 1
    fn(inputs + 1)
    root = tmp_path / "graphs/graph-0000"
    manifest = verify_native(root)
    assert manifest["request_count"] == 2
    hashes = {name: file_hash(root / name) for name in manifest["files"]}
    snapshot = weakref.ref(recorder.entries[0]["requests"][0][0])
    recorder.prepare(1)
    gc.collect()
    assert snapshot() is None and recorder.entries[0]["requests"] == []
    fn(inputs + 2)
    assert recorder.summary()[0]["calls"] == 3
    assert recorder.summary()[0]["steps"] == [0, 1]
    assert {name: file_hash(root / name) for name in manifest["files"]} == hashes
    assert verify_native(root) == manifest
    payload = torch.load(root / "tensors.pt", weights_only=True)
    assert torch.equal(payload["requests"][0][0], inputs)
    assert payload["requests"][0][0].stride() == inputs.stride()
