import hashlib

import pytest

from faultline import sample_text, sweep_models


def test_text_batches_are_offline_and_repeatable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sweep_models, "download", lambda *a: pytest.fail("text preparation accessed network")
    )
    first = sweep_models.real_inputs(tmp_path / "one", 20, modality="text")
    second = sweep_models.real_inputs(tmp_path / "two", 20, modality="text")
    assert first == second
    assert first["sources"][0]["transport"] == "bundled"
    assert (
        hashlib.sha256((tmp_path / "one/book.txt").read_bytes()).hexdigest() == sample_text.SHA256
    )
    assert len(first["batches"]) == 20
    assert len({tuple(b["texts"]) for b in first["batches"]}) == 20
    assert all(len(t.split()) == 200 for b in first["batches"] for t in b["texts"])


def test_corrupt_bundled_text_fails_before_writing(monkeypatch, tmp_path):
    monkeypatch.setattr(sample_text, "SHA256", "0" * 64)
    target = tmp_path / "book.txt"
    with pytest.raises(ValueError, match="checksum mismatch"):
        sample_text.bundled_text(target)
    assert not target.exists()
