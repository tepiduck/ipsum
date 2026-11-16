import argparse
import io
import urllib.error

import pytest

from faultline import sweep_models, watch, watch_ci
from faultline.storage import atomic_json, read_json
from faultline.watch_support import HistoryIncompatible, frozen_plan


def arguments(root):
    parser = argparse.ArgumentParser()
    watch.add_watch_command(parser.add_subparsers())
    return parser.parse_args(["watch", "--output", str(root), "--baseline-python", "baseline"])


@pytest.mark.parametrize(
    "code,outcome,workflow",
    [(0, "SAFE", 0), (1, "REGRESSION", 1), (2, "NEEDS_REVIEW", 0), (3, "INCONCLUSIVE", 0)],
)
def test_ci_exit_mapping_and_reasons(tmp_path, monkeypatch, capsys, code, outcome, workflow):
    root = tmp_path / "history"
    row = {
        "date": "2026-11-15",
        "model": "small",
        "reference": "stable",
        "verdict": outcome,
        "reasons": ["controlled reason"],
    }
    atomic_json(root / "history.json", {"records": [row], "planned": [row], "state": "complete"})
    monkeypatch.setattr(watch_ci, "_watch_command", lambda args: code)
    result = watch_ci.run(arguments(root))
    assert result["workflow_exit_code"] == workflow
    assert result["watch_exit_code"] == code
    assert result["outcome"] == outcome
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary"))
    watch_ci.publish(result)
    assert (tmp_path / "output").read_text() == f"exit_code={code}\n"
    if code in {2, 3}:
        assert "not a SAFE result" in (tmp_path / "summary").read_text(encoding="utf-8")
        assert "::warning::small" in capsys.readouterr().out
        assert "controlled reason" in (tmp_path / "summary").read_text(encoding="utf-8")


def test_ci_rotates_only_incompatible_identity_preserving_bytes(tmp_path, monkeypatch):
    root = tmp_path / "history"
    atomic_json(root / "watch.json", {"source": "old"})
    old = (root / "watch.json").read_bytes()
    calls = []

    def run(args):
        calls.append(args.output)
        if len(calls) == 1:
            raise HistoryIncompatible("source changed")
        assert not root.exists()
        atomic_json(root / "watch.json", {"source": "new"})
        return 0

    monkeypatch.setattr(watch_ci, "_watch_command", run)
    result = watch_ci.run(arguments(root))
    assert result["watch_exit_code"] == 0 and len(calls) == 2
    assert (tmp_path / result["archived_history"] / "watch.json").read_bytes() == old
    assert read_json(root / "watch.json")["source"] == "new"

    def corrupt(args):
        raise ValueError("committed evidence changed")

    monkeypatch.setattr(watch_ci, "_watch_command", corrupt)
    result = watch_ci.run(arguments(root))
    assert result["watch_exit_code"] == 3 and result["archived_history"] is None
    assert "committed evidence changed" in result["notes"][0]
    assert read_json(root / "watch.json")["source"] == "new"


def test_ci_exception_is_inconclusive_and_annotation_escapes(monkeypatch, tmp_path, capsys):
    def broken(args):
        raise RuntimeError("503\n::error::fake%")

    monkeypatch.setattr(watch_ci, "_watch_command", broken)
    result = watch_ci.run(arguments(tmp_path))
    assert result["workflow_exit_code"] == 0 and result["outcome"] == "INCONCLUSIVE"
    watch_ci.publish(result)
    assert "::warning::RuntimeError: 503%0A::error::fake%25" in capsys.readouterr().out


def test_latest_unpublished_falls_back_and_frozen_resume_keeps_selection(tmp_path):
    wheels = [{"date": f"2026-11-{day:02}"} for day in (13, 14, 15)]
    model = {"id": "small"}
    plan = frozen_plan(tmp_path, [model], [], 1, "2026-11-16", lambda: {"wheels": wheels})
    assert plan["selected_dates"] == ["2026-11-15"]
    assert plan["availability"]["fallback"]
    assert "2026-11-16" in plan["availability"]["reason"]
    assert "2026-11-15" in plan["availability"]["reason"]
    assert (
        frozen_plan(tmp_path, [model], [], 1, "2026-11-16", lambda: pytest.fail("rediscovered"))
        == plan
    )


@pytest.mark.parametrize("modality", ["vision", "text", "both"])
def test_input_preparation_downloads_only_required_modality(tmp_path, monkeypatch, modality):
    from PIL import Image

    urls = []

    def download(url, path):
        urls.append(url)
        if url == sweep_models.BOOK_URL:
            path.write_text(
                "Alice was beginning to get very tired\n\nsecond paragraph\n\n" + "word " * 4000
            )
        else:
            Image.new("RGB", (100, 8)).save(path)
        return {"url": url, "sha256": sweep_models.digest(path)}

    monkeypatch.setattr(sweep_models, "download", download)
    result = sweep_models.real_inputs(tmp_path / "inputs", 10, modality=modality)
    assert sweep_models.BOOK_URL not in urls
    assert (sweep_models.IMAGE_URLS[0] in urls) == (modality != "text")
    assert len(result["batches"]) == 10
    if modality != "text":
        assert [b["images"][0]["box"] for b in result["batches"]] == [
            [i * 10, 0, (i + 1) * 10, 8] for i in range(10)
        ]
    if modality != "vision":
        assert all(len(b["texts"]) == 2 for b in result["batches"])
    if modality == "both":
        assert "modality" not in result  # Existing sweep manifest shape is preserved.


def test_transient_download_retry_and_permanent_error(tmp_path, monkeypatch):
    attempts, sleeps = [], []

    def fetch(url, timeout):
        attempts.append(url)
        if len(attempts) < 3:
            raise urllib.error.HTTPError(url, 503, "unavailable", {}, None)
        return io.BytesIO(b"completed")

    monkeypatch.setattr(sweep_models.urllib.request, "urlopen", fetch)
    monkeypatch.setattr(sweep_models.time, "sleep", sleeps.append)
    target = tmp_path / "sample"
    sweep_models.download("https://example.org/sample", target)
    assert target.read_bytes() == b"completed" and sleeps == [1, 2]

    for status, count in [(404, 1), (503, 3)]:
        attempts.clear()

        def fail(url, timeout, status=status):
            attempts.append(url)
            raise urllib.error.HTTPError(url, status, "failed", {}, None)

        monkeypatch.setattr(sweep_models.urllib.request, "urlopen", fail)
        with pytest.raises(urllib.error.HTTPError):
            sweep_models.download("https://example.org/sample", tmp_path / "failed")
        assert len(attempts) == count
        assert not (tmp_path / "failed").exists()
