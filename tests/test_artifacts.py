import json
from datetime import datetime, timezone

from experiments import artifacts

UTC = timezone.utc


def test_write_run_emits_contract_files_and_updates_index(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "RUNS_DIR", tmp_path)
    created = datetime(2026, 6, 22, 12, 0, tzinfo=UTC)

    run_dir = artifacts.write_run(
        run_id="20260622-120000-A-synth",
        card="A",
        dataset="synth",
        created=created,
        config={"seed": 1},
        slope={
            "metric_name": "cluster_f1",
            "selection_rate_cap": 0.33,
            "series": [],
        },
        metrics={"cluster_f1": 0.5},
        controls={"admit_everything_cluster_f1": 0.2},
        abstractions={"snapshots": []},
        events={"events": []},
        headline_metric=0.5,
    )

    assert (run_dir / "meta.json").exists()
    assert (run_dir / "slope.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "abstractions.json").exists()
    assert (run_dir / "events.json").exists()

    with (tmp_path / "index.json").open() as f:
        index = json.load(f)
    assert index["schema_version"] == 1
    assert index["runs"][0]["run_id"] == "20260622-120000-A-synth"
    assert index["runs"][0]["headline_metric"] == 0.5


def test_make_run_id_uses_interface_format():
    created = datetime(2026, 6, 22, 12, 34, 56, tzinfo=UTC)

    assert artifacts.make_run_id("instrument", "synth", created) == (
        "20260622-123456-instrument-synth"
    )
