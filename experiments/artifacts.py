"""Write experiment artifacts in the shape defined by INTERFACE.md."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUNS_DIR = Path(__file__).resolve().parent / "runs"


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def make_run_id(card: str, dataset: str, created: datetime | None = None) -> str:
    timestamp = (created or utc_now()).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{card}-{dataset}"


def write_run(
    *,
    run_id: str,
    card: str,
    dataset: str,
    created: datetime,
    config: dict[str, Any],
    slope: dict[str, Any],
    metrics: dict[str, float],
    controls: dict[str, float],
    abstractions: dict[str, Any] | None = None,
    events: dict[str, Any] | None = None,
    headline_metric: float | None = None,
) -> Path:
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        run_dir / "meta.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "card": card,
            "dataset": dataset,
            "created": created.isoformat().replace("+00:00", "Z"),
            "git_sha": _git_sha(),
            "config": config,
        },
    )
    _write_json(run_dir / "slope.json", {"schema_version": 1, **slope})
    _write_json(
        run_dir / "metrics.json",
        {
            "schema_version": 1,
            "card": card,
            "metrics": metrics,
            "controls": controls,
        },
    )
    if abstractions is not None:
        _write_json(run_dir / "abstractions.json", {"schema_version": 1, **abstractions})
    if events is not None:
        _write_json(run_dir / "events.json", {"schema_version": 1, **events})

    _update_index(run_id, card, dataset, created, headline_metric)
    return run_dir


def _update_index(
    run_id: str,
    card: str,
    dataset: str,
    created: datetime,
    headline_metric: float | None,
) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / "index.json"
    if path.exists():
        with path.open() as f:
            index = json.load(f)
    else:
        index = {"schema_version": 1, "runs": []}

    runs = [run for run in index.get("runs", []) if run.get("run_id") != run_id]
    runs.insert(
        0,
        {
            "run_id": run_id,
            "card": card,
            "dataset": dataset,
            "created": created.isoformat().replace("+00:00", "Z"),
            "headline_metric": headline_metric,
        },
    )
    index = {"schema_version": 1, "runs": runs}
    _write_json(path, index)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()
