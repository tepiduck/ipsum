# INTERFACE.md — the contract between backend and frontend

This is the **single source of truth** shared by the backend (Codex, `AGENTS.md`)
and the frontend (Claude Code, `CLAUDE.md`). Neither side invents data shapes.
Change this file only by agreement; bump `schema_version` when you do.

## Boundary
- **Backend owns:** `src/`, `experiments/`, `data/`, `research/`. It *produces* run
  artifacts as JSON.
- **Frontend owns:** `frontend/`. It *reads* run artifacts and renders them. It
  never imports Python or reads `src/`.
- The only thing crossing the boundary is the JSON described here.

## Where artifacts live
Each experiment run writes a directory:

```
experiments/runs/<run_id>/
  meta.json
  slope.json
  abstractions.json
  metrics.json
  events.json        # optional
experiments/runs/index.json   # list of all runs, newest first
```

`run_id` = `<YYYYMMDD-HHMMSS>-<card>-<dataset>` (e.g. `20260622-184500-A-synth`).

The frontend reads these as static JSON (a dev script may copy/symlink
`experiments/runs/` into `frontend/public/runs/`). No live server required.

## Schemas (schema_version: 1)

### index.json
```json
{ "schema_version": 1,
  "runs": [
    { "run_id": "20260622-184500-A-synth", "card": "A", "dataset": "synth",
      "created": "2026-06-22T18:45:00Z", "headline_metric": 0.83 }
  ] }
```

### meta.json
```json
{ "schema_version": 1, "run_id": "...", "card": "A|B|C|instrument",
  "dataset": "synth|sling|okhttp|sonarqube|...",
  "created": "ISO-8601", "git_sha": "abc1234",
  "config": { "any": "knobs — synth params or dataset/model settings" } }
```

### slope.json  (the headline plot)
A long-format series. `system` is the comparison line.
```json
{ "schema_version": 1,
  "metric_name": "test_recall_at_selrate",
  "selection_rate_cap": 0.33,
  "series": [
    { "cycle": 100, "system": "weekly_retrain",        "value": 0.61 },
    { "cycle": 100, "system": "data_matched_control",  "value": 0.63 },
    { "cycle": 100, "system": "ipsum",                 "value": 0.64 }
  ] }
```
`system` ∈ {`weekly_retrain`, `data_matched_control`, `ipsum`}. The thesis figure
is `ipsum` vs `data_matched_control` (gap should widen); `weekly_retrain` is table stakes.

### abstractions.json  (the inspector)
Snapshots of the store over time — powers the abstraction inspector.
```json
{ "schema_version": 1,
  "snapshots": [
    { "cycle": 100,
      "abstractions": [
        { "name": "auth_cluster", "complexity": 1.0, "usefulness": 0.42,
          "admitted_cycle": 80, "evicted_cycle": null,
          "payload_summary": "files: a.py, b.py, c.py" }
      ] } ] }
```

### metrics.json  (per-card isolating metrics)
Free-form but flat key→number, plus the controls they beat.
```json
{ "schema_version": 1,
  "card": "A",
  "metrics": { "cluster_f1": 0.78, "held_out_ll_gain": 0.12 },
  "controls": { "admit_everything_cluster_f1": 0.31 } }
```

### events.json  (optional timeline)
```json
{ "schema_version": 1,
  "events": [
    { "cycle": 80,  "type": "admit",  "name": "auth_cluster" },
    { "cycle": 220, "type": "drift",  "detail": "clusters rewired" },
    { "cycle": 240, "type": "evict",  "name": "auth_cluster" }
  ] }
```

## Rules
- Backend MUST write `meta.json` + `slope.json` for every run; the other files are
  produced when the run has that data (e.g. `abstractions.json` only for runs with a store).
- Unknown fields are allowed and ignored — add freely, don't remove without a version bump.
- Frontend MUST degrade gracefully if an optional file is missing.
- Keep numbers raw (no pre-formatting); the frontend handles display.
