// Mock fixtures matching INTERFACE.md (schema_version 1). Used when no live run
// artifacts are present under /runs. Clearly synthetic — replace by reading real
// artifacts the backend emits to experiments/runs/.

import type {
  AbstractionsFile,
  EventsFile,
  IndexFile,
  Meta,
  MetricsFile,
  RunBundle,
  SlopeFile,
} from "../types";

const RUN_ID = "20260622-184500-A-synth";

// Build a plausible slope: ipsum pulls ahead of the data-matched control, which
// edges out weekly-retrain — and the ipsum-vs-control gap WIDENS over cycles.
function buildSlope(): SlopeFile {
  const series: SlopeFile["series"] = [];
  for (let cycle = 100; cycle <= 1000; cycle += 100) {
    const t = cycle / 1000;
    const weekly = 0.6 + 0.08 * t;
    const control = 0.61 + 0.12 * t;
    const ipsum = 0.61 + 0.12 * t + 0.18 * t * t; // widening gap vs control
    series.push({ cycle, system: "weekly_retrain", value: round(weekly) });
    series.push({ cycle, system: "data_matched_control", value: round(control) });
    series.push({ cycle, system: "ipsum", value: round(Math.min(ipsum, 0.97)) });
  }
  return { schema_version: 1, metric_name: "test_recall_at_selrate", selection_rate_cap: 0.33, series };
}

function round(x: number): number {
  return Math.round(x * 1000) / 1000;
}

const meta: Meta = {
  schema_version: 1,
  run_id: RUN_ID,
  card: "A",
  dataset: "synth",
  created: "2026-06-22T18:45:00Z",
  git_sha: "fixture",
  config: { n_files: 200, n_tests: 100, n_clusters: 8, p_flaky: 0.02, note: "FIXTURE DATA" },
};

const abstractions: AbstractionsFile = {
  schema_version: 1,
  snapshots: [400, 700, 1000].map((cycle) => ({
    cycle,
    abstractions: [
      { name: "auth_cluster", complexity: 1.0, usefulness: round(0.2 + cycle / 2500), admitted_cycle: 320, evicted_cycle: null, payload_summary: "files: auth.py, session.py, token.py" },
      { name: "io_cluster", complexity: 0.8, usefulness: round(0.15 + cycle / 4000), admitted_cycle: 500, evicted_cycle: null, payload_summary: "files: reader.py, writer.py" },
      { name: "legacy_cluster", complexity: 1.2, usefulness: cycle >= 1000 ? 0.05 : round(0.3 - cycle / 5000), admitted_cycle: 360, evicted_cycle: cycle >= 1000 ? 980 : null, payload_summary: "files: legacy/*.py (drifting)" },
    ].filter((a) => a.evicted_cycle === null || a.evicted_cycle <= cycle || cycle < 1000),
  })),
};

const metrics: MetricsFile = {
  schema_version: 1,
  card: "A",
  metrics: { cluster_f1: 0.78, held_out_ll_gain: 0.12 },
  controls: { admit_everything_cluster_f1: 0.31 },
};

const events: EventsFile = {
  schema_version: 1,
  events: [
    { cycle: 320, type: "admit", name: "auth_cluster" },
    { cycle: 360, type: "admit", name: "legacy_cluster" },
    { cycle: 500, type: "admit", name: "io_cluster" },
    { cycle: 900, type: "drift", detail: "clusters rewired" },
    { cycle: 980, type: "evict", name: "legacy_cluster" },
  ],
};

export const fixtureBundle: RunBundle = {
  meta,
  slope: buildSlope(),
  abstractions,
  metrics,
  events,
};

export const fixtureIndex: IndexFile = {
  schema_version: 1,
  runs: [
    { run_id: RUN_ID, card: "A", dataset: "synth", created: meta.created, headline_metric: 0.78 },
  ],
};
