// Types mirror INTERFACE.md (schema_version 1). Source of truth is that file —
// if these drift from it, INTERFACE.md wins. Parse defensively: unknown fields are
// ignored, optional artifacts may be absent.

export interface RunIndexEntry {
  run_id: string;
  card: string;
  dataset: string;
  created: string;
  headline_metric?: number;
}

export interface IndexFile {
  schema_version: number;
  runs: RunIndexEntry[];
}

export interface Meta {
  schema_version: number;
  run_id: string;
  card: string;
  dataset: string;
  created: string;
  git_sha?: string;
  config?: Record<string, unknown>;
}

// systems compared in the headline plot
export type SlopeSystem = "weekly_retrain" | "data_matched_control" | "ipsum";

export interface SlopePoint {
  cycle: number;
  system: string; // SlopeSystem, but tolerate unknowns
  value: number;
}

export interface SlopeFile {
  schema_version: number;
  metric_name: string;
  selection_rate_cap?: number;
  series: SlopePoint[];
}

export interface AbstractionRow {
  name: string;
  complexity: number;
  usefulness: number;
  admitted_cycle: number;
  evicted_cycle: number | null;
  payload_summary?: string;
}

export interface AbstractionSnapshot {
  cycle: number;
  abstractions: AbstractionRow[];
}

export interface AbstractionsFile {
  schema_version: number;
  snapshots: AbstractionSnapshot[];
}

export interface MetricsFile {
  schema_version: number;
  card: string;
  metrics: Record<string, number>;
  controls?: Record<string, number>;
}

export interface EventItem {
  cycle: number;
  type: string; // "admit" | "evict" | "drift" | ...
  name?: string;
  detail?: string;
}

export interface EventsFile {
  schema_version: number;
  events: EventItem[];
}

export interface RunBundle {
  meta: Meta;
  slope?: SlopeFile;
  abstractions?: AbstractionsFile;
  metrics?: MetricsFile;
  events?: EventsFile;
}
