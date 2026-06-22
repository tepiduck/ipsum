// Data layer. Tries to read live run artifacts from /runs/* (static JSON copied or
// symlinked into public/runs). Falls back to fixtures when none are present, so the
// app runs against a blank backend. Shapes are defined in INTERFACE.md / types.ts.

import { fixtureBundle, fixtureIndex } from "./fixtures/sampleRun";
import type {
  AbstractionsFile,
  EventsFile,
  IndexFile,
  Meta,
  MetricsFile,
  RunBundle,
  SlopeFile,
} from "./types";

export type Source = "live" | "fixture";

async function fetchJson<T>(url: string): Promise<T | null> {
  try {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) return null;
    return (await r.json()) as T;
  } catch {
    return null;
  }
}

export async function loadIndex(): Promise<{ index: IndexFile; source: Source }> {
  const live = await fetchJson<IndexFile>("/runs/index.json");
  if (live && Array.isArray(live.runs) && live.runs.length > 0) {
    return { index: live, source: "live" };
  }
  return { index: fixtureIndex, source: "fixture" };
}

export async function loadRun(runId: string, source: Source): Promise<RunBundle> {
  if (source === "fixture") return fixtureBundle;

  const base = `/runs/${runId}`;
  const meta = await fetchJson<Meta>(`${base}/meta.json`);
  if (!meta) return fixtureBundle; // safety net

  // optional artifacts degrade gracefully
  const [slope, abstractions, metrics, events] = await Promise.all([
    fetchJson<SlopeFile>(`${base}/slope.json`),
    fetchJson<AbstractionsFile>(`${base}/abstractions.json`),
    fetchJson<MetricsFile>(`${base}/metrics.json`),
    fetchJson<EventsFile>(`${base}/events.json`),
  ]);

  return {
    meta,
    slope: slope ?? undefined,
    abstractions: abstractions ?? undefined,
    metrics: metrics ?? undefined,
    events: events ?? undefined,
  };
}
