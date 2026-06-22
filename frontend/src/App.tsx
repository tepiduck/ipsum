import { useEffect, useState } from "react";
import AbstractionInspector from "./components/AbstractionInspector";
import CompoundingChart from "./components/CompoundingChart";
import RunsBrowser from "./components/RunsBrowser";
import { loadIndex, loadRun, type Source } from "./data";
import type { IndexFile, RunBundle } from "./types";

type Tab = "compounding" | "abstractions" | "runs";

const TABS: { id: Tab; label: string }[] = [
  { id: "compounding", label: "Compounding" },
  { id: "abstractions", label: "Abstraction Inspector" },
  { id: "runs", label: "Runs" },
];

export default function App() {
  const [index, setIndex] = useState<IndexFile | null>(null);
  const [source, setSource] = useState<Source>("fixture");
  const [runId, setRunId] = useState<string | null>(null);
  const [bundle, setBundle] = useState<RunBundle | null>(null);
  const [tab, setTab] = useState<Tab>("compounding");

  useEffect(() => {
    loadIndex().then(({ index, source }) => {
      setIndex(index);
      setSource(source);
      setRunId(index.runs[0]?.run_id ?? null);
    });
  }, []);

  useEffect(() => {
    if (!runId) return;
    loadRun(runId, source).then(setBundle);
  }, [runId, source]);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-3">
          <div className="flex items-baseline gap-3">
            <h1 className="text-lg font-bold tracking-tight">ipsum</h1>
            <span className="text-sm text-slate-500">expertise compounding dashboard</span>
          </div>
          <span
            className={`rounded-full px-2 py-0.5 text-xs font-medium ${
              source === "live" ? "bg-green-100 text-green-700" : "bg-amber-100 text-amber-700"
            }`}
            title={
              source === "live"
                ? "reading live artifacts from /runs"
                : "no /runs artifacts found — showing FIXTURE data (see src/fixtures)"
            }
          >
            {source === "live" ? "live data" : "fixture data"}
          </span>
        </div>
        <nav className="mx-auto flex max-w-6xl gap-1 px-6">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`-mb-px border-b-2 px-3 py-2 text-sm ${
                tab === t.id
                  ? "border-blue-600 font-medium text-blue-700"
                  : "border-transparent text-slate-500 hover:text-slate-700"
              }`}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-6">
        {!index || !bundle ? (
          <p className="text-slate-500">Loading…</p>
        ) : tab === "compounding" ? (
          <CompoundingChart slope={bundle.slope} />
        ) : tab === "abstractions" ? (
          <AbstractionInspector abstractions={bundle.abstractions} events={bundle.events} />
        ) : (
          <RunsBrowser
            index={index}
            selectedRunId={runId}
            onSelect={setRunId}
            meta={bundle.meta}
            metrics={bundle.metrics}
          />
        )}
      </main>
    </div>
  );
}
