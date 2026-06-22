import type { IndexFile, MetricsFile, Meta } from "../types";
import { Empty, Panel } from "./common";

export default function RunsBrowser({
  index,
  selectedRunId,
  onSelect,
  meta,
  metrics,
}: {
  index: IndexFile;
  selectedRunId: string | null;
  onSelect: (runId: string) => void;
  meta?: Meta;
  metrics?: MetricsFile;
}) {
  if (index.runs.length === 0) return <Empty msg="No runs in index.json." />;

  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <Panel title="Runs">
        <ul className="space-y-1 text-sm">
          {index.runs.map((r) => (
            <li key={r.run_id}>
              <button
                className={`w-full rounded px-2 py-1 text-left ${
                  r.run_id === selectedRunId ? "bg-blue-50 text-blue-700" : "hover:bg-slate-50"
                }`}
                onClick={() => onSelect(r.run_id)}
              >
                <div className="font-medium">{r.run_id}</div>
                <div className="text-slate-500">
                  card {r.card} · {r.dataset}
                  {r.headline_metric !== undefined ? ` · ${r.headline_metric}` : ""}
                </div>
              </button>
            </li>
          ))}
        </ul>
      </Panel>

      <div className="lg:col-span-2 space-y-4">
        <Panel title="Run">
          {meta ? (
            <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
              <dt className="text-slate-500">run_id</dt>
              <dd>{meta.run_id}</dd>
              <dt className="text-slate-500">card</dt>
              <dd>{meta.card}</dd>
              <dt className="text-slate-500">dataset</dt>
              <dd>{meta.dataset}</dd>
              <dt className="text-slate-500">created</dt>
              <dd>{meta.created}</dd>
              <dt className="text-slate-500">git_sha</dt>
              <dd>{meta.git_sha ?? "—"}</dd>
              <dt className="text-slate-500">config</dt>
              <dd className="font-mono text-xs text-slate-600">{JSON.stringify(meta.config ?? {})}</dd>
            </dl>
          ) : (
            <p className="text-sm text-slate-500">Select a run.</p>
          )}
        </Panel>

        <Panel title="Card metrics vs controls">
          {metrics ? (
            <table className="text-sm">
              <tbody>
                {Object.entries(metrics.metrics).map(([k, v]) => (
                  <tr key={k}>
                    <td className="py-1 pr-6 text-slate-500">{k}</td>
                    <td className="py-1 font-medium tabular-nums">{v}</td>
                  </tr>
                ))}
                {metrics.controls &&
                  Object.entries(metrics.controls).map(([k, v]) => (
                    <tr key={k} className="text-slate-400">
                      <td className="py-1 pr-6">{k} (control)</td>
                      <td className="py-1 tabular-nums">{v}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          ) : (
            <p className="text-sm text-slate-500">No metrics.json.</p>
          )}
        </Panel>
      </div>
    </div>
  );
}
