import { useState } from "react";
import type { AbstractionsFile, EventsFile } from "../types";
import { Empty, Panel } from "./common";

// The differentiating view: the "inspectable" bet is only real if a human can see
// the abstractions. Shows the store at a chosen cycle + an admit/evict/drift timeline.
// TODO (Claude Code): add a usefulness-over-time sparkline per abstraction by reading
//   the value across all snapshots; highlight stale (decaying) abstractions.
export default function AbstractionInspector({
  abstractions,
  events,
}: {
  abstractions?: AbstractionsFile;
  events?: EventsFile;
}) {
  const snaps = abstractions?.snapshots ?? [];
  const [idx, setIdx] = useState(Math.max(0, snaps.length - 1));

  if (snaps.length === 0) return <Empty msg="No abstractions.json for this run." />;
  const snap = snaps[Math.min(idx, snaps.length - 1)];

  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <div className="lg:col-span-2">
        <Panel title={`Abstraction store @ cycle ${snap.cycle}`}>
          <label className="mb-3 block text-sm text-slate-500">
            snapshot&nbsp;
            <select
              className="rounded border border-slate-300 px-2 py-1"
              value={idx}
              onChange={(e) => setIdx(Number(e.target.value))}
            >
              {snaps.map((s, i) => (
                <option key={s.cycle} value={i}>
                  cycle {s.cycle}
                </option>
              ))}
            </select>
          </label>
          <table className="w-full text-left text-sm">
            <thead className="text-slate-500">
              <tr>
                <th className="py-1">name</th>
                <th className="py-1">usefulness</th>
                <th className="py-1">complexity</th>
                <th className="py-1">admitted</th>
                <th className="py-1">evicted</th>
                <th className="py-1">payload</th>
              </tr>
            </thead>
            <tbody>
              {snap.abstractions.map((a) => (
                <tr key={a.name} className={a.evicted_cycle !== null ? "text-slate-400 line-through" : ""}>
                  <td className="py-1 font-medium">{a.name}</td>
                  <td className="py-1">
                    <UsefulnessBar value={a.usefulness} />
                  </td>
                  <td className="py-1">{a.complexity.toFixed(2)}</td>
                  <td className="py-1">{a.admitted_cycle}</td>
                  <td className="py-1">{a.evicted_cycle ?? "—"}</td>
                  <td className="py-1 text-slate-500">{a.payload_summary ?? ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      </div>

      <Panel title="Timeline">
        {events && events.events.length > 0 ? (
          <ul className="space-y-2 text-sm">
            {events.events
              .slice()
              .sort((a, b) => a.cycle - b.cycle)
              .map((e, i) => (
                <li key={i} className="flex items-baseline gap-2">
                  <span className="w-12 shrink-0 tabular-nums text-slate-400">{e.cycle}</span>
                  <span className={badge(e.type)}>{e.type}</span>
                  <span className="text-slate-600">{e.name ?? e.detail ?? ""}</span>
                </li>
              ))}
          </ul>
        ) : (
          <p className="text-sm text-slate-500">No events.json.</p>
        )}
      </Panel>
    </div>
  );
}

function UsefulnessBar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="flex items-center gap-2">
      <div className="h-2 w-24 rounded bg-slate-200">
        <div className="h-2 rounded bg-blue-500" style={{ width: `${pct}%` }} />
      </div>
      <span className="tabular-nums text-slate-500">{value.toFixed(2)}</span>
    </div>
  );
}

function badge(type: string): string {
  const base = "rounded px-1.5 py-0.5 text-xs font-medium ";
  if (type === "admit") return base + "bg-green-100 text-green-700";
  if (type === "evict") return base + "bg-red-100 text-red-700";
  if (type === "drift") return base + "bg-amber-100 text-amber-700";
  return base + "bg-slate-100 text-slate-600";
}
