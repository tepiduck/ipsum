import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { SlopeFile } from "../types";
import { Empty, Panel } from "./common";

const COLORS: Record<string, string> = {
  weekly_retrain: "#94a3b8",
  data_matched_control: "#f59e0b",
  ipsum: "#2563eb",
};
const LABELS: Record<string, string> = {
  weekly_retrain: "weekly retrain (table stakes)",
  data_matched_control: "data-matched control",
  ipsum: "ipsum",
};

// The headline view: value vs cycle, one line per system. The ipsum-vs-control gap
// is the thesis — widening = compounding from abstractions, not data.
export default function CompoundingChart({ slope }: { slope?: SlopeFile }) {
  if (!slope || slope.series.length === 0) return <Empty msg="No slope.json for this run." />;

  const byCycle = new Map<number, Record<string, number>>();
  const systems = new Set<string>();
  for (const p of slope.series) {
    systems.add(p.system);
    const row = byCycle.get(p.cycle) ?? { cycle: p.cycle };
    row[p.system] = p.value;
    byCycle.set(p.cycle, row);
  }
  const data = [...byCycle.values()].sort((a, b) => a.cycle - b.cycle);
  // draw ipsum last so it sits on top
  const order = ["weekly_retrain", "data_matched_control", "ipsum"];
  const sysList = [...systems].sort((a, b) => order.indexOf(a) - order.indexOf(b));

  return (
    <Panel title={`Compounding — ${slope.metric_name}${slope.selection_rate_cap ? ` @ sel≤${slope.selection_rate_cap}` : ""}`}>
      <div className="h-80 w-full">
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 8, right: 24, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="cycle" stroke="#64748b" label={{ value: "CI cycle", position: "insideBottom", offset: -4 }} />
            <YAxis stroke="#64748b" domain={[0, 1]} />
            <Tooltip />
            <Legend />
            {sysList.map((s) => (
              <Line
                key={s}
                type="monotone"
                dataKey={s}
                name={LABELS[s] ?? s}
                stroke={COLORS[s] ?? "#0f172a"}
                strokeWidth={s === "ipsum" ? 3 : 2}
                dot={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-3 text-sm text-slate-500">
        Thesis check: <span className="font-medium text-slate-700">ipsum</span> vs{" "}
        <span className="font-medium text-slate-700">data-matched control</span> — the gap must
        widen. Beating weekly-retrain alone proves nothing.
      </p>
    </Panel>
  );
}
