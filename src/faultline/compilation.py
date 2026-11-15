"""Compilation-structure advisories, separate from verdict policy and signatures."""

import html
import json

POLICY = {
    "name": "compilation-structure-v1",
    "compile_time_minimum_ratio": 1.5,
    "compile_time_minimum_increase_s": 10.0,
}
FIELDS = ("frame_count", "graph_count", "graph_break_count", "recompile_count", "compile_wall_s")


def compare_structure(rows, planned=None):
    planned = len(rows) if planned is None else planned
    pairs = []
    labels = set()
    for row in rows:
        a = row.get("baseline", {}).get("compilation_structure", {})
        b = row.get("candidate", {}).get("compilation_structure", {})
        pair = {"batch": row.get("batch"), "baseline": a, "candidate": b, "labels": []}
        pair["deltas"] = {
            key: b[key] - a[key]
            if isinstance(a.get(key), (int, float)) and isinstance(b.get(key), (int, float))
            else None
            for key in FIELDS
        }
        if all(x.get("coverage", {}).get("graph_breaks") for x in (a, b)):
            old = {json.dumps(r, sort_keys=True) for r in a.get("graph_break_reasons", [])}
            new = [
                r
                for r in b.get("graph_break_reasons", [])
                if json.dumps(r, sort_keys=True) not in old
            ]
            pair["new_graph_break_reasons"] = new
            if new:
                pair["labels"].append("new_graph_breaks")
        if (
            all(x.get("coverage", {}).get("recompiles") for x in (a, b))
            and (pair["deltas"]["recompile_count"] or 0) > 0
        ):
            pair["labels"].append("recompile_increase")
        labels.update(pair["labels"])
        pairs.append(pair)
    totals = {}
    for side in ("baseline", "candidate"):
        totals[side] = {
            key: sum(p[side][key] for p in pairs)
            if pairs and all(isinstance(p[side].get(key), (int, float)) for p in pairs)
            else None
            for key in FIELDS
        }
    timing_complete = (
        len(pairs) == planned
        and planned > 0
        and all(
            p[s].get("coverage", {}).get("compile_time")
            for p in pairs
            for s in ("baseline", "candidate")
        )
    )
    timing = {
        "status": "complete" if timing_complete else "unavailable",
        "paired": len(pairs),
        "planned": planned,
    }
    if timing_complete:
        a, b = (totals[s]["compile_wall_s"] / planned for s in ("baseline", "candidate"))
        timing.update(
            baseline_mean_s=a, candidate_mean_s=b, increase_s=b - a, ratio=b / a if a else None
        )
        if (
            b - a >= POLICY["compile_time_minimum_increase_s"]
            and b >= a * POLICY["compile_time_minimum_ratio"]
        ):
            labels.add("compile_time_increase")
    return {
        "schema_version": 1,
        "advisory_only": True,
        "policy": POLICY,
        "labels": sorted(labels),
        "totals": totals,
        "timing": timing,
        "pairs": pairs,
    }


def notes(structure):
    messages = {
        "new_graph_breaks": "Candidate has new graph-break reasons/locations; see compilation structure.",
        "recompile_increase": "Candidate has more recompilation events in at least one paired batch.",
        "compile_time_increase": "Mean observed compile time increased by at least 1.5x and 10 seconds per paired batch.",
    }
    return [
        {
            "code": key,
            "batch": None,
            "message": messages[key] + " Advisory only; verdict unchanged.",
        }
        for key in structure["labels"]
    ]


def structure_html(structure):
    if not structure:
        return ""
    rows = []
    for key in FIELDS:
        a, b = (structure["totals"][side][key] for side in ("baseline", "candidate"))
        delta = b - a if a is not None and b is not None else None
        values = [key, a, b, delta]
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(v) if v is not None else 'unavailable')}</td>"
                for v in values
            )
            + "</tr>"
        )
    return (
        "<h2>Compilation structure</h2><p>Advisory only. Totals sum independent primary workers; counts include existing repeats. Compile time includes tracing and cache lookup. No cross-batch recompile tracking.</p>"
        "<table><tr><th>Metric</th><th>Baseline</th><th>Candidate</th><th>Delta</th></tr>"
        + "".join(rows)
        + "</table>"
        f"<p>Labels: {html.escape(', '.join(structure['labels']) or 'none')}</p>"
        "<details><summary>Per-batch reasons, locations and coverage</summary><pre>"
        + html.escape(json.dumps(structure, indent=2))
        + "</pre></details>"
    )
