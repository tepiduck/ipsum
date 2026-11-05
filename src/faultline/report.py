"""Portable, escaped reports with direct links to raw evidence."""

import html
import json
from pathlib import Path

from .storage import read_json


def write_report(directory):
    directory = Path(directory)
    summary = read_json(directory / "summary.json")
    rows = []
    for item in summary["outcomes"]:
        key = html.escape(item["id"], quote=True)
        status = html.escape(item["status"], quote=True)
        details = {
            name: item[name]
            for name in (
                "fingerprint",
                "comparison",
                "fp64_advisory",
                "exception_type",
                "error",
                "rtol",
                "atol",
                "repeats",
                "input_strides",
                "peak_tree_rss_mb",
            )
            if name in item
        }
        rows.append(
            f'<tr data-status="{status}"><td><a href="cases/{key}/case.json">{key}</a></td>'
            f"<td>{html.escape(item['tag'])}</td><td><span class='status'>{status}</span></td>"
            f'<td><a href="cases/{key}/outcome.json">result</a> · '
            f'<a href="cases/{key}/stderr.log">log</a>'
            f"<details><summary>Inspect</summary><pre>{html.escape(json.dumps(details, indent=2))}"
            f"</pre></details></td></tr>"
        )
    filters = "".join(
        f'<option value="{html.escape(s, quote=True)}">{html.escape(s)}</option>'
        for s in sorted(summary["counts"])
    )
    counts = "".join(
        f'<div class="count"><strong>{int(n)}</strong>{html.escape(s)}</div>'
        for s, n in sorted(summary["counts"].items())
    )
    badge = (
        "Controlled fault injection" if summary["injected_fault"] else "Compiler differential tests"
    )
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Faultline run</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:48px auto;padding:0 20px;color:#19242d;
background:#fafbfc}}h1{{font-size:40px;margin-bottom:0}}table{{border-collapse:collapse;width:100%}}
td,th{{padding:12px;text-align:left;border-bottom:1px solid #d6dee5;vertical-align:top}}
a{{color:#005a93}}.note{{background:#eef5f9;padding:20px;border-radius:8px}}
code,pre{{overflow-wrap:anywhere}}pre{{white-space:pre-wrap;max-width:540px;font-size:12px}}
details{{margin-top:8px}}summary{{cursor:pointer;color:#005a93}}
.counts,.filters{{display:flex;flex-wrap:wrap;gap:16px;margin:24px 0}}
.count{{background:white;border:1px solid #d6dee5;border-radius:8px;padding:16px 24px}}
.count strong{{display:block;font-size:28px}}input,select{{padding:10px;border:1px solid #9aacb9;
border-radius:5px;font:inherit}}input{{min-width:240px}}label{{display:grid;gap:6px;font-size:14px}}
.table-wrap{{overflow:auto}}.status{{font-family:monospace;font-size:13px}}
tr[data-status="numerical_mismatch"],tr[data-status="compile_error"]{{background:#fff8e8}}
[hidden]{{display:none!important}}</style><h1>Faultline</h1><p>{badge}</p>
<p>{summary["completed"]} / {summary["planned"]} cases completed.</p>
<div class="counts">{counts}</div>
<p class="note">Discrepancies require investigation. Injected faults are validation fixtures.
Failure families are not confirmed unique bugs. Timing fields are diagnostic observations.</p>
<p><a href="manifest.json">Environment and configuration</a> · <a href="summary.json">Full JSON</a></p>
<div class="filters"><label>Search cases<input id="query" type="search" placeholder="Tag or case ID"></label>
<label>Outcome<select id="status"><option value="">All outcomes</option>{filters}</select></label></div>
<p id="visible" aria-live="polite"></p><div class="table-wrap">
<table><thead><tr><th>Case</th><th>Workload</th><th>Outcome</th><th>Evidence</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>
<p>Use a case's directory with <code>faultline replay</code>, <code>faultline reduce</code>, or
<code>faultline export</code>. The terminal interface (<code>faultline ui</code>) can select cases for you.</p>
<script type="module">
const query = document.querySelector('#query');
const status = document.querySelector('#status');
const rows = [...document.querySelectorAll('tbody tr')];
function filter() {{
  const text = query.value.toLowerCase();
  for (const row of rows) {{
    row.hidden = !row.textContent.toLowerCase().includes(text) ||
      (status.value !== '' && row.dataset.status !== status.value);
  }}
  document.querySelector('#visible').textContent =
    `${{rows.filter(row => !row.hidden).length}} of ${{rows.length}} cases shown`;
}}
query.addEventListener('input', filter); status.addEventListener('change', filter); filter();
</script></html>"""
    target = directory / "report.html"
    target.write_text(document, encoding="utf-8")
    return target
