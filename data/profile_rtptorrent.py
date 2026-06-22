#!/usr/bin/env python3
"""Profile RTPTorrent CSVs to confirm project selection (see data/README.md).

RTPTorrent stores, per project, a `<project>.csv` of per-(job, test) rows with
columns roughly: job_id, [commit_id], test, index, count, failures, errors,
skipped, duration  (see Figure 1 in the MSR'20 paper). This script is defensive
about exact column names.

Usage:
    python data/profile_rtptorrent.py /path/to/rtptorrent
        # expects per-project CSVs somewhere under that dir

It prints, per project: #jobs (CI cycles), #test rows, total failing test-method
observations (the positive labels that matter), and failing-per-job — then ranks
by total positives, which is the metric that decides usefulness for measuring a slope.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("pip install pandas  (or: pip install -e '.[experiments]')")


def _col(df, *candidates):
    """Return the first matching column name (case-insensitive), or None."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def profile_one(csv_path: Path) -> dict | None:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:  # noqa: BLE001
        print(f"  ! skip {csv_path.name}: {e}")
        return None

    job = _col(df, "job_id", "tr_job_id", "build", "build_id")
    fails = _col(df, "failures", "failed", "failure")
    errs = _col(df, "errors", "error")
    if job is None or fails is None:
        return None

    n_jobs = df[job].nunique()
    failing_mask = df[fails].fillna(0) > 0
    if errs is not None:
        failing_mask = failing_mask | (df[errs].fillna(0) > 0)
    total_positive = int(failing_mask.sum())

    return {
        "project": csv_path.stem,
        "jobs": int(n_jobs),
        "rows": len(df),
        "positives": total_positive,
        "fail_per_job": round(total_positive / n_jobs, 2) if n_jobs else 0.0,
    }


def main(root: str) -> None:
    root_path = Path(root)
    csvs = [p for p in root_path.rglob("*.csv") if "built_commits" not in p.name]
    if not csvs:
        sys.exit(f"No project CSVs found under {root_path}")

    rows = [r for p in sorted(csvs) if (r := profile_one(p))]
    rows.sort(key=lambda r: r["positives"], reverse=True)

    print(f"\n{'project':22}{'jobs':>10}{'rows':>12}{'positives':>12}{'fail/job':>10}")
    print("-" * 66)
    for r in rows:
        print(f"{r['project']:22}{r['jobs']:>10}{r['rows']:>12}{r['positives']:>12}{r['fail_per_job']:>10}")
    print("\nRanked by total positive labels (most useful for measuring a slope).")
    print("Tier-1 starters per data/README.md: sling, okhttp, sonarqube.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
