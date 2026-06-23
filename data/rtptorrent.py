"""Load RTPTorrent per-project CSVs into replayable CI cycles.

The per-test CSVs provide job/build IDs and test outcomes. They do not, by
themselves, provide changed-file sets; those must be joined from VCS metadata in
the later feature pipeline. This loader keeps that boundary explicit.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RTPTorrentOutcome:
    test_name: str
    failed: bool
    failures: int
    errors: int
    skipped: int
    duration: float | None


@dataclass(frozen=True)
class RTPTorrentCycle:
    cycle: int
    job_id: str
    commit_id: str | None
    outcomes: tuple[RTPTorrentOutcome, ...]
    changed_files: frozenset[str] = field(default_factory=frozenset)


def load_project_csv(
    csv_path: str | Path,
    max_cycles: int | None = None,
    changes_csv: str | Path | None = None,
) -> list[RTPTorrentCycle]:
    """Load one RTPTorrent project CSV as chronological CI cycles."""
    path = Path(csv_path)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")

        columns = _Columns(reader.fieldnames)
        grouped: dict[str, list[RTPTorrentOutcome]] = {}
        commit_ids: dict[str, str | None] = {}
        order: list[str] = []
        for row in reader:
            job_id = _required(row, columns.job, "job/build id")
            if job_id not in grouped:
                grouped[job_id] = []
                commit_ids[job_id] = _optional(row, columns.commit)
                order.append(job_id)
            failures = _int(row, columns.failures)
            errors = _int(row, columns.errors)
            skipped = _int(row, columns.skipped)
            grouped[job_id].append(
                RTPTorrentOutcome(
                    test_name=_required(row, columns.test, "test name"),
                    failed=failures > 0 or errors > 0,
                    failures=failures,
                    errors=errors,
                    skipped=skipped,
                    duration=_float_or_none(row, columns.duration),
                )
            )

    ordered_jobs = sorted(order, key=_job_sort_key)
    if max_cycles is not None:
        ordered_jobs = ordered_jobs[:max_cycles]
    cycles = [
        RTPTorrentCycle(
            cycle=idx,
            job_id=job_id,
            commit_id=commit_ids[job_id],
            outcomes=tuple(grouped[job_id]),
        )
        for idx, job_id in enumerate(ordered_jobs)
    ]
    if changes_csv is None:
        return cycles
    return attach_changed_files(cycles, changes_csv)


def attach_changed_files(
    cycles: list[RTPTorrentCycle],
    changes_csv: str | Path,
) -> list[RTPTorrentCycle]:
    """Return cycles with changed files joined from job- or commit-keyed metadata."""
    by_job, by_commit = load_changed_files_csv(changes_csv)
    joined: list[RTPTorrentCycle] = []
    for cycle in cycles:
        changed_files = by_job.get(cycle.job_id, frozenset())
        if not changed_files and cycle.commit_id is not None:
            changed_files = by_commit.get(cycle.commit_id, frozenset())
        joined.append(
            RTPTorrentCycle(
                cycle=cycle.cycle,
                job_id=cycle.job_id,
                commit_id=cycle.commit_id,
                outcomes=cycle.outcomes,
                changed_files=changed_files,
            )
        )
    return joined


def load_changed_files_csv(
    changes_csv: str | Path,
) -> tuple[dict[str, frozenset[str]], dict[str, frozenset[str]]]:
    """Load changed-file metadata indexed by job id and commit id.

    The project notes identify multiple possible metadata joins, so this parser
    accepts common aliases and both row-per-file and delimited-list formats.
    """
    path = Path(changes_csv)
    by_job: dict[str, set[str]] = {}
    by_commit: dict[str, set[str]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")

        columns = _ChangeColumns(reader.fieldnames)
        for row in reader:
            files = _files_from_row(row, columns.files)
            if not files:
                continue
            job_id = _optional(row, columns.job)
            commit_id = _optional(row, columns.commit)
            if job_id is None and commit_id is None:
                raise ValueError("changed-file metadata needs a job or commit id column")
            if job_id is not None:
                by_job.setdefault(job_id, set()).update(files)
            if commit_id is not None:
                by_commit.setdefault(commit_id, set()).update(files)

    return (
        {key: frozenset(files) for key, files in by_job.items()},
        {key: frozenset(files) for key, files in by_commit.items()},
    )


class _Columns:
    def __init__(self, fieldnames: Iterable[str]) -> None:
        self._lower = {name.lower(): name for name in fieldnames}
        self.job = self.pick("job_id", "tr_job_id", "build", "build_id")
        self.commit = self.pick("commit_id", "commit", "sha", "git_sha", required=False)
        self.test = self.pick("test", "test_name", "testcase", "test_case")
        self.failures = self.pick("failures", "failed", "failure", required=False)
        self.errors = self.pick("errors", "error", required=False)
        self.skipped = self.pick("skipped", "skip", required=False)
        self.duration = self.pick("duration", "time", "runtime", required=False)

    def pick(self, *candidates: str, required: bool = True) -> str | None:
        for candidate in candidates:
            if candidate in self._lower:
                return self._lower[candidate]
        if required:
            names = ", ".join(candidates)
            raise ValueError(f"missing required RTPTorrent column; expected one of: {names}")
        return None


class _ChangeColumns:
    def __init__(self, fieldnames: Iterable[str]) -> None:
        self._lower = {name.lower(): name for name in fieldnames}
        self.job = self.pick("job_id", "tr_job_id", "build", "build_id", required=False)
        self.commit = self.pick("commit_id", "commit", "sha", "git_sha", required=False)
        self.files = self.pick(
            "file",
            "file_path",
            "filepath",
            "path",
            "filename",
            "changed_file",
            "changed_files",
            "files",
        )

    def pick(self, *candidates: str, required: bool = True) -> str | None:
        for candidate in candidates:
            if candidate in self._lower:
                return self._lower[candidate]
        if required:
            names = ", ".join(candidates)
            raise ValueError(f"missing required changed-file column; expected one of: {names}")
        return None


def _required(row: dict[str, str], column: str | None, label: str) -> str:
    if column is None:
        raise ValueError(f"missing required {label} column")
    value = row.get(column, "").strip()
    if not value:
        raise ValueError(f"empty required {label} value")
    return value


def _optional(row: dict[str, str], column: str | None) -> str | None:
    if column is None:
        return None
    value = row.get(column, "").strip()
    return value or None


def _int(row: dict[str, str], column: str | None) -> int:
    if column is None:
        return 0
    value = row.get(column, "").strip()
    if not value:
        return 0
    return int(float(value))


def _float_or_none(row: dict[str, str], column: str | None) -> float | None:
    if column is None:
        return None
    value = row.get(column, "").strip()
    if not value:
        return None
    return float(value)


def _files_from_row(row: dict[str, str], column: str) -> set[str]:
    raw = row.get(column, "").strip()
    if not raw:
        return set()
    normalized = raw.replace("\n", ";").replace("|", ";").replace(",", ";")
    return {part.strip() for part in normalized.split(";") if part.strip()}


def _job_sort_key(job_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(job_id))
    except ValueError:
        return (1, job_id)
