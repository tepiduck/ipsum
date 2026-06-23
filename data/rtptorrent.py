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


@dataclass(frozen=True)
class RTPTorrentV1JoinStats:
    raw_jobs: int
    emitted_jobs: int
    jobs_with_commits: int
    jobs_with_changed_files: int
    dropped_large_change_jobs: int
    missing_commit_jobs: int
    missing_patch_jobs: int


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


def load_rtptorrent_v1_project(
    project_csv: str | Path,
    built_commits_csv: str | Path,
    patches_csv: str | Path,
    *,
    max_cycles: int | None = None,
    max_changed_files: int = 30,
) -> tuple[list[RTPTorrentCycle], RTPTorrentV1JoinStats]:
    """Load actual RTPTorrent v1 CSVs joined through commit patches.

    RTPTorrent v1 stores test outcomes, job->commit edges, and commit->file
    patches separately. Jobs may map to several commits; the changed-file set is
    the union of all patch names for those commits. Jobs whose union exceeds
    ``max_changed_files`` are dropped as infra/merge noise.
    """
    cycles = load_project_csv(project_csv, max_cycles=max_cycles)
    commits_by_job = load_built_commits_csv(built_commits_csv)
    files_by_commit = load_patches_csv(patches_csv)

    joined = []
    jobs_with_commits = 0
    jobs_with_changed_files = 0
    dropped_large = 0
    missing_commit = 0
    missing_patch = 0
    for cycle in cycles:
        commits = commits_by_job.get(cycle.job_id, ())
        if not commits:
            missing_commit += 1
        else:
            jobs_with_commits += 1
        changed_files: set[str] = set()
        missing_any_patch = False
        for commit in commits:
            files = files_by_commit.get(commit)
            if files is None:
                missing_any_patch = True
                continue
            changed_files.update(files)
        if commits and missing_any_patch and not changed_files:
            missing_patch += 1
        if len(changed_files) > max_changed_files:
            dropped_large += 1
            continue
        if changed_files:
            jobs_with_changed_files += 1
        joined.append(
            RTPTorrentCycle(
                cycle=len(joined),
                job_id=cycle.job_id,
                commit_id=";".join(commits) if commits else None,
                outcomes=cycle.outcomes,
                changed_files=frozenset(changed_files),
            )
        )

    return joined, RTPTorrentV1JoinStats(
        raw_jobs=len(cycles),
        emitted_jobs=len(joined),
        jobs_with_commits=jobs_with_commits,
        jobs_with_changed_files=jobs_with_changed_files,
        dropped_large_change_jobs=dropped_large,
        missing_commit_jobs=missing_commit,
        missing_patch_jobs=missing_patch,
    )


def load_built_commits_csv(path: str | Path) -> dict[str, tuple[str, ...]]:
    """Load RTPTorrent v1 ``tr_all_built_commits.csv``."""
    commits: dict[str, list[str]] = {}
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        lower = {name.lower(): name for name in reader.fieldnames}
        job_col = _pick_column(lower, "tr_job_id", "travisjobid", "job_id")
        commit_col = _pick_column(lower, "git_commit_id")
        for row in reader:
            job_id = _required(row, job_col, "job id")
            commit_id = _required(row, commit_col, "git commit id")
            commits.setdefault(job_id, []).append(commit_id)
    return {job_id: tuple(values) for job_id, values in commits.items()}


def load_patches_csv(path: str | Path) -> dict[str, frozenset[str]]:
    """Load RTPTorrent v1 ``<owner>@<project>-patches.csv``."""
    patches: dict[str, set[str]] = {}
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        lower = {name.lower(): name for name in reader.fieldnames}
        sha_col = _pick_column(lower, "sha")
        name_col = _pick_column(lower, "name")
        for row in reader:
            sha = _required(row, sha_col, "patch sha")
            name = _required(row, name_col, "patch file name")
            patches.setdefault(sha, set()).add(name)
    return {sha: frozenset(files) for sha, files in patches.items()}


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
        self.job = self.pick("travisjobid", "job_id", "tr_job_id", "build", "build_id")
        self.commit = self.pick("commit_id", "commit", "sha", "git_sha", required=False)
        self.test = self.pick("testname", "test", "test_name", "testcase", "test_case")
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


def _pick_column(lower: dict[str, str], *candidates: str) -> str:
    for candidate in candidates:
        if candidate in lower:
            return lower[candidate]
    names = ", ".join(candidates)
    raise ValueError(f"missing required RTPTorrent column; expected one of: {names}")


def _job_sort_key(job_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(job_id))
    except ValueError:
        return (1, job_id)
