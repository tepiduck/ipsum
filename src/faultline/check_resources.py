"""Memory planning for check; never changes worker outcomes or numerical policy."""

import math
import os
from pathlib import Path

import psutil

from .storage import atomic_json, read_json

DEFAULT_CAP_MB = 32768
HEADROOM = 1.25


def cpu_jobs():
    cores = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    )
    return max(1, min(4, cores // 2))


def available_memory_mb():
    available = psutil.virtual_memory().available // 1024**2
    # In containers, host MemAvailable may exceed the process's cgroup allowance.
    if Path("/sys/fs/cgroup/memory.max").is_file():
        try:
            limit = int(Path("/sys/fs/cgroup/memory.max").read_text())
            used = int(Path("/sys/fs/cgroup/memory.current").read_text())
            available = min(available, max(0, limit - used) // 1024**2)
        except (OSError, ValueError):
            pass  # "max" means no cgroup limit; use the OS availability.
    return available


def memory_budget(output, requested=None, *, resume=False):
    """Freeze the auto budget so fluctuating free RAM does not invalidate resume."""
    path = Path(output) / "memory-budget.json"
    if path.exists():
        record = read_json(path)
        if not resume or record["requested_mb"] != requested:
            raise ValueError("Memory budget exists or changed; choose a new output directory")
        return record
    available = available_memory_mb()
    chosen = requested if requested is not None else min(DEFAULT_CAP_MB, int(0.75 * available))
    if type(chosen) is not int or not 128 <= chosen <= 131072:
        raise ValueError("Insufficient available memory or invalid --max-rss-mb (128..131072)")
    record = {
        "requested_mb": requested,
        "available_mb": available,
        "cap_mb": DEFAULT_CAP_MB,
        "chosen_mb": chosen,
        "reason": "explicit" if requested is not None else "75_percent_available_capped_at_32_GiB",
    }
    atomic_json(path, record)
    return record


def memory_limited(result):
    # Never infer OOM from a generic crash/SIGKILL or timeout.
    return (
        result.get("status") in {"resource_limit", "diagnostic_skipped_memory"}
        or result.get("exception_type") in {"OutOfMemoryError", "MemoryError"}
        or "DefaultCPUAllocator: can't allocate memory" in str(result.get("error", ""))
    )


def measured_schedule(row, budget, maximum):
    outcomes = [row[side] for side in ("baseline", "candidate") if side in row]
    peaks = [outcome.get("peak_tree_rss_mb", 0) or 0 for outcome in outcomes]
    peak = max(peaks, default=0)
    valid = all(math.isfinite(value) and value > 0 for value in peaks)
    limited = any(memory_limited(outcome) for outcome in outcomes)
    jobs = max(1, min(maximum, math.floor(budget / (HEADROOM * peak)))) if valid else 1
    if limited:
        jobs = 1
    return {
        "mode": "auto",
        "measured_batch": row["batch"],
        "measured_peak_rss_mb": peak,
        "headroom_factor": HEADROOM,
        "maximum_jobs": maximum,
        "effective_jobs": jobs,
        "reason": "first_batch_memory_limit"
        if limited
        else "missing_peak_serial"
        if not valid
        else "measured_peak_with_25_percent_headroom",
    }
