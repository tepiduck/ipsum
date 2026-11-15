"""Version-dependent Dynamo telemetry adapter; never part of a numerical predicate.

Reads existing counters/metrics without an explain replay, logging reconfiguration,
counter reset, or compiler configuration changes. Missing APIs are advisory gaps.
"""

import contextlib
import copy
import math
import re
from collections import Counter
from pathlib import PureWindowsPath


def normalized(text):
    text = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", str(text))
    text = re.sub(
        r"(?:[A-Za-z]:[\\/]|/)(?:[^\s\"'<>:]+[\\/])+([^\s\"'<>:/\\]+\.py)",
        r"\1",
        text,
    )
    return re.sub(r"faultline_(?:factory|native)_[0-9a-f]+", "faultline_factory", text)


def location(filename, line, name=None):
    return {
        "file": PureWindowsPath(str(filename)).name,
        "line": line,
        "function": name,
    }


def union_seconds(intervals):
    total, end = 0.0, -math.inf
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total / 1_000_000


class CompilationTracker:
    def __init__(self, utils=None, version=None):
        self.errors = []
        self.calls = 0
        self.counters = {"frames": Counter(), "stats": Counter(), "graph_break": Counter()}
        self.breaks = []
        self.events = []
        self.intervals = []
        self.metrics_complete = True
        try:
            if utils is None:
                import torch
                from torch._dynamo import utils

                version = str(torch.__version__)
            for name in ("counters", "graph_break_reasons", "get_compilation_metrics"):
                if not hasattr(utils, name):
                    raise AttributeError(f"Dynamo adapter missing {name}")
            self.utils = utils
        except Exception as exc:
            self.utils = None
            self.errors.append(f"{type(exc).__name__}: {exc}")
        self.version = version

    def _snapshot(self):
        return (
            {name: dict(self.utils.counters.get(name, {})) for name in self.counters},
            list(self.utils.graph_break_reasons),
            list(self.utils.get_compilation_metrics()),
        )

    @contextlib.contextmanager
    def observe(self):
        before = None
        if self.utils is not None:
            try:
                before = self._snapshot()
            except Exception as exc:
                self.errors.append(f"snapshot: {type(exc).__name__}: {exc}")
        try:
            yield
        finally:
            self.calls += 1
            if before is not None:
                try:
                    self._collect(before, self._snapshot())
                except Exception as exc:
                    self.errors.append(f"collect: {type(exc).__name__}: {exc}")

    def _collect(self, before, after):
        for name, old in before[0].items():
            new = after[0][name]
            if any(new.get(key, 0) < value for key, value in old.items()):
                raise ValueError("Dynamo counters reset during a primary invocation")
            self.counters[name].update({key: value - old.get(key, 0) for key, value in new.items()})
        old_breaks = {id(item) for item in before[1]}
        for item in after[1]:
            if id(item) in old_breaks or not getattr(item, "graph_break", True):
                continue
            stack = getattr(item, "user_stack", ()) or ()
            self.breaks.append(
                {
                    "reason": str(item.reason),
                    "locations": [location(s.filename, s.lineno, s.name) for s in stack],
                }
            )
        old_metrics = {id(item) for item in before[2]}
        after_ids = {id(item) for item in after[2]}
        capacity = getattr(getattr(self.utils, "_compilation_metrics", None), "maxlen", None)
        if capacity and len(after[2]) >= capacity:
            self.metrics_complete = False
            self.errors.append("Compilation metric ring buffer reached capacity")
        if before[2] and id(before[2][-1]) not in after_ids:
            self.metrics_complete = False
            self.errors.append("Compilation metric ring buffer rolled over during invocation")
        for item in after[2]:
            if id(item) in old_metrics or getattr(item, "is_runtime", False):
                continue
            required = ("recompile_reason", "start_time_us", "duration_us", "is_forward")
            if any(not hasattr(item, name) for name in required):
                self.metrics_complete = False
                self.errors.append("Unsupported compilation metric fields")
                continue
            start, duration = item.start_time_us, item.duration_us
            if (
                not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (start, duration))
                or duration < 0
            ):
                self.metrics_complete = False
                start, duration = None, None
            else:
                self.intervals.append((start, start + duration))
            self.events.append(
                {
                    "compile_id": str(getattr(item, "compile_id", "unknown")),
                    "forward": item.is_forward,
                    "recompile_reason": item.recompile_reason,
                    "location": location(
                        getattr(item, "co_filename", "unknown"),
                        getattr(item, "co_firstlineno", None),
                        getattr(item, "co_name", None),
                    ),
                    "duration_us": duration,
                    "start_time_us": start,
                }
            )

    def summary(self):
        available = self.utils is not None and not self.errors
        breaks = list(self.breaks)
        represented = {item["reason"] for item in breaks}
        breaks += [
            {"reason": reason, "locations": []}
            for reason, count in self.counters["graph_break"].items()
            if count and reason not in represented
        ]
        unique = {}
        for item in breaks:
            item = {**item, "reason": normalized(item["reason"])}
            key = (item["reason"], str(item["locations"]))
            unique[key] = item
        recompiles = [e for e in self.events if e["forward"] is not False and e["recompile_reason"]]
        recompile_reasons = {
            (normalized(e["recompile_reason"]), str(e["location"])): {
                "reason": normalized(e["recompile_reason"]),
                "location": e["location"],
            }
            for e in recompiles
        }
        return {
            "schema_version": 1,
            "adapter": "dynamo-metrics-v1",
            "torch_version": self.version,
            "status": "complete"
            if available and self.metrics_complete
            else "partial"
            if self.utils
            else "unavailable",
            "coverage": {
                "counts": available,
                "graph_breaks": available,
                "recompiles": available and self.metrics_complete,
                "compile_time": available and self.metrics_complete,
            },
            "compiled_invocations": self.calls,
            "frame_count": self.counters["frames"].get("total", 0) if available else None,
            "graph_count": self.counters["stats"].get("unique_graphs", 0) if available else None,
            "graph_break_count": sum(self.counters["graph_break"].values()) if available else None,
            "graph_break_reasons": list(unique.values()),
            "recompile_count": len(recompiles) if available and self.metrics_complete else None,
            "recompile_reasons": list(recompile_reasons.values()),
            "compile_wall_s": union_seconds(self.intervals)
            if available and self.metrics_complete
            else None,
            "compile_events": copy.deepcopy(self.events),
            "errors": sorted(set(self.errors)),
            "scope": "Existing compiled primary invocations only; fresh workers do not track recompiles across input batches.",
        }
