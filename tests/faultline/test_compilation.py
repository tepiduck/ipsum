import copy
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from faultline.compilation import compare_structure
from faultline.compile_tracking import CompilationTracker, union_seconds
from faultline.portable import fingerprint


def test_interval_union_avoids_nested_compile_time():
    assert union_seconds([(0, 10e6), (2e6, 3e6), (9e6, 12e6), (15e6, 16e6)]) == 13


def fake_utils():
    metrics = []
    return SimpleNamespace(
        counters=defaultdict(dict), graph_break_reasons=[], get_compilation_metrics=lambda: metrics
    ), metrics


def test_tracker_deltas_reasons_missing_api_and_exception_passthrough():
    utils, metrics = fake_utils()
    utils.counters["frames"]["total"] = 100
    tracker = CompilationTracker(utils, "test")
    with pytest.raises(ValueError, match="original failure"), tracker.observe():
        utils.counters["frames"]["total"] += 1
        utils.counters["stats"]["unique_graphs"] = 1
        utils.counters["graph_break"]["reason"] = 2
        for _ in range(2):
            utils.graph_break_reasons.append(
                SimpleNamespace(reason="reason", user_stack=[], graph_break=True)
            )
        metrics.append(
            SimpleNamespace(
                recompile_reason="two guards failed",
                start_time_us=0,
                duration_us=100,
                is_forward=True,
            )
        )
        raise ValueError("original failure")
    result = tracker.summary()
    assert result["frame_count"] == result["graph_count"] == result["recompile_count"] == 1
    assert result["graph_break_count"] == 2
    assert len(result["graph_break_reasons"]) == 1
    assert utils.counters["frames"]["total"] == 101
    missing = CompilationTracker(SimpleNamespace(), "unknown")
    with missing.observe():
        pass
    assert missing.summary()["status"] == "unavailable"
    assert missing.summary()["frame_count"] is None


def observation(time, breaks=(), recompiles=0):
    return {
        "compilation_structure": {
            "coverage": dict.fromkeys(
                ("counts", "graph_breaks", "recompiles", "compile_time"), True
            ),
            "frame_count": 1,
            "graph_count": 1,
            "graph_break_count": len(breaks),
            "recompile_count": recompiles,
            "compile_wall_s": time,
            "graph_break_reasons": [{"reason": r, "locations": []} for r in breaks],
        }
    }


def test_advisory_fixed_thresholds_coverage_and_primary_fingerprint():
    row = {"baseline": observation(20), "candidate": observation(30, ["graph break"], 1)}
    structure = compare_structure([row], 1)
    assert structure["labels"] == [
        "compile_time_increase",
        "new_graph_breaks",
        "recompile_increase",
    ]
    assert "compile_time_increase" not in compare_structure([row], 2)["labels"]
    row["candidate"]["compilation_structure"]["compile_wall_s"] = 29.999
    assert "compile_time_increase" not in compare_structure([row])["labels"]
    row["baseline"] = {}
    assert compare_structure([row])["labels"] == []
    primary = {"status": "pass", "backend": "inductor"}
    assert fingerprint(primary) == fingerprint({**primary, "compilation_structure": structure})


class Broken(torch.nn.Module):
    def __init__(self, add_break):
        super().__init__()
        self.add_break = add_break

    def forward(self, x):
        y = x.sin()
        if self.add_break:
            torch._dynamo.graph_break()
        return y.cos()


def observed_run(model, requests, backend, dynamic=False):
    torch._dynamo.reset()
    tracker = CompilationTracker()
    compiled = torch.compile(model, backend=backend, dynamic=dynamic)
    outputs = []
    for x in requests:
        with tracker.observe():
            outputs.append(compiled(x))
    return tracker.summary(), outputs


@pytest.mark.parametrize("backend", ["eager", pytest.param("inductor", marks=pytest.mark.compiler)])
def test_actual_adapter_graph_break_and_shape_recompile(backend):
    x, y = torch.ones(2, 3), torch.ones(4, 3)
    a, original = observed_run(Broken(False), [x, y], backend, dynamic=True)
    b, changed = observed_run(Broken(True), [x, y], backend)
    assert a["status"] == b["status"] == "complete", (a, b)
    assert a["recompile_count"] == 0 and b["recompile_count"] >= 1
    assert b["graph_break_count"] >= 1 and b["graph_break_reasons"]
    assert any(reason["locations"] for reason in b["graph_break_reasons"])
    assert b["compile_wall_s"] > 0 and b["graph_count"] > a["graph_count"]
    assert torch.equal(original[0], changed[0])
    assert torch.equal(original[1], changed[1])
    row = {"baseline": {"compilation_structure": a}, "candidate": {"compilation_structure": b}}
    assert set(compare_structure([row])["labels"]) >= {"new_graph_breaks", "recompile_increase"}
    before = copy.deepcopy(b)
    assert (
        compare_structure([{"baseline": row["candidate"], "candidate": row["candidate"]}])["labels"]
        == []
    )
    assert before == b
