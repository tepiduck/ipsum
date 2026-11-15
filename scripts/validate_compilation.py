"""Exercise the version-dependent telemetry adapter with actual CPU Inductor."""

import argparse
import json
from pathlib import Path

import torch

from faultline.compilation import compare_structure
from faultline.compile_tracking import CompilationTracker
from faultline.storage import atomic_json


class Example(torch.nn.Module):
    def __init__(self, graph_break=False):
        super().__init__()
        self.graph_break = graph_break

    def forward(self, x):
        y = x.sin()
        if self.graph_break:
            torch._dynamo.graph_break()
        return y.cos()


def observe(graph_break, shapes, dynamic=False):
    torch._dynamo.reset()
    tracker = CompilationTracker()
    compiled = torch.compile(Example(graph_break), backend="inductor", dynamic=dynamic)
    values = []
    for shape in shapes:
        x = torch.ones(shape, 3, requires_grad=True)
        with tracker.observe():
            result = compiled(x)
            (gradient,) = torch.autograd.grad(result.sum(), x)
        values.append((result.detach(), gradient))
    return tracker.summary(), values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Use a new validation output")
    torch.set_num_threads(1)
    baseline, a = observe(False, [2, 4], dynamic=True)
    candidate, b = observe(True, [2, 4])
    assert baseline["status"] == candidate["status"] == "complete", (baseline, candidate)
    assert baseline["recompile_count"] == 0 and candidate["recompile_count"] > 0
    assert candidate["graph_break_count"] > baseline["graph_break_count"]
    assert any(not e["forward"] for e in candidate["compile_events"])
    assert baseline["compile_wall_s"] > 0 and candidate["compile_wall_s"] > 0
    assert all(torch.equal(x, y) for x, y in zip(a[0], b[0]))
    assert all(torch.equal(x, y) for x, y in zip(a[1], b[1]))
    structure = compare_structure(
        [
            {
                "baseline": {"compilation_structure": baseline},
                "candidate": {"compilation_structure": candidate},
            }
        ]
    )
    assert {"new_graph_breaks", "recompile_increase"} <= set(structure["labels"])
    atomic_json(
        args.output,
        {
            "torch": str(torch.__version__),
            "torch_commit": torch.version.git_version,
            "controlled": True,
            "passed": True,
            "description": "Deliberately changed graph-break/shape workload; numerical equality retained.",
            "structure": structure,
        },
    )
    print(
        json.dumps({"torch": str(torch.__version__), "passed": True, "labels": structure["labels"]})
    )


if __name__ == "__main__":
    main()
