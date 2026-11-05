"""Bounded sequence/graph reduction, with a fresh whole-trace predicate per attempt."""

from dataclasses import replace

from .portable import REDUCIBLE, same_failure
from .reduce import bypass, shapes_and_values


def proposals(trace):
    # Large deletions first, followed by individual-request removal.
    width = max(1, len(trace.steps) // 2)
    while width:
        for start in range(len(trace.steps) - width + 1):
            steps = trace.steps[:start] + trace.steps[start + width :]
            if steps:
                yield f"remove requests {start}:{start + width}", replace(trace, steps=steps)
        width //= 2
    for label, graph in bypass(trace.steps[0]):
        names = {i.name for i in graph.inputs}
        steps = tuple(
            replace(
                step,
                nodes=graph.nodes,
                outputs=graph.outputs,
                inputs=tuple(i for i in step.inputs if i.name in names),
            )
            for step in trace.steps
        )
        yield label, replace(trace, steps=steps)
    for index, step in enumerate(trace.steps):
        for label, candidate in shapes_and_values(step):
            yield (
                f"request {index}: {label}",
                replace(trace, steps=trace.steps[:index] + (candidate,) + trace.steps[index + 1 :]),
            )


def minimize_trace(trace, oracle, max_evaluations=64, progress=None, expected=None):
    if max_evaluations < 2:
        raise ValueError("reduction needs at least two evaluations")
    trace.validate()
    baseline = oracle(trace)
    if baseline["status"] not in REDUCIBLE or (
        expected is not None and not same_failure(expected, baseline)
    ):
        raise ValueError("Saved trace failure did not reproduce; cannot reduce this outcome")
    current, calls = trace, 1
    cache = {trace.digest(): baseline}
    history = []
    exhausted = False
    while True:
        accepted = False
        for label, candidate in proposals(current):
            try:
                candidate.validate()
            except (ValueError, TypeError):
                continue
            if candidate.complexity() >= current.complexity():
                continue
            key = candidate.digest()
            if key not in cache:
                if calls >= max_evaluations - 1:
                    exhausted = True
                    break
                cache[key] = oracle(candidate)
                calls += 1
            result = cache[key]
            accepted = same_failure(baseline, result)
            event = {
                "proposal": label,
                "case_hash": key,
                "complexity": candidate.complexity(),
                "status": result["status"],
                "accepted": accepted,
            }
            history.append(event)
            if progress:
                progress(event)
            if accepted:
                current = candidate
                break
        if exhausted or not accepted:
            break
    confirmation = oracle(current)
    calls += 1
    if not same_failure(baseline, confirmation):
        raise RuntimeError("Trace failure disappeared at independent final confirmation")
    return current, {
        "schema_version": 1,
        "kind": "trace-reduction",
        "baseline": baseline,
        "confirmation": confirmation,
        "history": history,
        "evaluations": calls,
        "original_complexity": trace.complexity(),
        "reduced_complexity": current.complexity(),
        "budget_exhausted": exhausted,
        "locally_irreducible": not exhausted,
        "minimality": "Local fixed point under sequence/graph/input proposals, not global minimum.",
    }
