"""Failure-preserving, bounded greedy reduction over a data-only tensor DAG.

This finds a local fixed point under the implemented transformations, not a
globally minimal program. The oracle must independently execute each candidate.
"""

import itertools
from dataclasses import replace

from .ir import numel
from .portable import REDUCIBLE, same_failure

__all__ = ["minimize", "same_failure"]


def bypass(case):
    shapes = case.validate()
    available = [i.name for i in case.inputs]
    for node in case.nodes:
        # Same-shape substitutions avoid conflating a discrepancy with a shape error.
        choices = list(dict.fromkeys((*node.args, *reversed(available))))
        for replacement in choices:
            if shapes[replacement] != shapes[node.name]:
                continue
            nodes = tuple(
                replace(n, args=tuple(replacement if a == node.name else a for a in n.args))
                for n in case.nodes
                if n.name != node.name
            )
            outputs = tuple(replacement if o == node.name else o for o in case.outputs)
            yield (
                f"bypass {node.name} with {replacement}",
                replace(case, nodes=nodes, outputs=outputs).prune(),
            )
        available.append(node.name)


def crop_values(spec, shape):
    if spec.values is None:
        return None
    strides = [numel(spec.shape[i + 1 :]) for i in range(len(spec.shape))]
    return tuple(
        spec.values[sum(i * stride for i, stride in zip(index, strides, strict=True))]
        for index in itertools.product(*(range(d) for d in shape))
    )


def shapes_and_values(case):
    # Coordinate equal dimensions across operands so matmul/broadcast relationships
    # survive many proposals. Each complete graph is still checked independently.
    sizes = sorted({d for spec in case.inputs for d in spec.shape if d > 1}, reverse=True)
    for size in sizes:
        for smaller in dict.fromkeys([1, max(1, size // 2)]):
            inputs = []
            for spec in case.inputs:
                shape = tuple(smaller if d == size else d for d in spec.shape)
                inputs.append(replace(spec, shape=shape, values=crop_values(spec, shape)))
            nodes = tuple(
                replace(
                    n, params={"shape": [smaller if d == size else d for d in n.params["shape"]]}
                )
                if n.op == "reshape"
                else n
                for n in case.nodes
            )
            yield (
                f"shrink dimensions {size} to {smaller}",
                replace(case, inputs=tuple(inputs), nodes=nodes),
            )
    for index, spec in enumerate(case.inputs):
        # Equal dimensions need not be semantically linked. Independent shrinking
        # can simplify a batch axis while retaining the reduction/contracting axis.
        for dim, size in enumerate(spec.shape):
            if size <= 1:
                continue
            for smaller in dict.fromkeys([1, max(1, size // 2)]):
                shape = spec.shape[:dim] + (smaller,) + spec.shape[dim + 1 :]
                inputs = list(case.inputs)
                inputs[index] = replace(spec, shape=shape, values=crop_values(spec, shape))
                yield (
                    f"shrink {spec.name} axis {dim} to {smaller}",
                    replace(case, inputs=tuple(inputs)),
                )
        if spec.layout != "contiguous":
            inputs = list(case.inputs)
            inputs[index] = replace(spec, layout="contiguous")
            yield f"contiguous {spec.name}", replace(case, inputs=tuple(inputs))
        if spec.values and any(v != 0 for v in spec.values):
            # Whole-input zeroing respects expanded-layout aliasing.
            inputs = list(case.inputs)
            inputs[index] = replace(spec, values=(0.0,) * len(spec.values))
            yield f"zero {spec.name}", replace(case, inputs=tuple(inputs))


def minimize(case, oracle, max_evaluations=64, progress=None, expected=None):
    from .sequence_reduce import minimize_trace
    from .trace import Trace

    if isinstance(case, Trace):
        return minimize_trace(case, oracle, max_evaluations, progress, expected)
    if max_evaluations < 2:
        raise ValueError("reduction needs at least two evaluations")
    case.validate()
    baseline = oracle(case)
    if expected is not None and not same_failure(expected, baseline):
        raise ValueError(
            "Saved failure did not reproduce; refusing to minimize a different outcome"
        )
    if baseline["status"] not in REDUCIBLE:
        raise ValueError(f"Cannot reduce {baseline['status']}; a stable discrepancy is required")
    current = case
    cache = {case.digest(): baseline}
    history = []
    evaluations = 1
    exhausted = False
    while True:
        changed = False
        for description, candidate in itertools.chain(bypass(current), shapes_and_values(current)):
            try:
                candidate.validate()
            except ValueError:
                continue
            if candidate.complexity() >= current.complexity():
                continue
            key = candidate.digest()
            if key in cache:
                result = cache[key]
            else:
                # Reserve one independent final evaluation; cached evidence is insufficient.
                if evaluations >= max_evaluations - 1:
                    exhausted = True
                    break
                result = oracle(candidate)
                evaluations += 1
                cache[key] = result
            accepted = same_failure(baseline, result)
            event = {
                "proposal": description,
                "case_hash": key,
                "accepted": accepted,
                "status": result["status"],
                "complexity": candidate.complexity(),
            }
            history.append(event)
            if progress:
                progress(event)
            if accepted:
                current = candidate
                changed = True
                break
        if exhausted or not changed:
            break
    final = oracle(current)
    evaluations += 1
    if not same_failure(baseline, final):
        raise RuntimeError("Reduced failure did not survive independent final confirmation")
    return current, {
        "schema_version": 1,
        "original_hash": case.digest(),
        "reduced_hash": current.digest(),
        "original_complexity": case.complexity(),
        "reduced_complexity": current.complexity(),
        "evaluations": evaluations,
        "budget_exhausted": exhausted,
        "locally_irreducible": not exhausted,
        "baseline": baseline,
        "confirmation": final,
        "history": history,
        "minimality": "Local fixed point under implemented proposals, not global minimum.",
    }
