"""Budgeted native graph localization and baseline-preserving delta debugging."""

import copy
import sys
from dataclasses import replace
from pathlib import Path

from .compare_envs import interpreter
from .native import NativeJob, seal_native, verify_native
from .native_artifacts import copy_workload, export_native
from .native_graph import load_native_graph, save_native_graph, tensor_metadata
from .portable import REDUCIBLE, same_failure
from .runner import run_case
from .storage import atomic_json


def graph_complexity(module, requests):
    return (
        len(requests),
        sum(node.op.startswith("call_") for node in module.graph.nodes),
        sum(x.numel() for x in requests[0] if hasattr(x, "numel")),
    )


def bypass_proposals(module, requests):
    """Prune chunks by replacing tensor-valued nodes with shape/dtype-compatible inputs."""
    calls = [node for node in module.graph.nodes if node.op.startswith("call_")]
    width = max(1, len(calls) // 2)
    seen = set()
    while width:
        for start in range(0, len(calls), width):
            names = {node.name for node in calls[start : start + width]}
            proposal = copy.deepcopy(module)
            changed = False
            for node in list(proposal.graph.nodes):
                if node.name not in names or not node.users:
                    continue
                metadata = tensor_metadata(node)
                choices = [
                    value
                    for value in node.all_input_nodes
                    if metadata and tensor_metadata(value) == metadata
                ]
                if choices:
                    node.replace_all_uses_with(choices[0])
                    changed = True
            if not changed:
                continue
            proposal.graph.eliminate_dead_code()
            placeholders = [node for node in proposal.graph.nodes if node.op == "placeholder"]
            indices = [i for i, node in enumerate(placeholders) if node.users]
            for node in placeholders:
                if not node.users:
                    proposal.graph.erase_node(node)
            proposal.graph.lint()
            proposal.recompile()
            arguments = [tuple(args[i] for i in indices) for args in requests]
            key = proposal.code
            if key not in seen and graph_complexity(proposal, arguments) < graph_complexity(
                module, requests
            ):
                seen.add(key)
                yield proposal, arguments
        width //= 2


def reduce_native(
    job,
    settings,
    expected,
    output,
    *,
    candidate_python=sys.executable,
    baseline_settings=None,
    baseline_python=None,
    max_evaluations=32,
    progress=print,
):
    if expected["status"] not in REDUCIBLE or expected["case_hash"] != job.digest():
        raise ValueError("Reduction requires a stable saved failure for this workload")
    if max_evaluations < 4:
        raise ValueError("Native reduction needs at least four evaluations")
    if (baseline_settings is None) != (baseline_python is None):
        raise ValueError("Supply both baseline settings and interpreter")
    if baseline_settings:
        baseline_policy = {k: v for k, v in baseline_settings.to_dict().items() if k != "backend"}
        candidate_policy = {k: v for k, v in settings.to_dict().items() if k != "backend"}
        if baseline_policy != candidate_policy:
            raise ValueError("Baseline and candidate comparison policies must match")
    candidate_python = interpreter(candidate_python)
    baseline_python = interpreter(baseline_python) if baseline_python else None
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    original = NativeJob.create(
        copy_workload(job.bundle, output / "original"),
        only_graph=job.only_graph,
        requests=job.requests,
        aot=job.aot,
        only_aot=job.only_aot,
    )
    attempts, calls = [], 0
    last_baseline = None

    def evaluate(proposal, label):
        nonlocal calls, last_baseline
        calls += 1
        directory = output / "attempts" / f"{calls:04d}"
        last_baseline = None
        if baseline_settings:
            last_baseline = run_case(
                proposal, baseline_settings, directory / "baseline", python=baseline_python
            )
        if last_baseline and last_baseline["status"] != "pass":
            result = {"status": "baseline_rejected", "baseline": last_baseline}
        else:
            result = run_case(proposal, settings, directory / "candidate", python=candidate_python)
        attempts.append(
            {
                "evaluation": calls,
                "stage": label,
                "status": result["status"],
                "case_hash": proposal.digest(),
                "baseline_status": last_baseline and last_baseline["status"],
            }
        )
        atomic_json(output / "attempts.json", attempts)
        if progress:
            progress(
                f"native reduction {calls}/{max_evaluations} ({label}): {result['status']}",
                flush=True,
            )
        return result, directory / "candidate"

    observed, _ = evaluate(original, "confirm-original")
    if not same_failure(expected, observed):
        evidence = {
            "confirmed": False,
            "reason": "Original failure did not survive fresh evaluation",
            "attempts": attempts,
        }
        atomic_json(output / "reduction.json", evidence)
        return evidence
    best, target = original, observed
    manifest = verify_native(original.bundle)
    count = len(best.requests) if best.requests else manifest["request_count"]
    selection = list(best.requests or range(count))
    width = max(1, count // 2)
    while width and calls < max_evaluations - 2:
        improved = False
        for start in range(0, len(selection), width):
            remaining = selection[:start] + selection[start + width :]
            if not remaining or calls >= max_evaluations - 2:
                continue
            proposal = replace(best, requests=tuple(remaining))
            result, _ = evaluate(proposal, "request-deletion")
            if same_failure(target, result):
                best, selection, target = proposal, remaining, result
                improved = True
                break
        if not improved:
            width //= 2

    localization = []
    native = manifest["kind"] == "native-graph"
    if not native:
        graphs = observed.get("aot_graphs", []) if job.aot else observed.get("graphs", [])
        if job.aot:
            preferred = "backward" if expected["status"] == "gradient_mismatch" else "forward"
            graphs = sorted(graphs, key=lambda entry: entry["stage"] != preferred)
        for graph in graphs:
            if not graph.get("portable") or calls >= max_evaluations - 2:
                continue
            isolated = replace(
                best,
                only_graph=graph["parent"] if job.aot else graph["id"],
                only_aot=graph["id"] if job.aot else None,
            )
            result, path = evaluate(isolated, "isolate-graph-in-original-model")
            record = {
                "graph": graph["id"],
                "stage": graph.get("stage", "dynamo"),
                "whole_model_failure_retained": same_failure(target, result),
            }
            localization.append(record)
            if not record["whole_model_failure_retained"]:
                continue
            graph_path = (
                path
                / "graphs"
                / graph.get("directory", f"graph-{graph['id']:04d}" if not job.aot else "missing")
            )
            if not (graph_path / "workload.json").exists() or calls >= max_evaluations - 1:
                continue
            proposal = NativeJob.create(graph_path)
            standalone, _ = evaluate(proposal, "graph-alone")
            # Graph outputs and lifted gradient names differ from full-model targets.
            # Record this separate predicate rather than pretending the fingerprints match.
            record["graph_status"] = standalone["status"]
            if standalone["status"] == target["status"]:
                best, target, native = proposal, standalone, True
                break

    before = after = None
    if native:
        module, requests, _ = load_native_graph(best.bundle)
        graph_manifest = verify_native(best.bundle)
        stage_metadata = {
            key: graph_manifest[key]
            for key in ("aot_stage", "aot_static_inputs", "aot_inference", "origin_aot_key")
            if key in graph_manifest
        }
        if best.requests is not None:
            requests = [requests[i] for i in best.requests]
        before = graph_complexity(module, requests)
        while calls < max_evaluations - 1:
            accepted = False
            for smaller, inputs in bypass_proposals(module, requests):
                if calls >= max_evaluations - 1:
                    break
                directory = output / "proposals" / f"{calls + 1:04d}"
                save_native_graph(directory, smaller, inputs)
                seal_native(
                    directory,
                    {
                        "schema_version": 2 if stage_metadata.get("aot_stage") else 1,
                        "kind": "native-graph",
                        "sources": [],
                        "seed": best.seed,
                        "trace": len(inputs) > 1,
                        "request_count": len(inputs),
                        "captured": True,
                        **stage_metadata,
                    },
                )
                proposal = NativeJob.create(directory)
                result, _ = evaluate(proposal, "graph-bypass")
                if same_failure(target, result):
                    best, target, module, requests = proposal, result, smaller, inputs
                    accepted = True
                    break
            if not accepted:
                break
        after = graph_complexity(module, requests)
    confirmation, _ = evaluate(best, "uncached-final-confirmation")
    confirmed = same_failure(target, confirmation)
    evidence = {
        "schema_version": 1,
        "confirmed": confirmed,
        "evaluations": calls,
        "budget_exhausted": calls >= max_evaluations,
        "native_graph": native,
        "graph_stage": verify_native(best.bundle).get("aot_stage", "dynamo"),
        "original_complexity": before,
        "reduced_complexity": after,
        "retained_requests": list(best.requests) if best.requests else None,
        "localization": localization,
        "confirmation": confirmation,
        "baseline_confirmation": last_baseline,
        "predicate": "same stable failure" + (" AND baseline passes" if baseline_settings else ""),
        "note": "Graph localization establishes a separate graph predicate after full-model isolation. Signatures do not prove identical root cause. No global minimality claim.",
        "injected_fault": settings.backend.startswith("mutant:"),
    }
    if confirmed:
        export_native(best, settings, output / "reproducer", confirmation)
        if baseline_settings:
            export_native(best, baseline_settings, output / "baseline-reproducer", last_baseline)
    atomic_json(output / "reduction.json", evidence)
    return evidence
