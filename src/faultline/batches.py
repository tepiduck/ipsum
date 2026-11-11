"""Frozen independent workloads and resumable batch comparisons."""

import hashlib
from collections import Counter
from pathlib import Path

import torch

from .native import capture_workload, file_hash, verify_native
from .native_runtime import tensor_leaves
from .replication import POLICY, accuracy_shift_replicated, batch_score
from .storage import atomic_json, campaign_lock, provenance, read_json


def tensor_digest(value):
    digest = hashlib.sha256()
    for name, tensor in tensor_leaves(value):
        digest.update(
            str(
                (
                    name,
                    str(tensor.dtype),
                    tuple(tensor.shape),
                    tensor.stride(),
                    tensor.storage_offset(),
                )
            ).encode()
        )
        digest.update(
            tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def capture_batches(
    factory, output, settings, count, *, include=(), python=None, seeded_factory=False
):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "planned": count, "policy": POLICY, "batches": []}
    atomic_json(output / "batches.json", manifest)
    seen = set()
    state_hash = None
    for index in range(count):
        try:
            root = capture_workload(
                factory,
                output / f"batch-{index:04d}",
                settings,
                include=include,
                python=python,
                input_batches=count,
                batch_index=index,
                **({"seeded_factory": True} if seeded_factory else {}),
            )
        except (ValueError, OSError) as exc:
            manifest.setdefault("preparation_errors", []).append(
                {"batch": index, "error": str(exc)}
            )
            atomic_json(output / "batches.json", manifest)
            continue
        payload = torch.load(root / "snapshot.pt", weights_only=True, map_location="cpu")
        current_state = tensor_digest(payload["state"])
        if state_hash is not None and current_state != state_hash:
            raise ValueError("Input batches must share frozen model weights")
        state_hash = current_state
        request = payload["requests"][0]
        inputs = {k: v for k, v in request.items() if k != "cotangents"}
        leaves = list(tensor_leaves(inputs))
        # Reject repeated samples, not merely repeated batch containers. Factories must
        # supply batch-major tensors; scalar/non-batch metadata is hashed with each row.
        sizes = {t.shape[0] for _, t in leaves if t.ndim}
        if len(sizes) != 1:
            raise ValueError("Independent inputs require a common leading sample dimension")
        size = next(iter(sizes), 0)
        if not size:
            raise ValueError("An input batch must contain samples")
        for i in range(size):
            sample = {name: (t[i] if t.ndim else t).contiguous().clone() for name, t in leaves}
            key = tensor_digest(sample)
            if key in seen:
                raise ValueError("Input batches contain a repeated sample")
            seen.add(key)
        manifest["batches"].append(
            {
                "batch": index,
                "bundle": root.name,
                "bundle_hash": file_hash(root / "workload.json"),
                "inputs_sha256": tensor_digest(inputs),
                "probe_sha256": tensor_digest(request["cotangents"]),
                "weights_sha256": state_hash,
            }
        )
        atomic_json(output / "batches.json", manifest)
    return output


def verify_batches(directory):
    directory = Path(directory).resolve()
    manifest = read_json(directory / "batches.json")
    count = manifest["planned"]
    if type(count) is not int or count < 1 or manifest.get("schema_version") != 1:
        raise ValueError("Invalid input batch manifest")
    ids = [row["batch"] for row in manifest["batches"]]
    if len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i < count for i in ids):
        raise ValueError("Duplicate or unplanned input batch")
    for row in manifest["batches"]:
        root = (directory / row["bundle"]).resolve()
        if (
            not root.is_relative_to(directory)
            or file_hash(root / "workload.json") != row["bundle_hash"]
        ):
            raise ValueError("Frozen input batch changed")
        verify_native(root)
        payload = torch.load(root / "snapshot.pt", weights_only=True, map_location="cpu")
        if len(payload["requests"]) != 1:
            raise ValueError("A batch must be an independent single request")
        request = payload["requests"][0]
        inputs = {k: v for k, v in request.items() if k != "cotangents"}
        expected = {
            "inputs_sha256": tensor_digest(inputs),
            "probe_sha256": tensor_digest(request.get("cotangents", {})),
            "weights_sha256": tensor_digest(payload["state"]),
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("Batch input/probe/weight identity changed")
    if len({row["weights_sha256"] for row in manifest["batches"]}) > 1:
        raise ValueError("Batch weights differ")
    return manifest


def compare_batches(
    bundle,
    output,
    baseline_python,
    candidate_python,
    settings,
    candidate_settings=None,
    reduce=False,
    max_evaluations=32,
    aot=False,
    resume=False,
):
    from .compare_envs import gate_status, interpreter
    from .native_artifacts import compare_native_environments, write_native_report
    from .sweep import runtime_environment

    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    manifest = verify_batches(bundle)
    baseline_python, candidate_python = interpreter(baseline_python), interpreter(candidate_python)
    candidate_settings = candidate_settings or settings
    config = {
        "manifest_sha256": file_hash(bundle / "batches.json"),
        "source_sha256": provenance()["faultline_source_sha256"],
        "settings": settings.to_dict(),
        "candidate_settings": candidate_settings.to_dict(),
        "baseline_python": baseline_python,
        "candidate_python": candidate_python,
        "baseline_environment": runtime_environment(baseline_python),
        "candidate_environment": runtime_environment(candidate_python),
        "aot": aot,
        "reduce": reduce,
        "max_evaluations": max_evaluations,
        "policy": POLICY,
    }
    with campaign_lock(output):
        path = output / "batch-comparison.json"
        if path.exists():
            if not resume or read_json(path) != config:
                raise ValueError(
                    "Batch comparison exists or inputs, source, settings or environments changed"
                )
        else:
            atomic_json(path, config)
        results, scores = [], []
        entries = {row["batch"]: row for row in manifest["batches"]}
        for index in range(manifest["planned"]):
            if provenance()["faultline_source_sha256"] != config["source_sha256"]:
                raise ValueError("Source changed during batch comparison")
            entry = entries.get(index)
            if entry is None:
                scores.append({"batch": index, "status": "missing"})
                continue
            root = output / f"batch-{index:04d}"
            commits = sorted(root.glob("attempt-*/commit.json"))
            if commits:
                directory = commits[-1].parent
                for name, expected in read_json(commits[-1])["files"].items():
                    target = (directory / name).resolve()
                    if not target.is_relative_to(directory) or file_hash(target) != expected:
                        raise ValueError("Committed batch evidence changed")
                result = read_json(directory / "summary.json")
            else:
                directory = root / f"attempt-{len(list(root.glob('attempt-*'))) + 1:04d}"
                result = compare_native_environments(
                    bundle / entry["bundle"],
                    directory,
                    baseline_python,
                    candidate_python,
                    settings,
                    candidate_settings,
                    reduce,
                    max_evaluations,
                    aot=aot,
                    shared_bundle=True,
                )
                names = [
                    "summary.json",
                    "baseline/outcome.json",
                    "candidate/outcome.json",
                    "baseline/job.json",
                    "candidate/job.json",
                ]
                atomic_json(
                    directory / "commit.json",
                    {"files": {name: file_hash(directory / name) for name in names}},
                )
            results.append({"batch": index, **result})
            scores.append({"batch": index, **batch_score(result["baseline"], result["candidate"])})
            summary = {
                "schema_version": 1,
                "planned": manifest["planned"],
                "batches": results,
                "accuracy_shift_replicated": accuracy_shift_replicated(scores, manifest["planned"]),
                "exit_code": gate_status(Counter(row["classification"] for row in results)),
            }
            write_native_report(output, summary)
        if not results:
            summary = {
                "schema_version": 1,
                "planned": manifest["planned"],
                "batches": [],
                "exit_code": 2,
                "accuracy_shift_replicated": accuracy_shift_replicated(scores, manifest["planned"]),
            }
        if len(entries) < manifest["planned"]:
            summary["exit_code"] = gate_status(
                Counter([r["classification"] for r in results] + ["inconclusive"])
            )
        summary["accuracy_shift_replicated"] = accuracy_shift_replicated(
            scores, manifest["planned"]
        )
        write_native_report(output, summary)
        return summary
