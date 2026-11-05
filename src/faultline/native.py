"""Supervised workloads that execute original models and capture native compiler graphs."""

import copy
import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch

from . import portable
from .environment import describe
from .native_graph import load_native_graph, mutate_native, save_native_graph
from .native_runtime import (
    check_native,
    compile_native_graph,
    load_native_factory,
    make_native_execution,
    request_from_inputs,
    tensor_leaves,
)
from .storage import atomic_json, read_json


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seal_native(directory, manifest):
    directory = Path(directory)
    names = manifest["sources"] + (["snapshot.pt"] if (directory / "snapshot.pt").exists() else [])
    if manifest["kind"] == "native-graph":
        names = ["graph.json", "tensors.pt", "forward.py"]
    manifest["files"] = {name: file_hash(directory / name) for name in names}
    atomic_json(directory / "workload.json", manifest)
    return manifest


def verify_native(directory):
    directory = Path(directory).resolve()
    manifest = read_json(directory / "workload.json")
    if manifest["schema_version"] != 1 or manifest["kind"] not in {"workload", "native-graph"}:
        raise ValueError("Unsupported workload bundle schema")
    for name, digest in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or file_hash(path) != digest:
            raise ValueError(f"Workload file is missing, modified, or outside the bundle: {name}")
    required = {"factory.py"} if manifest["kind"] == "workload" else {"graph.json", "tensors.pt"}
    if manifest.get("captured") and manifest["kind"] == "workload":
        required.add("snapshot.pt")
    if not required <= set(manifest["files"]):
        raise ValueError("Workload bundle is missing required files")
    return manifest


@dataclass(frozen=True)
class NativeJob:
    bundle: str
    bundle_hash: str
    operation: str = "evaluate"
    only_graph: int | None = None
    requests: tuple[int, ...] | None = None
    seed: int = 0

    @classmethod
    def create(cls, directory, **kwargs):
        directory = Path(directory).resolve()
        manifest = verify_native(directory)
        return cls(
            str(directory), file_hash(directory / "workload.json"), seed=manifest["seed"], **kwargs
        )

    def validate(self):
        if self.operation not in {"capture", "evaluate"}:
            raise ValueError("Unsupported native operation")
        if self.only_graph is not None and (
            type(self.only_graph) is not int or self.only_graph < 0
        ):
            raise ValueError("Invalid native graph selection")
        if self.requests is not None and (
            not self.requests or any(type(i) is not int or i < 0 for i in self.requests)
        ):
            raise ValueError("Invalid native request selection")
        if file_hash(Path(self.bundle) / "workload.json") != self.bundle_hash:
            raise ValueError("Workload manifest changed")
        verify_native(self.bundle)

    def to_dict(self):
        return {
            "kind": "native",
            "bundle": self.bundle,
            "bundle_hash": self.bundle_hash,
            "operation": self.operation,
            "only_graph": self.only_graph,
            "requests": self.requests,
            "seed": self.seed,
        }

    def digest(self):
        # Relocating an identical bundle does not change its workload identity.
        data = self.to_dict()
        data.pop("bundle")
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def capture_workload(factory, output, settings, *, trace=False, include=(), python=None):
    from .compare_envs import interpreter
    from .runner import run_case

    source, separator, function = factory.rpartition(":")
    if not separator or not function.isidentifier():
        raise ValueError("Expected model.py:factory")
    source = Path(source).resolve()
    if source.suffix != ".py" or not source.is_file():
        raise ValueError("Factory must be an existing Python file")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source, output / "factory.py")
    sources = ["factory.py"]
    for item in include:
        dependency = (source.parent / item).resolve()
        if not dependency.is_relative_to(source.parent) or not dependency.is_file():
            raise ValueError("Included file must be beneath the factory directory")
        relative = dependency.relative_to(source.parent)
        if relative.as_posix().casefold() in {
            name.casefold() for name in sources
        } or relative.parts[0].casefold() in {
            "snapshot.pt",
            "workload.json",
            "capture",
        }:
            raise ValueError("Included file collides with a reserved or existing bundle path")
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dependency, target)
        sources.append(relative.as_posix())
    seal_native(
        output,
        {
            "schema_version": 1,
            "kind": "workload",
            "factory": function,
            "sources": sources,
            "trace": trace,
            "seed": 0,
        },
    )
    job = NativeJob.create(output, operation="capture")
    result = run_case(
        job, settings, output / "capture", python=interpreter(python) if python else None
    )
    if result["status"] != "pass":
        raise ValueError(f"Workload capture returned {result['status']}; see {output / 'capture'}")
    return output


def capture_native(job):
    directory = Path(job.bundle)
    manifest = verify_native(directory)
    torch.manual_seed(job.seed)
    model, inputs = load_native_factory(directory, manifest["factory"])
    if not isinstance(model, torch.nn.Module) or any(m.training for m in model.modules()):
        raise ValueError("Native factories must return an eval() torch.nn.Module")
    requests = (
        [request_from_inputs(x) for x in inputs]
        if manifest["trace"]
        else [request_from_inputs(inputs)]
    )
    if not requests:
        raise ValueError("A workload needs at least one request")
    if any(t.is_complex() for _, t in tensor_leaves(requests)):
        raise ValueError("Complex-valued workloads are not supported")
    state = model.state_dict()
    model_storage = {t.untyped_storage().data_ptr() for t in state.values() if t.numel()}
    if any(
        t.numel() and t.untyped_storage().data_ptr() in model_storage
        for _, t in tensor_leaves(requests)
    ):
        raise ValueError("Inputs sharing storage with model state are not supported")
    trainable = {name: p.requires_grad for name, p in model.named_parameters()}
    payload = {
        "state": state,
        "requests": requests,
        "trainable": trainable,
        "rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    import numpy as np

    name, keys, position, gaussian, cached = np.random.get_state()
    payload["numpy_rng"] = (name, keys.tolist(), position, gaussian, cached)
    torch.save(payload, directory / "snapshot.pt")
    manifest.update(
        parameter_count=sum(p.numel() for p in model.parameters()),
        request_count=len(requests),
        snapshot_bytes=(directory / "snapshot.pt").stat().st_size,
        model_class=f"{type(model).__module__}.{type(model).__qualname__}",
        environment=describe(),
        captured=True,
    )
    seal_native(directory, manifest)
    return {
        "status": "pass",
        "parameter_count": manifest["parameter_count"],
        "request_count": len(requests),
        "snapshot_bytes": manifest["snapshot_bytes"],
    }


class GraphRecorder:
    def __init__(self, directory, settings, only_graph=None):
        self.directory = Path(directory)
        self.settings, self.only_graph = settings, only_graph
        self.entries = []
        self.enabled = True
        self.current_step = 0

    def backend(self, module, example_inputs):
        from torch._dynamo.backends.registry import lookup_backend

        index = len(self.entries)
        entry = {
            "id": index,
            "nodes": len(list(module.graph.nodes)),
            "calls": 0,
            "requests": [],
            "steps": [],
            "portable": False,
        }
        self.entries.append(entry)
        original = copy.deepcopy(module)
        backend = self.settings.backend
        mutant = backend.split(":", 1)[1] if backend.startswith("mutant:") else None
        selected = self.only_graph is None or index == self.only_graph
        transformed = mutate_native(module, mutant) if selected and mutant else module
        # Backend selection is explicit; the eager graphs here implement graph isolation.
        if not selected or backend == "eager" or mutant:
            compiled = transformed.forward
        else:
            compiled = lookup_backend(backend)(transformed, example_inputs)

        def execute(*args):
            entry["calls"] += 1
            if self.enabled:
                try:
                    frozen = tuple(int(a) if isinstance(a, torch.SymInt) else a for a in args)
                    # Snapshot before execution, so in-place operators cannot corrupt evidence.
                    import io

                    buffer = io.BytesIO()
                    torch.save(frozen, buffer)
                    buffer.seek(0)
                    snapshot = torch.load(buffer, map_location="cpu", weights_only=True)
                    entry["requests"].append(snapshot)
                    entry["steps"].append(self.current_step)
                    target = self.directory / f"graph-{index:04d}"
                    save_native_graph(target, original, entry["requests"])
                    seal_native(
                        target,
                        {
                            "schema_version": 1,
                            "kind": "native-graph",
                            "sources": [],
                            "seed": 0,
                            "trace": len(entry["requests"]) > 1,
                            "request_count": len(entry["requests"]),
                            "origin_graph": index,
                            "origin_steps": entry["steps"],
                            "captured": True,
                        },
                    )
                    entry["portable"] = True
                except Exception as exc:
                    entry["capture_error"] = f"{type(exc).__name__}: {exc}"
            return compiled(*args)

        return execute

    def summary(self):
        return [{k: v for k, v in entry.items() if k != "requests"} for entry in self.entries]


def evaluate_native(job, settings, directory):
    job.validate()
    settings.validate()
    unavailable = portable.configure(settings.device, settings.threads, job.seed)
    if unavailable:
        return unavailable
    manifest = verify_native(job.bundle)
    if job.operation == "capture":
        return capture_native(job)
    if settings.backend.startswith("mutant:") and settings.backend not in {
        "mutant:relu-as-neg",
        "mutant:sum-as-mean",
        "mutant:gradient-scale",
    }:
        raise ValueError("This mutant is not supported for native workloads")
    if not manifest.get("captured"):
        raise ValueError("Workload snapshot is incomplete")
    torch._dynamo.reset()
    bundle, directory = Path(job.bundle), Path(directory)
    recorders = []

    def make_execution(candidate=False):
        recorder = (
            GraphRecorder(directory / "graphs", settings, job.only_graph) if candidate else None
        )
        if recorder:
            recorders.append(recorder)

        def compile_model(model, is_graph):
            if not is_graph:
                return torch.compile(model, backend=recorder.backend, dynamic=settings.dynamic)
            return compile_native_graph(model, recorder.backend)

        return make_native_execution(
            bundle,
            manifest,
            settings.to_dict(),
            load_native_graph,
            compile_model if candidate else None,
            requests_selection=job.requests,
            before_run=(lambda n: setattr(recorder, "enabled", n == 0)) if recorder else None,
            step_hook=(lambda n: setattr(recorder, "current_step", n)) if recorder else None,
        )

    result = check_native(
        make_execution(), lambda: make_execution(True), settings.to_dict(), portable
    )
    graphs = [entry for recorder in recorders for entry in recorder.summary()]
    if not graphs and settings.backend in {"inductor", "aot_eager"} and result["status"] == "pass":
        result["status"] = "no_compiled_graph"
    atomic_json(directory / "graphs.json", graphs)
    result.update(
        graphs=graphs,
        graph_count=len(graphs),
        parameter_count=manifest.get("parameter_count"),
        request_count=len(job.requests) if job.requests else manifest["request_count"],
        native=True,
        full_workload_preserved=True,
        loss_definition="sum(output.square().mean()) over floating tensor outputs",
    )
    return result


def native_worker(data, settings, directory):
    data = {k: v for k, v in data.items() if k != "kind"}
    if data.get("requests") is not None:
        data["requests"] = tuple(data["requests"])
    job = NativeJob(**data)
    result = evaluate_native(job, settings, directory)
    result.update(
        schema_version=1,
        case_hash=job.digest(),
        backend=settings.backend,
        device=settings.device,
        gradients=settings.gradients,
        dynamic=settings.dynamic,
        trace=True,
        injected_fault=settings.backend.startswith("mutant:"),
        native=True,
        repeats=settings.repeats,
        rtol=settings.rtol,
        atol=settings.atol,
    )
    result["fingerprint"] = portable.fingerprint(result)
    return result
