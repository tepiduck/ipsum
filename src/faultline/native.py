"""Supervised workloads that execute original models and capture native compiler graphs."""

import copy
import hashlib
import json
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch

from . import portable
from .environment import describe
from .native_graph import load_native_graph, mutate_native, save_native_graph
from .native_runtime import (
    capture_cotangents,
    check_native,
    compile_native_graph,
    load_native_factory,
    make_native_execution,
    request_from_inputs,
    tensor_leaves,
    workload_imports,
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
    if manifest["schema_version"] not in {1, 2, 3} or manifest["kind"] not in {
        "workload",
        "native-graph",
    }:
        raise ValueError("Unsupported workload bundle schema")
    if manifest["schema_version"] == 2 and (
        manifest["kind"] != "native-graph"
        or manifest.get("aot_stage") not in {"forward", "backward"}
    ):
        raise ValueError("Version-2 bundles must identify a valid AOT graph stage")
    if manifest["schema_version"] == 3 and (
        manifest["kind"] != "workload" or manifest.get("backward_probe") != "vjp"
    ):
        raise ValueError("Version-3 bundles must declare captured VJP probes")
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
    aot: bool = False
    only_aot: str | None = None

    @classmethod
    def create(cls, directory, **kwargs):
        directory = Path(directory).resolve()
        manifest = verify_native(directory)
        return cls(
            str(directory), file_hash(directory / "workload.json"), seed=manifest["seed"], **kwargs
        )

    def validate(self):
        if self.operation not in {"capture", "evaluate", "boundaries"}:
            raise ValueError("Unsupported native operation")
        if type(self.aot) is not bool or (
            self.only_aot is not None
            and (
                not self.aot
                or not isinstance(self.only_aot, str)
                or not re.fullmatch(r"\d+:(?:forward|backward):\d+", self.only_aot)
            )
        ):
            raise ValueError("Invalid AOT graph selection")
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
        data = {
            "kind": "native",
            "bundle": self.bundle,
            "bundle_hash": self.bundle_hash,
            "operation": self.operation,
            "only_graph": self.only_graph,
            "requests": self.requests,
            "seed": self.seed,
        }
        if self.aot:
            data.update(aot=True, only_aot=self.only_aot)
        return data

    def digest(self):
        # Relocating an identical bundle does not change its workload identity.
        data = self.to_dict()
        data.pop("bundle")
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def capture_workload(
    factory,
    output,
    settings,
    *,
    trace=False,
    include=(),
    python=None,
    backward_probe="vjp",
    input_batches=1,
    batch_index=None,
    seeded_factory=False,
):
    from .compare_envs import interpreter
    from .runner import run_case

    if type(input_batches) is not int or input_batches < 1:
        raise ValueError("input_batches must be positive")
    if input_batches > 1 and batch_index is None:
        from .batches import capture_batches

        if trace or backward_probe != "vjp":
            raise ValueError("Independent input batches require VJP probes and are not a trace")
        return capture_batches(
            factory,
            output,
            settings,
            input_batches,
            include=include,
            python=python,
            seeded_factory=seeded_factory,
        )

    if backward_probe not in {"loss", "vjp"}:
        raise ValueError("Backward probe must be loss or vjp")

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
            "schema_version": 3 if backward_probe == "vjp" else 1,
            **({"backward_probe": "vjp"} if backward_probe == "vjp" else {}),
            "kind": "workload",
            "factory": function,
            "sources": sources,
            "trace": trace,
            "seed": 0,
            **({"factory_seed": batch_index or 0} if seeded_factory else {}),
            **(
                {"batch_index": batch_index, "input_batches": input_batches}
                if batch_index is not None
                else {}
            ),
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
    if "batch_index" in manifest:
        import numpy as np

        random.seed(job.seed)
        np.random.seed(job.seed)
    model, inputs = load_native_factory(
        directory, manifest["factory"], seed=manifest.get("factory_seed")
    )
    if "batch_index" in manifest and "factory_seed" not in manifest:
        if not isinstance(inputs, (tuple, list)) or len(inputs) != manifest["input_batches"]:
            raise ValueError("Multi-input factory must return (model, sequence of N input batches)")
        inputs = inputs[manifest["batch_index"]]
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
    if manifest.get("backward_probe") == "vjp":
        with workload_imports(directory):
            probes = capture_cotangents(model, requests, job.seed + manifest.get("batch_index", 0))
        # Probe execution can mutate buffers and inputs. Reload the untouched snapshot.
        payload = torch.load(directory / "snapshot.pt", weights_only=True)
        for request, probe in zip(payload["requests"], probes, strict=True):
            request["cotangents"] = probe
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
    def __init__(self, directory, settings, only_graph=None, *, aot=False, only_aot=None):
        self.directory = Path(directory)
        self.settings, self.only_graph = settings, only_graph
        self.entries = []
        self.aot_entries = []
        self.aot, self.only_aot = aot, only_aot
        self.enabled = True
        self.current_step = 0

    def record(self, entry, module, args, metadata=None):
        entry["calls"] += 1
        if not self.enabled:
            return
        try:
            import io

            frozen = tuple(int(a) if isinstance(a, torch.SymInt) else a for a in args)
            buffer = io.BytesIO()
            torch.save(frozen, buffer)
            buffer.seek(0)
            snapshot = torch.load(buffer, map_location="cpu", weights_only=True)
            entry["requests"].append(snapshot)
            entry["steps"].append(self.current_step)
            target = self.directory / entry["directory"]
            save_native_graph(target, module, entry["requests"])
            seal_native(
                target,
                {
                    "schema_version": 2 if metadata and metadata.get("aot_stage") else 1,
                    "kind": "native-graph",
                    "sources": [],
                    "seed": 0,
                    "trace": len(entry["requests"]) > 1,
                    "request_count": len(entry["requests"]),
                    "origin_graph": entry.get("parent", entry["id"]),
                    "origin_steps": entry["steps"],
                    "captured": True,
                    **(metadata or {}),
                },
            )
            entry["portable"] = True
        except Exception as exc:
            entry["capture_error"] = f"{type(exc).__name__}: {exc}"

    def stage_compiler(self, parent, module, inputs, stage, options):
        from .native_aot import compile_stage

        ordinal = sum(x["parent"] == parent and x["stage"] == stage for x in self.aot_entries)
        key = f"{parent}:{stage}:{ordinal}"
        entry = {
            "id": key,
            "parent": parent,
            "stage": stage,
            "nodes": len(list(module.graph.nodes)),
            "calls": 0,
            "requests": [],
            "steps": [],
            "portable": False,
            "directory": f"aot-{len(self.aot_entries):04d}-{stage}",
        }
        self.aot_entries.append(entry)
        original = copy.deepcopy(module)
        placeholders = [node.name for node in module.graph.nodes if node.op == "placeholder"]
        metadata = {
            "aot_stage": stage,
            "origin_aot_key": key,
            "aot_static_inputs": [placeholders[i] for i in options.get("static_input_idxs", [])],
            "aot_inference": options.get("is_inference", False),
        }
        return compile_stage(
            module,
            inputs,
            self.settings.backend,
            stage,
            options=options,
            selected=self.only_aot in (None, key),
            observe=lambda values: self.record(entry, original, values, metadata),
        )

    def backend(self, module, example_inputs):
        from .native_compile import compile_graph, mode_options

        index = len(self.entries)
        entry = {
            "id": index,
            "nodes": len(list(module.graph.nodes)),
            "calls": 0,
            "requests": [],
            "steps": [],
            "portable": False,
            "directory": f"graph-{index:04d}",
        }
        self.entries.append(entry)
        original = copy.deepcopy(module)
        backend = self.settings.backend
        mutant = backend.split(":", 1)[1] if backend.startswith("mutant:") else None
        selected = self.only_graph is None or index == self.only_graph
        transformed = mutate_native(module, mutant) if selected and mutant else module
        # Backend selection is explicit; the eager graphs here implement graph isolation.
        if selected and self.aot and backend != "eager":
            from .native_aot import compile_pipeline

            compiled = compile_pipeline(
                module,
                example_inputs,
                backend,
                lambda graph, args, stage, options: self.stage_compiler(
                    index, graph, args, stage, options
                ),
                config_patches=mode_options(self.settings.to_dict()),
            )
        elif not selected or backend == "eager" or mutant:
            compiled = transformed.forward
        else:
            compiled = compile_graph(transformed, example_inputs, self.settings.to_dict())

        def execute(*args):
            self.record(entry, original, args)
            return compiled(*args)

        return execute

    def summary(self, aot=False):
        return [
            {k: v for k, v in entry.items() if k != "requests"}
            for entry in (self.aot_entries if aot else self.entries)
        ]


def evaluate_native(job, settings, directory):
    job.validate()
    settings.validate()
    unavailable = portable.configure(
        settings.device, settings.threads, job.seed, settings.deterministic
    )
    if unavailable:
        return unavailable
    manifest = verify_native(job.bundle)
    if job.operation == "capture":
        return capture_native(job)
    if job.operation == "boundaries":
        from .native_boundaries import capture_boundaries

        return capture_boundaries(job, settings, directory)
    if settings.backend.startswith("mutant:") and settings.backend not in {
        "mutant:relu-as-neg",
        "mutant:sum-as-mean",
        "mutant:gradient-scale",
        "mutant:aot-backward-scale",
    }:
        raise ValueError("This mutant is not supported for native workloads")
    if not manifest.get("captured"):
        raise ValueError("Workload snapshot is incomplete")
    if job.aot and (not settings.gradients or manifest["kind"] != "workload"):
        raise ValueError("AOT capture requires a whole-model workload and --gradients")
    if settings.backend == "mutant:aot-backward-scale" and not (
        job.aot or manifest.get("aot_stage")
    ):
        raise ValueError("The backward compiler mutant requires --aot or a saved AOT graph")
    torch._dynamo.reset()
    bundle, directory = Path(job.bundle), Path(directory)
    recorders = []

    def make_execution(candidate=False, precision=None, request_transform=None, probe=None):
        recorder = (
            GraphRecorder(
                directory / "graphs", settings, job.only_graph, aot=job.aot, only_aot=job.only_aot
            )
            if candidate and probe is None
            else None
        )
        if recorder:
            recorders.append(recorder)

        def compile_model(model, is_graph):
            if probe is not None:
                return torch.compile(model, backend=probe.backend, dynamic=settings.dynamic)
            if is_graph and manifest.get("aot_stage"):
                from .native_aot import replay_stage_backend
                from .native_compile import mode_options

                return compile_native_graph(
                    model,
                    lambda gm, args: replay_stage_backend(
                        gm, args, settings.backend, manifest, mode_options(settings.to_dict())
                    ),
                )
            if not is_graph:
                return torch.compile(model, backend=recorder.backend, dynamic=settings.dynamic)
            return compile_native_graph(model, recorder.backend)

        return make_native_execution(
            bundle,
            manifest,
            settings.to_dict(),
            load_native_graph,
            compile_model if candidate or probe else None,
            requests_selection=job.requests,
            before_run=(lambda n: setattr(recorder, "enabled", n == 0)) if recorder else None,
            step_hook=(lambda n: setattr(probe, "step", n))
            if probe
            else (lambda n: setattr(recorder, "current_step", n))
            if recorder
            else None,
            precision=precision,
            request_transform=request_transform,
            gradient_hook=probe.observe_gradients if probe else None,
        )

    def advisory(eager, candidate, primary):
        from . import native_precision
        from .native_runtime import compare_native

        atomic_json(directory / "primary-result.json", {"case_hash": job.digest(), **primary})
        control, control_observation = {}, {}
        if settings.numerics:
            from . import native_numerics

            control = native_numerics.assess_control(
                lambda transform: make_execution(request_transform=transform),
                eager,
                candidate,
                settings.to_dict(),
                compare_native,
                portable,
                native_precision.error_stats,
                observation_sink=control_observation,
            )
        fp64 = native_precision.assess_fp64(
            lambda: make_execution(precision=native_precision),
            eager,
            candidate,
            compare_native,
            portable,
            settings.repeats,
            reference_is_factory=True,
            control=control_observation.pop("observation", None),
            control_name=control.get("control"),
        )
        if settings.numerics:
            from . import native_aot, native_compile, native_numerics
            from .native_masks import MaskProbe

            masks = {
                "status": "unavailable",
                "reason": "Requires a whole-model --aot --gradients run",
            }
            if job.aot and manifest["kind"] == "workload" and settings.backend != "eager":
                torch._dynamo.reset()
                probe = MaskProbe(
                    native_aot,
                    native_compile,
                    settings.to_dict(),
                    compare_native,
                    portable,
                    {"only_graph": job.only_graph, "only_aot": job.only_aot},
                )
                masks = native_numerics.assess_masks(
                    probe,
                    lambda p: make_execution(probe=p),
                    eager,
                    candidate,
                    settings.to_dict(),
                    compare_native,
                    portable,
                )
            primary["numerical_context"] = native_numerics.annotate(primary, fp64, control, masks)
        return fp64

    from .native_aot import AOTUnsupported, annotate_aot_result, capture_context

    try:
        with capture_context(job.aot):
            result = check_native(
                make_execution(),
                lambda: make_execution(True),
                settings.to_dict(),
                portable,
                advisory if settings.fp64_advisory or settings.numerics else None,
            )
    except AOTUnsupported as exc:
        result = {"status": "unsupported_aot", "error": str(exc)}
    graphs = [entry for recorder in recorders for entry in recorder.summary()]
    if (
        not graphs
        and not manifest.get("aot_stage")
        and settings.backend in {"inductor", "aot_eager", "aot_eager_decomp_partition"}
        and result["status"] == "pass"
    ):
        result["status"] = "no_compiled_graph"
    aot_graphs = [entry for recorder in recorders for entry in recorder.summary(aot=True)]
    annotate_aot_result(
        result, manifest, job.aot, aot_graphs if settings.backend != "eager" else None
    )
    from .native_compile import annotate_compiler_result

    annotate_compiler_result(result, settings.to_dict())
    atomic_json(directory / "graphs.json", graphs)
    if job.aot:
        atomic_json(directory / "aot-graphs.json", aot_graphs)
    result.update(
        graphs=graphs,
        graph_count=len(graphs),
        parameter_count=manifest.get("parameter_count"),
        request_count=len(job.requests) if job.requests else manifest["request_count"],
        native=True,
        full_workload_preserved=True,
        loss_definition=(
            "Saved first-order backward cotangents; no second derivatives"
            if manifest.get("aot_stage") == "backward"
            else "Saved AOT forward outputs; no additional differentiation"
            if manifest.get("aot_stage")
            else "Captured Rademacher cotangents; shared vector-Jacobian product"
            if manifest.get("backward_probe") == "vjp"
            else "sum(output.square().mean()) over floating tensor outputs"
        ),
        backward_probe=manifest.get("backward_probe", "loss")
        if not manifest.get("aot_stage")
        else "aot_outputs",
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
