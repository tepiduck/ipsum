"""Profile an isolated source copy; numerical values and comparison rules are unchanged."""

import atexit
import functools
import json
import os
import sys
import threading
import time
import weakref
from collections import defaultdict
from pathlib import Path

import psutil


def install():
    if "--job" not in sys.argv:
        return
    import torch
    import torch._inductor.compile_fx as compiler

    from faultline import check_diagnostics as diag
    from faultline import native, native_compile, portable
    from faultline import native_precision as precision
    from faultline import native_runtime as runtime

    destination = Path(sys.argv[sys.argv.index("--job") + 1]).parent / "profile.json"
    process = psutil.Process()
    totals = defaultdict(float)
    calls = defaultdict(int)
    stack = [("worker_overhead", False)]
    stamp = time.perf_counter()
    totals["worker_startup"] = time.time() - process.create_time()
    repeats = defaultdict(float)
    marks = []
    largest = {"tree_rss_mb": 0}
    subprocess_peak = {"rss_mb": 0, "phase": None}
    models = []
    tracked = []
    live_activations = []
    execution_stack = []
    lock = threading.RLock()
    done = threading.Event()

    def charge():
        nonlocal stamp
        now = time.perf_counter()
        category, repeated = stack[-1]
        totals[category] += now - stamp
        if repeated:
            repeats[category] += now - stamp
        stamp = now

    def tensors(value, seen=None):
        seen = set() if seen is None else seen
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, dict):
            for item in list(value.values()):
                yield from tensors(item, seen)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from tensors(item, seen)

    def saved_tensors(outputs):
        pending = [v.grad_fn for v in tensors(outputs) if v.grad_fn is not None]
        seen = set()
        while pending:
            node = pending.pop()
            if node in seen or node is None:
                continue
            seen.add(node)
            for name in dir(node):
                if name == "saved_tensors" or name.startswith("_saved_"):
                    try:
                        yield from tensors(getattr(node, name))
                    except (RuntimeError, AttributeError):
                        pass
            pending.extend(x[0] for x in getattr(node, "next_functions", ()))

    def memory(label=None):
        nonlocal largest
        try:
            own = process.memory_info().rss
            children = sum(p.memory_info().rss for p in process.children(recursive=True))
        except psutil.Error:
            return
        total = (own + children) / 1024**2
        if children / 1024**2 > subprocess_peak["rss_mb"]:
            subprocess_peak.update(rss_mb=children / 1024**2, phase=stack[-1][0])
        if label is None and total <= largest["tree_rss_mb"]:
            return
        regions = {}
        priority = {
            "weights": 0,
            "retained_observations": 1,
            "gradient_copies": 1,
            "raw_gradients": 2,
            "activations": 3,
            "inputs": 4,
            "fp64_copies": -1,
        }

        def add(value, category):
            for tensor in tensors(value):
                if tensor.device.type != "cpu" or tensor.layout != torch.strided:
                    continue
                try:
                    storage = tensor.untyped_storage()
                    key = storage._cdata
                    group = "fp64_copies" if tensor.dtype == torch.float64 else category
                    if key not in regions or priority[group] < priority[regions[key][0]]:
                        regions[key] = (group, storage.nbytes())
                except RuntimeError:
                    pass

        for ref in models:
            model = ref()
            if model is not None:
                add(list(model.parameters()) + list(model.buffers()), "weights")
        for ref in live_activations:
            value = ref()
            if value is not None:
                add(value, "activations")
        for ref, category in tuple(tracked):
            value = ref()
            if value is not None:
                add(value, category)
        owned = defaultdict(int)
        for category, size in regions.values():
            owned[category] += size
        row = {
            "label": label or "sampled_peak",
            "phase": stack[-1][0],
            "tree_rss_mb": total,
            "worker_rss_mb": own / 1024**2,
            "compiler_subprocess_rss_mb": children / 1024**2,
            "known_storage_mb": {k: v / 1024**2 for k, v in owned.items()},
            "unattributed_worker_rss_mb": (own - sum(owned.values())) / 1024**2,
        }
        if total > largest["tree_rss_mb"]:
            largest = row
        if label:
            marks.append(row)

    def track(value, category):
        tracked.extend((weakref.ref(tensor), category) for tensor in tensors(value))
        if len(tracked) > 10000:
            tracked[:] = [(ref, kind) for ref, kind in tracked if ref() is not None]

    def timed(category, fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            repeated = execution_stack[-1][1] if execution_stack else False
            selected = category
            # Compilation includes all its nested serialization, tracing and comparisons.
            if any(c == "inductor_compilation" for c, _ in stack):
                selected = "inductor_compilation"
            with lock:
                charge()
                stack.append((selected, repeated))
                calls[category] += 1
            try:
                return fn(*args, **kwargs)
            finally:
                with lock:
                    charge()
                    stack.pop()

        return wrapped

    original_make = runtime.make_native_execution

    @functools.wraps(original_make)
    def make(*args, **kwargs):
        fn = timed("model_setup", original_make)(*args, **kwargs)
        closure = {}
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
            try:
                closure[name] = cell.cell_contents
            except ValueError:
                pass
        models.append(weakref.ref(closure["model"]))
        kind = (
            "fp64"
            if closure.get("precision")
            else "compiled"
            if closure.get("compile_model")
            else "eager"
        )
        count = 0

        @functools.wraps(fn)
        def execute():
            nonlocal count
            execution_stack.append((kind, count > 0))
            try:
                return timed("reset_and_observation", fn)()
            finally:
                count += 1
                execution_stack.pop()

        return execute

    original_observe = runtime.observe_native

    @functools.wraps(original_observe)
    def observe(model, fn, requests, *args, **kwargs):
        kind = execution_stack[-1][0]
        forward = timed(kind + "_forward", fn)

        def run(*a, **k):
            result = forward(*a, **k)
            track(result, "activations")
            saved = list(saved_tensors(result))
            live_activations[:] = [weakref.ref(value) for value in saved]
            memory(kind + "_after_forward")
            return result

        try:
            result = original_observe(model, run, requests, *args, **kwargs)
            for name, value in result[0].items():
                track(value, "gradient_copies" if ":gradient:" in name else "retained_observations")
            memory(kind + "_after_observation")
            return result
        finally:
            live_activations.clear()

    original_grad = torch.autograd.grad

    @functools.wraps(original_grad)
    def grad(*args, **kwargs):
        kind = execution_stack[-1][0] if execution_stack else "other"
        result = timed(kind + "_backward", original_grad)(*args, **kwargs)
        track(result, "raw_gradients")
        memory(kind + "_after_backward")
        return result

    runtime.make_native_execution = native.make_native_execution = diag.make_native_execution = make
    runtime.observe_native = observe
    torch.autograd.grad = grad
    runtime.compare_native = diag.compare_native = timed("comparison", runtime.compare_native)
    portable.reference_health = timed("comparison", portable.reference_health)
    precision.error_stats = timed("fp64_statistics", precision.error_stats)
    precision.reference_rms = timed("fp64_statistics", precision.reference_rms)
    original_load = torch.load

    @functools.wraps(original_load)
    def load(*args, **kwargs):
        result = timed("snapshot_loading", original_load)(*args, **kwargs)
        if isinstance(result, dict) and "state" in result:
            track(result["state"], "weights")
            track(result.get("requests"), "inputs")
        return result

    torch.load = load
    torch.save = timed("serialization", torch.save)
    original_record = native.GraphRecorder.record

    @functools.wraps(original_record)
    def record(self, entry, *args, **kwargs):
        result = timed("graph_capture", original_record)(self, entry, *args, **kwargs)
        track(entry["requests"], "retained_observations")
        return result

    native.GraphRecorder.record = record
    compiler._compile_fx_inner = timed("inductor_compilation", compiler._compile_fx_inner)
    native_compile.compile_graph = timed("inductor_compilation", native_compile.compile_graph)

    def sample():
        while not done.wait(0.05):
            try:
                memory()
            except (RuntimeError, TypeError, KeyError, AttributeError):
                pass

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()

    def finish():
        done.set()
        thread.join(timeout=2)
        with lock:
            charge()
        destination.write_text(
            json.dumps(
                {
                    "seconds": dict(totals),
                    "repeat_seconds": dict(repeats),
                    "torch_compilation_seconds": dict(torch._dynamo.utils.compilation_time_metrics),
                    "calls": dict(calls),
                    "peak": largest,
                    "sampled_compiler_subprocess_peak": subprocess_peak,
                    "boundaries": marks,
                    "memory_note": "Known CPU tensor storage bytes, deduplicated across aliases, at sampled boundaries. Not an allocator census. Unattributed RSS includes libraries, allocator arenas, hidden native temporaries, serialization buffers and unobserved graph storage. fp64 is an exclusive category; saved activation views may already be freed by sampling.",
                },
                indent=2,
            )
        )

    atexit.register(finish)


def main(argv=None):
    import argparse
    import shutil
    import subprocess

    import faultline

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--package", type=Path, default=Path(faultline.__file__).parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    package = root / "source/faultline"
    package.mkdir(parents=True)
    for path in args.package.glob("*.py"):
        shutil.copyfile(path, package / path.name)
    shutil.copyfile(__file__, package / "_profile_worker.py")
    bootstrap = package / "bootstrap.py"
    source = bootstrap.read_text()
    needle = "        program,\n        *map(str, arguments),"
    if source.count(needle) != 1:
        raise ValueError("Unsupported bootstrap; the original package remains unchanged")
    bootstrap.write_text(
        source.replace(
            needle,
            '        "from faultline._profile_worker import install; install()\\n" + program,\n'
            "        *map(str, arguments),",
        )
    )
    env = dict(
        os.environ,
        PYTHONPATH=str(package.parent),
        OMP_NUM_THREADS="1",
        TORCHINDUCTOR_COMPILE_THREADS="1",
    )
    command = [
        sys.executable,
        "-m",
        "faultline",
        "check",
        str(args.bundle.resolve()),
        "--baseline",
        args.baseline,
        "--candidate",
        args.candidate,
        "--level",
        "standard",
        "--output",
        str(root / "check"),
    ]
    (root / "protocol.json").write_text(
        json.dumps(
            {
                "level": "standard",
                "jobs": "auto",
                "threads": 1,
                "note": "Instrumented wall time; repeats are subsets. Storage bytes and RSS are different measurements. No numerical rules changed.",
            },
            indent=2,
        )
    )
    with (root / "check.log").open("w") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
