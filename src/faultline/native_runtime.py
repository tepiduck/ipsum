"""Whole-model execution and bounded-memory comparisons; embedded in reproducers.

Only PyTorch and the standard library are imported here. The caller supplies the
comparison policy and compiler backend, so workers and exported scripts share it.
"""

import importlib.util
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch


@contextmanager
def workload_imports(directory):
    previous = sys.path[:]
    sys.path.insert(0, str(directory))
    try:
        yield
    finally:
        sys.path[:] = previous


def load_native_factory(directory, name):
    path = Path(directory) / "factory.py"
    spec = importlib.util.spec_from_file_location("faultline_workload_factory", path)
    module = importlib.util.module_from_spec(spec)
    previous, paths = sys.modules.get(spec.name), sys.path[:]
    try:
        sys.modules[spec.name] = module
        sys.path.insert(0, str(path.parent))
        spec.loader.exec_module(module)
        return getattr(module, name)()
    finally:
        sys.path[:] = paths
        if previous is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous


def tensor_leaves(value, path="", result=None):
    result = [] if result is None else result
    if isinstance(value, torch.Tensor):
        result.append((path or "value", value))
    elif isinstance(value, dict):
        for key, item in value.items():
            tensor_leaves(item, f"{path}[{key!r}]", result)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            tensor_leaves(item, f"{path}[{index}]", result)
    elif value is not None and type(value) not in (str, int, float, bool):
        raise ValueError(f"Unsupported structured value: {type(value).__name__}")
    return result


def structure(value):
    if isinstance(value, torch.Tensor):
        return "tensor"
    if isinstance(value, dict):
        if any(type(key) not in (str, int, float, bool, type(None)) for key in value):
            raise ValueError("Output dictionary keys must be primitive scalars")
        return {"dict": [[structure(k), structure(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return {"tuple" if isinstance(value, tuple) else "list": [structure(v) for v in value]}
    if value is None or type(value) in (str, int, float, bool):
        return {"scalar": value, "type": type(value).__name__}
    raise ValueError(f"Unsupported output structure: {type(value).__name__}")


def request_from_inputs(inputs):
    if isinstance(inputs, dict) and set(inputs) == {"args", "kwargs"}:
        args, kwargs = tuple(inputs["args"]), inputs["kwargs"]
    elif isinstance(inputs, dict):
        args, kwargs = (), inputs
    elif isinstance(inputs, (tuple, list)):
        args, kwargs = tuple(inputs), {}
    else:
        args, kwargs = (inputs,), {}
    tensor_leaves((args, kwargs))
    if any(not isinstance(k, str) for k in kwargs):
        raise ValueError("Keyword argument names must be strings")
    return {"args": args, "kwargs": kwargs}


def capture_cotangents(model, requests, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    probes = []
    with torch.no_grad():
        for request in requests:
            outputs = model(*request["args"], **request["kwargs"])
            probes.append(
                {
                    name: torch.randint(0, 2, value.shape, generator=generator)
                    .float()
                    .mul_(2)
                    .sub_(1)
                    for name, value in tensor_leaves(outputs, "output")
                    if value.is_floating_point()
                }
            )
    return probes


def observe_native(model, fn, requests, gradients=False, wrong_gradient=False, step_hook=None):
    values, structures = {}, []
    for step, request in enumerate(requests):
        if step_hook:
            step_hook(step)
        args, kwargs = request["args"], request["kwargs"]
        inputs, seen = [], set()
        if gradients:
            for path, value in tensor_leaves((args, kwargs), "input"):
                if value.is_floating_point() and id(value) not in seen:
                    names = request.get("input_names")
                    # Dynamo lifts buffers and frozen parameters into graph arguments.
                    # Their captured autograd role must survive independent graph replay.
                    if names and not value.requires_grad:
                        continue
                    if not value.is_leaf:
                        raise ValueError("Frozen native inputs must be autograd leaves")
                    value.requires_grad_(True)
                    if names:
                        path = next(
                            (
                                f"graph_input:{names[i]}"
                                for i, arg in enumerate(args)
                                if arg is value
                            ),
                            path,
                        )
                    inputs.append((path, value))
                    seen.add(id(value))
        outputs = fn(*args, **kwargs)
        for name, value in tensor_leaves((args, kwargs), "input"):
            values[f"step:{step}:input_after:{name}"] = value.detach().cpu().clone()
        structures.append(structure(outputs))
        leaves = tensor_leaves(outputs, "output")
        if not leaves:
            raise ValueError("A workload must return at least one tensor")
        for path, value in leaves:
            if value.is_complex():
                raise ValueError("Complex output comparisons are unsupported")
            values[f"step:{step}:{path}"] = value.detach().cpu().clone()
        if gradients:
            differentiable = [v for _, v in leaves if v.is_floating_point()]
            if not differentiable:
                raise ValueError("Gradient checking needs a floating-point output")
            probes = request.get("cotangents")
            if probes is None:
                loss = sum(value.square().mean() for value in differentiable)
                values[f"step:{step}:loss"] = loss.detach().cpu().clone()
                backward_outputs = [loss] if loss.requires_grad else []
                vectors = None
            else:
                floating = {name: value for name, value in leaves if value.is_floating_point()}
                if set(floating) != set(probes) or any(
                    floating[name].shape != probe.shape for name, probe in probes.items()
                ):
                    raise ValueError("Output contract differs from captured cotangents")
                backward_outputs = [value for value in floating.values() if value.requires_grad]
                vectors = [
                    probes[name].to(device=value.device, dtype=value.dtype)
                    for name, value in floating.items()
                    if value.requires_grad
                ]
            targets = inputs + [
                (f"parameter:{name}", p)
                for name, p in model.named_parameters()
                if p.requires_grad and id(p) not in seen
            ]
            derivatives = (
                torch.autograd.grad(
                    backward_outputs,
                    [p for _, p in targets],
                    grad_outputs=vectors,
                    allow_unused=True,
                )
                if backward_outputs and targets
                else (None,) * len(targets)
            )
            for (name, value), derivative in zip(targets, derivatives, strict=True):
                prefix = f"step:{step}:gradient:{name}"
                values[prefix + ":present"] = torch.tensor(derivative is not None)
                # Keep only actual gradients, rather than full-sized zero placeholders.
                if derivative is not None:
                    if wrong_gradient:
                        derivative = derivative * 2
                    values[prefix + ":value"] = derivative.detach().cpu().clone()
    for name, buffer in model.named_buffers():
        values[f"state:buffer:{name}"] = buffer.detach().cpu().clone()
    for name, parameter in model.named_parameters():
        values[f"state:parameter:{name}"] = parameter.detach().cpu().clone()
    return values, structures


def compile_native_graph(model, backend):
    """Adapt the backend's flat output-list ABI to the captured FX output tuple."""
    compiled = None
    output = next(node for node in model.graph.nodes if node.op == "output")
    tuple_output = isinstance(output.args[0], tuple)

    def call(*args):
        nonlocal compiled
        if compiled is None:
            compiled = backend(model, list(args))
        result = compiled(*args)
        return tuple(result) if tuple_output and isinstance(result, list) else result

    return call


def compare_native(reference, candidate, policy, rtol=None, atol=None):
    left, left_structure = reference
    right, right_structure = candidate
    outputs = []
    for index, name in enumerate(sorted(set(left) | set(right))):
        if name not in left or name not in right:
            item = {
                "matches": False,
                "kind": "structure",
                "missing": "reference" if name not in left else "candidate",
            }
        else:
            dtype = str(left[name].dtype).removeprefix("torch.")
            limits = (
                policy.tolerances(dtype, rtol, atol) if left[name].is_floating_point() else (0, 0)
            )
            if not left[name].is_floating_point():
                item = {
                    "matches": left[name].dtype == right[name].dtype
                    and torch.equal(left[name], right[name]),
                    "kind": "numeric"
                    if left[name].shape == right[name].shape
                    and left[name].dtype == right[name].dtype
                    else "structure",
                }
            # Empty tensors are valid real-world outputs; the small-IR comparator assumes nonempty.
            elif left[name].numel() == right[name].numel() == 0:
                item = {
                    "matches": left[name].shape == right[name].shape
                    and left[name].dtype == right[name].dtype,
                    "kind": "structure",
                    "elements": 0,
                }
            else:
                item = policy.compare((left[name],), (right[name],), *limits)["outputs"][0]
        outputs.append({**item, "index": index, "target": name})
    if left_structure != right_structure:
        outputs.append(
            {
                "index": len(outputs),
                "target": "output_structure",
                "matches": False,
                "kind": "structure",
            }
        )
    return {
        "matches": all(x["matches"] for x in outputs),
        "outputs": outputs,
        "kind": "structure"
        if any(not x["matches"] and x["kind"] == "structure" for x in outputs)
        else "numeric",
    }


def native_failure_key(comparison):
    return tuple(
        (item["target"], item["kind"], bool(item.get("candidate_nonfinite", 0)))
        for item in comparison["outputs"]
        if not item["matches"]
    )


def check_native(reference, candidate_factory, settings, policy, advisory=None):
    """Keep two anchor observations in memory, irrespective of the repeat count."""
    rtol, atol = settings.get("rtol"), settings.get("atol")
    repeats = settings["repeats"]
    try:
        first = reference()
        if not policy.reference_health(tuple(first[0].values())):
            return {"status": "reference_nonfinite"}
        for _ in range(repeats - 1):
            other = reference()
            if (
                not policy.reference_health(tuple(other[0].values()))
                or not compare_native(first, other, policy, rtol, atol)["matches"]
            ):
                return {"status": "unstable_reference"}
            del other
    except Exception as exc:
        return policy.exception_record(exc, "reference_error")

    def cold():
        candidate = candidate_factory()
        started = time.perf_counter()
        return candidate, candidate(), time.perf_counter() - started

    try:
        candidate, initial, duration = cold()
    except Exception as exc:
        return policy.confirm_exception(exc, "compile_error", cold, repeats)
    initial_comparison = compare_native(first, initial, policy, rtol, atol)
    key = native_failure_key(initial_comparison)
    times = []
    for _ in range(repeats):
        try:
            started = time.perf_counter()
            observed = candidate()
            times.append(time.perf_counter() - started)
        except Exception as exc:
            return policy.confirm_exception(exc, "runtime_error", candidate, repeats)
        compared = compare_native(first, observed, policy, rtol, atol)
        if (
            native_failure_key(compared) != key
            or not compare_native(initial, observed, policy, rtol, atol)["matches"]
        ):
            return {
                "status": "unstable_candidate",
                "reason": "Repeated observation or failure signature changed",
                "cold_comparison": initial_comparison,
                "comparison": compared,
            }
        del observed
    failures = [x for x in initial_comparison["outputs"] if not x["matches"]]
    status = (
        "pass"
        if not failures
        else "gradient_mismatch"
        if all(":gradient:" in x["target"] for x in failures)
        else "structural_mismatch"
        if initial_comparison["kind"] == "structure"
        else "numerical_mismatch"
    )
    result = {
        "status": status,
        "comparison": initial_comparison,
        "cold_execution_s": duration,
        "warm_execution_s": times,
        "timing_note": "Includes reset and observation overhead; not a benchmark.",
    }
    if advisory is not None:
        result["fp64_advisory"] = advisory(first, initial, result)
    return result


def reset_native_rng(payload, seed):
    random.setstate(payload["python_rng"]) if "python_rng" in payload else random.seed(seed)
    if "numpy_rng" in payload:
        import numpy as np

        name, keys, position, gaussian, cached = payload["numpy_rng"]
        np.random.set_state((name, np.array(keys, dtype="uint32"), position, gaussian, cached))
    torch.set_rng_state(payload["rng"].cpu())
    if torch.cuda.is_available() and payload.get("cuda_rng"):
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])
    else:
        torch.manual_seed(seed)
        torch.set_rng_state(payload["rng"].cpu())


def make_native_execution(
    bundle,
    manifest,
    settings,
    graph_loader,
    compile_model=None,
    requests_selection=None,
    before_run=None,
    step_hook=None,
    precision=None,
):
    """Construct one eager/compiled callable, restoring saved state for each whole trace."""
    bundle = Path(bundle)
    device = settings["device"]
    gradients = settings.get("gradients", False) and not manifest.get("aot_stage")
    is_graph = manifest["kind"] == "native-graph"
    if is_graph:
        model, _, spec = graph_loader(bundle, device)
        names = [node["name"] for node in spec["nodes"] if node["op"] == "placeholder"]
        initial_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    else:
        model, _ = load_native_factory(bundle, manifest["factory"])
        if any(module.training for module in model.modules()):
            raise ValueError("The workload factory no longer returns an eval model")
        model.to(device)
    model.eval()
    if precision:
        precision.promote_model(model)
    fn = compile_model(model, is_graph) if compile_model else model
    if settings.get("autocast") and not manifest.get("aot_stage") and not precision:
        forward = fn

        def autocast_forward(*args, **kwargs):
            # Scope AMP to forward; the loss and backward run outside autocast.
            with torch.autocast(
                device_type=torch.device(device).type, dtype=getattr(torch, settings["autocast"])
            ):
                return forward(*args, **kwargs)

        fn = autocast_forward
    invocations = 0

    def execute():
        nonlocal invocations
        if is_graph:
            payload = torch.load(bundle / "tensors.pt", map_location=device, weights_only=True)
            model.load_state_dict(initial_state)
            requests = [
                {"args": args, "kwargs": {}, "input_names": names} for args in payload["requests"]
            ]
            torch.manual_seed(manifest["seed"])
        else:
            payload = torch.load(bundle / "snapshot.pt", map_location=device, weights_only=True)
            model.load_state_dict(payload["state"], strict=True)
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(payload["trainable"][name])
            requests = payload["requests"]
            if manifest.get("backward_probe") == "vjp" and any(
                not isinstance(request.get("cotangents"), dict) for request in requests
            ):
                raise ValueError("Captured VJP bundle is missing its cotangents")
            reset_native_rng(payload, manifest["seed"])
        if requests_selection is not None:
            requests = [requests[i] for i in requests_selection]
        if precision:
            requests = precision.promote_tree(requests)
        if before_run:
            before_run(invocations)
        with (
            workload_imports(bundle),
            torch.enable_grad() if gradients else torch.no_grad(),
        ):
            result = observe_native(
                model,
                fn,
                requests,
                gradients,
                compile_model is not None and settings["backend"] == "mutant:gradient-scale",
                step_hook,
            )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        invocations += 1
        return result

    return execute
