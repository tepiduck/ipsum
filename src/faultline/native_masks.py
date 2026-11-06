"""Optional AOT saved-mask evidence, shared by workers and standalone reproducers.

Runs only in a separate diagnostic replay. Shadow eager graphs see cloned joint
inputs and restored RNG. The primary execution is never instrumented by this module.
"""

import copy
import io

import torch


def clone_arguments(values, limit_bytes=512 * 1024**2):
    storages = {}
    for value in values:
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            storages[(str(value.device), storage._cdata)] = storage.nbytes()
    if sum(storages.values()) > limit_bytes:
        raise ValueError("AOT shadow inputs exceed the 512 MiB snapshot limit")
    stream = io.BytesIO()
    torch.save(values, stream)
    stream.seek(0)
    return torch.load(stream, weights_only=True)


def tensor_key(value):
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        return None
    return (
        str(value.device),
        value.untyped_storage()._cdata,
        value.storage_offset(),
        tuple(value.shape),
        value.stride(),
        str(value.dtype),
    )


def mask_specs(module):
    """Recognize explicit derivative selectors, not arbitrary floating activations."""
    for node in module.graph.nodes:
        target = str(node.target)
        if node.op != "call_function":
            continue
        if target == "aten.threshold_backward.default":
            yield node, node.args[1], ("threshold", node.args[2])
        elif target == "aten.where.self":
            yield node, node.args[0], ("boolean", None)
        elif target == "aten.max_pool2d_with_indices_backward.default":
            yield node, node.args[7], ("indices", None)


_PREDICATE_OPS = {
    "aten.detach.default",
    "aten.alias.default",
    "aten.view.default",
    "aten.reshape.default",
    "aten.squeeze.default",
    "aten.unsqueeze.default",
    "aten.gt.Scalar",
    "aten.ge.Scalar",
    "aten.lt.Scalar",
    "aten.le.Scalar",
    "aten.gt.Tensor",
    "aten.ge.Tensor",
    "aten.lt.Tensor",
    "aten.le.Tensor",
    "aten.eq.Scalar",
    "aten.eq.Tensor",
    "aten.ne.Scalar",
    "aten.ne.Tensor",
    "aten.bitwise_and.Tensor",
    "aten.bitwise_or.Tensor",
    "aten.bitwise_not.default",
    "aten.logical_and.default",
    "aten.logical_or.default",
    "aten.logical_not.default",
}


def predicate_value(value, environment):
    if not isinstance(value, torch.fx.Node):
        return value
    if value.op == "placeholder":
        if value not in environment:
            raise ValueError(f"No paired forward value for {value.name}")
        return environment[value]
    if value.op != "call_function" or str(value.target) not in _PREDICATE_OPS:
        raise ValueError(f"Unsupported mask expression: {value.target}")
    args = torch.fx.map_arg(value.args, lambda node: predicate_value(node, environment))
    kwargs = torch.fx.map_arg(value.kwargs, lambda node: predicate_value(node, environment))
    return value.target(*args, **kwargs)


def downstream_outputs(module, source):
    reachable = {source}
    output = None
    for node in module.graph.nodes:
        if any(parent in reachable for parent in node.all_input_nodes):
            reachable.add(node)
        if node.op == "output":
            output = node.args[0]
    flat = output if isinstance(output, (tuple, list)) else (output,)
    return [i for i, value in enumerate(flat) if value in reachable]


class MaskProbe:
    def __init__(self, aot, compiler, settings, compare, policy, selection):
        self.aot, self.compiler, self.settings = aot, compiler, settings
        self.compare, self.policy, self.selection = compare, policy, selection
        self.graphs = 0
        self.ordinals = {}
        self.reset()

    def reset(self):
        self.pairs, self.owners = {}, []
        self.records, self.errors, self.backward_checks = [], [], []
        self.bytes = 0
        self.step = 0

    def shadow(self, module, values):
        devices = sorted(
            {
                v.device.index
                for v in values
                if isinstance(v, torch.Tensor) and v.device.type == "cuda"
            }
        )
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            return copy.deepcopy(module)(*clone_arguments(values))

    def around(self, module, stage, key, compiled, values):
        if stage == "backward":
            self.inspect_backward(module, values, key)
        try:
            shadow = self.shadow(module, values)
        except Exception as exc:  # noqa: BLE001 -- optional user-graph replay
            self.errors.append({"stage": key, "error": f"{type(exc).__name__}: {exc}"})
            return compiled(values)
        actual = compiled(values)  # May clear its boxed argument list.
        if stage == "forward":
            self.remember(shadow, actual, key)
        else:
            left, right = {}, {}
            for i, (a, b) in enumerate(zip(shadow, actual, strict=True)):
                if isinstance(a, torch.Tensor):
                    left[str(i)] = a.detach().cpu()
                if isinstance(b, torch.Tensor):
                    right[str(i)] = b.detach().cpu()
            compared = self.compare(
                (left, []),
                (right, []),
                self.policy,
                self.settings.get("rtol"),
                self.settings.get("atol"),
            )
            self.backward_checks.append({"stage": key, "step": self.step, **compared})
            # Sequential first-order VJPs consume one forward tape per request.
            consumed = key.split(":", 1)[0]
            self.pairs = {
                k: v
                for k, v in self.pairs.items()
                if v is not None and v[2].split(":", 1)[0] != consumed
            }
            self.owners = [v for v in self.owners if tensor_key(v) in self.pairs]
            self.bytes = sum(
                a.numel() * a.element_size() + b.numel() * b.element_size()
                for a, b, _, _ in self.pairs.values()
            )
        return actual

    def remember(self, eager, candidate, key):
        for index, (a, b) in enumerate(zip(eager, candidate, strict=True)):
            storage_key = tensor_key(b)
            if storage_key is None or not isinstance(a, torch.Tensor) or a.shape != b.shape:
                continue
            required = (a.numel() * a.element_size()) + (b.numel() * b.element_size())
            if self.bytes + required > 256 * 1024**2:
                self.errors.append(
                    {
                        "stage": key,
                        "output": index,
                        "error": "Saved-mask pairs exceed 256 MiB limit",
                    }
                )
                continue
            pair = (a.detach().cpu().clone(), b.detach().cpu().clone(), key, index)
            if storage_key in self.pairs and (
                self.pairs[storage_key] is None
                or not torch.equal(self.pairs[storage_key][0], pair[0])
            ):
                self.pairs[storage_key] = None  # Ambiguous alias; never infer a correspondence.
            elif storage_key not in self.pairs:
                self.pairs[storage_key] = pair
                self.owners.append(b)  # Prevent allocator reuse until backward consumes the tape.
                self.bytes += required

    def inspect_backward(self, module, values, key):
        left, right, origins = {}, {}, {}
        placeholders = [n for n in module.graph.nodes if n.op == "placeholder"]
        for node, value in zip(placeholders, values, strict=True):
            pair = self.pairs.get(tensor_key(value))
            if pair is not None and torch.equal(value.detach().cpu(), pair[1]):
                left[node], right[node] = pair[:2]
                origins[node.name] = {"forward_stage": pair[2], "output": pair[3]}
        for node, expression, (kind, threshold) in mask_specs(module):
            row = {
                "stage": key,
                "step": self.step,
                "node": node.name,
                "kind": kind,
                "affected_backward_outputs": downstream_outputs(module, node),
            }
            try:
                ancestors = set()

                def visit(value, ancestors=ancestors):
                    if isinstance(value, torch.fx.Node) and value not in ancestors:
                        ancestors.add(value)
                        for parent in value.all_input_nodes:
                            visit(parent)

                visit(expression)
                a, b = predicate_value(expression, left), predicate_value(expression, right)
                if kind == "threshold":
                    a, b = a > threshold, b > threshold
                    row["threshold"] = threshold
                if not isinstance(a, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
                    raise ValueError("Mask structure differs")
                if kind == "boolean" and a.dtype != torch.bool:
                    raise ValueError("Expected a boolean branch selector")
                changed = (a != b).reshape(-1)
                row.update(
                    status="complete",
                    elements=a.numel(),
                    flips=int(changed.sum()),
                    changed_flat_indices=changed.nonzero().flatten()[:16].tolist(),
                    paired_inputs={
                        n.name: origins[n.name]
                        for n in placeholders
                        if n in ancestors and n.name in origins
                    },
                )
            except (ValueError, RuntimeError, TypeError) as exc:
                row.update(status="unsupported", error=str(exc))
            self.records.append(row)

    def backend(self, module, inputs):
        parent = self.graphs
        self.graphs += 1
        if self.selection.get("only_graph") not in (None, parent):
            return module.forward

        def stage_compiler(graph, args, stage, options):
            ordinal = self.ordinals.get((parent, stage), 0)
            self.ordinals[parent, stage] = ordinal + 1
            key = f"{parent}:{stage}:{ordinal}"
            original = copy.deepcopy(graph)
            return self.aot.compile_stage(
                graph,
                args,
                self.settings["backend"],
                stage,
                options=options,
                selected=self.selection.get("only_aot") in (None, key),
                around=lambda call, values: self.around(original, stage, key, call, values),
            )

        return self.aot.compile_pipeline(
            module,
            inputs,
            self.settings["backend"],
            stage_compiler,
            self.compiler.mode_options(self.settings),
        )

    def summary(self):
        complete = [r for r in self.records if r["status"] == "complete"]
        return {
            "status": "partial"
            if self.errors or len(complete) != len(self.records)
            else "complete"
            if complete
            else "no_supported_masks",
            "masks": self.records,
            "flips": sum(r["flips"] for r in complete),
            "elements": sum(r["elements"] for r in complete),
            "errors": self.errors,
            "backward_same_inputs": self.backward_checks,
            "note": "AOT eager forward replay versus compiled saved activations; not original-module "
            "mask matching. Flips establish sensitivity evidence, not complete causal attribution. "
            "Counts are per backward selector; reused selectors can count the same element twice.",
        }
