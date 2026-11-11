"""Measure one frozen CPU input batch in one fresh process (PyTorch + transformers).

The assets directory contains model.json, weights.pt and identity.json; the input
file contains tensor-only `inputs` and a saved `probe`. See docs/input-accuracy.md
for their schema and the batch-level analysis. This records advisory measurements,
not a compiler-correctness verdict. Only load trusted local model assets.
"""

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class LastHidden(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, **inputs):
        return self.inner(**inputs).last_hidden_state


def load_model(assets, spec, identity):
    from transformers import AutoConfig, AutoModel

    torch.manual_seed(0)
    configuration = dict(spec["config"])
    config = AutoConfig.for_model(configuration.pop("model_type"), **configuration)
    config.use_cache = False
    model = LastHidden(
        AutoModel.from_config(config, attn_implementation="sdpa", **spec.get("model_options", {}))
    )
    model.load_state_dict(
        torch.load(assets / "weights.pt", map_location="cpu", weights_only=True), strict=True
    )
    parameters = dict(model.named_parameters())
    if parameters.keys() != identity["trainable"].keys():
        raise ValueError("Frozen parameter names differ")
    for name, parameter in parameters.items():
        parameter.requires_grad_(identity["trainable"][name])
    return model.eval()


def observe(model, inputs, probe, compiled):
    fn = torch.compile(model, backend="inductor", dynamic=False) if compiled else model
    values = fn(**inputs)
    if values.shape != probe.shape:
        raise ValueError("Saved VJP shape differs from the model output")
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    gradients = torch.autograd.grad(
        values, [p for _, p in parameters], grad_outputs=probe.to(values.dtype), allow_unused=True
    )
    return values.detach(), {
        name: value.detach() if value is not None else None
        for (name, _), value in zip(parameters, gradients, strict=True)
    }


def rms(value):
    value = value.double().abs()
    maximum = float(value.max()) if value.numel() else 0.0
    return maximum * float((value / maximum).square().mean().sqrt()) if maximum else 0.0


def metrics(reference, eager, compiled):
    row = {
        "reference_rms": rms(reference),
        "eager_rmse": rms(eager.double() - reference),
        "compiled_rmse": rms(compiled.double() - reference),
        "reference_sha256": hashlib.sha256(reference.contiguous().numpy().tobytes()).hexdigest(),
        "shape": list(reference.shape),
    }
    if not all(math.isfinite(row[k]) for k in ("reference_rms", "eager_rmse", "compiled_rmse")):
        raise ValueError("Nonfinite measurement")
    return row


def measure(args):
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = json.loads((args.assets / "model.json").read_text())
    identity = json.loads((args.assets / "identity.json").read_text())
    for name, key in (("model.json", "model_sha256"), ("weights.pt", "weights_sha256")):
        if file_hash(args.assets / name) != identity[key]:
            raise ValueError(f"Frozen asset changed: {name}")
    payload = torch.load(args.inputs, map_location="cpu", weights_only=True)
    inputs, probe = payload["inputs"], payload["probe"]
    if probe.dtype != torch.float32 or probe.shape != (2, 64, 768):
        raise ValueError("Expected the frozen fp32 2x64x768 probe")
    for value in inputs.values():
        if value.dtype != torch.int64 or value.shape != (2, 64):
            raise ValueError("Expected frozen int64 2x64 input tensors")
    model = load_model(args.assets, spec, identity)
    eager_out, eager = observe(model, inputs, probe, False)
    compiled_out, compiled = observe(model, inputs, probe, True)
    del model
    torch._dynamo.reset()
    gc.collect()
    model = load_model(args.assets, spec, identity).double()
    reference_out, reference = observe(model, inputs, probe, False)
    del model
    gc.collect()
    if not (reference.keys() == eager.keys() == compiled.keys()):
        raise ValueError("Gradient parameter names differ between paths")
    present = sorted(k for k, value in reference.items() if value is not None)
    if present != identity["expected_gradients"]:
        raise ValueError("Present gradients differ from the frozen workload")
    rows, absent = [], []
    for name, value in reference.items():
        if (value is None) != (eager[name] is None) or (value is None) != (compiled[name] is None):
            raise ValueError(f"Gradient presence changed: {name}")
        if value is None:
            absent.append(name)
        else:
            rows.append({"parameter": name, **metrics(value, eager[name], compiled[name])})
    return {
        "schema": 1,
        "advisory_only": True,
        "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "identity_sha256": file_hash(args.assets / "identity.json"),
        "inputs_sha256": file_hash(args.inputs),
        "worker_sha256": file_hash(Path(__file__)),
        "input_layouts": {
            k: {"shape": list(v.shape), "stride": list(v.stride()), "offset": v.storage_offset()}
            for k, v in inputs.items()
        },
        "threads": 1,
        "observations_per_path": 1,
        "gradients": rows,
        "absent_gradients": absent,
        "output": metrics(reference_out, eager_out, compiled_out),
        "note": "One input batch is one replicate. Parameter gradients are correlated. "
        "Each build uses its own fp64 eager reference; reference hashes expose changes. "
        "No correctness verdict, timing claim or compiler-defect claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file to preserve evidence")
    result = measure(args)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
