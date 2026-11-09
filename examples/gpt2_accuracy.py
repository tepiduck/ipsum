"""Standalone CPU GPT-2 accuracy measurement; requires PyTorch and transformers.

Downloads a pinned checkpoint unless --assets names an existing frozen source directory.
Run in each environment, then use --reference with the baseline JSON. Results are advisory;
this script has no compiler-correctness verdict and does not import Faultline.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch

REPOSITORY = "openai-community/gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
# Two 64-token excerpts of public-domain Alice's Adventures in Wonderland,
# frozen by the original pretrained sweep. No tokenizer/network input variation.
TOKENS = [
    [
        44484,
        373,
        3726,
        284,
        651,
        845,
        10032,
        286,
        5586,
        416,
        607,
        6621,
        319,
        262,
        3331,
        11,
        290,
        286,
        1719,
        2147,
        284,
        466,
        25,
        1752,
        393,
        5403,
        673,
        550,
        613,
        538,
        276,
        656,
        262,
        1492,
        607,
        6621,
        373,
        3555,
        11,
        475,
        340,
        550,
        645,
        5986,
        393,
        10275,
        287,
        340,
        11,
        564,
        250,
        392,
        644,
        318,
        262,
        779,
        286,
        257,
        1492,
        11,
        447,
        251,
        1807,
        14862,
    ],
    [
        2396,
        673,
        373,
        6402,
        287,
        607,
        898,
        2000,
        357,
        292,
        880,
        355,
        673,
        714,
        11,
        329,
        262,
        3024,
        1110,
        925,
        607,
        1254,
        845,
        42023,
        290,
        8531,
        828,
        1771,
        262,
        9476,
        286,
        1642,
        257,
        12379,
        13560,
        12,
        7983,
        561,
        307,
        2861,
        262,
        5876,
        286,
        1972,
        510,
        290,
        10868,
        262,
        12379,
        271,
        444,
        11,
        618,
        6451,
        257,
        2635,
        25498,
        351,
        11398,
        2951,
        4966,
        1969,
        416,
        607,
    ],
]


class LastHidden(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, **inputs):
        return self.inner(**inputs).last_hidden_state


def validate_assets(assets):
    specification = json.loads((assets / "model.json").read_text())
    if specification["repository"] != REPOSITORY or specification["revision"] != REVISION:
        raise ValueError("Expected the pinned GPT-2 checkpoint")
    for name, key in (("weights.pt", "weights_sha256"), ("inputs.pt", "inputs_sha256")):
        with (assets / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != specification[key]:
            raise ValueError(f"Frozen asset checksum differs: {name}")
    return specification


def validate_inputs(inputs):
    expected = torch.tensor(TOKENS, dtype=torch.int64)
    if set(inputs) != {"input_ids", "attention_mask"}:
        raise ValueError("Frozen input keys differ from the recorded investigation")
    for name, wanted in (("input_ids", expected), ("attention_mask", torch.ones_like(expected))):
        actual = inputs[name]
        if not isinstance(actual, torch.Tensor) or not (
            actual.dtype == wanted.dtype
            and actual.layout == wanted.layout
            and actual.stride() == wanted.stride()
            and actual.storage_offset() == wanted.storage_offset()
            and torch.equal(actual, wanted)
        ):
            raise ValueError(f"Frozen input values, dtype, or layout differ: {name}")


def load_model(assets, offline):
    specification = validate_assets(assets) if assets else None
    from transformers import AutoConfig, AutoModel

    torch.manual_seed(0)
    if assets:
        configuration = dict(specification["config"])
        kind = configuration.pop("model_type")
        config = AutoConfig.for_model(kind, **configuration)
        config.use_cache = False
        model = LastHidden(AutoModel.from_config(config, attn_implementation="sdpa"))
        model.load_state_dict(
            torch.load(assets / "weights.pt", map_location="cpu", weights_only=True), strict=True
        )
        inputs = torch.load(assets / "inputs.pt", map_location="cpu", weights_only=True)
        validate_inputs(inputs)
    else:
        inner = AutoModel.from_pretrained(
            REPOSITORY,
            revision=REVISION,
            dtype=torch.float32,
            use_safetensors=True,
            trust_remote_code=False,
            local_files_only=offline,
            attn_implementation="sdpa",
        )
        inner.config.use_cache = False
        model = LastHidden(inner)
    if model.inner.config.num_hidden_layers != 12:
        raise ValueError("This reproducer measures the full 12-layer model")
    return model.eval()


def observe(model, compiled):
    inputs = torch.tensor(TOKENS, dtype=torch.int64)
    fn = torch.compile(model, backend="inductor", dynamic=False) if compiled else model
    values = fn(input_ids=inputs, attention_mask=torch.ones_like(inputs))
    probe = torch.randint(0, 2, values.shape, generator=torch.Generator().manual_seed(0))
    probe = probe.to(dtype=values.dtype).mul_(2).sub_(1)
    parameters = list(model.named_parameters())
    gradients = torch.autograd.grad(values, [p for _, p in parameters], grad_outputs=probe)
    return {name: value.detach() for (name, _), value in zip(parameters, gradients, strict=True)}


def rms(value):
    value = value.double().abs()
    maximum = float(value.max()) if value.numel() else 0.0
    return maximum * float((value / maximum).square().mean().sqrt()) if maximum else 0.0


def measure(args):
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_model(args.assets, args.offline)
    eager = observe(model, False)
    compiled = observe(model, True)
    # Release compiler state before constructing the higher-precision model.
    del model
    torch._dynamo.reset()
    gc.collect()
    high_model = load_model(args.assets, args.offline).double()
    high = observe(high_model, False)
    rows = []
    for name, reference in high.items():
        magnitude = rms(reference)
        eager_error = rms(eager[name].double() - reference)
        compiled_error = rms(compiled[name].double() - reference)
        rows.append(
            {
                "parameter": name,
                "reference_rms": magnitude,
                "eager_rmse": eager_error,
                "compiled_rmse": compiled_error,
                "compiled_error_in_u_times_reference_rms": compiled_error / (2**-24 * magnitude)
                if magnitude
                else None,
            }
        )
    result = {
        "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "repository": REPOSITORY,
        "revision": REVISION,
        "unit_roundoff": 2**-24,
        "threads": 1,
        "probe_seed": 0,
        "shape": [2, 64, 768],
        "gradients": rows,
        "note": "One fresh-process observation. Repeat in three fresh processes per environment. "
        "No primary verdict or compiler defect claim. Relative error is RMSE/(u*RMS(reference)), "
        "not an elementwise ULP bound. The fp64 reference also has rounding error.",
    }
    if args.reference:
        baseline = json.loads(args.reference.read_text())
        for key in ("repository", "revision", "threads", "probe_seed", "shape"):
            if baseline[key] != result[key]:
                raise ValueError(f"Reference setting differs: {key}")
        errors = {r["parameter"]: r["compiled_rmse"] for r in baseline["gradients"]}
        values = []
        missing = []
        for name, error in errors.items():
            if not math.isfinite(error) or error <= 0:
                continue
            candidate = next((r["compiled_rmse"] for r in rows if r["parameter"] == name), None)
            if candidate is None or not math.isfinite(candidate) or candidate <= 0:
                missing.append(name)
            else:
                values.append(math.log2(candidate) - math.log2(error))
        result["comparison"] = {
            "paired_positive_gradients": len(values),
            "missing": missing,
            "median_log2_growth": statistics.median(values) if values and not missing else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, help="Frozen model.json, weights.pt and inputs.pt")
    parser.add_argument("--offline", action="store_true", help="Use only cached checkpoint files")
    parser.add_argument("--reference", type=Path, help="Baseline JSON from this script")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file to preserve earlier evidence")
    result = measure(args)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
