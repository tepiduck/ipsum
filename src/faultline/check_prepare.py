"""Trusted model preparation in a separate interpreter for faultline check."""

import argparse
import gc
import importlib.util
import inspect
import random
import sys
from pathlib import Path

import numpy as np
import torch

from .evaluate import Settings
from .native import capture_workload
from .native_runtime import workload_imports
from .storage import atomic_json, read_json


def prepare(specification, output):
    from . import sweep_models

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    count = specification["batches"]
    settings = Settings(
        backend="eager",
        repeats=2,
        timeout_s=specification["timeout"],
        max_rss_mb=specification["max_rss_mb"],
    )
    seeded = False
    collection = False
    if specification.get("torchvision") or specification.get("hf"):
        name = specification.get("torchvision")
        if name and name not in sweep_models.VISION_MODELS:
            raise ValueError(
                "Unknown torchvision catalog model. Run faultline sweep list, or supply a model.py:factory."
            )
        if specification.get("hf"):
            model_id = specification["hf"]
            name = next(
                (
                    key
                    for key, value in sweep_models.TEXT_MODELS.items()
                    if value == model_id or key == model_id
                ),
                "custom_hf",
            )
            if name == "custom_hf":
                sweep_models.TEXT_MODELS[name] = model_id
            if specification.get("revision"):
                sweep_models.TEXT_REVISIONS[name] = specification["revision"]
        sweep_models.real_inputs(output / "inputs", count)
        sweep_models.prepare_model(name, output / "source", output / "inputs")
        factory = f"{output / 'source/factory.py'}:factory"
        include = ["weights.pt", "inputs.pt", "model.json"]
    else:
        factory = specification["factory"]
        filename, _, function = factory.rpartition(":")
        path = Path(filename).resolve()
        if not path.is_file() or not function.isidentifier():
            raise ValueError("Use an existing Python file and function: model.py:factory.")
        module_spec = importlib.util.spec_from_file_location("faultline_check_factory_probe", path)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        with workload_imports(path.parent):
            module_spec.loader.exec_module(module)
            fn = getattr(module, function)
            seeded = "seed" in inspect.signature(fn).parameters
            model, inputs = fn(seed=0) if seeded else fn()
        # A list of tuples/dicts is explicitly a collection. Tensor lists are
        # ambiguous with nested model inputs and require --batches to opt in.
        collection = (
            isinstance(inputs, list)
            and inputs
            and (
                all(isinstance(x, (tuple, dict)) for x in inputs)
                or specification.get("explicit_batches")
            )
        )
        if collection:
            count = len(inputs)
        elif not seeded:
            count = 1
        del model, inputs, module
        gc.collect()
        include = specification.get("include", [])
    gc.collect()
    bundle = capture_workload(
        factory,
        output / "bundle",
        settings,
        input_batches=count,
        seeded_factory=seeded,
        include=include,
        batch_index=0 if collection and count == 1 else None,
    )
    atomic_json(
        output / "prepared.json",
        {
            "bundle": str(bundle),
            "input_batches": count,
            "seeded_factory": seeded,
            "requested_batches": specification["batches"],
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    prepare(read_json(args.specification), args.output)


if __name__ == "__main__":
    main()
