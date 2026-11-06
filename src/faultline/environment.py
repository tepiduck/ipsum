"""Collect relevant reproducibility facts without dumping credentials or environment variables."""

import importlib.metadata
import os
import shutil
import sysconfig
from pathlib import Path

import torch

from .storage import provenance


def describe():
    packages = {}
    for name in (
        "torch",
        "triton",
        "numpy",
        "psutil",
        "faultline-ml",
        "torchvision",
        "transformers",
        "safetensors",
        "pillow",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    available = torch.cuda.is_available()
    devices = []
    if available:
        for index in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": prop.name,
                    "total_memory": prop.total_memory,
                    "architecture": getattr(prop, "gcnArchName", None),
                }
            )
    return {
        **provenance(),
        "packages": packages,
        "torch_build": torch.__config__.show(),
        "logical_cpus": os.cpu_count(),
        "python_headers_available": (Path(sysconfig.get_path("include")) / "Python.h").is_file(),
        "cxx_compiler": shutil.which("g++") or shutil.which("clang++") or shutil.which("cl"),
        "hip_version": torch.version.hip,
        "cuda_version": torch.version.cuda,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "gpu_available": available,
        "devices": devices,
        "note": "GPU availability is discovery, not a successful compiler execution test.",
    }
