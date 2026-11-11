"""Explicit venv creation using the supervised bisection installer primitives."""

import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version

from .bisect_env import environment_identity, supervise
from .native import file_hash
from .storage import atomic_json, campaign_lock, read_json


def torch_requirement(specification):
    if specification == "nightly":
        return "torch"
    if specification.startswith("https://") or specification.endswith(".whl"):
        if specification.startswith("https://"):
            parsed = urlsplit(specification)
            if parsed.username or parsed.password or not parsed.path.endswith(".whl"):
                raise ValueError("Supply a wheel URL without embedded credentials")
            return specification
        path = Path(specification).resolve()
        if not path.is_file():
            raise ValueError("Local wheel does not exist")
        return str(path)
    try:
        return "torch==" + str(Version(specification))
    except InvalidVersion:
        requirement = Requirement(specification)
        if (
            requirement.name != "torch"
            or requirement.extras
            or requirement.marker
            or requirement.url
        ):
            raise ValueError("Use a torch version, nightly, torch==VERSION, or a wheel path/URL")
        pins = list(requirement.specifier)
        if len(pins) != 1 or pins[0].operator != "==" or "*" in pins[0].version:
            raise ValueError("Pin torch with an exact version")
        return str(requirement)


def create_environment(
    output,
    torch_spec,
    *,
    device="cpu",
    index_url=None,
    requirements=(),
    model_dependencies=True,
    resume=False,
    timeout=900,
):
    if device not in {"cpu", "cuda", "rocm"}:
        raise ValueError("Choose cpu, cuda, or rocm")
    if device != "cpu" and index_url is None:
        raise ValueError(
            "GPU wheels require --index-url for your CUDA/ROCm runtime. Copy the index from https://pytorch.org/get-started/locally/; Faultline will not guess your driver compatibility."
        )
    index_url = (
        index_url
        or "https://download.pytorch.org/whl/"
        + ("nightly/" if torch_spec == "nightly" else "")
        + "cpu"
    )
    parsed = urlsplit(index_url)
    if parsed.scheme != "https" or parsed.username or parsed.password:
        raise ValueError("Use an HTTPS wheel index without embedded credentials")
    requirement = torch_requirement(torch_spec)
    dependencies = ["numpy>=1.26,<3", "psutil>=5.9,<8", "packaging>=24,<27"]
    if model_dependencies:
        dependencies += ["transformers>=5,<6", "safetensors>=0.5,<1", "pillow>=11,<14"]
    for item in requirements:
        dependency = Requirement(item)
        if (
            dependency.name.lower().replace("_", "-")
            in {"torch", "torchvision", "torchaudio", "faultline-ml"}
            or dependency.url
        ):
            raise ValueError(
                "Additional dependencies cannot replace managed torch packages or use direct URLs"
            )
        dependencies.append(str(dependency))
    output = Path(output).resolve()
    specification = {
        "torch": requirement,
        "device": device,
        "index_url": index_url,
        "dependencies": dependencies,
        "model_dependencies": model_dependencies,
        "local_wheel_sha256": file_hash(requirement) if Path(requirement).is_file() else None,
    }
    with campaign_lock(output):
        ready = output / "ready.json"
        if ready.exists():
            record = read_json(ready)
            if not resume or record["specification"] != specification:
                raise ValueError(
                    "Environment exists or specification changed; use --resume unchanged or choose a new output"
                )
            probe = output / f"resume-probe-{len(list(output.glob('resume-probe-*'))):04d}"
            if environment_identity(record["python"], probe) != record["identity"]:
                raise ValueError("Installed environment changed")
            return record
        attempt = output / f"attempt-{len(list(output.glob('attempt-*'))) + 1:04d}"
        attempt.mkdir()
        python = attempt / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        torch_args = [requirement] + (["torchvision"] if model_dependencies else [])
        stages = [
            ("create", [sys.executable, "-m", "venv", str(python.parent.parent)]),
            (
                "torch",
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "--only-binary=:all:",
                    *(["--pre"] if torch_spec == "nightly" else []),
                    *torch_args,
                    "--index-url",
                    index_url,
                ],
            ),
        ]
        for name, command in stages:
            observed = supervise(command, attempt / name, timeout=timeout)
            if observed["status"] != "complete":
                raise ValueError(
                    f"Environment {name}: {observed['status']}. See {attempt / name / 'stderr.log'}"
                )
        initial = environment_identity(python, attempt / "initial-probe")
        constraints = attempt / "constraints.txt"
        constraints.write_text("torch==" + initial["torch"] + "\n", encoding="utf-8")
        for name, command in [
            (
                "dependencies",
                [str(python), "-m", "pip", "install", "-c", str(constraints), *dependencies],
            ),
            ("pip-check", [str(python), "-m", "pip", "check"]),
        ]:
            observed = supervise(command, attempt / name, timeout=timeout)
            if observed["status"] != "complete":
                raise ValueError(
                    f"Environment {name}: {observed['status']}. See {attempt / name / 'stderr.log'}"
                )
        identity = environment_identity(python, attempt / "final-probe")
        if identity["torch"] != initial["torch"]:
            raise ValueError("Dependency installation changed torch")
        probe = supervise(
            [
                str(python),
                "-c",
                "import json,torch;print(json.dumps({'cuda':torch.version.cuda,'hip':torch.version.hip}))",
            ],
            attempt / "device-probe",
            timeout=90,
        )
        if probe["status"] != "complete":
            raise ValueError("Cannot verify installed device support")
        build = read_json(attempt / "device-probe/stdout.log")
        observed_device = "rocm" if build["hip"] else "cuda" if build["cuda"] else "cpu"
        if observed_device != device:
            raise ValueError(
                f"Installed {observed_device} wheel, requested {device}; choose the matching wheel index"
            )
        record = {
            "schema_version": 1,
            "status": "ready",
            "python": str(python),
            "specification": specification,
            "identity": identity,
            "device_build": build,
        }
        atomic_json(ready, record)
        return record


def add_env_command(commands):
    parser = commands.add_parser("env", help="create isolated, supervised PyTorch environments")
    sub = parser.add_subparsers(dest="env_command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--torch", required=True)
    create.add_argument("--device", choices=("cpu", "cuda", "rocm"), default="cpu")
    create.add_argument("--index-url")
    create.add_argument("--requirement", action="append", default=[])
    create.add_argument(
        "--minimal", action="store_true", help="omit torchvision/transformers model dependencies"
    )
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--resume", action="store_true")
    create.add_argument("--timeout", type=float, default=900)


def env_command(args):
    result = create_environment(
        args.output,
        args.torch,
        device=args.device,
        index_url=args.index_url,
        requirements=args.requirement,
        model_dependencies=not args.minimal,
        resume=args.resume,
        timeout=args.timeout,
    )
    print(
        f"Environment ready: {result['python']}\nPyTorch: {result['identity']['torch']}\nManifest: {args.output / 'ready.json'}"
    )
    return 0
