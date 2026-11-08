"""Isolated, supervised CPU nightly installations for advisory version bisection."""

import importlib.metadata
import math
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta
from html.parser import HTMLParser
from pathlib import Path

from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

from .native import file_hash
from .runner import stop_tree, tree_rss
from .storage import atomic_json, read_json

INDEX = "https://download.pytorch.org/whl/nightly/cpu/torch/"


class WheelLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(value for key, value in attrs if key == "href")


def nightly_catalog(start, end, *, document=None, tags=None):
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last or (last - first).days > 366:
        raise ValueError("Nightly search requires an ordered window of at most 366 days")
    if document is None:
        with urllib.request.urlopen(INDEX, timeout=30) as response:
            document = response.read(16 * 1024**2 + 1)
        if len(document) > 16 * 1024**2:
            raise ValueError("Nightly index exceeds the size limit")
        document = document.decode("utf-8")
    parser = WheelLinks()
    parser.feed(document)
    compatible = set(sys_tags() if tags is None else tags)
    wheels = {}
    for link in parser.links:
        url = urllib.parse.urljoin(INDEX, link)
        parts = urllib.parse.urlsplit(url)
        if parts.scheme != "https" or parts.hostname not in {
            "download.pytorch.org",
            "download-r2.pytorch.org",
        }:
            continue
        filename = urllib.parse.unquote(parts.path.rsplit("/", 1)[-1])
        if not filename.endswith(".whl"):
            continue
        try:
            name, version, _, wheel_tags = parse_wheel_filename(filename)
        except ValueError:
            continue
        if name != "torch" or version.local != "cpu" or not wheel_tags & compatible:
            continue
        match = re.search(r"\.dev(\d{8})(?:\+|$)", str(version))
        if not match:
            continue
        day = date.fromisoformat(match[1])
        if not first <= day <= last:
            continue
        checksum = urllib.parse.parse_qs(parts.fragment).get("sha256", [None])[0]
        if checksum and not re.fullmatch(r"[0-9a-f]{64}", checksum):
            continue
        row = {
            "date": day.isoformat(),
            "version": str(version),
            "url": url,
            "wheel_sha256": checksum,
            "filename": filename,
        }
        key = day.isoformat()
        if key in wheels and wheels[key] != row:
            raise ValueError(f"Ambiguous compatible CPU wheels on {key}")
        wheels[key] = row
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    return {
        "index": INDEX,
        "wheels": [wheels[k] for k in sorted(wheels)],
        "missing_dates": [d.isoformat() for d in days if d.isoformat() not in wheels],
    }


def dependency_lock(text=None):
    if text is None:
        text = "\n".join(
            f"{name}=={importlib.metadata.version(name)}" for name in ("numpy", "psutil")
        )
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        item = Requirement(line)
        specs = list(item.specifier)
        if (
            item.url
            or item.marker
            or len(specs) != 1
            or specs[0].operator != "=="
            or "*" in specs[0].version
            or item.name.lower().replace("_", "-")
            in {"torch", "torchvision", "torchaudio", "faultline-ml"}
        ):
            raise ValueError(
                "Pin dependencies with name==version; PyTorch and Faultline are managed separately"
            )
        lines.append(str(item))
    return "\n".join(sorted(set(lines))) + "\n"


def supervise(command, directory, *, timeout=900, max_rss_mb=4096):
    if (
        not math.isfinite(timeout)
        or timeout <= 0
        or not math.isfinite(max_rss_mb)
        or max_rss_mb <= 0
    ):
        raise ValueError("Process limits must be positive")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    environment = dict(
        os.environ, PIP_NO_INPUT="1", PIP_DISABLE_PIP_VERSION_CHECK="1", PYTHONNOUSERSITE="1"
    )
    environment.pop("PYTHONPATH", None)
    options = (
        {"creationflags": subprocess.CREATE_NO_WINDOW}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    observed, peak, status = {}, 0, None
    started = time.monotonic()
    with (
        (directory / "stdout.log").open("wb") as stdout,
        (directory / "stderr.log").open("wb") as stderr,
    ):
        try:
            process = subprocess.Popen(
                command, stdout=stdout, stderr=stderr, env=environment, **options
            )
        except OSError as exc:
            result = {
                "status": "launch_error",
                "error": f"{type(exc).__name__}: {exc}",
                "returncode": None,
                "wall_s": time.monotonic() - started,
                "peak_tree_rss_mb": 0.0,
            }
            atomic_json(directory / "result.json", result)
            return result
        try:
            while process.poll() is None:
                peak = max(peak, tree_rss(process.pid, observed))
                if time.monotonic() - started > timeout:
                    status = "timeout"
                    break
                if peak > max_rss_mb * 1024**2:
                    status = "resource_limit"
                    break
                time.sleep(0.05)
        finally:
            stop_tree(process, observed)
    result = {
        "status": status or ("complete" if process.returncode == 0 else "command_failed"),
        "returncode": process.returncode,
        "wall_s": time.monotonic() - started,
        "peak_tree_rss_mb": peak / 1024**2,
    }
    atomic_json(directory / "result.json", result)
    return result


def environment_identity(python, directory):
    program = """import importlib.metadata, json, platform, sys, torch
print(json.dumps({'python':sys.version,'platform':platform.platform(),
 'torch':torch.__version__,'torch_git_version':torch.version.git_version,
 'packages':{d.metadata['Name'].lower().replace('_','-'):d.version for d in importlib.metadata.distributions()},
 'torch_build':torch.__config__.show()}))"""
    result = supervise([str(python), "-c", program], directory, timeout=90)
    if result["status"] != "complete":
        raise ValueError(f"Interpreter probe failed: {result['status']}")
    return read_json(Path(directory) / "stdout.log")


def ensure_environment(root, wheel, requirements, *, install_timeout=900):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    expected = {"wheel": wheel, "requirements_sha256": file_hash(requirements)}
    ready = sorted(root.glob("attempt-*/ready.json"))
    if ready:
        record = read_json(ready[-1])
        if record["specification"] != expected:
            raise ValueError("Nightly environment specification changed")
        attempt = ready[-1].parent
        if file_hash(attempt / "downloads" / wheel["filename"]) != record["download_sha256"]:
            raise ValueError("Downloaded wheel changed")
        probe = attempt / f"probe-{len(list(attempt.glob('probe-*'))):04d}"
        identity = environment_identity(record["python"], probe)
        if identity != record["identity"]:
            raise ValueError("Installed nightly environment changed")
        return record
    attempt = root / f"attempt-{len(list(root.glob('attempt-*'))) + 1:04d}"
    attempt.mkdir(exist_ok=False)
    venv = attempt / "venv"
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    commands = [
        ("create", [sys.executable, "-m", "venv", str(venv)]),
        (
            "download",
            [
                str(python),
                "-m",
                "pip",
                "download",
                "--no-deps",
                "--only-binary=:all:",
                "--dest",
                str(attempt / "downloads"),
                wheel["url"],
            ],
        ),
        (
            "install",
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--only-binary=:all:",
                str(attempt / "downloads" / wheel["filename"]),
                "-r",
                str(requirements),
            ],
        ),
        ("pip-check", [str(python), "-m", "pip", "check"]),
    ]
    for name, command in commands:
        result = supervise(command, attempt / name, timeout=install_timeout)
        if result["status"] != "complete":
            return {
                "status": "installation_inconclusive",
                "stage": name,
                "detail": result,
                "specification": expected,
            }
    try:
        identity = environment_identity(python, attempt / "probe-0000")
    except (ValueError, OSError) as exc:
        return {
            "status": "installation_inconclusive",
            "stage": "probe",
            "detail": str(exc),
            "specification": expected,
        }
    if identity["torch"] != wheel["version"]:
        raise ValueError("Installed PyTorch does not match the selected wheel")
    record = {
        "status": "ready",
        "python": str(python),
        "identity": identity,
        "specification": expected,
        "download_sha256": file_hash(attempt / "downloads" / wheel["filename"]),
    }
    atomic_json(attempt / "ready.json", record)
    return record
