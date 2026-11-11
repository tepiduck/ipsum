"""Cross-interpreter workers must not borrow the coordinator's dependencies."""

import json
import os
import subprocess
import sys

from faultline import bootstrap


def test_package_bootstrap_keeps_child_dependencies_and_arguments(tmp_path, monkeypatch):
    coordinator = tmp_path / "coordinator site-packages"
    package = coordinator / "faultline"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("marker = 'coordinator harness'\n")
    (package / "worker_example.py").write_text(
        "import json, sys, dependency_example\n"
        "from . import marker\n"
        "print(json.dumps([marker, dependency_example.version, sys.argv[1:]]))\n"
    )
    (coordinator / "dependency_example.py").write_text(
        "raise RuntimeError('coordinator dependencies leaked')\n"
    )
    candidate = tmp_path / "candidate site-packages"
    candidate.mkdir()
    (candidate / "dependency_example.py").write_text("version = 'candidate runtime'\n")
    (candidate / "faultline.py").write_text("raise RuntimeError('wrong harness loaded')\n")
    monkeypatch.setattr(bootstrap, "__file__", str(package / "bootstrap.py"))
    command = bootstrap.python_command(
        sys.executable,
        "import runpy;runpy.run_module('faultline.worker_example',run_name='__main__')",
        ["--job", "a path with spaces.json"],
    )
    result = subprocess.run(
        command,
        env={**os.environ, "PYTHONPATH": str(candidate)},
        cwd=tmp_path,
        check=True,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert json.loads(result.stdout) == [
        "coordinator harness",
        "candidate runtime",
        ["--job", "a path with spaces.json"],
    ]
