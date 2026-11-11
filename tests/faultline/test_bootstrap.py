"""Cross-interpreter workers must not borrow the coordinator's dependencies."""

import json
import os
import subprocess
import sys
import time

import psutil

from faultline import bootstrap
from faultline.runner import parent_alive


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


def test_supervised_bootstrap_exits_when_coordinator_disappears(tmp_path):
    program = """import json, subprocess, sys
from faultline.bootstrap import python_command
from faultline.runner import process_identity
child = subprocess.Popen(
    python_command(sys.executable, 'import time;time.sleep(60)', watch_parent=True),
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
print(json.dumps([child.pid, process_identity(child.pid)]), flush=True)
"""
    parent = subprocess.run(
        bootstrap.python_command(sys.executable, program),
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    pid, identity = json.loads(parent.stdout)
    if not parent_alive(pid, identity):
        return
    try:
        child = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    stopped = False
    try:
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            try:
                if not child.is_running() or child.status() == psutil.STATUS_ZOMBIE:
                    stopped = True
                    break
            except psutil.NoSuchProcess:
                stopped = True
                break
            time.sleep(0.05)
        assert stopped, "Orphaned diagnostic/preparation worker did not stop"
    finally:
        if not stopped and parent_alive(pid, identity):
            child.kill()
