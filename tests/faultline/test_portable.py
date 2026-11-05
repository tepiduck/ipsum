import json
import subprocess
import sys

import pytest
import torch

from faultline.evaluate import Settings, evaluate
from faultline.export import export_case
from faultline.generate import corpus
from faultline.lowering import freeze
from faultline.portable import check_execution


def test_near_threshold_classification_changes_are_inconclusive():
    # Within-path variation is below atol, yet only one repeat exceeds it.
    outputs = iter([0.009, 0.009, 0.011])
    candidate = lambda: (torch.tensor([next(outputs)]),)
    result, _, _ = check_execution(
        lambda: (torch.zeros(1),),
        lambda: candidate,
        (),
        device="cpu",
        repeats=2,
        rtol=0,
        atol=0.01,
    )
    assert result["status"] == "unstable_candidate"
    assert result["reason"] == "Failure classification changed between invocations"


def test_cold_failure_on_different_output_is_not_accepted_as_the_same_failure():
    outputs = iter([(0.011, 0.009), (0.009, 0.011), (0.009, 0.011)])
    candidate = lambda: tuple(torch.tensor([v]) for v in next(outputs))
    result, _, _ = check_execution(
        lambda: (torch.zeros(1), torch.zeros(1)),
        lambda: candidate,
        (),
        device="cpu",
        repeats=2,
        rtol=0,
        atol=0.01,
    )
    assert result["status"] == "unstable_candidate"


@pytest.mark.parametrize(
    "mode,status",
    [
        ("compile-transient", "unstable_candidate"),
        ("runtime-transient", "unstable_candidate"),
        ("runtime-stable", "runtime_error"),
    ],
)
def test_standalone_retries_backend_errors_without_faultline(tmp_path, mode, status):
    path = export_case(corpus()[0], Settings(backend="aot_eager"), tmp_path)
    # Replace only the compiler entry point in a separate Python interpreter.
    wrapper = """import runpy, sys, torch
path, mode = sys.argv[1:]
sys.argv = [path]
creations = 0
def compile_fixture(module, **kwargs):
    global creations
    creations += 1
    if mode == "compile-transient" and creations == 1:
        raise RuntimeError("transient compilation failure")
    calls = 0
    def candidate(*args):
        nonlocal calls
        calls += 1
        if mode.startswith("runtime-") and (calls == 2 or (calls > 2 and mode == "runtime-stable")):
            raise RuntimeError("runtime failure")
        return module(*args)
    return candidate
torch.compile = compile_fixture
runpy.run_path(path, run_name="__main__")
"""
    run = subprocess.run(
        [sys.executable, "-I", "-c", wrapper, str(path), mode],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert run.returncode == (1 if status == "runtime_error" else 2), run.stderr
    result = json.loads(run.stdout.splitlines()[-1])
    assert result["status"] == status
    if status == "runtime_error":
        assert result["exception_confirmations"] == 3


@pytest.mark.parametrize("backend", ["eager", "mutant:sum-as-mean", "mutant:nan", "mutant:crash"])
def test_standalone_matches_runner_classification_and_failure_signature(tmp_path, backend):
    case = freeze(corpus()[0])
    settings = Settings(backend=backend, repeats=3, rtol=3e-4, atol=2e-6)
    expected = evaluate(case, settings)
    path = export_case(case, settings, tmp_path, expected)
    run = subprocess.run(
        [sys.executable, "-I", str(path)], check=False, capture_output=True, text=True, timeout=60
    )
    assert run.returncode == (0 if backend == "eager" else 1), run.stderr
    observed = json.loads(run.stdout.splitlines()[-1])
    for key in ("status", "fingerprint", "comparison", "exception_confirmations", "rtol", "atol"):
        assert observed.get(key) == expected.get(key), key
    if backend != "eager":
        assert observed["reproduces_expected"]


@pytest.mark.parametrize("replacement", ["pass", "different_error"])
def test_standalone_does_not_claim_a_changed_failure_reproduces_the_original(
    tmp_path, monkeypatch, replacement
):
    def original_error(*args, **kwargs):
        raise RuntimeError("original compiler error")

    case = freeze(corpus()[0])
    settings = Settings(backend="aot_eager", repeats=2)
    with monkeypatch.context() as patch:
        patch.setattr(torch, "compile", original_error)
        expected = evaluate(case, settings)
    path = export_case(case, settings, tmp_path, expected)
    wrapper = """import runpy, sys, torch
path, replacement = sys.argv[1:]
sys.argv = [path]
def compiler(module, **kwargs):
    if replacement == "different_error":
        raise RuntimeError("unrelated new compiler error")
    return module
torch.compile = compiler
runpy.run_path(path, run_name="__main__")
"""
    run = subprocess.run(
        [sys.executable, "-I", "-c", wrapper, str(path), replacement],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert run.returncode == 2, run.stderr
    result = json.loads(run.stdout.splitlines()[-1])
    assert not result["reproduces_expected"]
    assert result["status"] == ("pass" if replacement == "pass" else "compile_error")
