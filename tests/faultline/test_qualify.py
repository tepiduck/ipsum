import pytest

from faultline import qualify
from faultline.cli import main
from faultline.evaluate import Settings


def test_unavailable_rocm_has_evidence_and_never_runs_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr(
        qualify,
        "describe",
        lambda: {"hip_version": None, "cuda_version": None, "gpu_available": False},
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Missing GPU must not execute CPU workloads")

    monkeypatch.setattr(qualify, "capture_workload", forbidden)
    result = qualify.qualify_device(
        tmp_path / "run", Settings(device="cuda:0", gradients=True), require_rocm=True
    )
    assert result["exit_code"] == 2 and result["status"] == "unavailable"
    assert result["device"] == "cuda:0" and not result["cases"]
    assert (tmp_path / "run" / "summary.json").is_file()


def test_cpu_qualification_cli_runs_real_aot_workers(tmp_path):
    assert (
        main(
            [
                "qualify",
                "--device",
                "cpu",
                "--backend",
                "aot_eager",
                "--dtype",
                "float32",
                "--workload",
                "mlp",
                "--repeats",
                "2",
                "--output",
                str(tmp_path / "run"),
            ]
        )
        == 0
    )
    assert (
        tmp_path / "run" / "cases" / "mlp-float32" / "check" / "run" / "aot-graphs.json"
    ).is_file()


@pytest.mark.parametrize(
    "environment",
    [
        {"hip_version": None, "cuda_version": None, "gpu_available": False},
        {"hip_version": "7.0", "cuda_version": None, "gpu_available": True},
        {"hip_version": None, "cuda_version": "13.0", "gpu_available": False},
    ],
)
def test_nvidia_requirement_never_accepts_cpu_or_rocm(tmp_path, monkeypatch, environment):
    monkeypatch.setattr(qualify, "describe", lambda: environment)
    monkeypatch.setattr(
        qualify,
        "capture_workload",
        lambda *a, **kw: pytest.fail("Must not execute a fallback workload"),
    )
    result = qualify.qualify_device(
        tmp_path / "check", Settings(device="cuda:0", gradients=True), require_cuda=True
    )
    assert result["status"] == "unavailable" and result["exit_code"] == 2
    assert not result["cases"]


def test_buffer_update_request_stream_on_cpu(tmp_path):
    result = qualify.qualify_device(
        tmp_path / "check",
        Settings(backend="aot_eager", gradients=True, repeats=2),
        dtypes=("float32",),
        workloads=("buffer_update",),
        aot=False,
        progress=None,
    )
    assert result["status"] == "pass", result


def test_qualification_exposes_bfloat16_skip_without_rewriting_primary(tmp_path, monkeypatch):
    warning = "Tesla T4 does not support bfloat16 compilation natively, skipping"
    monkeypatch.setattr(
        qualify,
        "describe",
        lambda: {"hip_version": None, "cuda_version": "13.2", "gpu_available": True},
    )
    monkeypatch.setattr(qualify, "capture_workload", lambda source, output, settings, **kw: output)

    def run(bundle, output, settings, **kwargs):
        (output / "run").mkdir(parents=True)
        (output / "run/stderr.log").write_text(
            f"/site-packages/torch/_inductor/compile_fx.py:1: UserWarning: {warning}\n"
            f"/site-packages/torch/_inductor/compile_fx.py:1: UserWarning: {warning}\n"
        )
        return {"status": "pass", "exit_code": 0}

    monkeypatch.setattr(qualify, "run_native", run)
    messages = []
    result = qualify.qualify_device(
        tmp_path / "check",
        Settings(device="cuda:0", gradients=True, autocast="bfloat16"),
        dtypes=("float32",),
        workloads=("mlp",),
        require_cuda=True,
        aot=False,
        progress=messages.append,
    )
    assert result["status"] == "pass" and result["exit_code"] == 0
    case = result["cases"][0]
    assert case["status"] == "pass" and case["exit_code"] == 0
    assert case["compilation_coverage"]["status"] == "skipped"
    assert case["compilation_coverage"]["warnings"] == [warning]
    assert any("not compiled validation" in message for message in messages)
    assert warning in (tmp_path / "check/report.html").read_text()


def test_missing_or_unrelated_logs_do_not_establish_compiler_coverage(tmp_path):
    log = tmp_path / "stderr.log"
    assert qualify.compilation_coverage(log)["status"] == "unavailable"
    log.write_text("UserWarning: unrelated runtime warning\n")
    coverage = qualify.compilation_coverage(log)
    assert coverage["status"] == "no_skip_reported"
    assert coverage["warnings"] == []
    assert coverage["advisory_only"]


def test_unreadable_coverage_log_does_not_replace_a_primary_result(tmp_path, monkeypatch):
    log = tmp_path / "stderr.log"
    log.write_text("unreadable log")
    read_text = type(log).read_text

    def denied(path, *args, **kwargs):
        if path == log:
            raise PermissionError("cannot read worker log")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(type(log), "read_text", denied)
    coverage = qualify.compilation_coverage(log)
    assert coverage == {
        "advisory_only": True,
        "status": "unavailable",
        "warnings": [],
        "error": "PermissionError",
    }
