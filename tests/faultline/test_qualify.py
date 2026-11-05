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
