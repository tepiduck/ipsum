"""Commands for generation, execution, reduction, export, and capability discovery."""

import argparse
import json
import sys
import webbrowser
from pathlib import Path

from . import __version__
from .campaign import run_campaign
from .environment import describe
from .evaluate import Settings
from .export import export_case
from .finding import load_finding, replay_finding
from .generate import corpus, generate, reduction_fixture
from .lowering import freeze
from .report import write_report
from .runner import run_case
from .storage import atomic_json, read_json
from .trace import Trace, program_from_dict
from .workflows import run_demo, run_reduction


def execution_options(parser, optional=False):
    parser.add_argument(
        "--backend",
        default="inductor",
        help="inductor, aot_eager, aot_eager_decomp_partition, eager, or mutant:<name>",
    )
    parser.add_argument("--device", default="cpu", help="cpu or cuda[:index]; ROCm uses cuda")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-rss-mb", type=int, default=4096)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--gradients", action="store_true", default=None if optional else False)
    parser.add_argument(
        "--dynamic", choices=("static", "auto", "dynamic"), default=None if optional else "static"
    )
    if optional:
        parser.set_defaults(
            backend=None, device=None, repeats=None, threads=None, timeout=None, max_rss_mb=None
        )


def settings_from(args):
    names = {
        "timeout": "timeout_s",
        "max_rss_mb": "max_rss_mb",
        "backend": "backend",
        "device": "device",
        "repeats": "repeats",
        "threads": "threads",
        "rtol": "rtol",
        "atol": "atol",
        "gradients": "gradients",
    }
    values = {
        key: getattr(args, arg) for arg, key in names.items() if getattr(args, arg) is not None
    }
    if args.dynamic is not None:
        values["dynamic"] = {"static": False, "auto": None, "dynamic": True}[args.dynamic]
    for name in ("compile_mode", "autocast", "fp64_advisory"):
        if getattr(args, name, None) is not None:
            values[name] = getattr(args, name)
    return Settings(**values)


def case_source(args):
    if args.case.is_dir():
        if any(
            getattr(args, name) is not None
            for name in (
                "backend",
                "device",
                "repeats",
                "threads",
                "timeout",
                "max_rss_mb",
                "rtol",
                "atol",
                "gradients",
                "dynamic",
            )
        ):
            raise ValueError("Saved findings retain their settings; overrides are not accepted")
        finding = load_finding(args.case)
        return finding.case, finding.settings, finding
    return freeze(program_from_dict(read_json(args.case))), settings_from(args), None


def exit_status(summary):
    counts = summary["counts"]
    if any(
        counts.get(s, 0)
        for s in (
            "numerical_mismatch",
            "gradient_mismatch",
            "structural_mismatch",
            "compile_error",
            "runtime_error",
        )
    ):
        return 1
    return 0 if set(counts) <= {"pass"} else 2


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Faultline — differential testing for tensor compilers"
    )
    parser.add_argument("--version", action="version", version=f"Faultline {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    from .model_cli import add_model_commands, model_command, native_options

    add_model_commands(commands, execution_options)
    benchmark = commands.add_parser(
        "benchmark", help="measure historical-issue detection and reduction"
    )
    benchmark_sub = benchmark.add_subparsers(dest="benchmark_command", required=True)
    benchmark_sub.add_parser("list", help="list the versioned issue fixtures")
    history = benchmark_sub.add_parser(
        "historical", help="run known issue patterns in two environments"
    )
    history.add_argument("--good-python", default=sys.executable)
    history.add_argument("--bad-python", required=True)
    history.add_argument(
        "--case", action="append", dest="issue_ids", help="issue ID; repeat to select a subset"
    )
    history.add_argument("--device", default="cpu")
    history.add_argument("--repeats", type=int, default=2)
    history.add_argument("--threads", type=int, default=1)
    history.add_argument("--timeout", type=float, default=300)
    history.add_argument("--max-rss-mb", type=int, default=4096)
    history.add_argument("--max-evaluations", type=int, default=16)
    history.add_argument("--no-reduce", action="store_false", dest="reduce")
    history.add_argument("--diagnose", action="store_true")
    history.add_argument(
        "--require-paired",
        action="store_true",
        help="exit 1 unless every case fails as expected with a passing control",
    )
    history.add_argument("--output", type=Path, required=True)
    qualify = commands.add_parser(
        "qualify", help="check a device/compiler across dtypes and backward graphs"
    )
    execution_options(qualify)
    native_options(qualify)
    qualify.set_defaults(device="cuda:0", gradients=True, timeout=300)
    qualify.add_argument("--dtype", action="append", choices=("float32", "float16", "bfloat16"))
    qualify.add_argument(
        "--workload",
        action="append",
        choices=("mlp", "convolution", "attention", "buffer_update", "shape_sequence"),
    )
    runtime = qualify.add_mutually_exclusive_group()
    runtime.add_argument("--require-rocm", action="store_true")
    runtime.add_argument("--require-cuda", action="store_true")
    qualify.add_argument(
        "--no-aot",
        action="store_false",
        dest="aot",
        help="use normal backend caching; omit diagnostic AOT snapshots",
    )
    qualify.add_argument("--output", type=Path, required=True)
    ui = commands.add_parser("ui", help="open the interactive terminal interface")
    ui.add_argument("--workspace", type=Path, default=Path("faultline-runs"))
    demo = commands.add_parser("demo", help="detect and shrink a controlled fault, end to end")
    demo.add_argument("--output", type=Path, required=True)
    doctor = commands.add_parser("doctor", help="show environment and optionally test compilation")
    execution_options(doctor)
    doctor.add_argument(
        "--probe", type=Path, metavar="OUTPUT", help="run a small compiler smoke test"
    )
    gen = commands.add_parser("generate", help="write a frozen, data-only case")
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--depth", type=int, default=10)
    gen.add_argument("--dtype", default="float32")
    gen.add_argument(
        "--fixture", action="store_true", help="noisy graph for controlled reduction tests"
    )
    gen.add_argument("--output", type=Path, required=True)
    capture = commands.add_parser("capture", help="capture a trusted local model.py:factory")
    capture.add_argument("factory", help="executes trusted Python returning (eval model, inputs)")
    capture.add_argument("--frontend", choices=("export", "fx"), default="export")
    capture.add_argument(
        "--trace", action="store_true", help="factory returns a sequence of input tuples"
    )
    capture.add_argument("--output", type=Path, required=True)
    trace = commands.add_parser(
        "trace", help="combine ordered, compatible cases into a request trace"
    )
    trace.add_argument("--case", type=Path, action="append", required=True)
    trace.add_argument("--output", type=Path, required=True)
    comparison = commands.add_parser("compare", help="gate compiler environment upgrades")
    execution_options(comparison)
    comparison.add_argument("--baseline-python", required=True)
    comparison.add_argument("--candidate-python", default=sys.executable)
    comparison.add_argument("--candidate-backend", help="optional explicit backend comparison")
    comparison.add_argument("--case", type=Path, action="append", default=[])
    comparison.add_argument("--corpus", action="store_true")
    comparison.add_argument("--output", type=Path, required=True)
    comparison.add_argument("--reduce", action="store_true")
    comparison.add_argument("--max-evaluations", type=int, default=64)
    run = commands.add_parser("run", help="run a corpus, generated cases, or JSON files")
    execution_options(run)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--case", type=Path, action="append", default=[])
    run.add_argument("--corpus", action="store_true")
    run.add_argument("--count", type=int, default=0)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--depth", type=int, default=10)
    run.add_argument("--dtype", default="float32")
    resume = commands.add_parser(
        "resume", help="continue a campaign with unchanged source/environment"
    )
    resume.add_argument("directory", type=Path)
    reduction = commands.add_parser("reduce", help="minimize a stable failure")
    execution_options(reduction, optional=True)
    reduction.add_argument("case", type=Path, help="case JSON or saved finding directory")
    reduction.add_argument("--output", type=Path, required=True)
    reduction.add_argument("--max-evaluations", type=int, default=64)
    export = commands.add_parser("export", help="export a standalone PyTorch reproducer")
    execution_options(export, optional=True)
    export.add_argument("case", type=Path, help="case JSON or saved finding directory")
    export.add_argument("--output", type=Path, required=True)
    replay = commands.add_parser("replay", help="recheck saved evidence with its original settings")
    replay.add_argument("directory", type=Path)
    replay.add_argument("--output", type=Path, required=True)
    report = commands.add_parser("report", help="render an existing campaign")
    report.add_argument("directory", type=Path)
    report.add_argument("--open", action="store_true", help="open the local report in your browser")
    args = parser.parse_args(argv)
    try:
        if args.command == "benchmark":
            from .historical import case_catalog, historical_benchmark

            if args.benchmark_command == "list":
                print(json.dumps(case_catalog(), indent=2))
                return 0
            policy = Settings(
                device=args.device,
                repeats=args.repeats,
                threads=args.threads,
                timeout_s=args.timeout,
                max_rss_mb=args.max_rss_mb,
            )
            result = historical_benchmark(
                args.output,
                args.good_python,
                args.bad_python,
                policy,
                cases=args.issue_ids,
                reduce=args.reduce,
                diagnose=args.diagnose,
                max_evaluations=args.max_evaluations,
                require_paired=args.require_paired,
            )
            print(
                json.dumps(
                    {
                        k: result[k]
                        for k in (
                            "status",
                            "selected_count",
                            "candidate_detections",
                            "paired_reproductions",
                            "counts",
                        )
                    },
                    indent=2,
                )
            )
            return result["exit_code"]
        if args.command == "qualify":
            from .qualify import qualify_device

            result = qualify_device(
                args.output,
                settings_from(args),
                dtypes=args.dtype or ("float32", "float16", "bfloat16"),
                workloads=args.workload or ("mlp", "convolution", "attention"),
                require_rocm=args.require_rocm,
                require_cuda=args.require_cuda,
                aot=args.aot,
            )
            print(
                json.dumps(
                    {
                        "status": result["status"],
                        "reason": result.get("reason"),
                        "report": str(args.output / "report.html"),
                    },
                    indent=2,
                )
            )
            return result["exit_code"]
        if args.command == "model":
            return model_command(args, settings_from)
        if args.command == "ui":
            from .ui import launch

            return launch(args.workspace)
        if args.command == "demo":
            result = run_demo(args.output, progress=lambda text: print(text, flush=True))
            print(
                f"\nControlled fault reduced: {result['original_complexity']} -> "
                f"{result['reduced_complexity']}"
            )
            print(f"Report: {args.output / result['report']}")
            print(f"Reproducer: {args.output / result['reproducer']}")
            print(
                "Demo completed. The standalone reproducer returns 1 for the injected discrepancy."
            )
        elif args.command == "doctor":
            environment = describe()
            if args.probe:
                if args.probe.exists():
                    raise ValueError("Probe output exists; choose a new directory")
                result = run_case(freeze(corpus()[0]), settings_from(args), args.probe)
                environment["probe"] = result
                print(json.dumps(environment, indent=2))
                return 0 if result["status"] == "pass" else 2
            print(json.dumps(environment, indent=2))
        elif args.command == "generate":
            case = (
                reduction_fixture() if args.fixture else generate(args.seed, args.depth, args.dtype)
            )
            atomic_json(args.output, freeze(case).to_dict())
            print(args.output)
        elif args.command == "capture":
            from .model_source import capture_factory

            if args.output.exists():
                raise ValueError("Capture output exists; choose a new path")
            case = capture_factory(args.factory, trace=args.trace, frontend=args.frontend)
            atomic_json(args.output, case.to_dict())
            print(args.output)
        elif args.command == "trace":
            if args.output.exists():
                raise ValueError("Trace output exists; choose a new path")
            steps = tuple(freeze(program_from_dict(read_json(path))) for path in args.case)
            if any(isinstance(step, Trace) for step in steps):
                raise ValueError("trace --case accepts individual cases, not nested traces")
            result = Trace(steps)
            result.validate()
            atomic_json(args.output, result.to_dict())
            print(args.output)
        elif args.command == "compare":
            from dataclasses import replace

            from .compare_envs import compare_environments

            settings = settings_from(args)
            cases = [program_from_dict(read_json(path)) for path in args.case]
            if args.corpus:
                cases.extend(corpus())
            candidate = (
                replace(settings, backend=args.candidate_backend)
                if args.candidate_backend
                else settings
            )
            result = compare_environments(
                cases,
                args.output,
                args.baseline_python,
                args.candidate_python,
                settings,
                candidate,
                args.reduce,
                args.max_evaluations,
            )
            print(args.output / "report.html")
            return result["exit_code"]
        elif args.command == "run":
            if not 0 <= args.count <= 10000:
                raise ValueError("count must be in [0, 10000]")
            cases = [program_from_dict(read_json(path)) for path in args.case]
            if args.corpus:
                cases.extend(corpus(args.dtype))
            cases.extend(generate(args.seed + i, args.depth, args.dtype) for i in range(args.count))
            summary = run_campaign(args.output, cases, settings_from(args))
            print(write_report(args.output))
            return exit_status(summary)
        elif args.command == "resume":
            summary = run_campaign(args.directory, resume=True)
            print(write_report(args.directory))
            return exit_status(summary)
        elif args.command == "report":
            path = write_report(args.directory)
            print(path)
            if args.open and not webbrowser.open(path.resolve().as_uri()):
                print("Could not open a browser; open the report path above.", file=sys.stderr)
        elif args.command == "replay":
            result = replay_finding(load_finding(args.directory), args.output)
            print(json.dumps(result, indent=2))
            return 1 if result["reproduces_expected"] else 2
        elif args.command == "export":
            case, settings, finding = case_source(args)
            if finding:
                print(
                    export_case(
                        case,
                        settings,
                        args.output,
                        finding.outcome,
                        environment=finding.environment,
                    )
                )
                atomic_json(args.output / "source.json", finding.source_record())
            else:
                print(export_case(case, settings, args.output))
        elif args.command == "reduce":
            case, settings, finding = case_source(args)
            repro, _ = run_reduction(
                case,
                settings,
                args.output,
                args.max_evaluations,
                finding=finding,
                progress=lambda text: print(text, flush=True),
            )
            print(repro)
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"faultline: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; completed campaign cases remain resumable.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
