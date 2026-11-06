"""CLI for original-model workloads, separate from the compact IR fuzzing commands."""

import json
import sys
from dataclasses import replace
from pathlib import Path


def native_options(parser, optional=False):
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=None if optional else False,
        help="require deterministic PyTorch algorithms and fixed BLAS workspace policy",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"),
        default=None if optional else "default",
    )
    parser.add_argument("--autocast", choices=("float16", "bfloat16"))
    parser.add_argument("--fp64-advisory", action="store_true", default=None if optional else False)
    parser.add_argument(
        "--numerics",
        action="store_true",
        default=None if optional else False,
        help="fp64 evidence and eager control; add --aot --gradients for saved masks",
    )


def add_model_commands(commands, execution_options):
    model = commands.add_parser("model", help="capture and check original PyTorch models")
    sub = model.add_subparsers(dest="model_command", required=True)
    capture = sub.add_parser("capture", help="freeze a trusted factory's state and nested inputs")
    capture.add_argument("factory", help="model.py:factory returning (eval model, inputs)")
    capture.add_argument("--trace", action="store_true")
    capture.add_argument("--backward-probe", choices=("vjp", "loss"), default="vjp")
    capture.add_argument(
        "--include", action="append", default=[], help="explicit local source or checkpoint asset"
    )
    capture.add_argument("--python", default=sys.executable)
    execution_options(capture)
    for name in ("run", "compare"):
        command = sub.add_parser(name, help="check a frozen native workload")
        command.add_argument("bundle", type=Path)
        execution_options(command)
        native_options(command)
        command.add_argument(
            "--aot",
            action="store_true",
            help="capture compiler forward/backward graphs; requires --gradients",
        )
        if name == "run":
            command.add_argument("--python", default=sys.executable)
        else:
            command.add_argument("--baseline-python", required=True)
            command.add_argument("--candidate-python", default=sys.executable)
            command.add_argument("--candidate-backend")
            command.add_argument("--reduce", action="store_true")
            command.add_argument("--max-evaluations", type=int, default=32)
        command.add_argument("--output", type=Path, required=True)
    for name in ("replay", "reduce", "export"):
        command = sub.add_parser(name, help=f"{name} a saved model run with its original settings")
        command.add_argument("directory", type=Path)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--python", default=sys.executable)
        if name == "reduce":
            command.add_argument("--max-evaluations", type=int, default=32)
    capture.add_argument("--output", type=Path, required=True)
    diagnose = sub.add_parser("diagnose", help="run the backend ladder on a bundle or saved run")
    diagnose.add_argument("directory", type=Path)
    diagnose.add_argument("--python", default=sys.executable)
    diagnose.add_argument("--output", type=Path, required=True)
    execution_options(diagnose, optional=True)
    native_options(diagnose, optional=True)


def model_command(args, settings_from):
    from .native import capture_workload
    from .native_artifacts import (
        compare_native_environments,
        export_native,
        load_native_run,
        run_native,
    )

    if args.model_command == "diagnose":
        from .diagnose import diagnose_native

        names = (
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
            "compile_mode",
            "autocast",
            "fp64_advisory",
            "numerics",
            "deterministic",
        )
        overridden = any(getattr(args, name) is not None for name in names)
        result = diagnose_native(
            args.directory,
            args.output,
            settings_from(args) if overridden else None,
            python=args.python,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "boundary": result.get("boundary"),
                    "report": str(args.output / "report.html"),
                },
                indent=2,
            )
        )
        return result["exit_code"]

    if args.model_command == "capture":
        path = capture_workload(
            args.factory,
            args.output,
            settings_from(args),
            trace=args.trace,
            include=args.include,
            python=args.python,
            backward_probe=args.backward_probe,
        )
        print(f"Captured workload: {path}")
        return 0
    if args.model_command == "run":
        result = run_native(
            args.bundle, args.output, settings_from(args), python=args.python, aot=args.aot
        )
    elif args.model_command == "compare":
        settings = settings_from(args)
        candidate = replace(settings, backend=args.candidate_backend or settings.backend)
        result = compare_native_environments(
            args.bundle,
            args.output,
            args.baseline_python,
            args.candidate_python,
            settings,
            candidate,
            args.reduce,
            args.max_evaluations,
            aot=args.aot,
        )
        print(
            json.dumps(
                {
                    "classification": result["classification"],
                    "accuracy_advisory": {
                        key: value
                        for key, value in result.get("accuracy_comparison", {}).items()
                        if key not in {"targets", "policy", "note"}
                    },
                    "report": str(args.output / "report.html"),
                },
                indent=2,
            )
        )
        return result["exit_code"]
    else:
        job, settings, expected = load_native_run(args.directory)
        if args.model_command == "export":
            print(export_native(job, settings, args.output, expected))
            return 0
        if args.model_command == "reduce":
            from .native_reduce import reduce_native

            result = reduce_native(
                job,
                settings,
                expected,
                args.output,
                candidate_python=args.python,
                max_evaluations=args.max_evaluations,
            )
            print(json.dumps(result, indent=2))
            return 0 if result["confirmed"] else 2
        result = run_native(
            job.bundle,
            args.output,
            settings,
            python=args.python,
            expected=expected,
            only_graph=job.only_graph,
            requests=job.requests,
            aot=job.aot,
            only_aot=job.only_aot,
        )
    print(json.dumps(result, indent=2))
    return result["exit_code"]
