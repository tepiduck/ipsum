"""Concise check CLI, with exit codes reserved for the summary verdict only."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from .bisect_env import supervise
from .check import PRESETS, report, run_check
from .compare_envs import interpreter
from .evaluate import Settings
from .storage import atomic_json, read_json
from .verdict import summarize


class CheckAwareParser(argparse.ArgumentParser):
    """Only the new check command assigns setup/argument errors exit code 3."""

    check_invocation = False

    def error(self, message):
        if self.check_invocation or self.prog.endswith(" check"):
            self.exit(3, f"INCONCLUSIVE - {message}\nNext: run faultline check --help.\n")
        super().error(message)


def add_check_command(commands):
    parser = commands.add_parser("check", help="check a model and summarize upgrade evidence")
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument(
        "model", nargs="?", help="model.py:factory or a captured bundle/collection"
    )
    sources.add_argument("--torchvision", metavar="NAME")
    sources.add_argument("--hf", metavar="MODEL_ID")
    parser.add_argument("--revision", help="Pinned Hugging Face revision")
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--baseline", "--baseline-python", dest="baseline")
    parser.add_argument(
        "--candidate", "--candidate-python", dest="candidate", default=sys.executable
    )
    parser.add_argument("--level", choices=tuple(PRESETS), default="standard")
    parser.add_argument("--batches", type=int)
    parser.add_argument("--backend", default="inductor")
    parser.add_argument("--baseline-backend", default="inductor")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--baseline-compile-mode", default="default")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--max-rss-mb", type=int, default=12288)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def next_step(verdict):
    return {
        "SAFE": "Keep this report with the tested model and environment versions.",
        "REGRESSION": "Review the candidate-only evidence and export/reduce the failing batch before upgrading.",
        "NEEDS_REVIEW": "Open the report; inspect contrary evidence or provide the missing input/diagnostic coverage.",
        "INCONCLUSIVE": "Fix the recorded setup or execution failure, then use a new output directory.",
    }[verdict]


def terminal(verdict, output):
    scope = verdict["scope"]
    lines = [
        f"{verdict['verdict']} - Faultline check",
        f"Checked: {scope['mode']}, {scope['level']}; {scope['completed_batches']}/{scope['planned_batches']} batches.",
    ]
    distinct = list(dict.fromkeys(reason["message"] for reason in verdict["reasons"]))
    for reason in distinct[:3]:
        compact = " ".join(reason.split())
        lines.append(f"Reason: {compact[:197] + '...' if len(compact) > 200 else compact}")
    if not distinct:
        lines.append("No blocking evidence in the requested checks.")
    if verdict["notes"]:
        lines.append(
            f"Notes: {len(verdict['notes'])} explained findings or unavailable optional diagnostics; see report."
        )
    if any(note["code"] == "diagnostic_skipped_memory" for note in verdict["notes"]):
        lines.append("Diagnostics: skipped (memory); completed primary evidence retained.")
    if verdict["injected_fault"]:
        lines.append(
            "CONTROLLED INJECTED FAULT: this is harness validation, not a compiler finding."
        )
    action = next_step(verdict["verdict"])
    if any("compile_error" in reason["message"] for reason in verdict["reasons"]):
        action = "Check the C++ toolchain with faultline doctor --backend inductor --probe NEW_DIRECTORY; inspect compiler stderr in the report evidence."
    elif any(reason["code"].endswith("_resource_limit") for reason in verdict["reasons"]):
        action = "Use smaller representative inputs or an explicit larger memory budget if available; keep this incomplete report."
    lines.extend(
        [
            f"Next: {action}",
            f"HTML: {output / 'report.html'}",
            f"JSON: {output / 'verdict.json'}",
        ]
    )
    for path in verdict["artifacts"].get("reproducers", [])[:2]:
        lines.append(f"Reproducer: {output / path}")
    lines.append("Scope: tested inputs and settings only; not a general correctness guarantee.")
    return "\n".join(lines)


def setup_help(error):
    if isinstance(error, subprocess.CalledProcessError):
        detail = (error.stderr or "").strip().splitlines()
        text = (
            f"Python environment probe failed: {detail[-1]}"
            if detail
            else f"Python environment probe exited with code {error.returncode}."
        )
    else:
        text = str(error)
    lower = text.lower()
    if "compiler" in lower or "cl.exe" in lower or "g++" in lower:
        return (
            text
            + " Install a C++ toolchain and Python headers; on Windows use WSL or a supported compiler. Run faultline doctor --backend inductor --probe NEW_DIRECTORY."
        )
    if "python" in lower or "torch" in lower or "module" in lower:
        return (
            text
            + " Check each interpreter path and install PyTorch plus the factory's dependencies in both environments."
        )
    if "memory" in lower or "resource_limit" in lower:
        return (
            text
            + " Close other workloads, reduce batch size or select --level quick; keep the saved primary evidence."
        )
    return (
        text
        + " Verify the factory returns an eval model and representative inputs; inspect the saved worker logs."
    )


def check_command(args):
    output = (args.output or Path("faultline-runs") / f"check-{time.time_ns()}").resolve()
    try:
        if not (args.model or args.torchvision or args.hf):
            raise ValueError(
                "Supply model.py:factory, a captured bundle, --torchvision NAME, or --hf MODEL_ID."
            )
        if args.batches is not None and not 1 <= args.batches <= 100:
            raise ValueError("--batches must be in [1, 100]")
        candidate = interpreter(args.candidate)
        baseline = interpreter(args.baseline) if args.baseline else None
        preset = PRESETS[args.level]
        common = {
            "gradients": preset["gradients"],
            "repeats": preset["repeats"],
            "device": args.device,
            "deterministic": args.deterministic,
            "threads": args.threads,
            "timeout_s": args.timeout,
            "max_rss_mb": args.max_rss_mb,
        }
        settings = Settings(**common, backend=args.backend, compile_mode=args.compile_mode)
        base_settings = Settings(
            **common, backend=args.baseline_backend, compile_mode=args.baseline_compile_mode
        )
        settings.validate()
        base_settings.validate()
        started = time.monotonic()
        if args.model and Path(args.model).is_dir():
            bundle = Path(args.model)
        else:
            specification = {
                "factory": args.model,
                "torchvision": args.torchvision,
                "hf": args.hf,
                "revision": args.revision,
                "batches": args.batches or preset["batches"],
                "explicit_batches": args.batches is not None,
                "include": args.include,
                "timeout": args.timeout,
                "max_rss_mb": args.max_rss_mb,
            }
            preparation = output / "preparation"
            if preparation.exists():
                if (
                    not args.resume
                    or read_json(output / "prepare-specification.json") != specification
                ):
                    raise ValueError(
                        "Preparation exists or its specification changed; choose a new output directory."
                    )
            else:
                output.mkdir(parents=True, exist_ok=True)
                atomic_json(output / "prepare-specification.json", specification)
                from .bootstrap import python_command

                command = python_command(
                    baseline or candidate,
                    "from faultline.check_prepare import main;main()",
                    [
                        "--specification",
                        str(output / "prepare-specification.json"),
                        "--output",
                        str(preparation),
                    ],
                )
                observed = supervise(
                    command,
                    output / "preparation-supervision",
                    timeout=max(args.timeout, 900),
                    max_rss_mb=args.max_rss_mb,
                )
                if observed["status"] != "complete":
                    log = output / "preparation-supervision/stderr.log"
                    detail = (
                        log.read_text(encoding="utf-8", errors="replace")[-2000:]
                        if log.exists()
                        else ""
                    )
                    raise ValueError(
                        f"Model preparation {observed['status']}: {detail.strip().splitlines()[-1] if detail.strip() else 'no worker error detail'}. See {log}"
                    )
            bundle = Path(read_json(preparation / "prepared.json")["bundle"])
        verdict = run_check(
            bundle,
            output,
            candidate_python=candidate,
            baseline_python=baseline,
            settings=settings,
            baseline_settings=base_settings,
            level=args.level,
            batches=args.batches,
            resume=args.resume,
        )
        verdict.setdefault("metrics", {})["command_wall_s"] = time.monotonic() - started
        report(output, verdict, read_json(output / "evidence.json"))
    except (
        ValueError,
        RuntimeError,
        OSError,
        KeyError,
        ImportError,
        subprocess.SubprocessError,
    ) as exc:
        if (output / "verdict.json").exists():
            print(
                f"INCONCLUSIVE - {setup_help(exc)}\nExisting report retained: {output / 'report.html'}",
                file=sys.stderr,
            )
            return 3
        output.mkdir(parents=True, exist_ok=True)
        verdict = summarize(
            [],
            args.batches or PRESETS[args.level]["batches"],
            mode="environment_comparison" if args.baseline else "single_environment",
            level=args.level,
        )
        verdict["reasons"] = [{"code": "setup_error", "batch": None, "message": setup_help(exc)}]
        report(output, verdict, {"setup_error": str(exc)})
    print(terminal(verdict, output))
    return verdict["exit_code"]
