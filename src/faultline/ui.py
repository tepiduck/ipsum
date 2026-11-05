"""Dependency-free terminal interface. All actions use the public CLI workflows."""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .storage import read_json


def new_output(workspace, kind):
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return workspace / f"{kind}-{stamp}-{uuid4().hex[:6]}"


def choose(title, labels, ask=input, say=print):
    say(f"\n{title}")
    for index, label in enumerate(labels, 1):
        say(f"  {index}. {label}")
    say("  0. Back")
    while True:
        answer = ask("Choose a number: ").strip()
        if answer == "0":
            return None
        if answer.isdecimal() and 1 <= int(answer) <= len(labels):
            return int(answer) - 1
        say("Enter one of the listed numbers.")


def campaigns(workspace):
    """Discover only direct children and demo subruns, without following symlinks."""
    found = []
    for path in sorted(workspace.iterdir()):
        if path.is_symlink() or not path.is_dir():
            continue
        candidates = [path]
        if (path / "demo.json").is_file():
            candidates.extend(path / name for name in ("control", "injected"))
        for candidate in candidates:
            if candidate.is_symlink():
                continue
            summary_path = candidate / "summary.json"
            if not summary_path.is_file():
                continue
            try:
                summary = read_json(summary_path)
                label = ", ".join(f"{n} {s}" for s, n in summary["counts"].items())
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                label = "unreadable summary"
            found.append((candidate, label))
    return found


def saved_case(workspace, ask, say):
    runs = campaigns(workspace)
    if not runs:
        say("No campaigns here yet. Run a demo or a corpus first.")
        return None
    selected = choose("Campaigns", [f"{p.relative_to(workspace)} — {s}" for p, s in runs], ask, say)
    if selected is None:
        return None
    path = runs[selected][0]
    # A malformed manifest must never turn a menu choice into an arbitrary path.
    from .campaign import validate_manifest

    manifest = read_json(path / "manifest.json")
    validate_manifest(manifest)
    entries = manifest["cases"]
    labels = []
    for entry in entries:
        case_dir = path / "cases" / entry["id"]
        if not case_dir.resolve().is_relative_to(path.resolve()):
            raise ValueError("Campaign case resolves outside its directory")
        result_path = case_dir / "outcome.json"
        status = read_json(result_path)["status"] if result_path.is_file() else "pending"
        labels.append(f"{entry['id']} — {entry['tag']} — {status}")
    selected = choose("Cases", labels, ask, say)
    return None if selected is None else path / "cases" / entries[selected]["id"]


def launch(workspace, execute=None, ask=input, say=print):
    if execute is None:
        from .cli import main

        execute = main
    workspace = Path(workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    say(f"Faultline | tensor compiler testing\nRuns: {workspace}")
    say("Each action creates new evidence. Ctrl+C or Ctrl+D leaves the interface.")
    options = [
        "Run the guided demo (controlled fault)",
        "Test the built-in corpus",
        "Test a JSON case",
        "Investigate a saved campaign",
        "Check this environment",
    ]
    try:
        while True:
            action = choose("What would you like to do?", options, ask, say)
            if action is None:
                return 0
            try:
                if action == 0:
                    command = ["demo", "--output", str(new_output(workspace, "demo"))]
                elif action in {1, 2}:
                    case = None
                    if action == 2:
                        case = Path(ask("Case JSON path: ").strip()).expanduser()
                        if not case.is_file():
                            say("That file does not exist.")
                            continue
                    backend = choose(
                        "Compiler backend", ["inductor", "aot_eager", "eager"], ask, say
                    )
                    if backend is None:
                        continue
                    device = ask("Device [cpu] (ROCm also uses cuda:0): ").strip() or "cpu"
                    command = [
                        "run",
                        "--backend",
                        ["inductor", "aot_eager", "eager"][backend],
                        "--device",
                        device,
                        "--output",
                        str(new_output(workspace, "run")),
                    ]
                    command.extend(["--case", str(case)] if case else ["--corpus"])
                elif action == 3:
                    case = saved_case(workspace, ask, say)
                    if case is None:
                        continue
                    operation = choose(
                        "Finding action", ["Replay", "Reduce", "Export", "Open report"], ask, say
                    )
                    if operation is None:
                        continue
                    if operation == 3:
                        command = ["report", str(case.parent.parent), "--open"]
                    else:
                        verb = ["replay", "reduce", "export"][operation]
                        command = [verb, str(case), "--output", str(new_output(workspace, verb))]
                else:
                    command = ["doctor"]
                code = execute(command)
                say(
                    {
                        0: "Completed.",
                        1: "A stable discrepancy was observed; review the evidence.",
                        2: "Inconclusive or configuration error; see the message above.",
                        130: "Cancelled.",
                    }.get(code, f"Exit code: {code}")
                )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                say(f"Could not complete that action: {exc}")
    except (EOFError, KeyboardInterrupt):
        say("\nLeaving Faultline. Saved evidence remains on disk.")
        return 0
