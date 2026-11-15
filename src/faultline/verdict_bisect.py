"""Replicated standard-check v2 bisection for watch regressions, with fresh confirmation."""

import html
import json
import shutil
import time
from pathlib import Path

from .batches import verify_batches
from .bisect import choose_next
from .bisect_env import environment_identity, nightly_catalog
from .native import file_hash
from .storage import atomic_json, campaign_lock, provenance, read_json
from .watch_support import (
    commit_record,
    ensure_wheel,
    execute_check,
    model_requirements,
    verify_record,
    wheel_index,
)


def export_collection(bundle, output, settings):
    """Export all independent inputs; one native repro cannot establish replication."""
    output = Path(output)
    if (output / "reproducer.json").exists():
        record = read_json(output / "reproducer.json")
        for name, digest in record["files"].items():
            target = (output / name).resolve()
            if not target.is_relative_to(output.resolve()) or file_hash(target) != digest:
                raise ValueError("Exported watch reproducer changed")
        return output / "repro_check.py"
    verify_batches(bundle)
    output.mkdir(parents=True, exist_ok=True)
    # An interrupted export gets a new directory; never merge partial tensor snapshots.
    destination = output / f"inputs-{len(list(output.glob('inputs-*'))) + 1:04d}"
    shutil.copytree(bundle, destination)
    verify_batches(destination)
    program = '''"""Run with Faultline and the factory dependencies in both environments."""
import argparse, hashlib, json
from pathlib import Path
from faultline.cli import main
root = Path(__file__).resolve().parent
record = json.loads((root / "reproducer.json").read_text())
for name, expected in record["files"].items():
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Invalid artifact path")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
            raise ValueError("Modified artifact: " + name)
p = argparse.ArgumentParser()
p.add_argument("--baseline", required=True)
p.add_argument("--candidate", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
raise SystemExit(main(["check", str(root / record["bundle"]), "--baseline", a.baseline,
 "--candidate", a.candidate, "--output", a.output, "--level", "standard",
 "--verdict-policy", "v2", "--device", record["settings"]["device"],
 "--max-rss-mb", str(record["settings"]["max_rss_mb"]),
 "--timeout", str(record["settings"]["timeout_s"])]
 + (["--deterministic"] if record["settings"]["deterministic"] else [])))
'''
    (output / "repro_check.py").write_text(program, encoding="utf-8")
    files = [output / "repro_check.py", *[p for p in destination.rglob("*") if p.is_file()]]
    atomic_json(
        output / "reproducer.json",
        {
            "bundle": destination.name,
            "settings": settings.to_dict(),
            "source_sha256": provenance()["faultline_source_sha256"],
            "files": {p.relative_to(output).as_posix(): file_hash(p) for p in files},
            "note": "Full frozen batch collection. Requires Faultline; existing per-batch repro.py exports need only PyTorch and factory dependencies. No compiler defect claim.",
        },
    )
    return output / "repro_check.py"


def report(output, summary):
    atomic_json(output / "summary.json", summary)
    (output / "report.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Replicated verdict bisection</title><h1>Replicated verdict bisection</h1><p>Standard check, unchanged v2 rules. A localized upgrade regression is not a confirmed compiler defect. Binary search assumes monotonicity; missing nightlies remain explicit.</p><a href="summary.json">JSON evidence</a><pre>'
        + html.escape(json.dumps(summary, indent=2))
        + "</pre>",
        encoding="utf-8",
    )
    return summary


def run_verdict_bisect(
    bundle,
    output,
    baseline_python,
    start,
    end,
    *,
    settings,
    index_url=None,
    environment_cache=None,
    jobs="auto",
    install_timeout=1200,
    resume=False,
    deadline=None,
    progress=print,
):
    output, bundle = Path(output).resolve(), Path(bundle).resolve()
    manifest = verify_batches(bundle)
    if manifest["planned"] != 10 or len(manifest["batches"]) != 10:
        raise ValueError("Verdict v2 bisection requires ten complete frozen input batches")
    if (
        settings.backend != "inductor"
        or settings.repeats != 2
        or not settings.gradients
        or settings.threads != 1
    ):
        raise ValueError("Verdict bisection uses unchanged standard-level Inductor settings")
    index_url, local, kind = wheel_index(settings.device, index_url)
    cache = Path(environment_cache).resolve() if environment_cache else output / "environments"
    deadline = deadline if deadline is not None else time.monotonic() + 21600
    with campaign_lock(output):
        identity = environment_identity(
            baseline_python, output / f"baseline-probe-{time.time_ns()}"
        )
        specification = {
            "source_sha256": provenance()["faultline_source_sha256"],
            "bundle_sha256": file_hash(bundle / "batches.json"),
            "baseline_python": str(baseline_python),
            "baseline_identity": identity,
            "settings": settings.to_dict(),
            "start": start,
            "end": end,
            "index_url": index_url,
            "cache": str(cache),
            "jobs": jobs,
            "predicate": "verdict-v2",
            "replication": "Existing batch rule and stable repeated primary failures, unchanged",
        }
        path = output / "bisect.json"
        if path.exists():
            configuration = read_json(path)
            if not resume or configuration["specification"] != specification:
                raise ValueError("Bisection source, inputs, environments or settings changed")
        else:
            configuration = {
                "specification": specification,
                "catalog": nightly_catalog(
                    start, end, index_url=index_url + "/torch/", local_version=local
                ),
            }
            atomic_json(path, configuration)
        summary = {
            "schema_version": 1,
            "status": "running",
            "exit_code": 2,
            "trials": {},
            "boundary": {},
            "catalog": configuration["catalog"],
            "specification": specification,
        }

        def evaluate(wheel, phase="search"):
            directory = output / phase / wheel["date"]
            if (directory / "commit.json").exists():
                return verify_record(output, directory)
            if time.monotonic() >= deadline:
                return None
            if provenance()["faultline_source_sha256"] != specification["source_sha256"]:
                raise ValueError("Source changed during bisection")
            environment = ensure_wheel(
                cache,
                wheel,
                index_url,
                kind,
                min(install_timeout, max(1, deadline - time.monotonic())),
                model_requirements(identity),
            )
            destination = directory / "check"
            checked = execute_check(
                bundle,
                destination,
                str(baseline_python),
                environment["python"],
                settings,
                jobs,
                deadline,
            )
            if checked is None or not checked["complete"]:
                return None
            classification = {"SAFE": "not_shifted", "REGRESSION": "shifted"}.get(
                checked["verdict"], "inconclusive"
            )
            result = {
                "date": wheel["date"],
                "classification": classification,
                "torch_git_version": environment["identity"]["torch_git_version"],
                **checked,
            }
            commit_record(
                output,
                directory,
                result,
                [destination / p for p in ("check.json", "verdict.json", "evidence.json")],
            )
            if progress:
                progress(f"{phase} {wheel['date']}: {result['verdict']}", flush=True)
            return result

        wheels = configuration["catalog"]["wheels"]
        if len(wheels) < 2 or wheels[0]["date"] != start or wheels[-1]["date"] != end:
            summary["status"] = "bracket_wheels_unavailable"
            return report(output, summary)
        lower, upper = 0, len(wheels) - 1
        for index in (lower, upper):
            result = evaluate(wheels[index])
            if result is None:
                summary["status"] = "budget_exhausted"
                return report(output, summary)
            summary["trials"][wheels[index]["date"]] = result
        if [summary["trials"][wheels[i]["date"]]["classification"] for i in (lower, upper)] != [
            "not_shifted",
            "shifted",
        ]:
            summary["status"] = "no_confirmed_bracket"
            return report(output, summary)
        while upper - lower > 1:
            index = choose_next(wheels, summary["trials"], lower, upper)
            if index is None:
                break
            result = evaluate(wheels[index])
            if result is None:
                summary["status"] = "budget_exhausted"
                return report(output, summary)
            summary["trials"][wheels[index]["date"]] = result
            if result["classification"] == "shifted":
                upper = index
            elif result["classification"] == "not_shifted":
                lower = index
            report(output, summary)
        confirmations = [evaluate(wheels[i], "confirmation") for i in (lower, upper)]
        if any(r is None for r in confirmations):
            summary["status"] = "budget_exhausted"
            return report(output, summary)
        confirmed = [r["classification"] for r in confirmations] == ["not_shifted", "shifted"]
        a, b = [r["torch_git_version"] for r in confirmations]
        summary["confirmations"] = confirmations
        summary["boundary"] = {
            "last_SAFE": wheels[lower]["date"],
            "first_REGRESSION": wheels[upper]["date"],
            "adjacent_available_builds": upper - lower == 1,
            "independently_confirmed": confirmed,
            "last_good_commit": a,
            "first_bad_commit": b,
            "upstream_compare_url": f"https://github.com/pytorch/pytorch/compare/{a}...{b}"
            if a and b
            else None,
        }
        summary["status"] = (
            "localized_upgrade_regression"
            if confirmed and upper - lower == 1
            else "inconclusive_boundary"
        )
        summary["exit_code"] = 0 if summary["status"] == "localized_upgrade_regression" else 2
        if confirmed:
            summary["reproducer"] = str(
                export_collection(bundle, output / "reproducer", settings).relative_to(output)
            )
        return report(output, summary)
