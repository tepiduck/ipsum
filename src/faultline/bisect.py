"""Advisory bisection of compiled CPU accuracy across isolated nightly builds."""

import hashlib
import html
import json
import math
import statistics
import sys
from pathlib import Path

from .bisect_env import dependency_lock, ensure_environment, environment_identity, nightly_catalog
from .compare_envs import interpreter
from .evaluate import Settings
from .native import NativeJob, file_hash, verify_native
from .native_artifacts import copy_workload
from .runner import run_case
from .storage import atomic_json, campaign_lock, now, provenance, read_json


def compiled_errors(result):
    if result.get("status") not in {"pass", "gradient_mismatch", "numerical_mismatch"}:
        return None
    evidence = result.get("fp64_advisory", {})
    if evidence.get("status") != "complete":
        return None
    return {
        row["target"]: row.get("candidate", {}).get("rmse")
        for row in evidence.get("outputs", [])
        if ":gradient:" in row["target"] and row.get("reference_dtype") == "torch.float64"
    }


def baseline_errors(results):
    observations = [compiled_errors(result) for result in results]
    if not observations or any(row is None for row in observations):
        return None
    targets = set(observations[0])
    if any(set(row) != targets for row in observations):
        return None
    return {
        target: statistics.median(row[target] for row in observations)
        for target in sorted(targets)
        if all(
            isinstance(row[target], (int, float)) and math.isfinite(row[target]) and row[target] > 0
            for row in observations
        )
    }


def growth_score(reference, result):
    errors = compiled_errors(result)
    rows = []
    for target, original in sorted(reference.items()):
        current = errors.get(target) if errors is not None else None
        value = (
            math.log2(current) - math.log2(original)
            if isinstance(current, (int, float)) and math.isfinite(current) and current > 0
            else None
        )
        rows.append(
            {
                "target": target,
                "baseline_compiled_rmse": original,
                "candidate_compiled_rmse": current
                if current is not None and math.isfinite(current)
                else None,
                "log2_growth": value,
            }
        )
    complete = bool(rows) and all(row["log2_growth"] is not None for row in rows)
    return {
        "status": "complete" if complete else "incomplete",
        "targets": rows,
        "matched": sum(row["log2_growth"] is not None for row in rows),
        "total": len(rows),
        "median_log2_growth": statistics.median(row["log2_growth"] for row in rows)
        if complete
        else None,
    }


def advisory_classification(scores, growth):
    values = [score.get("median_log2_growth") for score in scores]
    if not values or any(value is None for value in values):
        return "inconclusive"
    threshold = math.log2(growth)
    return (
        "shifted"
        if all(value >= threshold for value in values)
        else "not_shifted"
        if all(value < threshold for value in values)
        else "inconclusive"
    )


def choose_next(wheels, results, lower, upper):
    choices = [i for i in range(lower + 1, upper) if wheels[i]["date"] not in results]
    return min(choices, key=lambda i: (abs(i - (lower + upper) / 2), i)) if choices else None


def verify_commit(directory):
    marker = read_json(directory / "commit.json")
    for name, expected in marker["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or file_hash(path) != expected:
            raise ValueError("Completed bisection evidence changed")
    return read_json(directory / "result.json")


def commit_result(directory, result, files):
    atomic_json(directory / "result.json", result)
    atomic_json(
        directory / "commit.json",
        {"files": {name: file_hash(directory / name) for name in ["result.json", *files]}},
    )


def report(output, summary):
    atomic_json(output / "summary.json", summary)
    rows = []
    for day, result in sorted(summary.get("trials", {}).items()):
        scores = [row.get("median_log2_growth") for row in result.get("scores", [])]
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(value))}</td>"
                for value in [
                    day,
                    result["classification"],
                    scores,
                    result.get("torch_git_version"),
                ]
            )
            + "</tr>"
        )
    (output / "report.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Faultline accuracy bisection</title>'
        "<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px}"
        "td,th{padding:.5rem;text-align:left;overflow-wrap:anywhere}table{width:100%;table-layout:fixed}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>"
        "<h1>CPU accuracy bisection</h1><p>Advisory accuracy shifts, not compiler defect confirmations. "
        'Existing primary verdicts and exit codes are unchanged.</p><a href="summary.json">JSON evidence</a>'
        "<table><tr><th>Date</th><th>Advisory</th><th>Median log2 growth, fresh workers</th><th>PyTorch commit</th></tr>"
        + "".join(rows)
        + "</table><pre>"
        + html.escape(str(summary.get("boundary", {})))
        + "</pre>",
        encoding="utf-8",
    )


def config_digest(config):
    payload = {k: v for k, v in config.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def run_bisect(
    bundle,
    output,
    baseline_python,
    start,
    end,
    *,
    requirements=None,
    growth=1.4,
    observations=3,
    settings=None,
    install_timeout=900,
    resume=False,
    retry_installations=False,
    progress=print,
):
    if not math.isfinite(growth) or growth <= 1 or not 2 <= observations <= 10:
        raise ValueError("Choose a growth threshold above one and 2..10 fresh worker observations")
    if not math.isfinite(install_timeout) or install_timeout <= 0:
        raise ValueError("Installation timeout must be finite and positive")
    settings = settings or Settings(
        gradients=True, fp64_advisory=True, repeats=2, threads=1, timeout_s=600, max_rss_mb=12288
    )
    settings.validate()
    if (
        settings.device != "cpu"
        or settings.backend != "inductor"
        or not settings.gradients
        or not settings.fp64_advisory
        or settings.numerics
    ):
        raise ValueError(
            "Accuracy bisection requires CPU Inductor gradients and fp64 advisory only"
        )
    output, bundle = Path(output).resolve(), Path(bundle).resolve()
    baseline_python = interpreter(baseline_python)
    verify_native(bundle)
    lock = dependency_lock(Path(requirements).read_text(encoding="utf-8") if requirements else None)
    source = provenance()["faultline_source_sha256"]
    specification = {
        "schema_version": 1,
        "source_sha256": source,
        "workload_sha256": file_hash(bundle / "workload.json"),
        "baseline_python": baseline_python,
        "start": start,
        "end": end,
        "requirements": lock,
        "growth_threshold": growth,
        "observations": observations,
        "settings": settings.to_dict(),
        "install_timeout": install_timeout,
    }
    with campaign_lock(output):
        config_path = output / "bisect.json"
        if resume and config_path.exists():
            config = read_json(config_path)
            if config.get("sha256") != config_digest(config):
                raise ValueError("Bisection configuration or wheel catalog changed")
            if config["specification"] != specification:
                raise ValueError("Bisection source, workload, settings, or interpreter changed")
            probe = output / f"baseline-probe-{len(list(output.glob('baseline-probe-*'))):04d}"
            if environment_identity(baseline_python, probe) != config["baseline_environment"]:
                raise ValueError("Baseline environment changed")
            if (output / "requirements.txt").read_text(encoding="utf-8") != lock:
                raise ValueError("Dependency lock changed")
        else:
            if config_path.exists() or (output / "workload").exists():
                raise ValueError("Bisection exists; use --resume or a new output directory")
            catalog = nightly_catalog(start, end)
            (output / "requirements.txt").write_text(lock, encoding="utf-8")
            config = {
                "specification": specification,
                "catalog": catalog,
                "baseline_environment": environment_identity(
                    baseline_python,
                    output / f"baseline-probe-{len(list(output.glob('baseline-probe-*'))):04d}",
                ),
            }
            config["sha256"] = config_digest(config)
            atomic_json(config_path, config)
        if not (output / "workload").exists():
            staging = (
                output
                / "initialization"
                / f"attempt-{len(list((output / 'initialization').glob('attempt-*'))) + 1:04d}"
            )
            staging.mkdir(parents=True, exist_ok=False)
            copy_workload(bundle, staging / "workload")
            (staging / "workload").rename(output / "workload")
        verify_native(output / "workload")
        if file_hash(output / "workload/workload.json") != specification["workload_sha256"]:
            raise ValueError("Frozen bisection workload changed")
        summary = {
            "schema_version": 1,
            "status": "running",
            "exit_code": 2,
            "trials": {},
            "policy": specification,
            "catalog": config["catalog"],
            "boundary": {},
            "note": "Binary search assumes a monotone shift across available nightlies. Dates with missing/failed evidence are not good builds; the range may remain unresolved. A source interval is not causal attribution.",
        }
        report(output, summary)
        job = NativeJob.create(output / "workload")

        def guard():
            if provenance()["faultline_source_sha256"] != source:
                raise ValueError(
                    "Source changed during bisection; completed artifacts are preserved"
                )

        baseline_root = output / "baseline"
        committed = sorted(baseline_root.glob("attempt-*/commit.json"))
        if committed:
            reference_record = verify_commit(committed[-1].parent)
        else:
            directory = (
                baseline_root / f"attempt-{len(list(baseline_root.glob('attempt-*'))) + 1:04d}"
            )
            directory.mkdir(parents=True, exist_ok=False)
            results = []
            for repeat in range(observations):
                guard()
                results.append(
                    run_case(
                        job, settings, directory / f"observation-{repeat}", python=baseline_python
                    )
                )
            reference_record = {
                "errors": baseline_errors(results),
                "primary_statuses": [r["status"] for r in results],
            }
            commit_result(
                directory,
                reference_record,
                [f"observation-{i}/outcome.json" for i in range(observations)],
            )
        reference = reference_record["errors"]
        if not reference:
            summary["status"] = "inconclusive_baseline"
            report(output, summary)
            return summary

        def evaluate(wheel, phase="search"):
            guard()
            root = output / phase / wheel["date"]
            committed = sorted(root.glob("attempt-*/commit.json"))
            if committed:
                result = verify_commit(committed[-1].parent)
                if result.get("environment", {}).get("status") == "ready":
                    ensure_environment(
                        output / "environments" / wheel["date"],
                        wheel,
                        output / "requirements.txt",
                        install_timeout=install_timeout,
                    )
                    return result
                if not retry_installations:
                    return result
            directory = root / f"attempt-{len(list(root.glob('attempt-*'))) + 1:04d}"
            directory.mkdir(parents=True, exist_ok=False)
            environment = ensure_environment(
                output / "environments" / wheel["date"],
                wheel,
                output / "requirements.txt",
                install_timeout=install_timeout,
            )
            result = {
                "classification": "inconclusive",
                "date": wheel["date"],
                "environment": environment,
            }
            files = []
            if environment["status"] == "ready":
                scores, statuses = [], []
                for repeat in range(observations):
                    guard()
                    outcome = run_case(
                        job,
                        settings,
                        directory / f"observation-{repeat}",
                        python=environment["python"],
                    )
                    scores.append(growth_score(reference, outcome))
                    statuses.append(outcome["status"])
                    files.append(f"observation-{repeat}/outcome.json")
                result.update(
                    scores=scores,
                    primary_statuses=statuses,
                    torch_git_version=environment["identity"]["torch_git_version"],
                    classification=advisory_classification(scores, growth),
                )
            commit_result(directory, result, files)
            if progress:
                progress(f"{phase} {wheel['date']}: {result['classification']}", flush=True)
            return result

        wheels = config["catalog"]["wheels"]
        if len(wheels) < 2:
            summary["status"] = "insufficient_available_nightlies"
            report(output, summary)
            return summary
        lower, upper = 0, len(wheels) - 1
        for index in (lower, upper):
            result = evaluate(wheels[index])
            summary["trials"][wheels[index]["date"]] = result
            report(output, summary)
        if (
            summary["trials"][wheels[lower]["date"]]["classification"] != "not_shifted"
            or summary["trials"][wheels[upper]["date"]]["classification"] != "shifted"
        ):
            summary["status"] = "no_confirmed_bracket"
            report(output, summary)
            return summary
        while upper - lower > 1:
            index = choose_next(wheels, summary["trials"], lower, upper)
            if index is None:
                break
            result = evaluate(wheels[index])
            summary["trials"][wheels[index]["date"]] = result
            if result["classification"] == "shifted":
                upper = index
            elif result["classification"] == "not_shifted":
                lower = index
            report(output, summary)
        confirmations = [evaluate(wheels[i], "confirmation") for i in (lower, upper)]
        summary["confirmations"] = confirmations
        confirmed = [r["classification"] for r in confirmations] == ["not_shifted", "shifted"]
        a, b = (r.get("torch_git_version") for r in confirmations)
        summary["boundary"] = {
            "last_observed_not_shifted": wheels[lower]["date"],
            "first_observed_shifted": wheels[upper]["date"],
            "adjacent_available_builds": upper - lower == 1,
            "independently_confirmed": confirmed,
            "last_good_commit": a,
            "first_bad_commit": b,
            "upstream_compare_url": f"https://github.com/pytorch/pytorch/compare/{a}...{b}"
            if a and b
            else None,
        }
        summary["status"] = (
            "localized_accuracy_shift"
            if confirmed and upper - lower == 1
            else "inconclusive_boundary"
        )
        summary["exit_code"] = 0 if summary["status"] == "localized_accuracy_shift" else 2
        summary["finished_utc"] = now()
        report(output, summary)
        return summary


def add_bisect_command(commands):
    parser = commands.add_parser(
        "bisect", help="bisect a CPU accuracy shift across isolated nightly wheels"
    )
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--baseline-python", default=sys.executable)
    parser.add_argument("--start", required=True, help="first nightly date, YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="last nightly date, YYYY-MM-DD")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, help="exact pinned non-PyTorch dependencies")
    parser.add_argument("--growth", type=float, default=1.4)
    parser.add_argument("--observations", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--max-rss-mb", type=int, default=12288)
    parser.add_argument("--install-timeout", type=float, default=900)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-installations",
        action="store_true",
        help="on resume, retry failed installations in new attempt directories; retain measured trials",
    )


def bisect_command(args):
    if args.retry_installations and not args.resume:
        raise ValueError("--retry-installations requires --resume")
    result = run_bisect(
        args.bundle,
        args.output,
        args.baseline_python,
        args.start,
        args.end,
        requirements=args.requirements,
        growth=args.growth,
        observations=args.observations,
        settings=Settings(
            gradients=True,
            fp64_advisory=True,
            repeats=2,
            threads=1,
            timeout_s=args.timeout,
            max_rss_mb=args.max_rss_mb,
        ),
        install_timeout=args.install_timeout,
        resume=args.resume,
        retry_installations=args.retry_installations,
    )
    print(result["status"], result["boundary"])
    return result["exit_code"]
