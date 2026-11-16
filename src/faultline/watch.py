"""One resumable nightly-watch invocation, scheduled externally by CI/cron/systemd."""

import math
import subprocess
import sys
import time
from pathlib import Path

from packaging.version import Version

from .batches import verify_batches
from .bisect_env import environment_identity, supervise
from .bootstrap import python_command
from .check_cli import jobs_argument
from .check_resources import memory_budget
from .compare_envs import interpreter
from .environments import create_environment
from .evaluate import Settings
from .native import file_hash
from .storage import atomic_json, campaign_lock, provenance, read_json
from .watch_support import (
    HistoryIncompatible,
    catalog,
    commit_record,
    discover,
    ensure_wheel,
    execute_check,
    frozen_plan,
    model_requirements,
    render_history,
    validate_catalog,
    verify_record,
    wheel_index,
)


def prepared_model(root, model, baseline, settings, deadline):
    if "bundle" in model:
        bundle = Path(model["bundle"]).resolve()
    else:
        parent = root / "prepared" / model["id"]
        completed = sorted(parent.glob("attempt-*/prepared.json"))
        if completed:
            bundle = Path(read_json(completed[-1])["bundle"])
        else:
            target = parent / f"attempt-{len(list(parent.glob('attempt-*'))) + 1:04d}"
            specification = {
                **model,
                "batches": 10,
                "timeout": settings.timeout_s,
                "max_rss_mb": settings.max_rss_mb,
            }
            parent.mkdir(parents=True, exist_ok=True)
            atomic_json(parent / "specification.json", specification)
            observed = supervise(
                python_command(
                    baseline,
                    "from faultline.check_prepare import main;main()",
                    [
                        "--specification",
                        str(parent / "specification.json"),
                        "--output",
                        str(target),
                    ],
                    watch_parent=True,
                ),
                parent / f"supervision-{time.time_ns()}",
                timeout=min(max(settings.timeout_s, 900), max(1, deadline - time.monotonic())),
                max_rss_mb=settings.max_rss_mb,
            )
            if observed["status"] != "complete":
                detail = ""
                logs = sorted(parent.glob("supervision-*/stderr.log"))
                if logs:
                    with logs[-1].open("rb") as stream:
                        stream.seek(max(0, logs[-1].stat().st_size - 2048))
                        lines = stream.read().decode("utf-8", errors="replace").strip().splitlines()
                    if lines:
                        detail = f": {lines[-1][:500]}"
                raise ValueError(
                    f"Preparation {observed['status']}{detail}; logs retained in {parent}"
                )
            bundle = Path(read_json(target / "prepared.json")["bundle"])
    manifest = verify_batches(bundle)
    if manifest["planned"] != 10 or len(manifest["batches"]) != 10:
        raise ValueError(
            "Standard nightly watch requires exactly ten complete frozen input batches"
        )
    marker = root / "prepared" / model["id"] / "identity.json"
    identity = {"bundle": str(bundle), "manifest_sha256": file_hash(bundle / "batches.json")}
    if marker.exists() and read_json(marker) != identity:
        raise ValueError("Watch frozen input collection changed")
    atomic_json(marker, identity)
    return bundle


def run_watch(
    output,
    *,
    models=None,
    baseline_python=None,
    stable="2.15.0",
    device="cpu",
    index_url=None,
    environment_cache=None,
    seed_nightlies=1,
    through=None,
    jobs="auto",
    max_rss_mb=None,
    timeout=600,
    install_timeout=1200,
    max_wall_s=21600,
    deterministic=False,
    progress=print,
):
    if not 1 <= seed_nightlies <= 30:
        raise ValueError("Choose 1..30 nightlies per invocation")
    for value in (timeout, install_timeout, max_wall_s):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Watch time limits must be finite and positive")
    pinned = Version(stable)
    if pinned.is_prerelease or pinned.is_devrelease:
        raise ValueError("Stable baseline must be an exact stable version")
    root = Path(output).resolve()
    deadline = time.monotonic() + max_wall_s
    index_url, local, kind = wheel_index(device, index_url)
    models = catalog() if models is None else validate_catalog(models)
    cache = Path(environment_cache).resolve() if environment_cache else root / "environments"
    with campaign_lock(root):
        budget = memory_budget(root, max_rss_mb, resume=(root / "memory-budget.json").exists())
        settings = Settings(
            device=device,
            gradients=True,
            repeats=2,
            threads=1,
            timeout_s=timeout,
            max_rss_mb=budget["chosen_mb"],
            deterministic=deterministic,
        )
        settings.validate()
        if progress:
            progress(
                f"Watch memory budget: {budget['chosen_mb']} MiB; standard level, verdict v2.",
                flush=True,
            )
        if baseline_python is None:
            stable_root = root / "stable-environment"
            stable_index = index_url.replace("/nightly/", "/")
            baseline_python = create_environment(
                stable_root,
                stable,
                device=kind,
                index_url=stable_index,
                resume=(stable_root / "ready.json").exists(),
                timeout=install_timeout,
            )["python"]
        baseline_python = interpreter(baseline_python)
        identity = environment_identity(
            baseline_python, root / "probes" / f"stable-{time.time_ns()}"
        )
        if (
            Version(identity["torch"]).base_version != pinned.base_version
            or Version(identity["torch"]).is_prerelease
        ):
            raise ValueError("Baseline interpreter does not match --stable")
        configuration = {
            "schema_version": 1,
            "catalog": models,
            "baseline_python": baseline_python,
            "baseline_identity": identity,
            "stable": stable,
            "device": device,
            "index_url": index_url,
            "local_version": local,
            "environment_cache": str(cache),
            "settings": settings.to_dict(),
            "jobs": jobs,
            "source_sha256": provenance()["faultline_source_sha256"],
            "verdict_policy": "v2",
        }
        config_path = root / "watch.json"
        if config_path.exists():
            if read_json(config_path) != configuration:
                raise HistoryIncompatible(
                    "Watch source, catalog, baseline, device, settings or dependency identity changed; use a new history directory"
                )
        else:
            atomic_json(config_path, configuration)
        records = [
            verify_record(root, p.parent)
            for p in sorted((root / "records").glob("*/*/*/commit.json"))
        ]
        plan = frozen_plan(
            root,
            models,
            records,
            seed_nightlies,
            through,
            lambda: discover(through, index_url, local),
        )
        wheels = plan["inventory"]["wheels"]
        availability = plan.get("availability", {})
        if progress and availability.get("fallback"):
            progress(availability["reason"], flush=True)
        chosen = [w for w in wheels if w["date"] in plan["selected_dates"]]
        planned = [
            {"date": w["date"], "model": m["id"], "reference": rail}
            for w in chosen
            for m in models
            for rail in ("stable", "previous")
        ]
        render_history(root, configuration, records, "running", planned)
        prepared = {}
        environments = {}

        def environment(wheel):
            day = wheel["date"]
            if day not in environments:
                environments[day] = ensure_wheel(
                    cache,
                    wheel,
                    index_url,
                    kind,
                    min(install_timeout, max(1, deadline - time.monotonic())),
                    model_requirements(identity),
                )
            return environments[day]

        for wheel in chosen:
            previous = wheels[wheels.index(wheel) - 1]
            for model in models:
                name = model["id"]
                if name not in prepared and any(r["model"] == name for r in records):
                    prepared[name] = prepared_model(
                        root, model, baseline_python, settings, deadline
                    )
                for rail in ("stable", "previous"):
                    marker = root / "records" / wheel["date"] / name / rail
                    if (marker / "commit.json").exists():
                        continue
                    if time.monotonic() >= deadline:
                        return render_history(
                            root,
                            configuration,
                            records,
                            "budget_exhausted_resume_required",
                            planned,
                        )
                    if provenance()["faultline_source_sha256"] != configuration["source_sha256"]:
                        raise ValueError("Source changed during watch")
                    try:
                        if name not in prepared:
                            prepared[name] = prepared_model(
                                root, model, baseline_python, settings, deadline
                            )
                        candidate = environment(wheel)
                        reference = (
                            baseline_python if rail == "stable" else environment(previous)["python"]
                        )
                    except (ValueError, OSError, RuntimeError) as exc:
                        atomic_json(
                            root / "interruptions" / f"{time.time_ns()}.json",
                            {
                                "date": wheel["date"],
                                "model": name,
                                "reference": rail,
                                "error": str(exc),
                            },
                        )
                        render_history(
                            root,
                            configuration,
                            records,
                            "setup_inconclusive_resume_required",
                            planned,
                        )
                        raise
                    destination = root / "checks" / wheel["date"] / name / rail
                    if progress:
                        progress(f"{wheel['date']} {name} vs {rail}: checking", flush=True)
                    result = execute_check(
                        prepared[name],
                        destination,
                        reference,
                        candidate["python"],
                        settings,
                        jobs,
                        deadline,
                    )
                    if result is None or not result["complete"]:
                        if result:
                            atomic_json(root / "interruptions" / f"{time.time_ns()}.json", result)
                        return render_history(
                            root,
                            configuration,
                            records,
                            "check_interrupted_resume_required",
                            planned,
                        )
                    record = {
                        "date": wheel["date"],
                        "version": wheel["version"],
                        "model": name,
                        "reference": rail,
                        "reference_date": previous["date"] if rail == "previous" else None,
                        "candidate_commit": candidate["identity"]["torch_git_version"],
                        "bundle_sha256": file_hash(prepared[name] / "batches.json"),
                        "report": (destination / "report.html").relative_to(root).as_posix(),
                        **result,
                    }
                    if result["verdict"] == "REGRESSION":
                        from .verdict_bisect import export_collection

                        record["replicated_reproducer"] = str(
                            export_collection(
                                prepared[name], destination / "replicated-reproducer", settings
                            ).relative_to(root)
                        )
                        render_history(
                            root,
                            configuration,
                            records + [record],
                            "investigating_regression",
                            planned,
                        )
                        safe = [
                            r
                            for r in records
                            if r["model"] == name
                            and r["reference"] == rail
                            and r["verdict"] == "SAFE"
                            and r["date"] < wheel["date"]
                        ]
                        if safe:
                            start = max(r["date"] for r in safe)
                            from .verdict_bisect import run_verdict_bisect

                            bisect_root = (
                                root / "bisections" / name / rail / f"{start}-{wheel['date']}"
                            )
                            fixed_baseline = (
                                baseline_python
                                if rail == "stable"
                                else environment(next(w for w in wheels if w["date"] == start))[
                                    "python"
                                ]
                            )
                            bisection = run_verdict_bisect(
                                prepared[name],
                                bisect_root,
                                fixed_baseline,
                                start,
                                wheel["date"],
                                settings=settings,
                                index_url=index_url,
                                environment_cache=cache,
                                jobs=jobs,
                                install_timeout=install_timeout,
                                resume=(bisect_root / "bisect.json").exists(),
                                deadline=deadline,
                            )
                            if bisection["status"] == "budget_exhausted":
                                return render_history(
                                    root,
                                    configuration,
                                    records,
                                    "bisection_interrupted_resume_required",
                                    planned,
                                )
                            record["bisection"] = (
                                (bisect_root / "report.html").relative_to(root).as_posix()
                            )
                            record["bisection_status"] = bisection["status"]
                        else:
                            record["bisection_status"] = "no_observed_SAFE_nightly_bracket"
                    files = [
                        destination / p
                        for p in ("check.json", "evidence.json", "verdict.json", "report.html")
                    ]
                    commit_record(root, marker, record, files)
                    records.append(record)
                    render_history(root, configuration, records, "running", planned)
                    if progress:
                        progress(
                            f"{wheel['date']} {name} vs {rail}: {result['verdict']}", flush=True
                        )
        return render_history(root, configuration, records, "complete", planned)


def add_watch_command(commands):
    parser = commands.add_parser(
        "watch", help="run/resume a standard nightly watch; schedule externally"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--catalog",
        type=Path,
        help="JSON model list; defaults to five pretrained vision/text models",
    )
    parser.add_argument("--stable", default="2.15.0")
    parser.add_argument("--baseline-python")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--index-url", help="explicit nightly CPU/CUDA/ROCm wheel index")
    parser.add_argument("--environment-cache", type=Path)
    parser.add_argument("--seed-nightlies", type=int, default=1)
    parser.add_argument("--through", help="latest allowed nightly date, YYYY-MM-DD")
    parser.add_argument("--jobs", type=jobs_argument, default="auto")
    parser.add_argument("--max-rss-mb", type=int)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--install-timeout", type=float, default=1200)
    parser.add_argument("--max-wall-s", type=float, default=21600)
    parser.add_argument("--deterministic", action="store_true")


def watch_command(args):
    try:
        return _watch_command(args)
    except (
        ValueError,
        OSError,
        RuntimeError,
        KeyError,
        ImportError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"INCONCLUSIVE - {exc}\nWatch artifacts retained at {args.output}", file=sys.stderr)
        return 3


def _watch_command(args):
    result = run_watch(
        args.output,
        models=catalog(args.catalog),
        baseline_python=args.baseline_python,
        stable=args.stable,
        device=args.device,
        index_url=args.index_url,
        environment_cache=args.environment_cache,
        seed_nightlies=args.seed_nightlies,
        through=args.through,
        jobs=args.jobs,
        max_rss_mb=args.max_rss_mb,
        timeout=args.timeout,
        install_timeout=args.install_timeout,
        max_wall_s=args.max_wall_s,
        deterministic=args.deterministic,
    )
    print(
        f"Watch {result['state']}; {len(result['records'])} recorded checks. {args.output / 'index.html'}"
    )
    selected = [
        r
        for r in result["records"]
        if any(all(r[k] == p[k] for k in ("date", "model", "reference")) for p in result["planned"])
    ]
    if any(r["verdict"] == "REGRESSION" for r in selected):
        return 1
    if result["state"] != "complete" or any(r["verdict"] == "INCONCLUSIVE" for r in selected):
        return 3
    return 2 if any(r["verdict"] == "NEEDS_REVIEW" for r in selected) else 0
