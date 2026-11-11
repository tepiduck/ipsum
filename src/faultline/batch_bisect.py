"""Nightly bisection using the frozen batch-level advisory predicate."""

from pathlib import Path

from .batches import compare_batches, verify_batches
from .bisect import choose_next, config_digest, report
from .bisect_env import dependency_lock, ensure_environment, nightly_catalog
from .compare_envs import interpreter
from .native import file_hash
from .replication import POLICY
from .storage import atomic_json, campaign_lock, provenance, read_json
from .sweep import runtime_environment


def predicate(evidence):
    if evidence["accuracy_shift_replicated"]:
        return "shifted"
    if (
        evidence["missing"]
        or evidence["completed"] < POLICY["minimum_completed_batches"]
        or evidence.get("median_status") == "missing"
    ):
        return "inconclusive"
    return "not_shifted"


def run_replicated_bisect(
    bundle,
    output,
    baseline_python,
    start,
    end,
    *,
    settings,
    requirements=None,
    install_timeout=900,
    resume=False,
    retry_installations=False,
    progress=print,
):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    verify_batches(bundle)
    baseline_python = interpreter(baseline_python)
    lock = dependency_lock(Path(requirements).read_text(encoding="utf-8") if requirements else None)
    specification = {
        "predicate": "accuracy_shift_replicated",
        "policy": POLICY,
        "source_sha256": provenance()["faultline_source_sha256"],
        "batch_manifest_sha256": file_hash(bundle / "batches.json"),
        "baseline_python": baseline_python,
        "baseline_environment": runtime_environment(baseline_python),
        "settings": settings.to_dict(),
        "start": start,
        "end": end,
        "requirements": lock,
        "install_timeout": install_timeout,
    }
    with campaign_lock(output):
        path = output / "bisect.json"
        if path.exists():
            config = read_json(path)
            if (
                not resume
                or config.get("sha256") != config_digest(config)
                or config["specification"] != specification
            ):
                raise ValueError(
                    "Bisection exists or source, batches, environment, settings changed"
                )
            if (output / "requirements.txt").read_text(encoding="utf-8") != lock:
                raise ValueError("Dependency lock changed")
        else:
            config = {"specification": specification, "catalog": nightly_catalog(start, end)}
            config["sha256"] = config_digest(config)
            atomic_json(path, config)
            (output / "requirements.txt").write_text(lock, encoding="utf-8")
        summary = {
            "schema_version": 1,
            "status": "running",
            "exit_code": 2,
            "trials": {},
            "policy": specification,
            "catalog": config["catalog"],
            "boundary": {},
            "note": "Advisory batch-level accuracy predicate, not a compiler defect. Binary search assumes monotonicity. Confirmation uses fresh workers.",
        }

        def evaluate(wheel, phase="search"):
            if provenance()["faultline_source_sha256"] != specification["source_sha256"]:
                raise ValueError("Source changed during bisection")
            root = output / phase / wheel["date"]
            saved = root / "result.json"
            if saved.exists():
                record = read_json(saved)
                expected = read_json(root / "commit.json")["result_sha256"]
                if file_hash(saved) != expected:
                    raise ValueError("Bisection evidence changed")
                if record["environment"]["status"] != "ready" and not retry_installations:
                    return record
            previous = read_json(saved) if saved.exists() else None
            environment = ensure_environment(
                output / "environments" / wheel["date"],
                wheel,
                output / "requirements.txt",
                install_timeout=install_timeout,
            )
            record = {
                "date": wheel["date"],
                "environment": environment,
                "classification": "inconclusive",
            }
            if environment["status"] == "ready":
                comparison = root / "comparison"
                result = compare_batches(
                    bundle,
                    comparison,
                    baseline_python,
                    environment["python"],
                    settings,
                    resume=(comparison / "batch-comparison.json").exists(),
                )
                evidence = result["accuracy_shift_replicated"]
                record.update(
                    accuracy_shift_replicated=evidence,
                    classification=predicate(evidence),
                    torch_git_version=environment["identity"]["torch_git_version"],
                )
                if previous and previous["environment"]["status"] == "ready":
                    if record != previous:
                        raise ValueError("Completed bisection evidence changed")
                    return previous
            root.mkdir(parents=True, exist_ok=True)
            # Installation failures are preserved before a retry changes the current pointer.
            archive = root / f"result-{len(list(root.glob('result-*.json'))):04d}.json"
            atomic_json(archive, record)
            atomic_json(saved, record)
            atomic_json(root / "commit.json", {"result_sha256": file_hash(saved)})
            if progress:
                progress(f"{phase} {wheel['date']}: {record['classification']}", flush=True)
            return record

        wheels = config["catalog"]["wheels"]
        if len(wheels) < 2:
            summary["status"] = "insufficient_available_nightlies"
            report(output, summary)
            return summary
        lower, upper = 0, len(wheels) - 1
        for index in (lower, upper):
            summary["trials"][wheels[index]["date"]] = evaluate(wheels[index])
            report(output, summary)
        if [summary["trials"][wheels[i]["date"]]["classification"] for i in (lower, upper)] != [
            "not_shifted",
            "shifted",
        ]:
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
        confirmed = [r["classification"] for r in confirmations] == ["not_shifted", "shifted"]
        summary["confirmations"] = confirmations
        summary["boundary"] = {
            "last_observed_not_shifted": wheels[lower]["date"],
            "first_observed_shifted": wheels[upper]["date"],
            "adjacent_available_builds": upper - lower == 1,
            "independently_confirmed": confirmed,
            "last_good_commit": confirmations[0].get("torch_git_version"),
            "first_bad_commit": confirmations[1].get("torch_git_version"),
        }
        summary["status"] = (
            "localized_accuracy_shift"
            if confirmed and upper - lower == 1
            else "inconclusive_boundary"
        )
        summary["exit_code"] = 0 if summary["status"] == "localized_accuracy_shift" else 2
        report(output, summary)
        return summary
