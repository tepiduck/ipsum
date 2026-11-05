"""Restartable campaigns. Completed per-case outcomes are the source of truth."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from .environment import describe
from .evaluate import Settings
from .ir import Case
from .lowering import freeze
from .runner import run_case
from .storage import atomic_json, campaign_lock, now, provenance, read_json


def config_hash(settings):
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def validate_manifest(manifest):
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported campaign schema")
    if config_hash(manifest["settings"]) != manifest["settings_hash"]:
        raise ValueError("Campaign settings changed after creation")
    identifiers = [entry["id"] for entry in manifest["cases"]]
    if any(
        not isinstance(key, str) or not re.fullmatch(r"[0-9]{4,6}-[a-f0-9]{12}", key)
        for key in identifiers
    ) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Invalid or duplicate campaign case identifiers")


def summarize(directory, manifest):
    outcomes = []
    groups = {}
    for entry in manifest["cases"]:
        result_path = directory / "cases" / entry["id"] / "outcome.json"
        if not result_path.exists():
            continue
        result = read_json(result_path)
        if (
            result.get("case_hash") != entry["hash"]
            or result.get("settings_hash") != manifest["settings_hash"]
        ):
            raise ValueError(f"Stale or modified result: {result_path}")
        outcomes.append({"id": entry["id"], "tag": entry["tag"], **result})
        if result["status"] in {
            "numerical_mismatch",
            "structural_mismatch",
            "compile_error",
            "runtime_error",
        }:
            groups.setdefault(result["fingerprint"], []).append(entry["id"])
    summary = {
        "schema_version": 1,
        "updated_utc": now(),
        "planned": len(manifest["cases"]),
        "completed": len(outcomes),
        "counts": dict(Counter(x["status"] for x in outcomes)),
        "failure_families": groups,
        "outcomes": outcomes,
        "injected_fault": manifest["settings"]["backend"].startswith("mutant:"),
        "note": "Families are triage groups, not counts of unique confirmed compiler bugs.",
    }
    atomic_json(directory / "summary.json", summary)
    return summary


def run_campaign(
    directory, cases=None, settings=None, resume=False, stop_after=None, progress=print
):
    directory = Path(directory).resolve()
    with campaign_lock(directory):
        meta = directory / "manifest.json"
        environment = describe()
        if resume:
            manifest = read_json(meta)
            validate_manifest(manifest)
            previous = manifest["environment"]
            for key in (
                "faultline_source_sha256",
                "packages",
                "hip_version",
                "cuda_version",
                "devices",
                "python",
                "platform",
                "torch_build",
            ):
                if previous[key] != environment[key]:
                    raise ValueError(f"Cannot mix evidence after environment/source change: {key}")
            settings = Settings(**manifest["settings"])
        else:
            if meta.exists():
                raise ValueError("Campaign already exists; use resume or a new directory")
            if not cases or settings is None:
                raise ValueError("A new campaign needs cases and execution settings")
            settings.validate()
            entries = []
            for index, case in enumerate(cases):
                case = freeze(case)
                key = f"{index:04d}-{case.digest()[:12]}"
                atomic_json(directory / "cases" / key / "case.json", case.to_dict())
                entries.append({"id": key, "hash": case.digest(), "tag": case.tag})
            manifest = {
                "schema_version": 1,
                "created_utc": now(),
                "state": "pending",
                "settings": settings.to_dict(),
                "settings_hash": config_hash(settings.to_dict()),
                "environment": environment,
                "cases": entries,
            }
            atomic_json(meta, manifest)
        manifest["state"] = "running"
        atomic_json(meta, manifest)
        executed = 0
        try:
            for entry in manifest["cases"]:
                if (
                    provenance()["faultline_source_sha256"]
                    != environment["faultline_source_sha256"]
                ):
                    raise ValueError("Source changed during campaign; start a new run")
                path = directory / "cases" / entry["id"]
                case = Case.from_dict(read_json(path / "case.json"))
                if case.digest() != entry["hash"]:
                    raise ValueError("Input case changed after campaign creation")
                if (path / "outcome.json").exists():
                    result = read_json(path / "outcome.json")
                    if (
                        result.get("case_hash") == entry["hash"]
                        and result.get("settings_hash") == manifest["settings_hash"]
                    ):
                        continue
                    # A worker result not yet committed by the campaign is rerun.
                    if "settings_hash" in result:
                        raise ValueError("Completed evidence does not match the manifest")
                if stop_after is not None and executed >= stop_after:
                    break
                result = run_case(case, settings, path)
                result["settings_hash"] = manifest["settings_hash"]
                atomic_json(path / "outcome.json", result)
                executed += 1
                summarize(directory, manifest)
                if progress:
                    progress(f"{entry['id']} {entry['tag']}: {result['status']}")
        except BaseException:
            manifest["state"] = "interrupted"
            atomic_json(meta, manifest)
            raise
        summary = summarize(directory, manifest)
        manifest["state"] = (
            "completed" if summary["completed"] == summary["planned"] else "interrupted"
        )
        atomic_json(meta, manifest)
        return summary
