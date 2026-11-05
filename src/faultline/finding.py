"""Load and replay saved evidence without retyping its execution policy."""

from dataclasses import dataclass
from pathlib import Path

from .campaign import config_hash, validate_manifest
from .environment import describe
from .evaluate import Settings
from .ir import Case
from .portable import REDUCIBLE, fingerprint, same_failure, tolerances
from .runner import run_case
from .storage import atomic_json, now, read_json
from .trace import Trace, program_from_dict

ENVIRONMENT_KEYS = (
    "faultline_source_sha256",
    "packages",
    "hip_version",
    "cuda_version",
    "devices",
    "python",
    "platform",
    "torch_build",
)


@dataclass(frozen=True)
class Finding:
    directory: Path
    case: Case | Trace
    settings: Settings
    outcome: dict
    environment: dict | None

    def source_record(self):
        return {
            "schema_version": 1,
            "directory": str(self.directory),
            "case_hash": self.case.digest(),
            "settings_hash": config_hash(self.settings.to_dict()),
            "outcome": self.outcome,
            "environment": self.environment,
        }


def seal_bundle(directory):
    """Record content hashes to detect accidental edits; not an authenticity signature."""
    directory = Path(directory)
    names = ("case.json", "settings.json", "outcome.json", "environment.json")
    atomic_json(
        directory / "bundle.json",
        {
            "schema_version": 1,
            "hashes": {
                name: config_hash(read_json(directory / name))
                for name in names
                if (directory / name).exists()
            },
        },
    )


def load_finding(directory):
    directory = Path(directory).resolve()
    try:
        return _load(directory)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"Malformed finding in {directory}: {exc}") from exc


def _load(directory):
    job_path = directory / "job.json"
    settings_path = directory / "settings.json"
    case_path = directory / "case.json"
    job = read_json(job_path) if job_path.exists() else None
    raw_case = read_json(case_path) if case_path.exists() else job["case"] if job else None
    raw_settings = (
        read_json(settings_path) if settings_path.exists() else (job["settings"] if job else None)
    )
    if raw_case is None or raw_settings is None:
        raise ValueError("Finding needs case/settings JSON or a worker job.json")
    case = program_from_dict(raw_case)
    settings = Settings(**raw_settings)
    settings.validate()
    if any(spec.values is None for spec in case.inputs):
        raise ValueError("Saved findings must contain frozen input values")
    if job and (
        program_from_dict(job["case"]).digest() != case.digest()
        or job["settings"] != settings.to_dict()
    ):
        raise ValueError("Saved case or settings disagree with the worker job")
    outcome = read_json(directory / "outcome.json")
    if outcome["schema_version"] != 1 or outcome["case_hash"] != case.digest():
        raise ValueError("Saved outcome does not match the case")
    expected = {
        "backend": settings.backend,
        "device": settings.device,
        "injected_fault": settings.backend.startswith("mutant:"),
    }
    if any(outcome[key] != value for key, value in expected.items()):
        raise ValueError("Saved outcome does not match execution settings")
    for key, value in {
        "gradients": settings.gradients,
        "dynamic": settings.dynamic,
        "trace": isinstance(case, Trace),
    }.items():
        if outcome.get(key, False) != value:
            raise ValueError("Saved outcome does not match gradient/trace execution settings")
    rtol, atol = tolerances(case.inputs[0].dtype, settings.rtol, settings.atol)
    if any(
        outcome.get(key, value) != value
        for key, value in {
            "rtol": rtol,
            "atol": atol,
            "repeats": settings.repeats,
            "settings_hash": config_hash(settings.to_dict()),
        }.items()
    ):
        raise ValueError("Saved outcome does not match execution settings")
    if outcome["status"] in REDUCIBLE and outcome["fingerprint"] != fingerprint(outcome):
        raise ValueError("Saved failure fingerprint does not match its comparison")

    environment_path = directory / "environment.json"
    environment = (
        read_json(environment_path)
        if environment_path.exists()
        else outcome.get("worker_environment")
    )
    manifest_path = directory.parent.parent / "manifest.json"
    if directory.parent.name == "cases" and manifest_path.exists():
        manifest = read_json(manifest_path)
        validate_manifest(manifest)
        entries = [e for e in manifest["cases"] if e["id"] == directory.name]
        if len(entries) != 1 or entries[0]["hash"] != case.digest():
            raise ValueError("Case is not the recorded campaign entry")
        if (
            manifest["settings"] != settings.to_dict()
            or outcome.get("settings_hash") != manifest["settings_hash"]
        ):
            raise ValueError("Finding is not committed with the campaign settings")
        environment = manifest["environment"]
    if (directory / "bundle.json").exists():
        bundle = read_json(directory / "bundle.json")
        allowed = {"case.json", "settings.json", "outcome.json", "environment.json"}
        if (
            bundle["schema_version"] != 1
            or set(bundle["hashes"]) != allowed
            or any(
                config_hash(read_json(directory / name)) != digest
                for name, digest in bundle["hashes"].items()
            )
        ):
            raise ValueError("Finding bundle integrity check failed")
    return Finding(directory, case, settings, outcome, environment)


def replay_finding(finding, output):
    if finding.outcome["status"] not in REDUCIBLE:
        raise ValueError("Replay requires a saved stable discrepancy or compiler exception")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = describe()
    atomic_json(output / "source.json", finding.source_record())
    atomic_json(output / "case.json", finding.case.to_dict())
    atomic_json(output / "settings.json", finding.settings.to_dict())
    atomic_json(output / "environment.json", environment)
    result = run_case(finding.case, finding.settings, output)
    reproduced = same_failure(finding.outcome, result)
    classification = (
        "reproduced"
        if reproduced
        else (
            "not_observed"
            if result["status"] == "pass"
            else "changed_failure"
            if result["status"] in REDUCIBLE
            else "inconclusive"
        )
    )
    previous = finding.environment
    replay = {
        "schema_version": 1,
        "created_utc": now(),
        "classification": classification,
        "reproduces_expected": reproduced,
        "expected_fingerprint": finding.outcome["fingerprint"],
        "observed_fingerprint": result["fingerprint"],
        "original_environment_available": previous is not None,
        "environment_changes": {
            key: {"before": previous.get(key), "after": environment.get(key)}
            for key in ENVIRONMENT_KEYS
            if previous is not None and previous.get(key) != environment.get(key)
        },
        "note": "A matching symptom is not proof of a common root cause; source evidence is unchanged.",
    }
    atomic_json(output / "replay.json", replay)
    seal_bundle(output)
    return replay
