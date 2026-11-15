"""Frozen watch catalogs, supervised checks and committed timeline records."""

import html
import re
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit

from .bisect_env import environment_identity, nightly_catalog, supervise
from .bootstrap import python_command
from .environments import create_environment
from .native import file_hash
from .storage import atomic_json, now, read_json

DEFAULT_CATALOG = [
    {"id": "resnet50", "torchvision": "resnet50"},
    {"id": "mobilenet_v3_small", "torchvision": "mobilenet_v3_small"},
    {"id": "bert", "hf": "google-bert/bert-base-uncased"},
    {"id": "gpt2", "hf": "openai-community/gpt2"},
    {"id": "vit_b_16", "torchvision": "vit_b_16"},
]


def catalog(path=None):
    rows = read_json(path) if path else DEFAULT_CATALOG
    return validate_catalog(rows)


def validate_catalog(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Watch catalog must be a nonempty JSON list")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or any(
            not isinstance(value, str) or not value for value in row.values()
        ):
            raise ValueError("Catalog entries must contain nonempty string values")
        name = row.get("id", "")
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", name) or name in seen:
            raise ValueError("Catalog ids must be unique safe directory names")
        if len(set(row) & {"torchvision", "hf", "bundle"}) != 1:
            raise ValueError("Each model needs exactly one of torchvision, hf or frozen bundle")
        if set(row) - {"id", "torchvision", "hf", "bundle", "revision"}:
            raise ValueError("Unknown watch catalog fields")
        seen.add(name)
    return rows


def wheel_index(device, index_url=None):
    if index_url is None:
        if device != "cpu":
            raise ValueError("GPU watch requires an explicit matching nightly --index-url")
        index_url = "https://download.pytorch.org/whl/nightly/cpu"
    parsed = urlsplit(index_url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in {"download.pytorch.org", "download-r2.pytorch.org"}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Use an official HTTPS PyTorch nightly wheel index")
    local = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    if "/nightly/" not in parsed.path or not re.fullmatch(r"cpu|cu[0-9]+|rocm[0-9.]+", local):
        raise ValueError("Index must identify nightly/cpu, nightly/cuNNN or nightly/rocmN.N")
    kind = "cpu" if local == "cpu" else "rocm" if local.startswith("rocm") else "cuda"
    if (device == "cpu") != (kind == "cpu"):
        raise ValueError("Requested execution device and nightly wheel type differ")
    return index_url.rstrip("/"), local, kind


def discover(through, index_url, local):
    end = datetime.fromisoformat(through).date() if through else datetime.now(UTC).date()
    return nightly_catalog(
        (end - timedelta(days=90)).isoformat(),
        end.isoformat(),
        index_url=index_url + "/torch/",
        local_version=local,
    )


def frozen_plan(root, models, records, seed_nightlies, through, discover_catalog):
    """Finish an interrupted immutable selection before discovering a newer nightly."""
    pointer = root / "active-plan.json"
    if pointer.exists():
        marker = read_json(pointer)
        path = (root / marker["file"]).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != marker["sha256"]:
            raise ValueError("Frozen watch plan changed")
        previous = read_json(path)
        completed = {(r["date"], r["model"], r["reference"]) for r in records}
        expected = {
            (day, model["id"], rail)
            for day in previous["selected_dates"]
            for model in models
            for rail in ("stable", "previous")
        }
        if not expected <= completed:
            if previous["seed_nightlies"] != seed_nightlies or previous["through"] != through:
                raise ValueError(
                    "Finish the incomplete watch plan with its original --seed-nightlies and --through options"
                )
            return previous
    inventory = discover_catalog()
    wheels = inventory["wheels"]
    if len(wheels) < seed_nightlies + 1:
        raise ValueError(
            "Not enough available nightly wheels, including the previous-nightly anchor"
        )
    plan = {
        "schema_version": 1,
        "inventory": inventory,
        "selected_dates": [w["date"] for w in wheels[-seed_nightlies:]],
        "seed_nightlies": seed_nightlies,
        "through": through,
    }
    path = root / "plans" / f"{time.time_ns()}.json"
    atomic_json(path, plan)
    atomic_json(pointer, {"file": path.relative_to(root).as_posix(), "sha256": file_hash(path)})
    return plan


def model_requirements(identity):
    names = (
        "numpy",
        "psutil",
        "packaging",
        "transformers",
        "safetensors",
        "pillow",
        "tokenizers",
        "huggingface-hub",
    )
    return [
        f"{name}=={identity['packages'][name]}"
        for name in names
        if name in identity.get("packages", {})
    ]


def ensure_wheel(root, wheel, index_url, kind, timeout, requirements=()):
    target = Path(root) / wheel["date"]
    if (target / "ready.json").exists():
        result = read_json(target / "ready.json")
        spec = result["specification"]
        if (
            spec["torch"] != wheel["url"]
            or spec["index_url"] != index_url
            or spec["device"] != kind
        ):
            raise ValueError("Watch environment cache specification changed")
        actual = environment_identity(result["python"], target / f"watch-probe-{time.time_ns()}")
        if actual != result["identity"]:
            raise ValueError("Installed watch environment changed")
    else:
        result = create_environment(
            target,
            wheel["url"],
            device=kind,
            index_url=index_url,
            timeout=timeout,
            requirements=requirements,
        )
    if result["identity"]["torch"] != wheel["version"]:
        raise ValueError("Installed watch wheel does not match the frozen nightly catalog")
    for requirement in requirements:
        name, version = requirement.split("==", 1)
        if result["identity"]["packages"].get(name) != version:
            raise ValueError(f"Cached nightly has different {name}; choose a new environment cache")
    return result


def verify_record(root, directory):
    directory = Path(directory)
    commit = read_json(directory / "commit.json")
    for name, digest in commit["files"].items():
        target = (Path(root) / name).resolve()
        if not target.is_relative_to(Path(root).resolve()) or file_hash(target) != digest:
            raise ValueError("Committed watch evidence changed")
    return read_json(directory / "record.json")


def commit_record(root, directory, record, files=()):
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "record.json", record)
    paths = [directory / "record.json", *files]
    atomic_json(
        directory / "commit.json",
        {"files": {p.relative_to(root).as_posix(): file_hash(p) for p in paths}},
    )


def execute_check(bundle, destination, baseline, candidate, settings, jobs, deadline):
    from .evaluate import Settings

    expected = Settings(
        device=settings.device,
        deterministic=settings.deterministic,
        gradients=True,
        repeats=2,
        threads=1,
        timeout_s=settings.timeout_s,
        max_rss_mb=settings.max_rss_mb,
    )
    if expected.to_dict() != settings.to_dict():
        raise ValueError("Watch execution requires unchanged standard-level settings")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None
    args = [
        "check",
        str(bundle),
        "--baseline",
        baseline,
        "--candidate",
        candidate,
        "--level",
        "standard",
        "--verdict-policy",
        "v2",
        "--jobs",
        str(jobs),
        "--device",
        settings.device,
        "--timeout",
        str(settings.timeout_s),
        "--max-rss-mb",
        str(settings.max_rss_mb),
        "--output",
        str(destination),
    ]
    if settings.deterministic:
        args.append("--deterministic")
    if (destination / "check.json").exists() or (destination / "memory-budget.json").exists():
        args.append("--resume")
    observed = supervise(
        python_command(
            sys.executable,
            "from faultline.cli import main;raise SystemExit(main())",
            args,
            watch_parent=True,
        ),
        destination.parent / f"supervision-{destination.name}-{time.time_ns()}",
        timeout=remaining,
        max_rss_mb=settings.max_rss_mb + 2048,
    )
    path = destination / "verdict.json"
    verdict = read_json(path) if path.exists() else {}
    if verdict.get("execution_state") != "complete":
        return {
            "complete": False,
            "supervision": observed,
            "verdict": verdict.get("verdict", "INCONCLUSIVE"),
        }
    evidence = read_json(destination / "evidence.json")
    structure = evidence.get("compilation_structure", {})
    return {
        "complete": True,
        "verdict": verdict["verdict"],
        "reasons": verdict["reasons"],
        "notes": verdict["notes"],
        "scope": verdict["scope"],
        "metrics": verdict.get("metrics", {}),
        "accuracy_shift_replicated": verdict["accuracy_shift_replicated"],
        "scheduling": evidence.get("scheduling", {}),
        "compilation_structure": {k: v for k, v in structure.items() if k != "pairs"},
        "reproducers": verdict["artifacts"]["reproducers"],
        "supervision": observed,
    }


def render_history(root, config, records, state, planned=()):
    root = Path(root)
    summary = {
        "schema_version": 1,
        "updated_utc": now(),
        "state": state,
        "models": [r["id"] for r in config["catalog"]],
        "records": records,
        "planned": list(planned),
        "policy": "v2",
        "source_sha256": config["source_sha256"],
        "note": "Tested frozen inputs only. Compilation structure is advisory; no automatic upstream filing.",
    }
    atomic_json(root / "history.json", summary)
    links = []
    style = "<style>body{font:16px system-ui;max-width:1100px;margin:3rem auto;padding:1rem}table{border-collapse:collapse}td,th{padding:.5rem;border:1px solid #ccc;text-align:left}</style>"
    for model in config["catalog"]:
        name = model["id"]
        rows = []
        for record in records:
            if record["model"] != name:
                continue
            report = record.get("report")
            values = [
                record["date"],
                record["reference"],
                record["verdict"],
                ", ".join(record.get("compilation_structure", {}).get("labels", [])) or "none",
            ]
            cells = "".join("<td>" + html.escape(str(v)) + "</td>" for v in values)
            link = (
                f'<a href="../{html.escape(report, quote=True)}">Report</a>'
                if report
                else html.escape(record.get("error", "unavailable"))
            )
            bisect = record.get("bisection")
            if bisect:
                link += f' · <a href="../{html.escape(bisect, quote=True)}">Bisection</a>'
            rows.append(f"<tr>{cells}<td>{link}</td></tr>")
        target = root / "timeline" / f"{name}.html"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            '<!doctype html><meta charset="utf-8">'
            + style
            + f"<title>{html.escape(name)} nightly watch</title><h1>{html.escape(name)}</h1><p>Standard checks, verdict v2. Compilation notes do not change verdicts.</p><table><tr><th>Nightly</th><th>Reference</th><th>Verdict</th><th>Advisory notes</th><th>Evidence</th></tr>"
            + "".join(rows)
            + "</table>",
            encoding="utf-8",
        )
        links.append(f'<li><a href="timeline/{name}.html">{html.escape(name)}</a></li>')
    (root / "index.html").write_text(
        '<!doctype html><meta charset="utf-8">'
        + style
        + "<title>Faultline nightly watch</title><h1>Nightly watch</h1><p>"
        + html.escape(state)
        + '</p><p><a href="history.json">JSON history</a></p><ul>'
        + "".join(links)
        + "</ul>",
        encoding="utf-8",
    )
    return summary
