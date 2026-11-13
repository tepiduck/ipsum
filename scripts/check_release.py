"""Audit the staged release or built archives before distributing them.

This is a strict publication allowlist, not a general-purpose secret scanner.
It rejects unexpected files before reading their contents.
"""

import argparse
import re
import subprocess
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

FILES = {
    ".gitattributes",
    ".gitignore",
    ".github/workflows/ci.yml",
    ".github/workflows/gpu.yml",
    ".github/workflows/nightly.yml",
    ".github/workflows/check.yml",
    "examples/ci/upgrade-check.yml",
    "docs/verdict-policy.md",
    "docs/verdict-schema.md",
    "docs/verdict.schema.json",
    "docs/verdict-policy-v2.md",
    "docs/verdict-v2.schema.json",
    "docs/research/rescore-017.json",
    "docs/research/rescore-017.csv",
    "docs/research/validation-017.json",
    "docs/research/usability-017.md",
    "scripts/rescore_check_v2.py",
    "scripts/validate_check_v2.py",
    "scripts/summarize_check_v2.py",
    "docs/user-guide.md",
    "docs/troubleshooting.md",
    "docs/research/README.md",
    "docs/research/usability-016.md",
    "docs/research/quickstart-test.md",
    "docs/research/calibration-016.json",
    "docs/research/calibration-016.csv",
    "docs/research/cost-016.json",
    "README.md",
    "RESULTS.md",
    "pyproject.toml",
    "MANIFEST.in",
    "docs/architecture.md",
    "docs/cli.md",
    "docs/diagnostics.md",
    "docs/numerics.md",
    "docs/sweeps.md",
    "docs/validation.json",
    "docs/gpt2-accuracy.csv",
    "docs/accuracy-shift-010.csv",
    "docs/accuracy-shift-011.csv",
    "docs/bisection-011.csv",
    "docs/bisect.md",
    "docs/v011-policy.json",
    "docs/accuracy-investigation.md",
    "docs/inductor-config-012.diff",
    "docs/inductor-code-012.json",
    "docs/inductor-code-012.diff",
    "docs/inductor-fx-012.diff",
    "docs/gradient-scales-012.csv",
    "docs/accuracy-investigation-012.csv",
    "docs/upstream-report-draft.md",
    "docs/upstream-report-draft-012.md",
    "docs/attention-investigation.md",
    "docs/v013-policy.json",
    "docs/v013-kernel-controls.json",
    "docs/attribution-013.csv",
    "docs/attention-capture-013.json",
    "docs/attention-kernel-013.csv",
    "docs/attention-heads-013.csv",
    "docs/attention-rows-013.csv",
    "docs/attention-association-013.csv",
    "docs/attention-consistency-013.csv",
    "docs/exp-accuracy-013.csv",
    "docs/exp-timing-013.csv",
    "docs/upstream-report-draft-013.md",
    "docs/gelu-investigation.md",
    "docs/v014-policy.json",
    "docs/gelu-code-014.json",
    "docs/gelu-capture-014.json",
    "docs/gelu-ranges-014.csv",
    "docs/gelu-exp-grid-014.csv",
    "docs/gelu-block12-014.csv",
    "docs/upstream-report-draft-014.md",
    "docs/v015-policy.json",
    "docs/v016-policy.json",
    "examples/input_batches.py",
    "docs/input-accuracy.md",
    "docs/input-batches-015.json",
    "docs/input-accuracy-015.csv",
    "docs/input-accuracy-015.json",
    "examples/capture_mlp.py",
    "examples/models.py",
    "examples/native_models.py",
    "examples/relu_sensitivity.py",
    "examples/gpt2_accuracy.py",
    "examples/sdpa_accuracy.py",
    "examples/exp_accuracy.py",
    "examples/cpu_exp_probe.cpp",
    "examples/gelu_ranges.py",
    "examples/input_batch_accuracy.py",
    "scripts/validate_faultline.py",
    "scripts/validate_upgrades.py",
    "scripts/validate_native.py",
    "scripts/validate_aot.py",
    "scripts/compare_inductor.py",
    "scripts/summarize_input_accuracy.py",
    "scripts/summarize_check_calibration.py",
    "scripts/check_release.py",
}
SOURCE_PREFIXES = ("src/faultline/", "tests/faultline/")
PRIVATE_PATTERNS = (
    re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\s\\/\"']+", re.IGNORECASE),
    re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+/"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b"),
)


def public_source(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        return False
    return name in FILES or (name.startswith(SOURCE_PREFIXES) and path.suffix == ".py")


def archive_allowed(name, wheel):
    if wheel:
        if name.startswith("faultline/"):
            return public_source("src/" + name)
        parts = PurePosixPath(name).parts
        return bool(
            parts
            and parts[0].startswith("faultline_ml-")
            and parts[0].endswith(".dist-info")
            and "/".join(parts[1:])
            in {
                "METADATA",
                "WHEEL",
                "RECORD",
                "entry_points.txt",
                "top_level.txt",
            }
        )
    if public_source(name) or name in {"PKG-INFO", "setup.cfg"}:
        return True
    return name.startswith("src/faultline_ml.egg-info/") and PurePosixPath(name).name in {
        "PKG-INFO",
        "SOURCES.txt",
        "dependency_links.txt",
        "entry_points.txt",
        "requires.txt",
        "top_level.txt",
    }


def inspect_text(name, data):
    text = data.decode("utf-8")
    if any(pattern.search(text) for pattern in PRIVATE_PATTERNS):
        raise ValueError(f"Potential personal path or credential in {name}; content not printed")


def check_index():
    result = subprocess.check_output(["git", "ls-files", "-z"])
    names = [name.decode("utf-8") for name in result.split(b"\0") if name]
    unexpected = [name for name in names if not public_source(name)]
    if unexpected:
        raise ValueError("Unexpected staged paths: " + ", ".join(unexpected))
    for name in names:
        inspect_text(name, subprocess.check_output(["git", "show", f":{name}"]))
    return len(names)


def check_archive(path):
    records = []
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            for entry in archive.infolist():
                if entry.is_dir():
                    continue
                if not archive_allowed(entry.filename, True):
                    raise ValueError(f"Unexpected wheel member: {entry.filename}")
                records.append((entry.filename, archive.read(entry)))
    else:
        with tarfile.open(path, "r:gz") as archive:
            for entry in archive.getmembers():
                if entry.isdir():
                    continue
                name = entry.name.partition("/")[2]
                if not entry.isfile() or not archive_allowed(name, False):
                    raise ValueError(f"Unexpected source archive member: {entry.name}")
                records.append((name, archive.extractfile(entry).read()))
    for name, data in records:
        inspect_text(name, data)
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", action="store_true")
    parser.add_argument("archives", type=Path, nargs="*")
    args = parser.parse_args()
    if not args.index and not args.archives:
        parser.error("choose --index and/or distribution archives")
    try:
        if args.index:
            print(f"Public index: {check_index()} files checked")
        for path in args.archives:
            print(f"{path.name}: {check_archive(path)} members checked")
    except (ValueError, OSError, UnicodeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Release audit failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
