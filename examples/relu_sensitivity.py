"""Inspect ResNet's numerical gradient sensitivity, using frozen random weights.

Requires torchvision and an Inductor toolchain. This is a numerical diagnostic,
not a known compiler regression or an accuracy benchmark. Exit 1 preserves a
stable mismatch even when the advisory finds activation-mask sensitivity.
"""

import argparse
import json
from pathlib import Path

import torch


def factory():
    from torchvision.models import resnet18

    torch.manual_seed(21)
    return resnet18(weights=None).eval(), (torch.randn(1, 3, 224, 224),)


def main():
    from faultline.evaluate import Settings
    from faultline.native import capture_workload
    from faultline.native_artifacts import run_native

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    settings = Settings(
        device=args.device,
        gradients=True,
        numerics=True,
        repeats=2,
        timeout_s=600,
        max_rss_mb=12288,
    )
    bundle = capture_workload(
        f"{Path(__file__).resolve()}:factory", args.output / "workload", settings
    )
    summary = run_native(bundle, args.output / "analysis", settings, aot=True)
    result = json.loads((args.output / "analysis/run/outcome.json").read_text())
    context = result.get("numerical_context", {})
    masks = context.get("masks", {})
    print(
        json.dumps(
            {
                "primary_status": result["status"],
                "labels": context.get("labels"),
                "mask_status": masks.get("status"),
                "mask_flips": masks.get("flips"),
                "checked_selector_elements": masks.get("elements"),
                "changed_selectors": [r for r in masks.get("masks", []) if r.get("flips")],
                "report": str(args.output / "analysis/report.html"),
            },
            indent=2,
        )
    )
    return summary["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
