"""Advisory numerical context; none of these labels changes the failure predicate."""

import math
from contextlib import nullcontext

import torch


def perturb_inputs(requests):
    """One representable step toward +infinity, once per floating input storage.

    Operates only on freshly loaded requests. Retains dtype, aliases and strides,
    and never alters integer/boolean inputs or captured output cotangents.
    """
    storages = {}

    def collect(value):
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided or value.is_complex():
                raise ValueError("ULP control requires real strided inputs")
            storage = value.untyped_storage()
            key = (str(value.device), storage._cdata)
            if key in storages and storages[key].dtype != value.dtype:
                raise ValueError("ULP control rejects mixed-dtype storage aliases")
            storages[key] = value
        elif isinstance(value, dict):
            for child in value.values():
                collect(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect(child)

    for request in requests:
        collect((request["args"], request["kwargs"]))
    changed = 0
    with torch.no_grad():
        for value in storages.values():
            if not value.is_floating_point():
                continue
            storage = value.untyped_storage()
            flat = torch.empty(0, dtype=value.dtype, device=value.device).set_(
                storage, 0, (storage.nbytes() // value.element_size(),), (1,)
            )
            shifted = torch.nextafter(flat, torch.full_like(flat, math.inf))
            mask = torch.isfinite(flat) & torch.isfinite(shifted) & (flat != shifted)
            changed += int(mask.sum())
            flat.copy_(torch.where(mask, shifted, flat))
    if not changed:
        raise ValueError("No finite floating input changed in the ULP control")


def control_policy(settings):
    if (
        settings["device"] == "cpu"
        and torch.backends.mkldnn.is_available()
        and torch.backends.mkldnn.enabled
    ):
        return "onednn_disabled"
    return "input_one_ulp_up"


def control_context(name):
    return (
        torch.backends.mkldnn.flags(enabled=False) if name == "onednn_disabled" else nullcontext()
    )


def mismatch_targets(comparison):
    return [(x["target"], x["kind"]) for x in comparison["outputs"] if not x["matches"]]


def assess_control(execution_factory, eager, candidate, settings, compare, policy, stats):
    name = control_policy(settings)
    base = {
        "control": name,
        "note": "Sensitivity evidence only. A similar eager-control error "
        "does not exclude an independent compiler defect. Inputs change only for the ULP control.",
    }
    try:
        with control_context(name):
            execution = execution_factory(perturb_inputs if name == "input_one_ulp_up" else None)
            first = execution()
            if not policy.reference_health(tuple(first[0].values())):
                return {**base, "status": "control_nonfinite"}
            comparison = compare(eager, first, policy, settings.get("rtol"), settings.get("atol"))
            for _ in range(settings["repeats"] - 1):
                repeated = execution()
                if not compare(first, repeated, policy, settings.get("rtol"), settings.get("atol"))[
                    "matches"
                ] or mismatch_targets(comparison) != mismatch_targets(
                    compare(eager, repeated, policy, settings.get("rtol"), settings.get("atol"))
                ):
                    return {**base, "status": "unstable_control"}
        comparison = compare(eager, first, policy, settings.get("rtol"), settings.get("atol"))
        records = []
        for row in comparison["outputs"]:
            name = row["target"]
            original, alternate, compiled = (
                eager[0].get(name),
                first[0].get(name),
                candidate[0].get(name),
            )
            if (
                original is None
                or alternate is None
                or compiled is None
                or not original.is_floating_point()
                or name.startswith("state:")
                or ":input_after:" in name
            ):
                continue
            if original.shape != alternate.shape or original.shape != compiled.shape:
                continue
            a, b = stats(alternate, original), stats(compiled, original)
            ratio = a.get("rmse", 0) / b["rmse"] if b.get("rmse") else None
            records.append(
                {
                    "target": name,
                    "control_vs_eager": a,
                    "candidate_vs_eager": b,
                    "control_to_candidate_rmse": ratio
                    if ratio is not None and math.isfinite(ratio)
                    else None,
                    "exceeds_primary_tolerance": not row["matches"],
                }
            )
        return {**base, "status": "complete", "comparison": comparison, "outputs": records}
    except Exception as exc:  # noqa: BLE001 -- optional arbitrary-model control
        return {**base, "status": "control_error", "error": f"{type(exc).__name__}: {exc}"}


def assess_masks(probe, execution_factory, eager, candidate, settings, compare, policy):
    try:
        with probe.aot.capture_context(True):
            run = execution_factory(probe)
            anchor = None
            for _ in range(settings["repeats"]):
                probe.reset()
                observed = run()
                result = probe.summary()
                replay = compare(
                    candidate, observed, policy, settings.get("rtol"), settings.get("atol")
                )
                original = compare(
                    eager, candidate, policy, settings.get("rtol"), settings.get("atol")
                )
                current = compare(
                    eager, observed, policy, settings.get("rtol"), settings.get("atol")
                )
                if not replay["matches"] or mismatch_targets(original) != mismatch_targets(current):
                    return {
                        **result,
                        "status": "diagnostic_replay_changed",
                        "primary_candidate_comparison": replay,
                    }
                if anchor is not None and result != anchor:
                    return {**result, "status": "unstable_masks"}
                anchor = result
            return {**anchor, "replay_matches_primary": True, "repeats": settings["repeats"]}
    except Exception as exc:  # noqa: BLE001 -- optional version-dependent AOT diagnostics
        return {"status": "diagnostic_error", "error": f"{type(exc).__name__}: {exc}"}


def annotate(primary, fp64, control, masks):
    failed = {
        r["target"] for r in primary.get("comparison", {}).get("outputs", []) if not r["matches"]
    }
    fp64_rows = [r for r in fp64.get("outputs", []) if r["target"] in failed]
    comparable = [
        r["target"]
        for r in fp64_rows
        if r.get("accuracy_label") == "comparable_to_eager_against_fp64"
    ]
    sensitive = [
        r["target"]
        for r in control.get("outputs", [])
        if r["target"] in failed
        and r["exceeds_primary_tolerance"]
        and r.get("control_to_candidate_rmse") is not None
        and r["control_to_candidate_rmse"] >= 0.5
    ]
    labels = []
    if comparable:
        labels.append("comparable_to_eager_against_fp64")
    if sensitive:
        labels.append("eager_control_sensitive")
    if (
        primary["status"] == "gradient_mismatch"
        and masks.get("replay_matches_primary")
        and masks.get("flips", 0)
    ):
        labels.append("discontinuity_sensitive")
    return {
        "schema_version": 1,
        "labels": labels,
        "fp64_comparable_targets": comparable,
        "eager_sensitive_targets": sensitive,
        "eager_control": control,
        "masks": masks,
        "note": "Advisory labels describe evidence; the primary status, tolerances, failure "
        "signature, and exit code are unchanged. Mask flips do not rule out a second defect.",
    }
