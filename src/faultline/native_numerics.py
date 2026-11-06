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
    comparable = sorted(
        {
            r["target"]
            for r in fp64_rows
            if fp64.get("status") == "complete"
            and r.get("reference_dtype") == "torch.float64"
            and r.get("accuracy_label") == "comparable_to_eager_against_fp64"
        }
    )
    sensitive = sorted(
        {
            r["target"]
            for r in control.get("outputs", [])
            if control.get("status") == "complete"
            and r["target"] in failed
            and r["exceeds_primary_tolerance"]
            and r.get("control_to_candidate_rmse") is not None
            and r["control_to_candidate_rmse"] >= 0.5
        }
    )
    labels = []
    if failed and set(comparable) == failed:
        labels.append("comparable_to_eager_against_fp64")
    if failed and set(sensitive) == failed:
        labels.append("eager_control_sensitive")
    flipped_outputs = {
        (row["step"], row["stage"], index)
        for row in masks.get("masks", [])
        if row.get("status") == "complete" and row.get("flips", 0)
        for index in row.get("affected_backward_outputs", [])
    }
    downstream, mapped = [], []
    for target in sorted(failed):
        mapping = masks.get("gradient_mapping", {}).get(target, {})
        origins = mapping.get("origins", [])
        if mapping.get("status") == "mapped" and len(origins) == 1:
            mapped.append(target)
            origin = origins[0]
            if (origin["step"], origin["stage"], origin["output"]) in flipped_outputs:
                downstream.append(target)
    checks = masks.get("backward_same_inputs", [])
    backward_agrees = None
    if any(row.get("fp64_relative", {}).get("agrees") is False for row in checks):
        backward_agrees = False
    elif (
        checks
        and all(row.get("fp64_relative", {}).get("agrees") is True for row in checks)
        and not any(":backward:" in row.get("stage", "") for row in masks.get("errors", []))
    ):
        backward_agrees = True
    if (
        primary["status"] == "gradient_mismatch"
        and masks.get("replay_matches_primary")
        and masks.get("flips", 0)
    ):
        labels.append(
            "explained_by_discontinuity"
            if backward_agrees and failed and set(downstream) == failed
            else "discontinuity_present"
        )
    coverage = {
        "comparable_to_eager_against_fp64": {
            "matched": len(comparable),
            "total": len(failed),
            "targets": comparable,
        },
        "eager_control_sensitive": {
            "matched": len(sensitive),
            "total": len(failed),
            "targets": sensitive,
            "control": control.get("control"),
        },
        "discontinuity_downstream": {
            "matched": len(downstream),
            "total": len(failed),
            "targets": downstream,
            "backward_same_inputs_agree": backward_agrees,
        },
    }
    context = {
        "schema_version": 3,
        "labels": labels,
        "label_coverage": coverage,
        "failing_targets": sorted(failed),
        "control_policy": control.get("control"),
        "fp64_comparable_targets": comparable,
        "eager_sensitive_targets": sensitive,
        "eager_control": control,
        "masks": masks,
        "note": "Advisory labels describe evidence; the primary status, tolerances, failure "
        "signature, and exit code are unchanged. Whole-finding labels require all failing targets. "
        "The discontinuity explanation requires same-input backward agreement and mapped downstream "
        "coverage of every failing gradient; it is an evidence criterion, not a correctness proof.",
    }
    context["triage"] = triage_evidence(primary, fp64, control, masks, context, mapped)
    return context


def triage_evidence(primary, fp64, control, masks, context, mapped):
    """Missing diagnostics never become contrary evidence by implication."""
    failed = set(context["failing_targets"])
    contrary, missing = [], []
    replay_valid = masks.get("replay_matches_primary") is True
    checks = masks.get("backward_same_inputs", [])
    if replay_valid and any(row.get("fp64_relative", {}).get("agrees") is False for row in checks):
        contrary.append("same_input_backward_farther_from_fp64")
    downstream = set(context["label_coverage"]["discontinuity_downstream"]["targets"])
    if (
        replay_valid
        and masks.get("flips", 0)
        and masks.get("status") == "complete"
        and set(mapped) - downstream
    ):
        contrary.append("failing_gradients_not_downstream_of_observed_flips")
    if not replay_valid:
        missing.append("mask_replay_unavailable_or_unstable")
    if masks.get("status") not in {"complete", "no_supported_masks"}:
        missing.append("mask_coverage_incomplete")
    if not checks or any(row.get("fp64_relative", {}).get("agrees") is None for row in checks):
        missing.append("same_input_backward_fp64_incomplete")
    fp_rows = (
        {
            row["target"]: row
            for row in fp64.get("outputs", [])
            if row.get("reference_dtype") == "torch.float64"
            and row.get("accuracy_label") != "unavailable"
        }
        if fp64.get("status") == "complete"
        else {}
    )
    if not failed <= set(fp_rows):
        missing.append("fp64_targets_incomplete")
    control_rows = (
        {
            row["target"]: row
            for row in control.get("outputs", [])
            if row.get("control_to_candidate_rmse") is not None
        }
        if control.get("status") == "complete"
        else {}
    )
    if not failed <= set(control_rows):
        missing.append("eager_control_incomplete")
    if primary["status"] == "pass":
        classification = "agreement"
    elif contrary:
        classification = "unexplained_contrary_evidence"
    elif set(context["labels"]) & {
        "explained_by_discontinuity",
        "comparable_to_eager_against_fp64",
        "eager_control_sensitive",
    }:
        classification = "explained"
    else:
        # A completed check can rule out its own proposed explanation without establishing a bug.
        if any(
            fp_rows[t].get("accuracy_label")
            in {"candidate_farther_from_fp64", "eager_exact_candidate_differs"}
            for t in failed & set(fp_rows)
        ):
            contrary.append("fp64_comparability_does_not_explain_all_targets")
        if any(t not in context["eager_sensitive_targets"] for t in failed & set(control_rows)):
            contrary.append("eager_control_does_not_explain_all_targets")
        classification = (
            "unexplained_contrary_evidence" if contrary else "unexplained_missing_evidence"
        )
        if not contrary and not missing:
            missing.append("no_complete_supported_explanation")
    return {
        "classification": classification,
        "contrary_evidence": contrary,
        "missing_evidence": missing,
        "note": "Advisory triage, not a correctness judgment. A complete explanation can coexist "
        "with missing optional diagnostics. Verified contrary backward/selector evidence takes precedence.",
    }
