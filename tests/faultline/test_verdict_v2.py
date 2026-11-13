import copy
import itertools
import json
import math
from pathlib import Path

import jsonschema
import pytest

from faultline.replication import accuracy_shift_replicated, encoded
from faultline.replication_stop import decision_complete, replication_evidence, stopping_certificate
from faultline.verdict import summarize


def score(value, batch):
    return {"batch": batch, "status": "complete", **encoded(value)}


def primary(status="gradient_mismatch", target="output:gradient:parameter:w"):
    return {
        "status": status,
        "comparison": {"outputs": [{"target": target, "matches": status == "pass"}]},
    }


def pair(baseline="gradient_mismatch", candidate="gradient_mismatch", batch=0):
    return {"batch": batch, "baseline": primary(baseline), "candidate": primary(candidate)}


def verdict(
    rows,
    *,
    level="quick",
    replication=None,
    planned=None,
    policy="v2",
    mode="environment_comparison",
):
    return summarize(
        rows, planned or len(rows), mode=mode, level=level, replication=replication, policy=policy
    )


def test_recurring_and_changed_signatures_are_not_change_attribution():
    rows = [pair(), pair(candidate="numerical_mismatch", batch=1)]
    rows[1]["candidate"]["comparison"]["outputs"][0]["target"] = "new-output"
    saved = copy.deepcopy(rows)
    assert verdict(rows, policy="v1")["verdict"] == "NEEDS_REVIEW"
    result = verdict(rows)
    assert result["verdict"] == "SAFE"
    assert result["pre_existing_differences"]["batches"] == 2
    assert result["pre_existing_differences"]["candidate_failed_targets"] == 2
    assert all(item["links"] for item in result["pre_existing_differences"]["details"])
    assert rows == saved
    schema = json.loads((Path(__file__).parents[2] / "docs/verdict-v2.schema.json").read_text())
    jsonschema.validate(result, schema)


@pytest.mark.parametrize("policy", ["v1", "v2"])
def test_single_environment_keeps_v1(policy):
    result = verdict([pair()], mode="single_environment", policy=policy)
    assert (result["verdict"], result["policy_version"]) == ("NEEDS_REVIEW", 1)


def test_candidate_only_and_replicated_shift_remain_regressions():
    assert verdict([pair("pass", "numerical_mismatch")])["verdict"] == "REGRESSION"
    rows = [pair(batch=i) for i in range(10)]
    replication = accuracy_shift_replicated([score(math.log2(1.5), i) for i in range(10)], 10)
    assert verdict(rows, level="standard", replication=replication)["verdict"] == "REGRESSION"


def test_missing_replication_distinguishes_worsening_trend():
    rows = [pair(batch=i) for i in range(10)]
    for value, expected in [(0, "INCONCLUSIVE"), (1, "NEEDS_REVIEW"), (-1, "INCONCLUSIVE")]:
        replication = accuracy_shift_replicated([score(value, i) for i in range(9)], 10)
        assert verdict(rows, level="standard", replication=replication)["verdict"] == expected
    replication = accuracy_shift_replicated([score(0, i) for i in range(10)], 10)
    assert verdict(rows, level="standard", replication=replication)["verdict"] == "SAFE"


def test_unstable_numerical_change_is_review_but_generic_instability_is_inconclusive():
    rows = [pair("pass", "unstable_candidate")]
    assert verdict(rows)["verdict"] == "NEEDS_REVIEW"
    rows[0]["candidate"].pop("comparison")
    assert verdict(rows)["verdict"] == "INCONCLUSIVE"
    rows[0]["baseline"] = primary("unstable_reference")
    assert verdict(rows)["verdict"] == "INCONCLUSIVE"
    assert verdict([pair("pass", "resource_limit")])["verdict"] == "INCONCLUSIVE"
    assert verdict([pair()], planned=2)["verdict"] == "INCONCLUSIVE"


def test_stopping_needs_a_valid_certificate_and_all_primaries():
    prefix = [score(0, i) for i in range(3)]
    certificate = stopping_certificate(prefix, 10)
    assert certificate["outcome"] == "not_flagged"
    replication = replication_evidence(prefix, 10, certificate=certificate)
    assert replication["completed"] == 3 and replication["deliberately_skipped"] == 7
    rows = [pair(batch=i) for i in range(10)]
    assert verdict(rows, level="standard", replication=replication)["verdict"] == "SAFE"
    assert (
        verdict(rows[:3], planned=10, level="standard", replication=replication)["verdict"]
        == "INCONCLUSIVE"
    )
    rows[-1] = pair("pass", "numerical_mismatch", 9)
    assert verdict(rows, level="standard", replication=replication)["verdict"] == "REGRESSION"
    replication["stopping"]["prefix_batches"] = [0]
    assert not decision_complete(replication, 10)


def test_stopping_bounds_cover_every_completion_and_do_not_flag_below_ten():
    for prefix_values in ([0, 0, 0], [0.1] * 6, [1] * 9, [1] * 10, [1] * 10 + [0]):
        planned = 12
        rows = [score(value, i) for i, value in enumerate(prefix_values)]
        certificate = stopping_certificate(rows, planned)
        if not certificate:
            continue
        expected = certificate["outcome"] == "flagged"
        assert not expected or len(rows) >= 10
        # Includes future unavailable measurements: the denominator is completed scores.
        for extra in range(planned - len(rows) + 1):
            for values in itertools.product(
                (-math.inf, 0, math.log2(1.25), math.inf), repeat=extra
            ):
                future = rows + [score(value, len(rows) + i) for i, value in enumerate(values)]
                assert (
                    accuracy_shift_replicated(future, planned)["accuracy_shift_replicated"]
                    == expected
                )
    assert stopping_certificate([score(0, 0), {"batch": 1, "status": "incomplete"}], 10) is None
    assert stopping_certificate([score(0, 1)], 10) is None
    assert stopping_certificate([score(0, i) for i in range(3)], 3) is None


def test_infinity_median_cannot_create_false_positive_certificate():
    rows = [score(-math.inf, 0)] + [score(math.inf, i) for i in range(1, 10)]
    certificate = stopping_certificate(rows, 11)
    assert certificate is not None and certificate["outcome"] == "flagged"
    evidence = replication_evidence(rows, 11, certificate=certificate)
    assert decision_complete(evidence, 11)
