from experiments import artifacts
from experiments.card_d_coverage import (
    CardDConfig,
    coverage_decision,
    run,
)
from experiments.card_d_skew_sweep import (
    CardDSkewSweepConfig,
    SkewLevel,
    run as run_skew_sweep,
)
from ipsum.abstractions import Abstraction


def test_strict_guard_refuses_thin_coverage_candidate():
    candidate = Abstraction("thin", {"files": [1, 2]}, complexity=0.01)
    commits = [frozenset({1}), frozenset({3}), frozenset({4})]

    decision = coverage_decision(
        mode="strict_guard",
        candidate=candidate,
        commits=commits,
        ll_gain=0.2,
        min_count=4,
        provisional_min_count=1,
    )

    assert not decision.admitted
    assert decision.coverage_count == 1


def test_provisional_guard_keeps_post_drift_latency_near_no_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "RUNS_DIR", tmp_path)
    config = CardDConfig(
        n_files=40,
        n_tests=20,
        n_clusters=5,
        train_cycles=120,
        heldout_cycles=40,
        drift_post_cycles=120,
        admission_interval=40,
        coverage_skew=(8.0, 4.0, 1.0, 0.5, 0.25),
        max_candidates=64,
        seed=31,
        seeds=(31,),
    )

    result = run(config)

    assert result["metrics"]["provisional_latency_extra_vs_no_guard_mean"] <= (
        config.max_provisional_latency_extra
    )


def test_provisional_guard_quarantine_changes_prediction_and_not_false_rate(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(artifacts, "RUNS_DIR", tmp_path)
    config = CardDConfig(
        n_files=40,
        n_tests=20,
        n_clusters=5,
        train_cycles=120,
        heldout_cycles=40,
        drift_post_cycles=120,
        admission_interval=20,
        coverage_skew=(8.0, 4.0, 1.0, 0.5, 0.25),
        max_candidates=64,
        seed=31,
        seeds=(31,),
    )

    result = run(config)
    metrics = result["metrics"]

    assert metrics["provisional_guard_provisional_count_mean"] > 0.0
    assert metrics["provisional_guard_thin_heldout_ll_mean"] != (
        metrics["no_guard_thin_heldout_ll_mean"]
    )
    assert metrics["provisional_false_admission_rate_thin_reduction_mean"] == 0.0


def test_skew_sweep_reports_uncertainty_and_negative_seed_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "RUNS_DIR", tmp_path)
    base = CardDConfig(
        n_files=32,
        n_tests=16,
        n_clusters=4,
        train_cycles=80,
        heldout_cycles=24,
        drift_post_cycles=80,
        admission_interval=20,
        coverage_skew=(2.0, 1.5, 0.8, 0.5),
        max_candidates=48,
        seed=41,
        seeds=(41, 42),
    )
    config = CardDSkewSweepConfig(
        base=base,
        levels=(
            SkewLevel("mild", 1.0, (2.0, 1.5, 0.8, 0.5)),
            SkewLevel("extreme", 2.0, (8.0, 5.0, 0.4, 0.2)),
        ),
    )

    result = run_skew_sweep(config)
    metrics = result["metrics"]
    first_row = result["level_results"][0]["summary"][
        "provisional_thin_undercovered_harm_reduction"
    ]

    assert "card_d_keystone_demonstrated" in metrics
    assert "provisional_thin_undercovered_harm_reduction_skew_slope" in metrics
    assert "se" in first_row
    assert "ci95_low" in first_row
    assert "negative_seed_count" in first_row
