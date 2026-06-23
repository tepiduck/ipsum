from experiments import artifacts
from experiments.card_b_eviction import (
    CardBEvictionConfig,
    EvictionSelector,
    OracleDrift,
    _eviction_quality,
    _eviction_latencies,
    run,
)


def test_eviction_latency_counts_hand_built_stale_abstraction():
    config = CardBEvictionConfig(n_tests=4, cycles=40)
    system = EvictionSelector(
        system_name="ipsum_evict",
        n_tests=4,
        selection_rate_cap=0.5,
        evict=True,
        config=config,
    )
    system._birth_clusters["files_1_2"] = (frozenset({1, 2, 3}),)
    system._birth_files["files_1_2"] = frozenset({1, 2})
    system._admitted_cycle["files_1_2"] = 5
    system._evicted_cycle["files_1_2"] = 15
    system._events.append(
        {
            "cycle": 15,
            "type": "evict",
            "system": "ipsum_evict",
            "name": "files_1_2",
        }
    )
    drift = OracleDrift(
        cycle=10,
        before_clusters=(frozenset({1, 2, 3}),),
        after_clusters=(frozenset({1, 4}), frozenset({2, 5})),
    )

    latencies, stale_fraction = _eviction_latencies(system, [drift], config)
    quality = _eviction_quality(system, [drift], config)

    assert latencies == [5.0]
    assert stale_fraction == 1.0
    assert quality["precision"] == 1.0
    assert quality["recall"] == 1.0


def test_no_drift_stable_period_evict_matches_append_only(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "RUNS_DIR", tmp_path)
    config = CardBEvictionConfig(
        n_files=32,
        n_tests=16,
        n_clusters=4,
        cycles=260,
        drift_schedule=(),
        admission_warmup=120,
        admission_interval=40,
        validation_cycles=30,
        adaptation_window=100,
        max_candidates=32,
        min_support=2,
        seed=19,
        seeds=(19,),
    )

    result = run(config)

    assert result["metrics"]["stable_accuracy_delta_mean"] >= -config.stable_margin
