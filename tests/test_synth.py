from ipsum.synth import Synth, SynthConfig


def _stream(world: Synth, n_cycles: int):
    return [world.step() for _ in range(n_cycles)]


def test_synth_stream_is_reproducible():
    cfg = SynthConfig(n_files=24, n_tests=12, n_clusters=4, seed=13)

    left = Synth(cfg)
    right = Synth(cfg)

    assert left.true_clusters() == right.true_clusters()
    assert [left.true_deps(t) for t in range(cfg.n_tests)] == [
        right.true_deps(t) for t in range(cfg.n_tests)
    ]
    assert _stream(left, 8) == _stream(right, 8)


def test_step_output_does_not_expose_oracle_cause():
    cfg = SynthConfig(n_files=12, n_tests=4, n_clusters=3, p_hit=1.0, p_flaky=0.0, seed=2)
    world = Synth(cfg)

    commit, outcomes = world.step()

    assert commit.cycle == 0
    assert commit.changed_files
    assert len(outcomes) == cfg.n_tests
    assert not hasattr(outcomes[0], "cause")


def test_cause_oracle_matches_failures_when_noise_is_off():
    cfg = SynthConfig(n_files=36, n_tests=18, n_clusters=6, p_hit=1.0, p_flaky=0.0, seed=8)
    world = Synth(cfg)

    for _ in range(10):
        commit, outcomes = world.step()
        for outcome in outcomes:
            cause = world.cause(outcome)
            if not outcome.failed:
                assert cause is None
                continue

            assert cause is not None
            causal_files = world.true_clusters()[cause]
            assert causal_files & commit.changed_files & world.true_deps(outcome.test)


def test_delay_holds_outcomes_until_reveal_cycle():
    cfg = SynthConfig(
        n_files=16,
        n_tests=5,
        n_clusters=4,
        p_hit=1.0,
        p_flaky=0.0,
        delay_mean=2.0,
        seed=5,
    )
    world = Synth(cfg)

    _, cycle_0 = world.step()
    _, cycle_1 = world.step()
    _, cycle_2 = world.step()

    assert cycle_0 == []
    assert cycle_1 == []
    assert len(cycle_2) == cfg.n_tests
    assert {outcome.cycle_decided for outcome in cycle_2} == {0}
    assert {outcome.cycle_revealed for outcome in cycle_2} == {2}


def test_scheduled_drift_changes_oracle_state():
    cfg = SynthConfig(n_files=40, n_tests=20, n_clusters=5, drift_schedule=(1,), seed=21)
    world = Synth(cfg)
    clusters_before = world.true_clusters()
    deps_before = [world.true_deps(t) for t in range(cfg.n_tests)]

    world.step()
    world.step()

    clusters_after = world.true_clusters()
    deps_after = [world.true_deps(t) for t in range(cfg.n_tests)]
    assert clusters_after != clusters_before or deps_after != deps_before
    assert world.drift_schedule() == (1,)


def test_coverage_skew_biases_commit_clusters():
    cfg = SynthConfig(
        n_files=40,
        n_tests=20,
        n_clusters=4,
        coverage_skew=(12.0, 1.0, 1.0, 1.0),
        seed=9,
    )
    world = Synth(cfg)
    counts = [0, 0, 0, 0]

    for _ in range(200):
        commit, _ = world.step()
        overlaps = [
            len(commit.changed_files & cluster)
            for cluster in world.true_clusters()
        ]
        counts[max(range(len(overlaps)), key=lambda idx: overlaps[idx])] += 1

    assert counts[0] > max(counts[1:]) * 3


def test_cause_history_can_be_bounded():
    cfg = SynthConfig(
        n_files=12,
        n_tests=4,
        n_clusters=3,
        p_hit=1.0,
        p_flaky=0.0,
        cause_history_cycles=2,
        seed=3,
    )
    world = Synth(cfg)

    caused_failure = None
    for _ in range(10):
        _, outcomes = world.step()
        caused_failure = next(
            (outcome for outcome in outcomes if world.cause(outcome) is not None),
            None,
        )
        if caused_failure is not None:
            break
    assert caused_failure is not None

    for _ in range(5):
        world.step()

    assert world.cause(caused_failure) is None
