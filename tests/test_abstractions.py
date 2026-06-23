import pytest

from ipsum.abstractions import Abstraction, AbstractionStore


def test_candidates_use_observed_cochange_only():
    store = AbstractionStore(min_support=2, cochange_threshold=0.5)
    store.observe_commit({1, 2, 3})
    store.observe_commit({1, 2})
    store.observe_commit({4})

    candidates = store.candidates()

    assert candidates
    assert any(set(candidate.payload["files"]) >= {1, 2} for candidate in candidates)


def test_admit_requires_gain_over_complexity():
    store = AbstractionStore()
    candidate = Abstraction(name="files_1_2", payload={"files": [1, 2]}, complexity=0.2)

    assert not store.admit(candidate, ll_gain=0.1)
    assert len(store) == 0

    assert store.admit(candidate, ll_gain=0.3)
    assert len(store) == 1
    assert next(iter(store)).usefulness == pytest.approx(0.1)


def test_reinforce_and_decay_evict():
    store = AbstractionStore(decay=0.5)
    candidate = Abstraction(name="files_1_2", payload={"files": [1, 2]}, complexity=0.1)

    assert store.admit(candidate, ll_gain=0.5)
    store.reinforce("files_1_2", ll_gain=0.2)
    assert next(iter(store)).usefulness > 0.1

    evicted = []
    for _ in range(5):
        evicted.extend(store.decay_and_evict())

    assert evicted == ["files_1_2"]
    assert len(store) == 0
