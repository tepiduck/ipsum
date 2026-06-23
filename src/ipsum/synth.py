"""Synthetic testbed — a controllable mock of "a codebase under CI".

Its ONLY purpose is to give every research mechanism a ground-truth oracle to be
validated against (RESEARCH.md §1). The model under test never sees the oracle;
evaluation does. Debug mechanisms HERE before touching RTPTorrent — on real data
a null result is ambiguous (bad mechanism vs. noise); here you know the answer.

Everything is seeded and reproducible.

Interfaces + knobs are fixed. Keep the oracle accessors strictly separate from
`step()` output so they can never leak into a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

import numpy as np


@dataclass
class SynthConfig:
    n_files: int = 200
    n_tests: int = 100
    n_clusters: int = 8           # number of TRUE latent abstractions to recover
    p_hit: float = 0.9            # P(fail) when commit touches a true dependency
    p_flaky: float = 0.02         # noise knob: P(fail) otherwise
    delay_mean: float = 0.0       # feedback delay in cycles (Card C)
    delay_var: float = 0.0
    drift_schedule: tuple[int, ...] = ()   # cycles at which deps/clusters mutate (Card B)
    cause_history_cycles: int | None = 10_000
    seed: int = 0


@dataclass
class Commit:
    """One CI cycle's change."""

    cycle: int
    changed_files: frozenset[int]


@dataclass
class Outcome:
    """A revealed test result. May arrive `delay` cycles after the decision."""

    cycle_decided: int
    cycle_revealed: int
    test: int
    failed: bool


@dataclass
class Synth:
    """Generative world. `step()` advances one cycle and returns (commit, outcomes).

    The mechanisms consume commits/outcomes. Evaluation calls the oracle methods.
    """

    cfg: SynthConfig
    _cycle: int = 0
    # ground truth (private — never returned by step()):
    _deps: dict[int, frozenset[int]] = field(default_factory=dict)        # test -> files
    _clusters: list[frozenset[int]] = field(default_factory=list)         # file groups
    _pending: list[Outcome] = field(default_factory=list)
    _rng: np.random.Generator = field(init=False, repr=False)
    _file_to_cluster: dict[int, int] = field(default_factory=dict, repr=False)
    _causes: dict[tuple[int, int], int | None] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.reset()

    # --- world setup -------------------------------------------------------
    def reset(self) -> None:
        """Sample true clusters, dependencies, and RNG from cfg.seed."""
        self._validate_config()
        self._cycle = 0
        self._rng = np.random.default_rng(self.cfg.seed)
        self._pending = []
        self._causes = {}

        shuffled_files = [int(f) for f in self._rng.permutation(self.cfg.n_files)]
        partitions = np.array_split(shuffled_files, self.cfg.n_clusters)
        self._clusters = [frozenset(int(f) for f in part.tolist()) for part in partitions]
        self._rebuild_file_index()

        self._deps = {
            test: self._sample_deps_for_test(test) for test in range(self.cfg.n_tests)
        }

    def step(self) -> tuple[Commit, list[Outcome]]:
        """Advance one CI cycle.

        - sample a cluster-correlated commit
        - schedule each test's outcome under the p_hit / p_flaky model
        - apply any drift due this cycle
        - return this cycle's commit and the outcomes that become visible NOW
          (i.e. decided at cycle - delay).
        """
        cycle = self._cycle
        if cycle in self.cfg.drift_schedule:
            self._apply_drift()

        commit = self._sample_commit(cycle)
        for test in range(self.cfg.n_tests):
            outcome = self._sample_outcome(commit, test)
            self._pending.append(outcome)

        visible = [outcome for outcome in self._pending if outcome.cycle_revealed <= cycle]
        self._pending = [outcome for outcome in self._pending if outcome.cycle_revealed > cycle]
        self._prune_causes(cycle)
        self._cycle += 1
        return commit, visible

    # --- ORACLE (evaluation only; must NOT leak to any model) ---------------
    def true_clusters(self) -> list[frozenset[int]]:
        """Ground-truth abstraction membership. Score admission against this."""
        return list(self._clusters)

    def true_deps(self, test: int) -> frozenset[int]:
        """Ground-truth dependency set for a test."""
        if test not in self._deps:
            raise KeyError(f"unknown test id: {test}")
        return self._deps[test]

    def cause(self, outcome: Outcome) -> int | None:
        """Which cluster actually caused this failure (or None if flaky/noise).

        Score credit-assignment attribution against this. TODO.
        """
        if not outcome.failed:
            return None
        return self._causes.get((outcome.cycle_decided, outcome.test))

    def drift_schedule(self) -> tuple[int, ...]:
        """Cycles at which the world mutates. Score eviction latency against this."""
        return self.cfg.drift_schedule

    # --- private mechanics -------------------------------------------------
    def _validate_config(self) -> None:
        if self.cfg.n_files < 1:
            raise ValueError("n_files must be positive")
        if self.cfg.n_tests < 1:
            raise ValueError("n_tests must be positive")
        if not 1 <= self.cfg.n_clusters <= self.cfg.n_files:
            raise ValueError("n_clusters must be between 1 and n_files")
        if not 0.0 <= self.cfg.p_hit <= 1.0:
            raise ValueError("p_hit must be in [0, 1]")
        if not 0.0 <= self.cfg.p_flaky <= 1.0:
            raise ValueError("p_flaky must be in [0, 1]")
        if self.cfg.delay_mean < 0.0:
            raise ValueError("delay_mean must be non-negative")
        if self.cfg.delay_var < 0.0:
            raise ValueError("delay_var must be non-negative")
        if any(cycle < 0 for cycle in self.cfg.drift_schedule):
            raise ValueError("drift_schedule cycles must be non-negative")
        if self.cfg.cause_history_cycles is not None and self.cfg.cause_history_cycles < 1:
            raise ValueError("cause_history_cycles must be positive or None")

    def _rebuild_file_index(self) -> None:
        self._file_to_cluster = {}
        for cluster_id, files in enumerate(self._clusters):
            for file_id in files:
                self._file_to_cluster[file_id] = cluster_id

    def _sample_deps_for_test(self, test: int) -> frozenset[int]:
        primary = test % self.cfg.n_clusters
        dep_cluster_ids = {primary}
        if self.cfg.n_clusters > 1 and self._rng.random() < 0.3:
            dep_cluster_ids.add(int(self._rng.integers(0, self.cfg.n_clusters)))

        deps: set[int] = set()
        for cluster_id in dep_cluster_ids:
            files = sorted(self._clusters[cluster_id])
            n_take = max(1, int(round(len(files) * self._rng.uniform(0.2, 0.6))))
            chosen = self._rng.choice(files, size=min(n_take, len(files)), replace=False)
            deps.update(int(file_id) for file_id in chosen.tolist())
        return frozenset(deps)

    def _sample_commit(self, cycle: int) -> Commit:
        primary = int(self._rng.integers(0, self.cfg.n_clusters))
        active_clusters = {primary}
        for cluster_id in range(self.cfg.n_clusters):
            if cluster_id != primary and self._rng.random() < 0.08:
                active_clusters.add(cluster_id)

        changed: set[int] = set()
        for cluster_id in active_clusters:
            files = sorted(self._clusters[cluster_id])
            min_take = max(1, int(round(len(files) * 0.25)))
            max_take = max(min_take, min(len(files), int(round(len(files) * 0.7))))
            n_take = int(self._rng.integers(min_take, max_take + 1))
            chosen = self._rng.choice(files, size=n_take, replace=False)
            changed.update(int(file_id) for file_id in chosen.tolist())

        for file_id in range(self.cfg.n_files):
            if file_id not in changed and self._rng.random() < 0.01:
                changed.add(file_id)

        return Commit(cycle=cycle, changed_files=frozenset(changed))

    def _sample_outcome(self, commit: Commit, test: int) -> Outcome:
        hit_files = commit.changed_files & self._deps[test]
        hit = bool(hit_files)
        failed = bool(self._rng.random() < (self.cfg.p_hit if hit else self.cfg.p_flaky))
        delay = self._sample_delay()
        outcome = Outcome(
            cycle_decided=commit.cycle,
            cycle_revealed=commit.cycle + delay,
            test=test,
            failed=failed,
        )

        cause = None
        if failed and hit:
            hit_clusters = sorted({self._file_to_cluster[file_id] for file_id in hit_files})
            cause = int(self._rng.choice(hit_clusters))
        self._causes[(commit.cycle, test)] = cause
        return outcome

    def _sample_delay(self) -> int:
        if self.cfg.delay_var == 0.0:
            return max(0, int(round(self.cfg.delay_mean)))
        delay = self._rng.normal(self.cfg.delay_mean, sqrt(self.cfg.delay_var))
        return max(0, int(round(delay)))

    def _apply_drift(self) -> None:
        self._drift_clusters()
        self._deps = {
            test: (
                self._sample_deps_for_test(test)
                if self._rng.random() < 0.25
                else self._refresh_deps_after_cluster_drift(self._deps[test])
            )
            for test in range(self.cfg.n_tests)
        }

    def _drift_clusters(self) -> None:
        if self.cfg.n_clusters == 1:
            return

        memberships = dict(self._file_to_cluster)
        cluster_sizes = [len(files) for files in self._clusters]
        n_moves = max(1, self.cfg.n_files // 10)
        for file_id in self._rng.choice(range(self.cfg.n_files), size=n_moves, replace=False):
            file_id = int(file_id)
            old_cluster = memberships[file_id]
            if cluster_sizes[old_cluster] <= 1:
                continue
            choices = [c for c in range(self.cfg.n_clusters) if c != old_cluster]
            new_cluster = int(self._rng.choice(choices))
            memberships[file_id] = new_cluster
            cluster_sizes[old_cluster] -= 1
            cluster_sizes[new_cluster] += 1

        grouped = [set() for _ in range(self.cfg.n_clusters)]
        for file_id, cluster_id in memberships.items():
            grouped[cluster_id].add(file_id)
        self._clusters = [frozenset(files) for files in grouped]
        self._rebuild_file_index()

    def _refresh_deps_after_cluster_drift(self, deps: frozenset[int]) -> frozenset[int]:
        kept = {
            file_id for file_id in deps if self._rng.random() >= 0.15
        }
        if kept:
            return frozenset(kept)
        return frozenset({int(self._rng.integers(0, self.cfg.n_files))})

    def _prune_causes(self, cycle: int) -> None:
        if self.cfg.cause_history_cycles is None:
            return
        cutoff = cycle - self.cfg.cause_history_cycles
        if cutoff <= 0:
            return
        self._causes = {
            key: cause
            for key, cause in self._causes.items()
            if key[0] >= cutoff
        }
