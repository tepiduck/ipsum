"""Synthetic testbed — a controllable mock of "a codebase under CI".

Its ONLY purpose is to give every research mechanism a ground-truth oracle to be
validated against (RESEARCH.md §1). The model under test never sees the oracle;
evaluation does. Debug mechanisms HERE before touching RTPTorrent — on real data
a null result is ambiguous (bad mechanism vs. noise); here you know the answer.

Everything is seeded and reproducible.

This module is a SKELETON. Interfaces + knobs are fixed; bodies are TODO. It is a
good first agent task (Card A depends on it). Keep the oracle accessors strictly
separate from `step()` output so they can never leak into a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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

    # --- world setup -------------------------------------------------------
    def reset(self) -> None:
        """Sample true clusters, dependencies, and RNG from cfg.seed. TODO."""
        raise NotImplementedError

    def step(self) -> tuple[Commit, list[Outcome]]:
        """Advance one CI cycle.

        - sample a cluster-correlated commit
        - schedule each test's outcome under the p_hit / p_flaky model
        - apply any drift due this cycle
        - return this cycle's commit and the outcomes that become visible NOW
          (i.e. decided at cycle - delay). TODO.
        """
        raise NotImplementedError

    # --- ORACLE (evaluation only; must NOT leak to any model) ---------------
    def true_clusters(self) -> list[frozenset[int]]:
        """Ground-truth abstraction membership. Score admission against this. TODO."""
        raise NotImplementedError

    def true_deps(self, test: int) -> frozenset[int]:
        """Ground-truth dependency set for a test. TODO."""
        raise NotImplementedError

    def cause(self, outcome: Outcome) -> int | None:
        """Which cluster actually caused this failure (or None if flaky/noise).

        Score credit-assignment attribution against this. TODO.
        """
        raise NotImplementedError

    def drift_schedule(self) -> tuple[int, ...]:
        """Cycles at which the world mutates. Score eviction latency against this."""
        return self.cfg.drift_schedule
