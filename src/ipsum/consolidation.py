"""Consolidate the prior without forgetting — EWC style.

Borrowed (see research/07-ewc.md). A Fisher-weighted quadratic anchor keeps
important parameters near their consolidated values as the prior updates on new
experience: "posterior of yesterday becomes prior for today." The Fisher
importance also doubles as a signal for what the abstraction store should
protect vs. release.

This module is a SKELETON. Interfaces are real; bodies are TODO.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Consolidator:
    lam: float = 1.0  # penalty strength
    _anchor: dict[str, float] = field(default_factory=dict)  # consolidated values
    _fisher: dict[str, float] = field(default_factory=dict)  # per-param importance

    def estimate_fisher(self, experiences) -> None:
        """Diagonal Fisher information over recent experiences. TODO."""
        raise NotImplementedError

    def consolidate(self, params: dict[str, float]) -> None:
        """Snapshot current params as the new anchor. TODO."""
        raise NotImplementedError

    def penalty(self, params: dict[str, float]) -> float:
        """sum_i lam/2 * F_i * (param_i - anchor_i)^2. TODO."""
        raise NotImplementedError
