"""Delayed, noisy credit assignment.

The hardest open problem (DESIGN.md 3.2) and the one no borrowed paper solves
for this setting. CI outcomes arrive minutes-to-hours after the decision and are
label-noisy (flaky tests). We must map a landed outcome back to the
abstractions that informed the decision, AFTER de-flaking.

This module is a SKELETON. Interfaces are real; bodies are TODO.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PendingDecision:
    decision_id: str
    used_abstractions: list[str]
    timestamp: float


@dataclass
class CreditAssigner:
    """Holds eligibility for in-flight decisions until outcomes land."""

    _pending: dict[str, PendingDecision] = field(default_factory=dict)

    def record(self, decision_id: str, used_abstractions: list[str], t: float) -> None:
        """Open an eligibility entry when a decision is made. TODO."""
        raise NotImplementedError

    def deflake(self, raw_outcome) -> bool:
        """Resolve a possibly-flaky outcome into a trustworthy label.

        Predictive Test Selection retries a failure up to 10x and only counts
        consistent failures. TODO.
        """
        raise NotImplementedError

    def settle(self, decision_id: str, outcome) -> dict[str, float]:
        """Match a landed (de-flaked) outcome to its decision and return the
        per-abstraction LL-gain credit to apply. TODO."""
        raise NotImplementedError
