"""Inspectable abstraction store — admit, score, and EVICT abstractions.

This is where ipsum departs from the literature. DreamCoder and Voyager only
ever ADD abstractions, and assume exact/immediate outcomes. ipsum must:

  - admit under UNCERTAINTY (Bayesian model selection, not exact MDL)
  - track each abstraction's usefulness with decay
  - EVICT stale abstractions as the domain drifts

See DESIGN.md sections 3.1 and 3.3.

This module is a SKELETON. Interfaces are real; bodies are TODO.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Abstraction:
    """An explicit, human-readable abstraction.

    ``name`` and ``payload`` are deliberately inspectable — the bet is on
    abstractions you can read, not weights you can't.
    """

    name: str
    payload: object
    complexity: float  # c(a): the description cost the abstraction must earn back
    usefulness: float = 0.0  # decaying trace of held-out predictive-LL contribution


@dataclass
class AbstractionStore:
    decay: float = 0.99
    _items: dict[str, Abstraction] = field(default_factory=dict)

    def candidates(self) -> list[Abstraction]:
        """Propose new candidate abstractions from recent patterns. TODO."""
        raise NotImplementedError

    def admit(self, a: Abstraction, ll_gain: float) -> bool:
        """Admit iff held-out predictive-LL gain exceeds complexity cost.

        Bayesian model selection replaces DreamCoder's exact MDL criterion:
        admit iff ``ll_gain - a.complexity > threshold`` with confidence. TODO.
        """
        raise NotImplementedError

    def reinforce(self, name: str, ll_gain: float) -> None:
        """Update an abstraction's usefulness trace from a (credited) outcome. TODO."""
        raise NotImplementedError

    def decay_and_evict(self) -> list[str]:
        """Decay all usefulness traces; evict those that fall below their cost.

        Returns the names evicted. This is the anti-staleness mechanism that
        neither DreamCoder nor Voyager has. TODO.
        """
        raise NotImplementedError
