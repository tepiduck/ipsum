"""Amortized prior over a domain — Conditional Neural Process style.

Borrowed, not novel (see research/03-neural-processes.md). The point is:
conditioning on a new experience is an O(1) update of a pooled latent, and the
predictor returns calibrated uncertainty we can use as a selection signal.

This module is a SKELETON. Interfaces are real; bodies are TODO.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence


class Experience(Protocol):
    """One observed interaction with the domain.

    For the v1 test-selection wedge: a (code_change, test, outcome, context) tuple.
    """

    ...


@dataclass
class Prediction:
    """A decision-relevant prediction with uncertainty."""

    mean: float
    std: float


@dataclass
class AmortizedPrior:
    """Encode each experience -> pool into a latent -> decode predictions.

    The pooled latent ``z`` IS the prior state. ``condition`` folds a new
    experience into ``z`` in O(1) (running mean), which is the cheap recursive
    update at the heart of the thesis.
    """

    latent_dim: int = 64
    _z: list[float] = field(default_factory=list)
    _n: int = 0

    def encode(self, e: Experience) -> Sequence[float]:
        """h_theta(e) -> r_i. TODO: learned encoder."""
        raise NotImplementedError

    def condition(self, e: Experience) -> None:
        """Fold one experience into the pooled latent (running mean). O(1)."""
        raise NotImplementedError

    def predict(self, query: Experience) -> Prediction:
        """g_theta(query, z) -> (mean, std). TODO: learned decoder."""
        raise NotImplementedError
