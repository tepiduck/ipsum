"""Smoke tests — the package imports and core types construct.

Keeps CI green from day one without asserting on unimplemented mechanisms.
"""

import ipsum
from ipsum import abstractions, consolidation, credit, prior


def test_version():
    assert ipsum.__version__


def test_constructs():
    prior.AmortizedPrior()
    abstractions.AbstractionStore()
    consolidation.Consolidator()
    credit.CreditAssigner()


def test_abstraction_is_inspectable():
    a = abstractions.Abstraction(name="auth_cluster", payload={"files": ["a", "b"]}, complexity=1.0)
    assert a.name == "auth_cluster"
    assert a.usefulness == 0.0
