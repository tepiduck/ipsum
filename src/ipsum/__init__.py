"""ipsum — expertise as a cheaply-updated, inspectable prior that compounds.

The package is intentionally organized around the four moving parts of the thesis:

    prior          -- an amortized prior over "what matters" (CNP-style)
    abstractions   -- an inspectable store; admit / evict abstractions
    consolidation  -- update the prior without forgetting (EWC-style)
    credit         -- assign delayed, noisy outcomes back to abstractions

See DESIGN.md for how these compose and which parts are borrowed vs. open.
"""

__version__ = "0.0.1"

__all__ = ["prior", "abstractions", "consolidation", "credit"]
