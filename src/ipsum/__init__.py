"""ipsum — expertise as a cheaply-updated, inspectable prior that compounds.

The package is intentionally organized around the four moving parts of the thesis:

    prior          -- an amortized prior over "what matters" (CNP-style)
    abstractions   -- an inspectable store; admit / evict abstractions
    consolidation  -- update the prior without forgetting (EWC-style)
    credit         -- assign delayed, noisy outcomes back to abstractions
    synth          -- synthetic testbed with ground-truth oracles (debug here first)

See DESIGN.md for how these compose, RESEARCH.md for how to work the open
mechanisms, and research/00-synthesis.md for which parts are borrowed vs. open.
"""

__version__ = "0.0.1"

__all__ = ["prior", "abstractions", "consolidation", "credit", "synth"]
