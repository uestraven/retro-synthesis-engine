"""Composition planning and generation.

Phase 3's rule-based composer lives here. It exists before any model so that
rendering and export can be proven independently of model quality, and so that
Phase 4 has a baseline it must beat (proposal 11.3).
"""

from rse.generation.loops import LoopReport, validate_loop
from rse.generation.plan import CompositionPlan, PlanSummary, make_plan
from rse.generation.procedural import GeneratedComposition, generate
from rse.generation.theory import Key

__all__ = [
    "CompositionPlan",
    "GeneratedComposition",
    "Key",
    "LoopReport",
    "PlanSummary",
    "generate",
    "make_plan",
    "validate_loop",
]
