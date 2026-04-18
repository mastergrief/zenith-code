"""Domain facades for the substrate.

A facade encapsulates: imports/exports (channel + sub-head allocations),
PT (NL → structure), Router (PT → compute dispatch), compute cards
(exact evaluation), and VerificationHook (compute → Gemma logits).
Install one facade per domain; multiple facades compose on the same
GemmaSubstrate by reserving disjoint channel / sub-head ranges.
"""

from calm.llm_computer.facades.hub_l23 import HubInjectionCard
from calm.llm_computer.facades.math_addition import MathAdditionFacade
from calm.llm_computer.facades.multi_step_composition import (
    MultiStepCompositionFacade,
    MultiStepCompositionResult,
)

__all__ = [
    "HubInjectionCard",
    "MathAdditionFacade",
    "MultiStepCompositionFacade",
    "MultiStepCompositionResult",
]
