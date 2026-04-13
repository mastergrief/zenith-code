"""
Auto-CALM Error Recovery — graceful degradation when knowledge is lacking.

Instead of hallucinating or refusing, provide useful partial information
with explicit uncertainty markers. The difference between "I don't know"
(useless) and "I'm not certain, but here's what I do know, and here's
how to find out" (useful).

Usage:
    from calm.error_recovery import ErrorRecovery
    er = ErrorRecovery()
    result = er.assess_response("I don't have information about that.")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class RecoveryAssessment:
    """Assessment of how well the model handles uncertainty."""
    has_refusal: bool = False       # "I can't help with that"
    has_hallucination_risk: bool = False  # confident without evidence
    has_partial_info: bool = False  # provides what it does know
    has_next_steps: bool = False    # suggests how to find out
    has_uncertainty_marker: bool = False  # explicitly flags uncertainty
    recovery_quality: str = "none"  # "none", "poor", "adequate", "good"
    suggestions: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return f"recovery={self.recovery_quality}, refusal={self.has_refusal}, partial={self.has_partial_info}, next_steps={self.has_next_steps}"


# Refusal patterns
_REFUSAL_PATTERNS = [
    re.compile(r"I (?:can't|cannot|don't|do not|am not able to|am unable to)\s+(?:help|assist|provide|answer|find|access|determine)", re.IGNORECASE),
    re.compile(r"I (?:don't|do not) have (?:information|data|knowledge|access)", re.IGNORECASE),
    re.compile(r"(?:not available|no (?:information|data) available|outside my (?:scope|knowledge|training))", re.IGNORECASE),
    re.compile(r"I'm (?:sorry|afraid) (?:but )?I (?:can't|cannot|don't)", re.IGNORECASE),
]

# Hallucination risk patterns (confident without hedging on uncertain topics)
_HALLUCINATION_RISK = [
    re.compile(r"(?:definitely|certainly|absolutely|undoubtedly|without question)\s+.{10,}", re.IGNORECASE),
    re.compile(r"(?:the answer is|the solution is|you should definitely)\s+", re.IGNORECASE),
]

# Partial information patterns
_PARTIAL_INFO = [
    re.compile(r"(?:what I (?:do |can )know|here's what I know|I can tell you that|based on what I know)", re.IGNORECASE),
    re.compile(r"(?:while I (?:can't|don't).*?,\s*I (?:can|do))", re.IGNORECASE),
    re.compile(r"(?:however|that said|nonetheless),?\s+(?:I can|there are|you could)", re.IGNORECASE),
]

# Next steps patterns
_NEXT_STEPS = [
    re.compile(r"(?:you (?:could|might|should|can) (?:try|check|look|consult|refer|search|ask|read))", re.IGNORECASE),
    re.compile(r"(?:I recommend|I suggest|consider|try)\s+(?:checking|looking|consulting|reading|searching)", re.IGNORECASE),
    re.compile(r"(?:the documentation|official docs|manual|reference|source)\s+(?:for|at|on)", re.IGNORECASE),
    re.compile(r"(?:for more (?:information|details|context)|to learn more|for the latest)", re.IGNORECASE),
]

# Uncertainty markers
_UNCERTAINTY_MARKERS = [
    re.compile(r"(?:I'm not (?:certain|sure|confident)|I believe|to my knowledge|as far as I know)", re.IGNORECASE),
    re.compile(r"(?:this may|this might|this could|it's possible|it appears|it seems)", re.IGNORECASE),
    re.compile(r"(?:please verify|double-check|confirm this|take this with)", re.IGNORECASE),
]


class ErrorRecovery:
    """Assesses how well the model handles uncertainty and edge cases."""

    def assess_response(self, response: str) -> RecoveryAssessment:
        """Assess recovery quality of a response."""
        result = RecoveryAssessment()

        # Check each pattern category
        result.has_refusal = any(p.search(response) for p in _REFUSAL_PATTERNS)
        result.has_hallucination_risk = any(p.search(response) for p in _HALLUCINATION_RISK)
        result.has_partial_info = any(p.search(response) for p in _PARTIAL_INFO)
        result.has_next_steps = any(p.search(response) for p in _NEXT_STEPS)
        result.has_uncertainty_marker = any(p.search(response) for p in _UNCERTAINTY_MARKERS)

        # Determine quality
        if result.has_refusal:
            if result.has_partial_info and result.has_next_steps:
                result.recovery_quality = "good"
            elif result.has_partial_info or result.has_next_steps:
                result.recovery_quality = "adequate"
            else:
                result.recovery_quality = "poor"
                result.suggestions.append("Instead of refusing, share what you DO know and suggest where to find the rest.")
        elif result.has_hallucination_risk and not result.has_uncertainty_marker:
            result.recovery_quality = "poor"
            result.suggestions.append("High confidence without evidence — add uncertainty markers or cite sources.")
        else:
            result.recovery_quality = "good"

        return result

    def suggest_recovery(self, topic: str) -> str:
        """Generate a recovery template for a topic the model isn't sure about."""
        return (
            f"While I'm not certain about the specifics of {topic}, "
            f"here's what I do know: [partial information]. "
            f"To get a definitive answer, you could: "
            f"1) check the official documentation, "
            f"2) search for recent discussions, or "
            f"3) test it directly in your environment."
        )
