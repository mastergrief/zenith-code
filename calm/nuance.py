"""
Auto-CALM Nuance — context-dependent qualification detection.

Detects when an answer genuinely "depends" and forces structured
branching instead of letting the model hedge vaguely. When the engine
detects multiple valid interpretations or context-dependent answers,
it produces a decision tree rather than a single answer.

Usage:
    from calm.nuance import NuanceDetector
    nd = NuanceDetector()
    result = nd.analyze("Is Python faster than JavaScript?")
    print(result.needs_qualification)  # True
    print(result.dimensions)  # ["runtime", "use case", "measurement"]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class QualificationBranch:
    """One branch of a context-dependent answer."""
    condition: str    # "If X..." or "For Y..."
    answer: str       # the answer under this condition
    confidence: float = 0.0  # 0-1


@dataclass
class NuanceResult:
    """Result of nuance analysis."""
    needs_qualification: bool = False
    dimensions: List[str] = field(default_factory=list)
    branches: List[QualificationBranch] = field(default_factory=list)
    hedges_detected: List[str] = field(default_factory=list)
    summary: str = ""

    @property
    def is_well_qualified(self) -> bool:
        """Whether the answer properly addresses multiple dimensions."""
        return len(self.branches) >= 2 or not self.needs_qualification


# Patterns that indicate the answer depends on context
_DEPENDS_PATTERNS = [
    re.compile(r'\b(?:it depends|depends on|that depends|depending on)\b', re.IGNORECASE),
    re.compile(r'\b(?:it varies|varies by|varies depending)\b', re.IGNORECASE),
    re.compile(r'\b(?:in some cases|in certain|under certain|in specific)\b', re.IGNORECASE),
    re.compile(r'\b(?:not always|not necessarily|not in all cases)\b', re.IGNORECASE),
    re.compile(r'\b(?:generally|typically|usually|often|sometimes|rarely)\b', re.IGNORECASE),
]

# Patterns for vague hedging (bad — model should commit or branch)
_HEDGE_PATTERNS = [
    re.compile(r'\b(?:sort of|kind of|more or less|somewhat|arguably)\b', re.IGNORECASE),
    re.compile(r'\b(?:it.s complicated|it.s complex|it.s nuanced)\b', re.IGNORECASE),
    re.compile(r'\b(?:you could say|one might argue|it could be argued)\b', re.IGNORECASE),
    re.compile(r'\b(?:in a way|in some sense|to some extent)\b', re.IGNORECASE),
    re.compile(r'\b(?:probably|possibly|perhaps|maybe|might be)\b', re.IGNORECASE),
]

# Patterns that indicate proper qualification (good — structured branching)
_BRANCH_PATTERNS = [
    re.compile(r'\b(?:if|when)\s+.+?,\s*(?:then\s+)?.+', re.IGNORECASE),
    re.compile(r'\b(?:for|in)\s+.+?,\s*.+', re.IGNORECASE),
    re.compile(r'\b(?:however|on the other hand|conversely|alternatively)\b', re.IGNORECASE),
    re.compile(r'\b(?:in contrast|by comparison|whereas)\b', re.IGNORECASE),
]

# Questions that inherently need qualification
_INHERENTLY_CONTEXTUAL = [
    re.compile(r'\b(?:faster|slower|better|worse|easier|harder)\s+than\b', re.IGNORECASE),
    re.compile(r'\b(?:should I use|which is better|what.s the best)\b', re.IGNORECASE),
    re.compile(r'\b(?:is it worth|is it good|is it bad)\b', re.IGNORECASE),
    re.compile(r'\b(?:pros and cons|advantages|disadvantages|tradeoffs?)\b', re.IGNORECASE),
]


class NuanceDetector:
    """Detects when answers need qualification and verifies branching."""

    def analyze_prompt(self, prompt: str) -> NuanceResult:
        """Analyze a prompt to determine if the answer needs qualification."""
        result = NuanceResult()

        # Check if the question inherently needs qualification
        for pat in _INHERENTLY_CONTEXTUAL:
            if pat.search(prompt):
                result.needs_qualification = True
                break

        # Extract comparison dimensions from the prompt
        result.dimensions = self._extract_dimensions(prompt)
        if len(result.dimensions) > 1:
            result.needs_qualification = True

        if result.needs_qualification:
            result.summary = (
                f"This question needs qualification across: {', '.join(result.dimensions) or 'context'}"
            )
        else:
            result.summary = "Direct answer appropriate"

        return result

    def analyze_response(self, response: str) -> NuanceResult:
        """Analyze a response for hedge quality — does it qualify properly?"""
        result = NuanceResult()

        # Detect hedging
        for pat in _DEPENDS_PATTERNS:
            for m in pat.finditer(response):
                result.needs_qualification = True

        for pat in _HEDGE_PATTERNS:
            for m in pat.finditer(response):
                result.hedges_detected.append(m.group(0))

        # Detect proper branching
        for pat in _BRANCH_PATTERNS:
            for m in pat.finditer(response):
                result.branches.append(QualificationBranch(
                    condition=m.group(0)[:80],
                    answer="(extracted from context)",
                ))

        # Score
        if result.needs_qualification:
            if result.branches:
                result.summary = (
                    f"Properly qualified: {len(result.branches)} branches"
                )
                if result.hedges_detected:
                    result.summary += f", but {len(result.hedges_detected)} vague hedges remain"
            else:
                result.summary = (
                    f"Says 'it depends' but doesn't branch — "
                    f"{len(result.hedges_detected)} vague hedges without structure"
                )
        else:
            result.summary = "Direct answer (no qualification needed)"

        return result

    def _extract_dimensions(self, text: str) -> List[str]:
        """Extract comparison dimensions from a prompt."""
        dimensions = []

        # "X vs Y" or "X or Y"
        vs_match = re.search(r'(\w+)\s+(?:vs\.?|versus|or|compared to)\s+(\w+)', text, re.IGNORECASE)
        if vs_match:
            dimensions.extend([vs_match.group(1), vs_match.group(2)])

        # Common dimension keywords
        dim_keywords = {
            "performance": ["fast", "slow", "speed", "latency", "throughput", "performance"],
            "scalability": ["scale", "scalab", "large", "growth", "concurrent"],
            "cost": ["cost", "price", "expensive", "cheap", "free", "budget"],
            "complexity": ["complex", "simple", "easy", "hard", "difficult", "learn"],
            "ecosystem": ["library", "framework", "community", "support", "ecosystem"],
            "use case": ["use case", "scenario", "application", "context", "situation"],
        }
        for dim, keywords in dim_keywords.items():
            if any(kw in text.lower() for kw in keywords):
                dimensions.append(dim)

        return list(dict.fromkeys(dimensions))  # deduplicate preserving order

    def qualify_check(self, prompt: str, response: str) -> str:
        """Full check: does the response properly qualify a context-dependent question?"""
        prompt_analysis = self.analyze_prompt(prompt)
        response_analysis = self.analyze_response(response)

        if not prompt_analysis.needs_qualification:
            return "OK: question doesn't require qualification"

        if response_analysis.is_well_qualified:
            return f"OK: {response_analysis.summary}"

        return (
            f"NEEDS WORK: Question requires qualification across "
            f"{', '.join(prompt_analysis.dimensions) or 'context'}, but response "
            f"{response_analysis.summary}"
        )
