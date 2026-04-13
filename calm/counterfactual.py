"""
Auto-CALM Counterfactual Reasoning — trace alternative timelines.

"What would happen if X were different?" is crucial for debugging,
planning, and risk assessment. This module structures counterfactual
analysis by identifying the change, tracing dependencies, and
predicting downstream effects.

Usage:
    from calm.counterfactual import CounterfactualEngine
    ce = CounterfactualEngine()
    result = ce.analyze("What if we used PostgreSQL instead of MySQL?")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CounterfactualScenario:
    """A structured counterfactual analysis."""
    original: str            # "We use MySQL"
    alternative: str         # "We use PostgreSQL"
    change_type: str = ""   # "replacement", "removal", "addition", "modification"
    affected_dimensions: List[str] = field(default_factory=list)
    likely_effects: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    benefits: List[str] = field(default_factory=list)
    unknowns: List[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"Change: {self.original} → {self.alternative} ({self.change_type})"]
        if self.affected_dimensions:
            parts.append(f"Affects: {', '.join(self.affected_dimensions)}")
        if self.benefits:
            parts.append(f"Benefits: {', '.join(self.benefits[:3])}")
        if self.risks:
            parts.append(f"Risks: {', '.join(self.risks[:3])}")
        if self.unknowns:
            parts.append(f"Unknowns: {', '.join(self.unknowns[:3])}")
        return ". ".join(parts)


_COUNTERFACTUAL_PATTERNS = [
    # "What if X instead of Y?"
    re.compile(r'[Ww]hat if (?:we |I )?(?:used?|chose?|picked?|went with)\s+(.{3,40}?)\s+instead of\s+(.{3,40}?)(?:\?|$)', re.IGNORECASE),
    # "What would happen if X?"
    re.compile(r'[Ww]hat would happen if\s+(.{5,60}?)(?:\?|$)', re.IGNORECASE),
    # "What if X didn't/doesn't Y?"
    re.compile(r'[Ww]hat if\s+(.{3,30}?)\s+(?:didn.t|doesn.t|wasn.t|isn.t|weren.t|aren.t)\s+(.{3,40}?)(?:\?|$)', re.IGNORECASE),
    # "What if X were/was Y?"
    re.compile(r'[Ww]hat if\s+(.{3,30}?)\s+(?:were|was)\s+(.{3,40}?)(?:\?|$)', re.IGNORECASE),
    # "If we had X, ..."
    re.compile(r'[Ii]f (?:we|I) had\s+(.{5,40}?)(?:,|$)', re.IGNORECASE),
]

_DIMENSION_KEYWORDS = {
    "performance": ["speed", "latency", "throughput", "fast", "slow", "optimization"],
    "cost": ["cost", "price", "budget", "expensive", "cheap", "license", "hosting"],
    "complexity": ["complex", "simple", "easy", "hard", "learning curve", "maintain"],
    "scalability": ["scale", "growth", "capacity", "horizontal", "vertical"],
    "compatibility": ["compatible", "integrate", "migration", "backward", "ecosystem"],
    "reliability": ["reliable", "uptime", "fault", "crash", "recovery", "redundancy"],
    "security": ["secure", "vulnerability", "attack", "encrypt", "auth"],
    "developer_experience": ["DX", "tooling", "debug", "documentation", "community"],
}


class CounterfactualEngine:
    """Structures counterfactual analysis."""

    def analyze(self, question: str) -> CounterfactualScenario:
        """Analyze a counterfactual question."""
        scenario = CounterfactualScenario(
            original="current state",
            alternative="proposed change",
        )

        # Extract the counterfactual
        for pat in _COUNTERFACTUAL_PATTERNS:
            m = pat.search(question)
            if m:
                groups = m.groups()
                if len(groups) == 2:
                    scenario.alternative = groups[0].strip()
                    scenario.original = groups[1].strip()
                    scenario.change_type = "replacement"
                elif len(groups) == 1:
                    scenario.alternative = groups[0].strip()
                    scenario.change_type = "modification"
                break

        # Detect affected dimensions
        question_lower = question.lower()
        for dim, keywords in _DIMENSION_KEYWORDS.items():
            if any(kw in question_lower for kw in keywords):
                scenario.affected_dimensions.append(dim)

        # If no specific dimensions detected, suggest common ones
        if not scenario.affected_dimensions:
            scenario.affected_dimensions = ["performance", "cost", "complexity", "compatibility"]

        # Generate structured analysis prompts
        scenario.unknowns = [
            f"How would {dim} be affected by changing from {scenario.original} to {scenario.alternative}?"
            for dim in scenario.affected_dimensions
        ]

        return scenario

    def compare_scenarios(self, scenarios: List[CounterfactualScenario]) -> str:
        """Compare multiple counterfactual scenarios."""
        if not scenarios:
            return "No scenarios to compare"

        lines = ["Scenario comparison:"]
        all_dims = set()
        for s in scenarios:
            all_dims.update(s.affected_dimensions)

        for dim in sorted(all_dims):
            lines.append(f"\n  {dim}:")
            for s in scenarios:
                affected = "affected" if dim in s.affected_dimensions else "unaffected"
                lines.append(f"    {s.alternative}: {affected}")

        return "\n".join(lines)

    def generate_analysis_plan(self, scenario: CounterfactualScenario) -> List[str]:
        """Generate a plan for analyzing a counterfactual scenario."""
        plan = [
            f"1. Baseline: document current state with {scenario.original}",
            f"2. Change: identify all touchpoints where {scenario.original} is used",
            f"3. Impact: for each touchpoint, assess effect of switching to {scenario.alternative}",
        ]
        for i, dim in enumerate(scenario.affected_dimensions, 4):
            plan.append(f"{i}. Assess {dim} impact specifically")
        plan.append(f"{len(scenario.affected_dimensions) + 4}. Decision: weigh benefits vs risks vs unknowns")
        return plan
