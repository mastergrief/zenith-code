"""
Auto-CALM Abstraction Ladder — detect and adjust abstraction level.

Conversations fail when they're at the wrong level: too abstract =
hand-wavy, too concrete = missing the forest for the trees. This
module detects the current level and suggests when to move up or down.

Levels (low → high):
  1. Implementation: specific code, line numbers, exact values
  2. Component: functions, classes, modules, APIs
  3. Architecture: systems, services, data flow, patterns
  4. Concept: principles, tradeoffs, mental models
  5. Strategy: goals, priorities, constraints, business value

Usage:
    from calm.abstraction import AbstractionDetector
    ad = AbstractionDetector()
    level = ad.detect("Use a HashMap<String, Vec<u32>> for the cache")
    # → "implementation"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class AbstractionAssessment:
    """Assessment of abstraction level."""
    level: str              # implementation/component/architecture/concept/strategy
    level_number: int       # 1-5
    confidence: float       # 0-1
    mismatch: bool = False  # True if question and answer are at different levels
    suggestion: str = ""    # how to adjust

    def __str__(self):
        return f"level={self.level} ({self.level_number}/5), confidence={self.confidence:.0%}"


# Signals for each abstraction level
_LEVEL_SIGNALS = {
    "implementation": [
        re.compile(r'\b(?:line \d+|byte|offset|pointer|register|instruction|opcode)\b', re.IGNORECASE),
        re.compile(r'\b(?:int|float|string|bool|char|void|null|undefined|None)\b'),
        re.compile(r'[{}\[\]();]'),  # code syntax
        re.compile(r'\b(?:for|while|if|else|return|break|continue|switch|case)\b'),
        re.compile(r'\b(?:0x[0-9a-f]+|\d{3,})\b', re.IGNORECASE),  # hex/large numbers
        re.compile(r'(?:\.py|\.js|\.rs|\.go|\.java|\.cpp|\.ts)\b'),  # file extensions
        re.compile(r'[A-Z][a-z]+[A-Z]|[a-z]+_[a-z]+'),  # camelCase/snake_case identifiers
    ],
    "component": [
        re.compile(r'\b(?:function|method|class|module|package|library|framework|API|endpoint)\b', re.IGNORECASE),
        re.compile(r'\b(?:interface|abstract|inherit|implement|extend|import|export)\b', re.IGNORECASE),
        re.compile(r'\b(?:constructor|destructor|getter|setter|factory|singleton)\b', re.IGNORECASE),
        re.compile(r'\b(?:parameter|argument|return value|exception|error handling)\b', re.IGNORECASE),
    ],
    "architecture": [
        re.compile(r'\b(?:service|microservice|monolith|server|client|database|cache|queue|broker)\b', re.IGNORECASE),
        re.compile(r'\b(?:REST|GraphQL|gRPC|WebSocket|HTTP|TCP|UDP)\b', re.IGNORECASE),
        re.compile(r'\b(?:load balancer|reverse proxy|CDN|container|kubernetes|docker)\b', re.IGNORECASE),
        re.compile(r'\b(?:event.driven|message.queue|pub.sub|CQRS|event.sourcing)\b', re.IGNORECASE),
        re.compile(r'\b(?:data flow|pipeline|middleware|orchestrat|choreograph)\b', re.IGNORECASE),
    ],
    "concept": [
        re.compile(r'\b(?:principle|pattern|paradigm|philosophy|approach|methodology|practice)\b', re.IGNORECASE),
        re.compile(r'\b(?:tradeoff|trade.off|balance|tension|compromise|constraint)\b', re.IGNORECASE),
        re.compile(r'\b(?:coupling|cohesion|encapsulation|abstraction|modularity|separation)\b', re.IGNORECASE),
        re.compile(r'\b(?:DRY|SOLID|KISS|YAGNI|composition over inheritance)\b', re.IGNORECASE),
        re.compile(r'\b(?:mental model|analogy|metaphor|framework|lens|perspective)\b', re.IGNORECASE),
    ],
    "strategy": [
        re.compile(r'\b(?:goal|objective|priority|roadmap|milestone|deadline|timeline)\b', re.IGNORECASE),
        re.compile(r'\b(?:stakeholder|user|customer|business|market|revenue|cost)\b', re.IGNORECASE),
        re.compile(r'\b(?:risk|opportunity|investment|ROI|impact|value|outcome)\b', re.IGNORECASE),
        re.compile(r'\b(?:strategy|tactic|initiative|decision|direction|vision)\b', re.IGNORECASE),
        re.compile(r'\b(?:why|should we|what if|long.term|short.term|sustainable)\b', re.IGNORECASE),
    ],
}

_LEVEL_NUMBERS = {
    "implementation": 1,
    "component": 2,
    "architecture": 3,
    "concept": 4,
    "strategy": 5,
}


class AbstractionDetector:
    """Detects and assesses abstraction levels in text."""

    def detect(self, text: str) -> AbstractionAssessment:
        """Detect the abstraction level of text."""
        scores = {}
        for level, patterns in _LEVEL_SIGNALS.items():
            count = sum(len(pat.findall(text)) for pat in patterns)
            scores[level] = count

        if not any(scores.values()):
            return AbstractionAssessment(
                level="component", level_number=2,
                confidence=0.3, suggestion="Could not determine abstraction level",
            )

        # Normalize scores
        total = sum(scores.values())
        best_level = max(scores, key=scores.get)
        confidence = scores[best_level] / total if total > 0 else 0

        return AbstractionAssessment(
            level=best_level,
            level_number=_LEVEL_NUMBERS[best_level],
            confidence=confidence,
        )

    def check_mismatch(self, question: str, answer: str) -> AbstractionAssessment:
        """Check if question and answer are at different abstraction levels."""
        q_level = self.detect(question)
        a_level = self.detect(answer)

        diff = abs(q_level.level_number - a_level.level_number)
        result = a_level

        if diff >= 2:
            result.mismatch = True
            if a_level.level_number < q_level.level_number:
                result.suggestion = (
                    f"Answer is too concrete ({a_level.level}) for the question "
                    f"({q_level.level}). Consider zooming out to address the "
                    f"higher-level concern first, then dive into details."
                )
            else:
                result.suggestion = (
                    f"Answer is too abstract ({a_level.level}) for the question "
                    f"({q_level.level}). Consider providing specific, actionable "
                    f"details at the {q_level.level} level."
                )

        return result

    def suggest_level(self, context: str) -> str:
        """Suggest the appropriate abstraction level for a context."""
        current = self.detect(context)

        suggestions = {
            "implementation": "Good for: code review, debugging, optimization. "
                            "Consider: does the user need this level of detail?",
            "component": "Good for: API design, refactoring, feature planning. "
                        "Consider: zoom in for implementation or out for architecture.",
            "architecture": "Good for: system design, scaling discussions, integration. "
                          "Consider: zoom in for component details or out for strategy.",
            "concept": "Good for: teaching, design principles, tradeoff analysis. "
                      "Consider: zoom in for actionable specifics.",
            "strategy": "Good for: planning, prioritization, stakeholder communication. "
                       "Consider: zoom in for how to actually implement.",
        }

        return suggestions.get(current.level, "Unknown level")
