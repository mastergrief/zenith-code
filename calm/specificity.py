"""
Auto-CALM Specificity Check — detect generic advice that lacks actionable detail.

Models give platitudes: "use caching", "add indexes", "write tests", "follow
best practices" — without specifics. This module detects low-specificity
advice and flags it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class GenericAdvice:
    """A piece of advice that's too generic to be actionable."""
    text: str
    category: str       # "platitude", "generic_tool", "missing_details", "hand_wave"
    suggestion: str     # how to make it specific
    severity: float


@dataclass
class SpecificityResult:
    """Result of specificity analysis."""
    issues: List[GenericAdvice] = field(default_factory=list)
    specific_count: int = 0     # count of specific/actionable items
    generic_count: int = 0      # count of generic items
    score: float = 1.0          # 1.0 = highly specific

    def summary(self) -> str:
        if not self.issues:
            return f"specific ({self.specific_count} actionable items)"
        return (f"{len(self.issues)} generic items "
                f"({self.specific_count} specific, {self.generic_count} generic)")


# Generic advice patterns — things that sound helpful but aren't actionable
_GENERIC_PATTERNS = [
    # Performance platitudes
    (r'\b(?:use\s+caching)\b(?!\s+(?:with|via|using|like|such as|e\.g\.|for example))',
     "platitude", "Specify: what to cache (queries? sessions? API responses?), what cache (Redis? in-memory?), TTL strategy"),
    (r'\b(?:add\s+(?:an?\s+)?index(?:es)?)\b(?!\s+(?:on|for|to|like))',
     "platitude", "Specify: which column(s), what index type (B-tree? GIN? partial?), expected query pattern"),
    (r'\b(?:optimize\s+(?:your|the)\s+(?:code|query|queries|database|performance))\b',
     "platitude", "Specify: which part, what metric to improve, what profiling showed"),
    (r'\b(?:scale\s+(?:horizontally|vertically))\b(?!\s+(?:by|using|with|via))',
     "platitude", "Specify: what component to scale, expected load, scaling mechanism (sharding? read replicas?)"),

    # Testing platitudes
    (r'\b(?:write\s+(?:more\s+)?(?:unit\s+)?tests)\b(?!\s+(?:for|that|which|to))',
     "platitude", "Specify: which functions, what edge cases, what coverage target"),
    (r'\b(?:follow\s+(?:best|coding|industry)\s+practices)\b',
     "platitude", "Name the specific practices and why they apply here"),
    (r'\b(?:use\s+(?:proper|good|better)\s+(?:error|exception)\s+handling)\b',
     "platitude", "Specify: which errors, how to handle each (retry? fallback? propagate?)"),

    # Architecture hand-waves
    (r'\b(?:use\s+(?:a\s+)?(?:microservice|microservices?))\b(?!\s+(?:for|to|because|when))',
     "hand_wave", "Specify: which services to extract, how they communicate, what data they own"),
    (r'\b(?:use\s+(?:a\s+)?(?:message\s+queue|event\s+bus))\b(?!\s+(?:like|such as|e\.g\.|for))',
     "hand_wave", "Specify: which queue (RabbitMQ? Kafka? SQS?), what messages, delivery guarantees"),
    (r'\b(?:add\s+(?:a\s+)?(?:load\s+balancer|reverse\s+proxy))\b(?!\s+(?:like|such as|using))',
     "hand_wave", "Specify: L4 vs L7, what balancing algorithm, health check strategy"),
    (r'\b(?:use\s+(?:a\s+)?(?:CDN))\b(?!\s+(?:like|such as|for|to))',
     "hand_wave", "Specify: which CDN, what to cache (static? API?), invalidation strategy"),

    # Security platitudes
    (r'\b(?:sanitize\s+(?:user\s+)?input)\b(?!\s+(?:using|with|by|via))',
     "platitude", "Specify: what sanitization (parameterized queries? HTML escaping? allowlist?)"),
    (r'\b(?:use\s+(?:encryption|HTTPS))\b(?!\s+(?:with|for|because|via))',
     "platitude", "Specify: encrypt what (at rest? in transit?), what algorithm, key management"),
    (r'\b(?:implement\s+(?:authentication|authorization|auth))\b(?!\s+(?:using|with|via))',
     "platitude", "Specify: what mechanism (JWT? sessions? OAuth2?), where to enforce, token storage"),

    # Process platitudes
    (r'\b(?:do\s+(?:a\s+)?code\s+review)\b',
     "platitude", "Specify: what to look for, checklist items, review scope"),
    (r'\b(?:refactor\s+(?:the|your)\s+code)\b(?!\s+(?:to|by|into))',
     "platitude", "Specify: what refactoring (extract method? rename? decompose class?), why"),
]

# Patterns indicating specific/actionable advice
_SPECIFIC_PATTERNS = [
    r'\b(?:for example|e\.g\.|such as|like|specifically|in particular)\b',
    r'\b(?:step\s+\d|first|second|third|then|next|finally)\b',
    r'```',  # code blocks indicate specificity
    r'\b(?:because|since|this is because|the reason)\b',
    r'\b\d+\s*(?:ms|seconds?|MB|GB|%|requests?|users?|queries)\b',  # concrete numbers
    r'\b(?:CREATE\s+INDEX|SELECT|INSERT|ALTER|DROP)\b',  # actual SQL
    r'\b(?:import|from|def |class |function |const |let |var )\b',  # actual code
]


class SpecificityChecker:
    """Detect generic advice that lacks actionable specifics."""

    def check(self, response: str) -> SpecificityResult:
        result = SpecificityResult()
        text = str(response)

        # Count specific indicators
        for pattern in _SPECIFIC_PATTERNS:
            result.specific_count += len(re.findall(pattern, text, re.IGNORECASE))

        # Find generic advice
        for pattern, category, suggestion in _GENERIC_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                result.generic_count += 1
                result.issues.append(GenericAdvice(
                    text=m.group(0),
                    category=category,
                    suggestion=suggestion,
                    severity=0.5,
                ))

        # Deduplicate
        seen = set()
        unique = []
        for issue in result.issues:
            key = issue.text.lower()
            if key not in seen:
                seen.add(key)
                unique.append(issue)
        result.issues = unique

        # Score: ratio of specific to generic
        total = result.specific_count + result.generic_count
        if total > 0:
            result.score = round(result.specific_count / total, 2)
        else:
            result.score = 0.5  # no signals either way

        return result
