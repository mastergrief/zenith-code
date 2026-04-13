"""
Auto-CALM Confidence Check — detect overconfidence on uncertain topics.

Models state opinions as facts, express certainty about implementation details
they don't know, and fail to hedge appropriately on contested/nuanced topics.

This module detects:
1. Absolute certainty on debatable topics ("X is always better than Y")
2. Confident implementation claims without caveats ("X uses Y internally")
3. Missing uncertainty hedging on predictions/recommendations
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class ConfidenceIssue:
    """An overconfidence issue found in text."""
    text: str           # the problematic text
    issue_type: str     # "absolute", "unhedged_claim", "false_certainty", "missing_caveat"
    suggestion: str     # how to fix it
    severity: float     # 0-1


@dataclass
class ConfidenceCheckResult:
    """Result of confidence analysis."""
    issues: List[ConfidenceIssue] = field(default_factory=list)
    hedging_count: int = 0      # how many hedges the response uses
    absolute_count: int = 0     # how many absolutes
    score: float = 1.0          # 1.0 = well-calibrated

    def summary(self) -> str:
        if not self.issues:
            return f"well-calibrated ({self.hedging_count} hedges, {self.absolute_count} absolutes)"
        return (f"{len(self.issues)} overconfidence issues "
                f"({self.absolute_count} absolutes, {self.hedging_count} hedges)")


# Patterns indicating overconfidence
_ABSOLUTE_PATTERNS = [
    (r'\b(?:always|never|impossible|guaranteed|certainly|definitely|undoubtedly|unquestionably)\b',
     "absolute", "Consider: are there exceptions? Add qualifiers like 'typically', 'in most cases'"),
    (r'\b(?:the\s+(?:only|best|worst|fastest|slowest|correct|right|wrong)\s+(?:way|approach|method|option|choice|solution))\b',
     "absolute", "Consider: 'one of the best' or 'a common approach' instead of 'the only/best'"),
    (r'\b(?:you\s+(?:must|should)\s+(?:always|never))\b',
     "absolute", "Replace with 'generally should' or 'in most cases should'"),
    (r'\b(?:there\s+is\s+no\s+(?:reason|way|need|point))\b',
     "absolute", "Consider edge cases where this might not hold"),
    (r'\b(?:everyone\s+(?:knows?|agrees?|uses?))\b',
     "false_certainty", "This is an appeal to popularity, not evidence"),
    (r'\b(?:it\s+is\s+(?:well.known|obvious|clear|evident)\s+that)\b',
     "false_certainty", "If it were obvious, the question wouldn't be asked"),
]

# Patterns indicating appropriate hedging
_HEDGE_PATTERNS = [
    r'\b(?:typically|usually|generally|often|commonly|in most cases)\b',
    r'\b(?:it depends|depends on|trade.?off|consider|context)\b',
    r'\b(?:may|might|could|can|tend to|likely|probably)\b',
    r'\b(?:in my (?:experience|opinion)|one approach|one option)\b',
    r'\b(?:however|although|but|that said|on the other hand)\b',
    r'\b(?:for most|for many|for some|in some cases|sometimes)\b',
]

# Topics where confidence should be LOW (debatable/context-dependent)
_DEBATABLE_TOPICS = [
    (r'(?:which|what)\s+(?:language|framework|database|tool)\s+(?:is|should)',
     "missing_caveat", "Tool/language choice is context-dependent — mention the tradeoffs"),
    (r'(?:best\s+practice|best way|right way)\s+(?:to|for)',
     "missing_caveat", "'Best practice' varies by context — qualify with 'for X type of projects'"),
    (r'(?:should\s+(?:I|you|we)\s+use)\s+(?:microservices|monolith|serverless|NoSQL|SQL)',
     "missing_caveat", "Architecture choice depends heavily on team, scale, and requirements"),
]


class ConfidenceChecker:
    """Detect overconfidence and missing uncertainty hedging."""

    def check(self, response: str) -> ConfidenceCheckResult:
        result = ConfidenceCheckResult()
        text = str(response)

        # Count absolutes
        for pattern, issue_type, suggestion in _ABSOLUTE_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                result.absolute_count += 1
                result.issues.append(ConfidenceIssue(
                    text=m.group(0),
                    issue_type=issue_type,
                    suggestion=suggestion,
                    severity=0.6,
                ))

        # Count hedges
        for pattern in _HEDGE_PATTERNS:
            result.hedging_count += len(re.findall(pattern, text, re.IGNORECASE))

        # Check debatable topics without hedging
        for pattern, issue_type, suggestion in _DEBATABLE_TOPICS:
            if re.search(pattern, text, re.IGNORECASE):
                # Check if there's hedging nearby
                sentences = re.split(r'[.!?]+', text)
                for sent in sentences:
                    if re.search(pattern, sent, re.IGNORECASE):
                        has_hedge = any(re.search(hp, sent, re.IGNORECASE) for hp in _HEDGE_PATTERNS)
                        if not has_hedge:
                            result.issues.append(ConfidenceIssue(
                                text=sent.strip()[:60],
                                issue_type=issue_type,
                                suggestion=suggestion,
                                severity=0.5,
                            ))

        # Deduplicate by text
        seen = set()
        unique = []
        for issue in result.issues:
            key = issue.text.lower()
            if key not in seen:
                seen.add(key)
                unique.append(issue)
        result.issues = unique

        # Score: penalize high absolute:hedge ratio
        if result.absolute_count > 0:
            ratio = result.hedging_count / max(result.absolute_count, 1)
            if ratio < 0.5:
                result.score = max(0.3, 1.0 - result.absolute_count * 0.15)
            else:
                result.score = max(0.6, 1.0 - result.absolute_count * 0.05)
        else:
            result.score = 1.0

        return result
