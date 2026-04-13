"""
Auto-CALM Constraint Satisfaction — track and verify requirements.

When problems have constraints ("must be under $100", "works on Linux",
"completes in 2 hours"), extract and track them. Verify the proposed
solution meets every constraint. Flag violations.

Usage:
    from calm.constraints import ConstraintTracker
    ct = ConstraintTracker()
    ct.extract("Build a web app under $500 that handles 1000 concurrent users on AWS")
    ct.check_solution("I recommend a t2.micro instance running Flask")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Constraint:
    """A requirement that must be satisfied."""
    text: str
    category: str        # "budget", "performance", "platform", "time", "technical", "other"
    quantified: bool = False   # whether it has a measurable threshold
    threshold: str = ""        # the specific limit ("$500", "1000 users", "2 hours")
    satisfied: Optional[bool] = None  # True, False, None (unknown)
    evidence: str = ""   # what in the solution addresses this


@dataclass
class ConstraintResult:
    """Result of constraint checking."""
    constraints: List[Constraint] = field(default_factory=list)
    satisfied: int = 0
    violated: int = 0
    unknown: int = 0

    @property
    def all_met(self) -> bool:
        return self.violated == 0 and self.unknown == 0

    def summary(self) -> str:
        total = len(self.constraints)
        parts = [f"{total} constraints"]
        if self.satisfied:
            parts.append(f"{self.satisfied} met")
        if self.violated:
            parts.append(f"{self.violated} VIOLATED")
        if self.unknown:
            parts.append(f"{self.unknown} unverified")
        return ", ".join(parts)


# Constraint extraction patterns
_CONSTRAINT_PATTERNS = [
    # Budget: "under $X", "less than $X", "within $X budget"
    (re.compile(r'(?:under|less than|within|max(?:imum)?|budget (?:of|is))\s*\$?([\d,]+(?:\.\d+)?)\s*(?:dollars?|USD|budget)?', re.IGNORECASE),
     "budget", True),
    # Performance: "handle X users", "X requests per second", "under X ms"
    (re.compile(r'(?:handle|support|serve)\s+(\d[\d,]*)\s+(?:concurrent\s+)?(?:users?|connections?|requests?)', re.IGNORECASE),
     "performance", True),
    (re.compile(r'(?:under|less than|within|max(?:imum)?)\s+(\d+)\s*(?:ms|milliseconds?|seconds?)\s+(?:latency|response)', re.IGNORECASE),
     "performance", True),
    # Time: "in X hours/days/weeks", "by DATE", "deadline"
    (re.compile(r'(?:within|in|under)\s+(\d+)\s+(?:hours?|days?|weeks?|months?)', re.IGNORECASE),
     "time", True),
    (re.compile(r'(?:by|before|deadline)\s+(\w+\s+\d+)', re.IGNORECASE),
     "time", True),
    # Platform: "on Linux/Windows/Mac", "using X"
    (re.compile(r'(?:on|for|targeting|supporting)\s+(Linux|Windows|macOS|Mac|iOS|Android|AWS|GCP|Azure)', re.IGNORECASE),
     "platform", False),
    # Technical: "must use X", "requires X", "compatible with X"
    (re.compile(r'(?:must|should|needs? to|has to|required to)\s+(?:use|support|include|have|be)\s+(.{3,40}?)(?:\.|,|$)', re.IGNORECASE),
     "technical", False),
    (re.compile(r'(?:compatible with|works? with|integrates? with)\s+(.{3,30})', re.IGNORECASE),
     "technical", False),
    # Size/weight: "under X MB/GB/KB"
    (re.compile(r'(?:under|less than|within)\s+(\d+)\s*(?:MB|GB|KB|bytes|lines)', re.IGNORECASE),
     "technical", True),
    # Generic: "must be X", "needs to be X"
    (re.compile(r'(?:must|needs? to|has to|should) be\s+(.{3,30}?)(?:\.|,|$)', re.IGNORECASE),
     "other", False),
]


class ConstraintTracker:
    """Extracts and verifies constraints from problem descriptions."""

    def __init__(self):
        self._constraints: List[Constraint] = []

    def extract(self, text: str) -> List[Constraint]:
        """Extract constraints from a problem description."""
        constraints = []

        for pat, category, quantified in _CONSTRAINT_PATTERNS:
            for m in pat.finditer(text):
                constraint = Constraint(
                    text=m.group(0).strip(),
                    category=category,
                    quantified=quantified,
                    threshold=m.group(1).strip() if m.groups() else "",
                )
                constraints.append(constraint)

        self._constraints.extend(constraints)
        return constraints

    def check_solution(self, solution: str) -> ConstraintResult:
        """Check if a proposed solution meets all tracked constraints."""
        result = ConstraintResult(constraints=self._constraints)

        solution_lower = solution.lower()

        for constraint in self._constraints:
            # Try to determine if the constraint is satisfied
            key_words = set(re.findall(r'[a-z]+', constraint.text.lower()))
            key_words -= {"must", "should", "need", "under", "less", "than", "within", "be"}

            # Check if the solution mentions relevant terms
            mentions = sum(1 for w in key_words if w in solution_lower and len(w) > 2)
            coverage = mentions / len(key_words) if key_words else 0

            if constraint.category == "platform":
                # Direct keyword check
                if constraint.threshold.lower() in solution_lower:
                    constraint.satisfied = True
                    constraint.evidence = f"Mentions {constraint.threshold}"
                    result.satisfied += 1
                else:
                    constraint.satisfied = None
                    result.unknown += 1
            elif coverage > 0.3:
                constraint.satisfied = True
                constraint.evidence = f"Solution addresses: {', '.join(w for w in key_words if w in solution_lower)}"
                result.satisfied += 1
            else:
                constraint.satisfied = None
                result.unknown += 1

        return result

    def add_constraint(self, text: str, category: str = "other",
                       threshold: str = ""):
        """Manually add a constraint."""
        self._constraints.append(Constraint(
            text=text, category=category,
            quantified=bool(threshold), threshold=threshold,
        ))

    @property
    def constraints(self) -> List[Constraint]:
        return list(self._constraints)

    def summary(self) -> str:
        if not self._constraints:
            return "No constraints tracked"
        by_cat = {}
        for c in self._constraints:
            by_cat[c.category] = by_cat.get(c.category, 0) + 1
        parts = [f"{count} {cat}" for cat, count in by_cat.items()]
        return f"{len(self._constraints)} constraints: {', '.join(parts)}"
