"""
Auto-CALM Consistency — cross-turn contradiction detection.

Tracks every factual claim the model makes across a conversation.
When a new claim contradicts a prior one, flags it. This is the
foundation of trustworthiness — without consistency, nothing else
is reliable.

Two types of contradiction:
  1. Direct: "X is 5" then later "X is 7"
  2. Logical: "X > Y" then later "Y > X"

Usage:
    from calm.consistency import ConsistencyTracker
    ct = ConsistencyTracker()
    ct.add_claims("The capital of France is Paris. Python is interpreted.")
    issues = ct.add_claims("The capital of France is Lyon.")
    # → [Contradiction: 'capital of France' was 'Paris', now 'Lyon']
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TrackedClaim:
    """A claim stored in the tracker."""
    subject: str        # normalized subject ("capital of France")
    predicate: str      # what's claimed ("Paris")
    original: str       # original text
    turn: int = 0       # which turn it was made in
    verified: bool = False  # whether a backend verified it


@dataclass
class Contradiction:
    """A detected contradiction between claims."""
    subject: str
    old_value: str
    new_value: str
    old_turn: int
    new_turn: int
    severity: str = "direct"  # "direct" or "logical"

    def __str__(self):
        return (f"Contradiction: '{self.subject}' was '{self.old_value}' "
                f"(turn {self.old_turn}), now '{self.new_value}' (turn {self.new_turn})")


# Patterns for extracting subject-predicate claims from text.
_CLAIM_EXTRACTORS = [
    # "X is Y" / "X are Y"
    re.compile(r'(?:the\s+)?(\w[\w\s]{2,30}?)\s+(?:is|are|was|were)\s+(?:a\s+|an\s+|the\s+)?(\w[\w\s,]{1,50}?)(?:\.|,|\n|$)', re.IGNORECASE),
    # "X = Y"
    re.compile(r'(\w[\w\s]{2,30}?)\s*=\s*(\w[\w\s,.]{1,50}?)(?:\.|,|\n|$)'),
    # "X equals Y"
    re.compile(r'(\w[\w\s]{2,30}?)\s+equals?\s+(\w[\w\s,.]{1,50}?)(?:\.|,|\n|$)', re.IGNORECASE),
    # "X has Y"
    re.compile(r'(?:the\s+)?(\w[\w\s]{2,30}?)\s+has\s+(?:a\s+|an\s+)?(\w[\w\s,]{1,50}?)(?:\.|,|\n|$)', re.IGNORECASE),
]

# Comparison patterns for logical contradictions
_COMPARISON_EXTRACTORS = [
    # "X is greater/larger/more than Y"
    re.compile(r'(\w[\w\s]{1,20}?)\s+is\s+(?:greater|larger|bigger|more|higher|faster)\s+than\s+(\w[\w\s]{1,20})', re.IGNORECASE),
    # "X is less/smaller/fewer than Y"
    re.compile(r'(\w[\w\s]{1,20}?)\s+is\s+(?:less|smaller|fewer|lower|slower)\s+than\s+(\w[\w\s]{1,20})', re.IGNORECASE),
]

# Subjects to ignore (too generic to track meaningfully)
_IGNORE_SUBJECTS = {
    "it", "this", "that", "there", "here", "i", "you", "we", "they",
    "what", "which", "who", "how", "why", "result", "answer", "value",
    "output", "following", "example", "code", "function", "method",
}


class ConsistencyTracker:
    """Tracks claims across turns and detects contradictions."""

    def __init__(self):
        self._claims: Dict[str, TrackedClaim] = {}  # subject → claim
        self._comparisons: List[Tuple[str, str, str]] = []  # (a, op, b)
        self._turn: int = 0
        self._contradictions: List[Contradiction] = []

    def _normalize_subject(self, subject: str) -> str:
        """Normalize a subject for comparison."""
        s = subject.strip().lower()
        # Remove articles
        s = re.sub(r'^(?:the|a|an)\s+', '', s)
        # Collapse whitespace
        s = re.sub(r'\s+', ' ', s)
        return s

    def _normalize_value(self, value: str) -> str:
        """Normalize a predicate value for comparison."""
        v = value.strip().lower().rstrip('.,;:')
        v = re.sub(r'^(?:the|a|an)\s+', '', v)
        return v

    def _is_trackable(self, subject: str) -> bool:
        """Whether a subject is specific enough to track."""
        s = self._normalize_subject(subject)
        if s in _IGNORE_SUBJECTS:
            return False
        if len(s) < 3:
            return False
        # Skip if it's just a number
        if re.match(r'^\d+$', s):
            return False
        return True

    def add_claims(self, text: str, verified: bool = False) -> List[Contradiction]:
        """Extract claims from text, check against prior claims.
        Returns list of contradictions found."""
        self._turn += 1
        contradictions = []

        # Extract subject-predicate claims
        for pat in _CLAIM_EXTRACTORS:
            for m in pat.finditer(text):
                subject = m.group(1)
                predicate = m.group(2)

                if not self._is_trackable(subject):
                    continue

                norm_subj = self._normalize_subject(subject)
                norm_val = self._normalize_value(predicate)

                # Check for contradiction with prior claim
                if norm_subj in self._claims:
                    old = self._claims[norm_subj]
                    old_val = self._normalize_value(old.predicate)

                    # Same value = reinforcement, not contradiction
                    if old_val == norm_val:
                        continue

                    # Different value on same subject = contradiction
                    # But only if both are specific enough
                    if len(old_val) > 1 and len(norm_val) > 1:
                        c = Contradiction(
                            subject=norm_subj,
                            old_value=old.predicate.strip(),
                            new_value=predicate.strip(),
                            old_turn=old.turn,
                            new_turn=self._turn,
                            severity="direct",
                        )
                        contradictions.append(c)
                        self._contradictions.append(c)

                # Store/update the claim
                self._claims[norm_subj] = TrackedClaim(
                    subject=norm_subj,
                    predicate=predicate.strip(),
                    original=m.group(0).strip(),
                    turn=self._turn,
                    verified=verified,
                )

        # Extract comparison claims (X > Y)
        for pat in _COMPARISON_EXTRACTORS:
            for m in pat.finditer(text):
                a = self._normalize_subject(m.group(1))
                b = self._normalize_subject(m.group(2))
                op = ">" if "greater" in pat.pattern or "larger" in pat.pattern else "<"

                # Check for contradiction with prior comparisons
                for old_a, old_op, old_b in self._comparisons:
                    if a == old_b and b == old_a and op == old_op:
                        # "A > B" then "B > A" = contradiction
                        c = Contradiction(
                            subject=f"{a} vs {b}",
                            old_value=f"{old_a} {old_op} {old_b}",
                            new_value=f"{a} {op} {b}",
                            old_turn=0,
                            new_turn=self._turn,
                            severity="logical",
                        )
                        contradictions.append(c)
                        self._contradictions.append(c)

                self._comparisons.append((a, op, b))

        return contradictions

    def check(self, text: str) -> List[Contradiction]:
        """Check text for contradictions without storing claims."""
        # Temporarily store, check, then revert
        old_claims = dict(self._claims)
        old_comparisons = list(self._comparisons)
        old_turn = self._turn

        contradictions = self.add_claims(text)

        self._claims = old_claims
        self._comparisons = old_comparisons
        self._turn = old_turn

        return contradictions

    def get_claim(self, subject: str) -> Optional[TrackedClaim]:
        """Look up what we've tracked about a subject."""
        return self._claims.get(self._normalize_subject(subject))

    @property
    def all_contradictions(self) -> List[Contradiction]:
        """All contradictions found so far."""
        return list(self._contradictions)

    @property
    def claim_count(self) -> int:
        return len(self._claims)

    def summary(self) -> str:
        """Summary of tracked state."""
        return (f"{self.claim_count} claims tracked across {self._turn} turns, "
                f"{len(self._contradictions)} contradictions found")

    def reset(self):
        """Clear all tracked state."""
        self._claims.clear()
        self._comparisons.clear()
        self._contradictions.clear()
        self._turn = 0
