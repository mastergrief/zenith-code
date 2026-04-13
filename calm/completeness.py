"""
Auto-CALM Completeness — verify all parts of multi-part questions answered.

Models frequently answer the first part of a question and forget the rest.
"What is X, how does it work, and when should I use it?" → only answers
"what is X."

Usage:
    from calm.completeness import CompletenessChecker
    cc = CompletenessChecker()
    result = cc.check("What is Redis, how does it work, and when should I use it?",
                      "Redis is an in-memory data store.")
    print(result.answered)   # ["What is Redis"]
    print(result.unanswered) # ["how does it work", "when should I use it"]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Set


@dataclass
class CompletenessResult:
    """Completeness assessment."""
    parts: List[str] = field(default_factory=list)
    answered: List[str] = field(default_factory=list)
    unanswered: List[str] = field(default_factory=list)
    score: float = 0.0           # 0-1
    label: str = "unknown"       # "complete", "partial", "incomplete"

    def summary(self) -> str:
        return (f"{self.label}: {len(self.answered)}/{len(self.parts)} parts answered"
                + (f", missing: {', '.join(self.unanswered[:3])}" if self.unanswered else ""))


_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "do", "does",
    "it", "its", "and", "or", "of", "to", "in", "for", "with", "on",
    "at", "by", "from", "this", "that", "you", "your", "we", "i", "my",
}


class CompletenessChecker:
    """Verifies all parts of multi-part questions are answered."""

    def extract_parts(self, question: str) -> List[str]:
        """Extract sub-questions from a multi-part question."""
        parts = []

        # Split on explicit conjunctions: "X, Y, and Z?"
        # "What is X, how does Y, and when Z?"
        conj_parts = re.split(r'[,;]\s+(?:and\s+)?|\s+and\s+', question)
        if len(conj_parts) >= 2:
            for part in conj_parts:
                part = part.strip().rstrip('?')
                if len(part) > 5:
                    parts.append(part)
            return parts

        # Split on multiple question marks
        q_parts = re.split(r'\?\s+', question)
        if len(q_parts) >= 2:
            for part in q_parts:
                part = part.strip().rstrip('?')
                if len(part) > 5:
                    parts.append(part)
            return parts

        # Single question
        parts.append(question.strip().rstrip('?'))
        return parts

    def check(self, question: str, response: str) -> CompletenessResult:
        """Check if all parts of a question are answered."""
        result = CompletenessResult()
        result.parts = self.extract_parts(question)

        if len(result.parts) <= 1:
            result.score = 1.0
            result.label = "complete"
            result.answered = result.parts
            return result

        response_lower = response.lower()

        for part in result.parts:
            # Extract key content words from this part
            words = set(re.findall(r'[a-z]+', part.lower())) - _STOP_WORDS
            significant = {w for w in words if len(w) > 3}

            if not significant:
                result.answered.append(part)
                continue

            # Check if response covers this part's keywords
            found = sum(1 for w in significant if w in response_lower)
            coverage = found / len(significant) if significant else 0

            if coverage >= 0.4:
                result.answered.append(part)
            else:
                result.unanswered.append(part)

        result.score = len(result.answered) / len(result.parts) if result.parts else 1.0

        if result.score >= 1.0:
            result.label = "complete"
        elif result.score >= 0.5:
            result.label = "partial"
        else:
            result.label = "incomplete"

        return result
