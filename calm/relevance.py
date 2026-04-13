"""
Auto-CALM Relevance — detect correct but irrelevant information.

Models often include accurate information that doesn't answer the
actual question. "What time is it?" → 3-paragraph history of
timekeeping. This module checks if the response addresses the
question directly.

Usage:
    from calm.relevance import RelevanceChecker
    rc = RelevanceChecker()
    score = rc.check("What is 2+2?", "The history of mathematics...")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Set


@dataclass
class RelevanceResult:
    """Relevance assessment."""
    score: float = 0.0            # 0-1
    question_keywords: Set[str] = field(default_factory=set)
    response_keywords: Set[str] = field(default_factory=set)
    overlap: Set[str] = field(default_factory=set)
    tangent_sections: List[str] = field(default_factory=list)
    label: str = "unknown"        # "direct", "mostly_relevant", "tangential", "off_topic"

    def summary(self) -> str:
        return f"{self.label} ({self.score:.0%}), {len(self.tangent_sections)} tangent sections"


_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "and",
    "but", "or", "not", "no", "so", "yet", "both", "either", "neither",
    "each", "every", "all", "any", "few", "more", "most", "other", "some",
    "than", "too", "very", "just", "also", "then", "that", "this", "these",
    "those", "it", "its", "they", "them", "their", "we", "our", "you",
    "your", "he", "she", "him", "her", "his", "what", "which", "who",
    "how", "why", "when", "where",
}

# Tangent signals
_TANGENT_PATTERNS = [
    re.compile(r'(?:As an aside|On a related note|Interestingly|Fun fact|By the way|Incidentally|Speaking of which|It.s worth noting that|As a side note)', re.IGNORECASE),
    re.compile(r'(?:A bit of (?:history|background|context)|The history of|Historically|For some background)', re.IGNORECASE),
]


class RelevanceChecker:
    """Checks if responses are relevant to questions."""

    def _keywords(self, text: str) -> Set[str]:
        words = re.findall(r'[a-z]+', text.lower())
        return {w for w in words if w not in _STOP_WORDS and len(w) > 2}

    def check(self, question: str, response: str) -> RelevanceResult:
        """Check relevance of response to question."""
        result = RelevanceResult()
        result.question_keywords = self._keywords(question)
        result.response_keywords = self._keywords(response)
        result.overlap = result.question_keywords & result.response_keywords

        # Keyword overlap score
        if result.question_keywords:
            coverage = len(result.overlap) / len(result.question_keywords)
        else:
            coverage = 0.5

        # Detect tangent sections
        paragraphs = re.split(r'\n\n|\n(?=#+\s)', response)
        for para in paragraphs:
            para_kw = self._keywords(para)
            if result.question_keywords:
                para_overlap = len(para_kw & result.question_keywords)
                if para_overlap == 0 and len(para) > 50:
                    result.tangent_sections.append(para[:80])

            for pat in _TANGENT_PATTERNS:
                if pat.search(para):
                    result.tangent_sections.append(para[:80])
                    break

        # Tangent penalty
        tangent_ratio = len(result.tangent_sections) / max(len(paragraphs), 1)

        # Final score
        result.score = max(0, min(1, coverage * 0.6 + (1 - tangent_ratio) * 0.4))

        if result.score > 0.8:
            result.label = "direct"
        elif result.score > 0.6:
            result.label = "mostly_relevant"
        elif result.score > 0.3:
            result.label = "tangential"
        else:
            result.label = "off_topic"

        return result
