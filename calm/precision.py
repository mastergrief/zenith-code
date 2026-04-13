"""
Auto-CALM Precision Language — convert vague claims to measurable ones.

Models say "fast", "scalable", "secure" without specifying what that
means. This module detects vague terms and suggests precise alternatives.

Usage:
    from calm.precision import PrecisionChecker
    pc = PrecisionChecker()
    result = pc.check("The API is fast and scalable")
    # → vague terms: "fast" (suggest: latency target), "scalable" (suggest: load target)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class VagueTerm:
    """A vague term with suggested precise alternative."""
    term: str
    context: str
    category: str       # "performance", "quality", "scope", "time", "quantity"
    suggestion: str     # how to make it precise
    question: str       # the question that would pin it down


@dataclass
class PrecisionResult:
    """Precision analysis result."""
    vague_terms: List[VagueTerm] = field(default_factory=list)
    precision_score: float = 0.0   # 0-1 (1 = fully precise)
    label: str = "unknown"

    def summary(self) -> str:
        if not self.vague_terms:
            return f"precise ({self.precision_score:.0%})"
        return (f"{self.label} ({self.precision_score:.0%}), "
                f"{len(self.vague_terms)} vague terms")


# Vague terms → precise alternatives
_VAGUE_TERMS = {
    # Performance
    "fast": ("performance", "Specify: latency target (e.g., <100ms p99)", "How fast in milliseconds?"),
    "slow": ("performance", "Specify: what latency/throughput is unacceptable", "What response time is too slow?"),
    "scalable": ("performance", "Specify: target load and growth rate (e.g., 10x current with linear cost)", "Scalable to what load? What's the cost curve?"),
    "efficient": ("performance", "Specify: which resource (CPU, memory, I/O) and target utilization", "Efficient in what dimension? Compared to what?"),
    "performant": ("performance", "Specify: benchmark and target metric", "What benchmark? What target number?"),
    "responsive": ("performance", "Specify: interaction latency target (e.g., <200ms for clicks)", "What response time feels responsive for this UI?"),
    "lightweight": ("performance", "Specify: memory/CPU/binary size targets", "Lightweight compared to what? In what dimension?"),
    "high-performance": ("performance", "Specify: exact throughput or latency targets", "What metric defines high performance here?"),
    # Quality
    "good": ("quality", "Specify: what criteria define 'good' in this context", "Good by what measure?"),
    "bad": ("quality", "Specify: what makes it bad — errors? UX? performance?", "Bad in what way specifically?"),
    "clean": ("quality", "Specify: formatting rules, linting config, or style guide", "Clean by which standard?"),
    "robust": ("quality", "Specify: what failure modes it handles", "Robust against which failure scenarios?"),
    "reliable": ("quality", "Specify: uptime target and MTTR (e.g., 99.9%, <5min recovery)", "What uptime percentage? What recovery time?"),
    "stable": ("quality", "Specify: what stability means (no crashes? no API changes? no regressions?)", "Stable in what sense?"),
    "secure": ("quality", "Specify: threat model and security controls", "Secure against which threats?"),
    "production-ready": ("quality", "Specify: the checklist of what 'production-ready' requires", "What's on the production readiness checklist?"),
    # Scope
    "simple": ("scope", "Specify: simple for whom? (user? developer? operator?)", "Simple for which audience?"),
    "complex": ("scope", "Specify: what makes it complex (many components? edge cases? algorithms?)", "Complex in what dimension?"),
    "easy": ("scope", "Specify: easy for whom with what background", "Easy for someone with what experience level?"),
    "hard": ("scope", "Specify: what makes it hard (technically? organizationally?)", "Hard because of what specifically?"),
    "flexible": ("scope", "Specify: what needs to be configurable/extensible", "Flexible in what ways?"),
    "modular": ("scope", "Specify: what modules and what interfaces", "What are the module boundaries?"),
    # Time
    "soon": ("time", "Specify: date or number of days/weeks", "By when exactly?"),
    "quickly": ("time", "Specify: time target", "In how many minutes/hours/days?"),
    "eventually": ("time", "Specify: timeline and what triggers completion", "By when? What determines when?"),
    # Quantity
    "many": ("quantity", "Specify: approximate count or order of magnitude", "How many approximately?"),
    "few": ("quantity", "Specify: count or range", "How few? 2? 5? 10?"),
    "large": ("quantity", "Specify: size in concrete units", "How large in bytes/rows/users?"),
    "small": ("quantity", "Specify: size threshold", "How small? What's the upper bound?"),
    "significant": ("quantity", "Specify: magnitude or percentage", "Significant by how much? What's the baseline?"),
    "minimal": ("quantity", "Specify: what the minimum is and why", "What's the actual minimum?"),
}


class PrecisionChecker:
    """Detects vague language and suggests precise alternatives."""

    def check(self, text: str) -> PrecisionResult:
        """Check text for vague language."""
        result = PrecisionResult()
        text_lower = text.lower()
        word_count = len(text.split())

        for term, (category, suggestion, question) in _VAGUE_TERMS.items():
            # Find the term as a whole word
            pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            for m in pattern.finditer(text):
                # Get surrounding context
                start = max(0, m.start() - 30)
                end = min(len(text), m.end() + 30)
                context = text[start:end].strip()

                result.vague_terms.append(VagueTerm(
                    term=term,
                    context=context,
                    category=category,
                    suggestion=suggestion,
                    question=question,
                ))

        # Deduplicate by term
        seen = set()
        unique = []
        for vt in result.vague_terms:
            if vt.term not in seen:
                seen.add(vt.term)
                unique.append(vt)
        result.vague_terms = unique

        # Precision score
        vague_density = len(result.vague_terms) / max(word_count / 20, 1)
        result.precision_score = max(0, min(1, 1 - vague_density * 0.3))

        if result.precision_score > 0.8:
            result.label = "precise"
        elif result.precision_score > 0.6:
            result.label = "mostly precise"
        elif result.precision_score > 0.3:
            result.label = "vague"
        else:
            result.label = "very vague"

        return result
