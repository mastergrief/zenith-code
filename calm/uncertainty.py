"""
Auto-CALM Uncertainty Propagation — trace how uncertainty spreads.

When one fact is uncertain, all conclusions that depend on it are also
uncertain. This module tracks which claims are certain vs uncertain
and propagates uncertainty through dependency chains.

The dual of causal reasoning: causal traces effects of changes,
uncertainty traces effects of doubt.

Usage:
    from calm.uncertainty import UncertaintyTracker
    ut = UncertaintyTracker()
    ut.set_certain("RAM is 32GB")
    ut.set_uncertain("VRAM is 8GB", confidence=0.7)
    ut.add_dependency("model fits in VRAM", "VRAM is 8GB")
    print(ut.confidence("model fits in VRAM"))  # ≤ 0.7
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional


@dataclass
class UncertainFact:
    """A fact with associated certainty."""
    statement: str
    confidence: float     # 0-1 (1 = certain, 0 = unknown)
    source: str = ""      # where this came from
    is_assumption: bool = False


@dataclass
class UncertaintyReport:
    """Report on uncertainty in a reasoning chain."""
    facts: List[UncertainFact] = field(default_factory=list)
    weakest_link: Optional[UncertainFact] = None
    overall_confidence: float = 1.0
    tainted_conclusions: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.facts:
            return "No facts tracked"
        certain = sum(1 for f in self.facts if f.confidence >= 0.9)
        uncertain = sum(1 for f in self.facts if f.confidence < 0.9)
        parts = [f"{len(self.facts)} facts ({certain} certain, {uncertain} uncertain)"]
        if self.weakest_link:
            parts.append(f"weakest: '{self.weakest_link.statement}' ({self.weakest_link.confidence:.0%})")
        parts.append(f"overall: {self.overall_confidence:.0%}")
        return ", ".join(parts)


# Uncertainty signals in text
_CERTAINTY_SIGNALS = {
    "high": re.compile(r'\b(?:definitely|certainly|always|proven|confirmed|verified|guaranteed|known|fact|established)\b', re.IGNORECASE),
    "medium": re.compile(r'\b(?:likely|probably|usually|typically|generally|expected|should|often|common)\b', re.IGNORECASE),
    "low": re.compile(r'\b(?:maybe|perhaps|possibly|might|could|uncertain|unclear|unknown|debatable|controversial|estimated|approximately|roughly)\b', re.IGNORECASE),
}

_ASSUMPTION_SIGNALS = re.compile(
    r'\b(?:assuming|assume|if we assume|given that|suppose|let.s say|presume)\b',
    re.IGNORECASE,
)


class UncertaintyTracker:
    """Tracks and propagates uncertainty through reasoning chains."""

    def __init__(self):
        self._facts: Dict[str, UncertainFact] = {}
        self._dependencies: Dict[str, Set[str]] = defaultdict(set)  # conclusion → set of premises

    def _normalize(self, statement: str) -> str:
        return statement.strip().lower()[:80]

    def set_certain(self, statement: str, source: str = ""):
        """Record a certain fact (confidence = 1.0)."""
        key = self._normalize(statement)
        self._facts[key] = UncertainFact(
            statement=statement, confidence=1.0, source=source,
        )

    def set_uncertain(self, statement: str, confidence: float = 0.5,
                      source: str = "", is_assumption: bool = False):
        """Record an uncertain fact."""
        key = self._normalize(statement)
        self._facts[key] = UncertainFact(
            statement=statement, confidence=confidence,
            source=source, is_assumption=is_assumption,
        )

    def add_dependency(self, conclusion: str, premise: str):
        """Record that conclusion depends on premise."""
        c_key = self._normalize(conclusion)
        p_key = self._normalize(premise)
        self._dependencies[c_key].add(p_key)

    def confidence(self, statement: str) -> float:
        """Get the effective confidence of a statement, considering dependencies."""
        key = self._normalize(statement)
        return self._compute_confidence(key, set())

    def _compute_confidence(self, key: str, visited: Set[str]) -> float:
        """Recursively compute confidence through dependency chain."""
        if key in visited:
            return 0.5  # cycle guard
        visited.add(key)

        # Base case: we have a direct confidence for this fact
        base = self._facts.get(key)
        base_conf = base.confidence if base else 0.5  # unknown defaults to 0.5

        # If this fact depends on others, its confidence is bounded by
        # the weakest premise (chain is as strong as weakest link)
        premises = self._dependencies.get(key, set())
        if not premises:
            return base_conf

        premise_confs = [self._compute_confidence(p, visited) for p in premises]
        min_premise = min(premise_confs) if premise_confs else 1.0

        # Effective confidence = min(own confidence, weakest premise)
        return min(base_conf, min_premise)

    def analyze_text(self, text: str) -> List[UncertainFact]:
        """Extract facts with uncertainty levels from text."""
        facts = []

        # Split into sentences
        sentences = re.split(r'[.!]\s+', text)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:
                continue

            # Detect certainty level
            is_assumption = bool(_ASSUMPTION_SIGNALS.search(sent))
            if is_assumption:
                confidence = 0.4
            elif _CERTAINTY_SIGNALS["high"].search(sent):
                confidence = 0.95
            elif _CERTAINTY_SIGNALS["low"].search(sent):
                confidence = 0.3
            elif _CERTAINTY_SIGNALS["medium"].search(sent):
                confidence = 0.7
            else:
                confidence = 0.6  # default for unqualified claims

            fact = UncertainFact(
                statement=sent,
                confidence=confidence,
                is_assumption=is_assumption,
            )
            facts.append(fact)

            # Record in tracker
            key = self._normalize(sent)
            self._facts[key] = fact

        return facts

    def report(self) -> UncertaintyReport:
        """Generate a report on tracked uncertainty."""
        result = UncertaintyReport(
            facts=list(self._facts.values()),
        )

        if self._facts:
            # Find weakest link
            result.weakest_link = min(self._facts.values(), key=lambda f: f.confidence)

            # Overall confidence = product of all independent fact confidences
            # (simplified: use average for readability)
            confs = [f.confidence for f in self._facts.values()]
            result.overall_confidence = sum(confs) / len(confs)

            # Find conclusions tainted by uncertainty
            for conclusion, premises in self._dependencies.items():
                eff_conf = self.confidence(conclusion)
                if eff_conf < 0.5:
                    result.tainted_conclusions.append(
                        f"'{conclusion}' (confidence {eff_conf:.0%})"
                    )

        return result
