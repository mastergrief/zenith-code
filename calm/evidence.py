"""
Auto-CALM Evidence Tracking — support strength for claims.

Tracks what evidence supports each claim, distinguishes strong evidence
from weak, flags unsupported claims. When the model says "X is true,"
this module asks: "based on what?"

Evidence hierarchy (strongest → weakest):
  1. Backend-verified: CPU-computed, deterministic
  2. Multi-source confirmed: multiple independent sources agree
  3. Single source: one reference
  4. Reasoning-based: logical inference from other facts
  5. Assertion: claimed without evidence

Usage:
    from calm.evidence import EvidenceTracker
    et = EvidenceTracker()
    et.add_claim("17 * 23 = 391", evidence_type="verified", source="math_ops")
    et.add_claim("Python is popular", evidence_type="assertion")
    print(et.unsupported_claims())
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Evidence:
    """A piece of evidence supporting a claim."""
    source: str            # where it came from
    evidence_type: str     # "verified", "multi_source", "single_source", "reasoning", "assertion"
    confidence: float      # 0-1
    text: str = ""         # supporting text

    @property
    def strength(self) -> int:
        """Numeric strength: 5 (verified) → 1 (assertion)."""
        return {
            "verified": 5,
            "multi_source": 4,
            "single_source": 3,
            "reasoning": 2,
            "assertion": 1,
        }.get(self.evidence_type, 0)


@dataclass
class TrackedAssertion:
    """A claim with its supporting evidence."""
    claim: str
    evidence: List[Evidence] = field(default_factory=list)

    @property
    def best_evidence(self) -> Optional[Evidence]:
        return max(self.evidence, key=lambda e: e.strength) if self.evidence else None

    @property
    def strength_label(self) -> str:
        if not self.evidence:
            return "unsupported"
        best = self.best_evidence
        if best.evidence_type == "verified":
            return "verified"
        if best.evidence_type == "multi_source":
            return "confirmed"
        if best.evidence_type == "single_source":
            return "sourced"
        if best.evidence_type == "reasoning":
            return "inferred"
        return "asserted"

    @property
    def is_supported(self) -> bool:
        return any(e.strength >= 2 for e in self.evidence)


# Patterns that indicate evidence is being cited
_EVIDENCE_PATTERNS = [
    (re.compile(r'(?:according to|as per|based on|per|from)\s+(.{5,50}?)(?:,|\.|$)', re.IGNORECASE), "single_source"),
    (re.compile(r'(?:the documentation|the docs|the spec|the standard|RFC \d+)\s+(?:says?|states?|specifies?)', re.IGNORECASE), "single_source"),
    (re.compile(r'(?:studies show|research shows|data shows|evidence suggests)', re.IGNORECASE), "single_source"),
    (re.compile(r'(?:multiple sources|several studies|consensus|widely accepted|well.established)', re.IGNORECASE), "multi_source"),
    (re.compile(r'(?:because|since|therefore|thus|given that)\s+(.{10,}?)(?:,|\.|$)', re.IGNORECASE), "reasoning"),
]

# Patterns that indicate unsupported assertions
_ASSERTION_PATTERNS = [
    re.compile(r'(?:it is|this is|that is)\s+(?:clearly|obviously|undeniably|certainly|definitely)\b', re.IGNORECASE),
    re.compile(r'\b(?:everyone knows|common knowledge|goes without saying|needless to say)\b', re.IGNORECASE),
    re.compile(r'\b(?:of course|naturally|clearly|obviously)\b', re.IGNORECASE),
]


class EvidenceTracker:
    """Tracks evidence strength for claims."""

    def __init__(self):
        self._claims: Dict[str, TrackedAssertion] = {}

    def add_claim(self, claim: str, evidence_type: str = "assertion",
                  source: str = "", confidence: float = 0.5):
        """Add a claim with evidence."""
        key = claim.strip().lower()[:80]
        if key not in self._claims:
            self._claims[key] = TrackedAssertion(claim=claim)

        self._claims[key].evidence.append(Evidence(
            source=source,
            evidence_type=evidence_type,
            confidence=confidence,
            text=claim,
        ))

    def analyze_text(self, text: str) -> Dict[str, str]:
        """Analyze text for evidence patterns. Returns {claim: evidence_type}."""
        results = {}

        # Find evidence-backed claims
        for pat, etype in _EVIDENCE_PATTERNS:
            for m in pat.finditer(text):
                # The claim is the surrounding sentence
                start = max(0, text.rfind('.', 0, m.start()) + 1)
                end = text.find('.', m.end())
                if end == -1:
                    end = len(text)
                sentence = text[start:end].strip()
                if len(sentence) > 10:
                    results[sentence] = etype

        # Find unsupported assertions
        for pat in _ASSERTION_PATTERNS:
            for m in pat.finditer(text):
                start = max(0, text.rfind('.', 0, m.start()) + 1)
                end = text.find('.', m.end())
                if end == -1:
                    end = len(text)
                sentence = text[start:end].strip()
                if sentence and sentence not in results:
                    results[sentence] = "assertion"

        return results

    def unsupported_claims(self) -> List[TrackedAssertion]:
        """Return claims without sufficient evidence."""
        return [c for c in self._claims.values() if not c.is_supported]

    def verified_claims(self) -> List[TrackedAssertion]:
        """Return backend-verified claims."""
        return [c for c in self._claims.values()
                if any(e.evidence_type == "verified" for e in c.evidence)]

    def evidence_summary(self) -> str:
        """Summary of evidence landscape."""
        if not self._claims:
            return "No claims tracked"

        by_strength = {"verified": 0, "confirmed": 0, "sourced": 0,
                       "inferred": 0, "asserted": 0, "unsupported": 0}
        for claim in self._claims.values():
            by_strength[claim.strength_label] = by_strength.get(claim.strength_label, 0) + 1

        parts = []
        for label, count in by_strength.items():
            if count > 0:
                parts.append(f"{count} {label}")

        return f"{len(self._claims)} claims: {', '.join(parts)}"

    def strength_score(self) -> float:
        """Overall evidence quality score (0-1)."""
        if not self._claims:
            return 0.0
        total_strength = sum(
            (c.best_evidence.strength if c.best_evidence else 0)
            for c in self._claims.values()
        )
        max_possible = len(self._claims) * 5
        return total_strength / max_possible if max_possible > 0 else 0.0
