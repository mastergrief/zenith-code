"""
Auto-CALM Provenance — trace where each fact came from.

Every claim in a response has a source: backend-verified, model-generated,
user-provided, or precomputed. Provenance tracking labels each fact with
its origin, enabling different trust levels.

Usage:
    from calm.provenance import ProvenanceTracker
    pt = ProvenanceTracker()
    pt.tag("17 * 23 = 391", source="math_ops", trust="verified")
    pt.tag("Python is popular", source="model", trust="generated")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ProvenanceTag:
    """Origin tag for a fact."""
    claim: str
    source: str           # "backend", "precompute", "model", "user", "learned"
    backend_name: str = "" # which specific backend (if applicable)
    trust_level: str = "unknown"  # "verified", "precomputed", "generated", "provided", "learned"
    confidence: float = 0.5

    @property
    def trust_score(self) -> float:
        return {
            "verified": 1.0,
            "precomputed": 0.95,
            "provided": 0.8,
            "learned": 0.7,
            "generated": 0.5,
            "unknown": 0.3,
        }.get(self.trust_level, 0.3)


@dataclass
class ProvenanceReport:
    """Full provenance report for a response."""
    tags: List[ProvenanceTag] = field(default_factory=list)
    overall_trust: float = 0.0

    @property
    def by_source(self) -> Dict[str, int]:
        counts = {}
        for t in self.tags:
            counts[t.trust_level] = counts.get(t.trust_level, 0) + 1
        return counts

    def summary(self) -> str:
        if not self.tags:
            return "no provenance data"
        parts = [f"{count} {level}" for level, count in sorted(self.by_source.items())]
        return f"{len(self.tags)} facts: {', '.join(parts)} (trust: {self.overall_trust:.0%})"


class ProvenanceTracker:
    """Tracks where each fact in a response came from."""

    def __init__(self):
        self._tags: List[ProvenanceTag] = []

    def tag(self, claim: str, source: str, trust: str = "unknown",
            backend: str = ""):
        """Tag a claim with its provenance."""
        self._tags.append(ProvenanceTag(
            claim=claim,
            source=source,
            backend_name=backend,
            trust_level=trust,
        ))

    def tag_from_autocalm(self, precomputed: dict, claims_verified: int,
                           claims_corrected: int, response: str):
        """Auto-tag based on Auto-CALM results."""
        # Precomputed facts = highest trust
        for expr, value in precomputed.items():
            self.tag(
                f"{expr} = {value}",
                source="backend",
                trust="precomputed",
                backend=expr.split("(")[0] if "(" in expr else "",
            )

        # Verified claims
        for _ in range(claims_verified):
            self.tag("(verified claim)", source="backend", trust="verified")

        # Corrected claims = were wrong, now fixed
        for _ in range(claims_corrected):
            self.tag("(corrected claim)", source="backend", trust="verified")

        # Everything else in the response = model-generated
        # Rough estimate: count sentences not covered by precompute
        sentences = re.split(r'[.!?]\s+', response)
        precompute_coverage = len(precomputed)
        generated = max(0, len(sentences) - precompute_coverage - claims_verified)
        for _ in range(min(generated, 20)):  # cap to avoid huge lists
            self.tag("(model-generated content)", source="model", trust="generated")

    def report(self) -> ProvenanceReport:
        """Generate provenance report."""
        report = ProvenanceReport(tags=self._tags)
        if self._tags:
            report.overall_trust = sum(t.trust_score for t in self._tags) / len(self._tags)
        return report

    def trust_breakdown(self) -> str:
        """Human-readable trust breakdown."""
        report = self.report()
        lines = [f"Overall trust: {report.overall_trust:.0%}"]
        for level in ["verified", "precomputed", "provided", "learned", "generated", "unknown"]:
            count = report.by_source.get(level, 0)
            if count > 0:
                score = {"verified": 100, "precomputed": 95, "provided": 80,
                         "learned": 70, "generated": 50, "unknown": 30}.get(level, 30)
                lines.append(f"  {level}: {count} facts ({score}% trust each)")
        return "\n".join(lines)

    def reset(self):
        self._tags.clear()
