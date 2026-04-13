"""
Auto-CALM Prioritization — rank options by impact and effort.

Models list things in arbitrary order. This module provides structured
prioritization using impact/effort matrices, dependency ordering,
and urgency detection.

Usage:
    from calm.prioritize import Prioritizer
    p = Prioritizer()
    ranked = p.rank([
        {"name": "Add caching", "impact": 8, "effort": 3},
        {"name": "Rewrite in Rust", "impact": 9, "effort": 9},
        {"name": "Fix N+1 query", "impact": 7, "effort": 1},
    ])
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class PrioritizedItem:
    """An item with priority score."""
    name: str
    impact: float = 5.0      # 1-10
    effort: float = 5.0      # 1-10 (higher = more effort)
    urgency: float = 5.0     # 1-10
    risk: float = 5.0        # 1-10 (higher = riskier)
    priority_score: float = 0.0
    quadrant: str = ""       # "quick-win", "major-project", "fill-in", "thankless"
    reasoning: str = ""


@dataclass
class PrioritizationResult:
    """Result of prioritization."""
    items: List[PrioritizedItem] = field(default_factory=list)

    @property
    def ranked(self) -> List[PrioritizedItem]:
        return sorted(self.items, key=lambda x: x.priority_score, reverse=True)

    def summary(self) -> str:
        lines = []
        for i, item in enumerate(self.ranked, 1):
            lines.append(f"  {i}. {item.name} (score={item.priority_score:.1f}, {item.quadrant})")
        return "\n".join(lines)


# Keywords that signal urgency
_URGENCY_SIGNALS = {
    "high": re.compile(r'\b(?:critical|urgent|blocking|outage|down|broken|crash|data loss|security|vulnerability|exploit)\b', re.IGNORECASE),
    "medium": re.compile(r'\b(?:important|significant|degraded|slow|bug|error|failing|regression)\b', re.IGNORECASE),
    "low": re.compile(r'\b(?:nice to have|cosmetic|minor|cleanup|refactor|tech debt|improvement|enhancement)\b', re.IGNORECASE),
}

# Keywords that signal effort
_EFFORT_SIGNALS = {
    "high": re.compile(r'\b(?:rewrite|redesign|migrate|overhaul|rebuild|replace|rearchitect|from scratch)\b', re.IGNORECASE),
    "medium": re.compile(r'\b(?:implement|build|create|develop|add feature|integrate|extend)\b', re.IGNORECASE),
    "low": re.compile(r'\b(?:fix|patch|tweak|adjust|configure|toggle|update|bump|rename)\b', re.IGNORECASE),
}


class Prioritizer:
    """Ranks items by impact, effort, urgency, and risk."""

    def rank(self, items: List[Dict]) -> PrioritizationResult:
        """Rank items. Each dict should have 'name' and optionally 'impact', 'effort', etc."""
        result = PrioritizationResult()

        for item_dict in items:
            item = PrioritizedItem(
                name=item_dict.get("name", "unnamed"),
                impact=float(item_dict.get("impact", 5)),
                effort=float(item_dict.get("effort", 5)),
                urgency=float(item_dict.get("urgency", 5)),
                risk=float(item_dict.get("risk", 5)),
            )

            # Priority score: high impact + high urgency + low effort + low risk
            # Weighted formula: impact matters most, effort is the main discount
            item.priority_score = (
                item.impact * 0.4 +
                item.urgency * 0.25 +
                (10 - item.effort) * 0.25 +
                (10 - item.risk) * 0.1
            )

            # Quadrant classification (Eisenhower matrix variant)
            if item.impact >= 7 and item.effort <= 4:
                item.quadrant = "quick-win"
            elif item.impact >= 7 and item.effort > 4:
                item.quadrant = "major-project"
            elif item.impact < 7 and item.effort <= 4:
                item.quadrant = "fill-in"
            else:
                item.quadrant = "thankless"

            result.items.append(item)

        return result

    def rank_from_text(self, items_text: List[str]) -> PrioritizationResult:
        """Rank text descriptions by detecting signals."""
        items = []
        for text in items_text:
            item = {
                "name": text[:60],
                "impact": self._estimate_impact(text),
                "effort": self._estimate_effort(text),
                "urgency": self._estimate_urgency(text),
                "risk": 5.0,
            }
            items.append(item)
        return self.rank(items)

    def _estimate_urgency(self, text: str) -> float:
        if _URGENCY_SIGNALS["high"].search(text):
            return 9.0
        if _URGENCY_SIGNALS["medium"].search(text):
            return 6.0
        if _URGENCY_SIGNALS["low"].search(text):
            return 3.0
        return 5.0

    def _estimate_effort(self, text: str) -> float:
        if _EFFORT_SIGNALS["high"].search(text):
            return 8.0
        if _EFFORT_SIGNALS["medium"].search(text):
            return 5.0
        if _EFFORT_SIGNALS["low"].search(text):
            return 2.0
        return 5.0

    def _estimate_impact(self, text: str) -> float:
        # Impact correlates with urgency for critical issues
        if _URGENCY_SIGNALS["high"].search(text):
            return 9.0
        if _URGENCY_SIGNALS["medium"].search(text):
            return 7.0
        return 5.0

    def format_matrix(self, result: PrioritizationResult) -> str:
        """Format as an impact/effort matrix."""
        quadrants = {
            "quick-win": [], "major-project": [],
            "fill-in": [], "thankless": [],
        }
        for item in result.items:
            quadrants[item.quadrant].append(item.name)

        lines = [
            "           High Effort          Low Effort",
            "High    | major-project        | quick-win",
            "Impact  | " + ", ".join(quadrants["major-project"][:2] or ["(none)"]),
            "        | " + ", ".join(quadrants["quick-win"][:2] or ["(none)"]),
            "Low     | thankless            | fill-in",
            "Impact  | " + ", ".join(quadrants["thankless"][:2] or ["(none)"]),
            "        | " + ", ".join(quadrants["fill-in"][:2] or ["(none)"]),
        ]
        return "\n".join(lines)
