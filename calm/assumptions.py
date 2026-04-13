"""
Auto-CALM Assumption Detection — extract unstated assumptions.

Every argument depends on assumptions. Making them explicit is half
of critical thinking. This module identifies hidden assumptions in
reasoning and flags them for verification.

Types:
  1. Implicit premises: "X so Y" assumes a link between X and Y
  2. Default assumptions: "deploy to production" assumes prod exists
  3. Scope assumptions: "this is faster" assumes a specific benchmark

Usage:
    from calm.assumptions import AssumptionDetector
    ad = AssumptionDetector()
    assumptions = ad.detect("Just add an index and the query will be fast")
    # → ["adding an index will improve THIS query (may not if full scan is needed)"]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Assumption:
    """A detected unstated assumption."""
    text: str              # what's being assumed
    category: str          # "implicit_premise", "default", "scope", "technical"
    trigger: str           # what text triggered the detection
    risk: str = "low"      # "low", "medium", "high" — how likely to be wrong
    question: str = ""     # the question that would validate/invalidate it


# Technical assumptions that are often wrong
_TECHNICAL_ASSUMPTIONS = [
    # Resource assumptions
    (re.compile(r'\b(?:just|simply)\s+(?:add|use|deploy|install|upgrade)\b', re.IGNORECASE),
     "default", "medium",
     "Assumes the change is trivial — may have hidden complexity, dependencies, or side effects",
     "What are the prerequisites and potential side effects?"),
    # Performance assumptions
    (re.compile(r'\b(?:will be faster|is faster|speeds up|improves performance)\b', re.IGNORECASE),
     "scope", "medium",
     "Assumes performance improvement without specifying the workload or measurement",
     "Faster for which workload? How was it measured?"),
    # Availability assumptions
    (re.compile(r'\b(?:always available|100% uptime|never fails|guaranteed)\b', re.IGNORECASE),
     "default", "high",
     "Assumes perfect reliability — no system is 100% available",
     "What is the actual SLA? What happens when it fails?"),
    # Scale assumptions
    (re.compile(r'\b(?:scales? (?:well|linearly|infinitely)|handles? any (?:load|amount))\b', re.IGNORECASE),
     "scope", "high",
     "Assumes unlimited scalability — every system has limits",
     "What is the actual throughput ceiling? Where does it break?"),
    # Security assumptions
    (re.compile(r'\b(?:is secure|completely safe|unhackable|bulletproof)\b', re.IGNORECASE),
     "default", "high",
     "Assumes perfect security — no system is completely secure",
     "What threat model? What attack vectors remain?"),
    # Compatibility assumptions
    (re.compile(r'\b(?:works? everywhere|cross-platform|backward.compatible|all (?:browsers?|platforms?))\b', re.IGNORECASE),
     "scope", "medium",
     "Assumes universal compatibility — check specific targets",
     "Which specific platforms/versions were tested?"),
    # Data assumptions
    (re.compile(r'\b(?:data is (?:clean|valid|correct)|no (?:null|missing|bad) (?:data|values))\b', re.IGNORECASE),
     "default", "high",
     "Assumes clean data — real data is messy",
     "What happens with nulls, duplicates, or malformed input?"),
    # Concurrency assumptions
    (re.compile(r'\b(?:no (?:race condition|deadlock)|thread.safe|atomic)\b', re.IGNORECASE),
     "technical", "high",
     "Assumes correct concurrent behavior — verify with testing",
     "What shared mutable state exists? What ordering guarantees?"),
    # Cost assumptions
    (re.compile(r'\b(?:free|no cost|cheap|inexpensive|affordable)\b', re.IGNORECASE),
     "default", "medium",
     "Assumes low cost — hidden costs (ops, maintenance, learning curve) add up",
     "What are the total cost of ownership including ops and training?"),
    # Stationarity assumptions
    (re.compile(r'\b(?:won.t change|stays? the same|constant|static|fixed)\b', re.IGNORECASE),
     "scope", "medium",
     "Assumes a static environment — requirements and conditions change",
     "What happens when this assumption breaks?"),
]

# Implicit premise patterns (X so Y — assumes link)
_IMPLICIT_PREMISE_PATTERNS = [
    re.compile(r'(.{10,50}?)\s+(?:so|therefore|thus|hence)\s+(.{10,50})', re.IGNORECASE),
    re.compile(r'(?:since|because)\s+(.{10,50}?)\s*,\s*(.{10,50})', re.IGNORECASE),
    re.compile(r'(.{10,50}?)\s+(?:means|implies|suggests)\s+(?:that\s+)?(.{10,50})', re.IGNORECASE),
]

# Scope modifiers that indicate hidden assumptions
_SCOPE_MODIFIERS = [
    (re.compile(r'\b(?:always|never|every|all|none|no one|everyone)\b', re.IGNORECASE),
     "Absolute scope — does this truly apply in ALL cases?"),
    (re.compile(r'\b(?:obviously|clearly|of course|naturally|certainly)\b', re.IGNORECASE),
     "Presented as self-evident — but is it actually obvious?"),
    (re.compile(r'\b(?:should|must|need to|have to|ought to)\b', re.IGNORECASE),
     "Normative claim — what standard or requirement is this based on?"),
    (re.compile(r'\b(?:best|worst|optimal|ideal|perfect)\b', re.IGNORECASE),
     "Superlative — best by what criteria and for whom?"),
]


class AssumptionDetector:
    """Detects unstated assumptions in reasoning."""

    def detect(self, text: str) -> List[Assumption]:
        """Detect all assumptions in text."""
        assumptions = []

        # Technical assumptions
        for pat, category, risk, assumption_text, question in _TECHNICAL_ASSUMPTIONS:
            for m in pat.finditer(text):
                assumptions.append(Assumption(
                    text=assumption_text,
                    category=category,
                    trigger=m.group(0),
                    risk=risk,
                    question=question,
                ))

        # Implicit premises
        for pat in _IMPLICIT_PREMISE_PATTERNS:
            for m in pat.finditer(text):
                premise = m.group(1).strip()
                conclusion = m.group(2).strip()
                assumptions.append(Assumption(
                    text=f"Assumes '{premise}' directly leads to '{conclusion}'",
                    category="implicit_premise",
                    trigger=m.group(0),
                    risk="medium",
                    question=f"Is the link between these actually causal?",
                ))

        # Scope issues
        for pat, warning in _SCOPE_MODIFIERS:
            for m in pat.finditer(text):
                # Get surrounding context
                start = max(0, m.start() - 20)
                end = min(len(text), m.end() + 40)
                context = text[start:end].strip()
                assumptions.append(Assumption(
                    text=warning,
                    category="scope",
                    trigger=context,
                    risk="low",
                    question="What are the actual boundaries of this claim?",
                ))

        # Deduplicate by text
        seen = set()
        unique = []
        for a in assumptions:
            if a.text not in seen:
                seen.add(a.text)
                unique.append(a)

        return unique

    def summarize(self, assumptions: List[Assumption]) -> str:
        """Summarize detected assumptions."""
        if not assumptions:
            return "No hidden assumptions detected"

        high = [a for a in assumptions if a.risk == "high"]
        medium = [a for a in assumptions if a.risk == "medium"]
        low = [a for a in assumptions if a.risk == "low"]

        parts = [f"{len(assumptions)} assumptions detected"]
        if high:
            parts.append(f"{len(high)} high-risk")
        if medium:
            parts.append(f"{len(medium)} medium-risk")

        return ", ".join(parts)
