"""
Auto-CALM Conflict Resolution — reconcile disagreements between modules.

When scope says "overgeneralized" but the user asked for simple, which
wins? When risk says "critical" but prioritize says "quick-win", how
to reconcile? This module provides a framework for resolving inter-module
conflicts based on context and user intent.

Usage:
    from calm.conflict_resolution import ConflictResolver
    cr = ConflictResolver()
    result = cr.resolve([
        ("scope", "overgeneralized", 0.8),
        ("nuance", "properly qualified", 0.9),
    ])
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class ModuleOpinion:
    """One module's assessment."""
    module: str
    assessment: str
    confidence: float       # 0-1
    severity: str = "info"  # "info", "warning", "error"


@dataclass
class Conflict:
    """A disagreement between two modules."""
    module_a: ModuleOpinion
    module_b: ModuleOpinion
    conflict_type: str       # "contradiction", "tension", "scope_mismatch"
    resolution: str = ""
    winner: str = ""         # which module's opinion prevails


@dataclass
class ResolutionResult:
    """Result of conflict resolution."""
    opinions: List[ModuleOpinion] = field(default_factory=list)
    conflicts: List[Conflict] = field(default_factory=list)
    consensus: Optional[str] = None
    action: str = ""

    def summary(self) -> str:
        if not self.conflicts:
            return f"consensus: {self.consensus or 'no conflicts'}"
        parts = [f"{len(self.conflicts)} conflicts"]
        for c in self.conflicts:
            parts.append(f"  {c.module_a.module} vs {c.module_b.module}: {c.resolution}")
        return "\n".join(parts)


# Module priority hierarchy: higher number = more weight in conflicts
_MODULE_PRIORITY = {
    # Safety-critical: highest priority
    "consistency": 10,
    "logic": 10,
    "security": 9,
    # Correctness
    "chain_verify": 8,
    "evidence": 8,
    "calibration": 7,
    # Quality
    "completeness": 6,
    "relevance": 6,
    "scope": 5,
    "precision": 5,
    "nuance": 5,
    # Style (lowest priority — defer to user preference)
    "density": 3,
    "abstraction": 3,
    "communication": 2,
    "creativity": 2,
}

# Known conflict pairs and resolution strategies
_RESOLUTION_RULES = {
    # Scope vs user request for simplicity
    ("scope", "communication"): {
        "rule": "If user asks for simple/brief, scope warnings become informational only",
        "winner_fn": lambda a, b, ctx: "communication" if "simple" in ctx or "brief" in ctx else "scope",
    },
    # Risk vs priority
    ("risk", "prioritize"): {
        "rule": "Risk warnings override priority ranking for critical severity",
        "winner_fn": lambda a, b, ctx: "risk" if "critical" in a.assessment.lower() else "prioritize",
    },
    # Density vs completeness
    ("density", "completeness"): {
        "rule": "Completeness wins — answer everything, then trim filler",
        "winner_fn": lambda a, b, ctx: "completeness",
    },
    # Abstraction vs user level
    ("abstraction", "communication"): {
        "rule": "Match the user's abstraction level, not what seems 'right'",
        "winner_fn": lambda a, b, ctx: "communication",
    },
    # Nuance vs precision
    ("nuance", "precision"): {
        "rule": "Both can coexist — be precise about each branch of nuance",
        "winner_fn": lambda a, b, ctx: "both",
    },
}


class ConflictResolver:
    """Resolves disagreements between cognitive modules."""

    def resolve(self, opinions: List[Tuple[str, str, float]],
                context: str = "") -> ResolutionResult:
        """Resolve conflicts between module opinions.

        Args:
            opinions: List of (module_name, assessment, confidence)
            context: User's original question/context
        """
        result = ResolutionResult()

        # Convert to ModuleOpinion objects
        for module, assessment, confidence in opinions:
            severity = self._classify_severity(assessment)
            result.opinions.append(ModuleOpinion(
                module=module,
                assessment=assessment,
                confidence=confidence,
                severity=severity,
            ))

        # Find conflicts: opposing assessments between modules
        for i in range(len(result.opinions)):
            for j in range(i + 1, len(result.opinions)):
                a = result.opinions[i]
                b = result.opinions[j]

                if self._are_conflicting(a, b):
                    conflict = self._resolve_pair(a, b, context)
                    result.conflicts.append(conflict)

        # Determine consensus
        if not result.conflicts:
            # No conflicts — aggregate opinions
            assessments = [o.assessment for o in result.opinions]
            if all("good" in a.lower() or "valid" in a.lower() or "pass" in a.lower()
                   for a in assessments):
                result.consensus = "all modules agree: positive"
            elif any("error" in a.lower() or "invalid" in a.lower() or "fail" in a.lower()
                     for a in assessments):
                result.consensus = "some modules flag issues"
            else:
                result.consensus = "mixed signals"
        else:
            result.consensus = f"{len(result.conflicts)} conflicts resolved"

        # Determine action
        errors = [o for o in result.opinions if o.severity == "error"]
        if errors:
            result.action = f"Address {len(errors)} error(s) first: " + \
                           ", ".join(f"{e.module}: {e.assessment}" for e in errors[:3])
        else:
            result.action = "Proceed with awareness of warnings"

        return result

    def _classify_severity(self, assessment: str) -> str:
        """Classify assessment severity."""
        lower = assessment.lower()
        if any(w in lower for w in ["invalid", "broken", "fail", "error", "critical", "wrong"]):
            return "error"
        if any(w in lower for w in ["warning", "issue", "concern", "missing", "vague", "weak"]):
            return "warning"
        return "info"

    def _are_conflicting(self, a: ModuleOpinion, b: ModuleOpinion) -> bool:
        """Check if two opinions conflict."""
        # Different severities on the same aspect
        if a.severity != b.severity and a.severity != "info" and b.severity != "info":
            return True

        # Positive vs negative assessment
        positive = {"good", "valid", "pass", "correct", "sound", "complete", "precise"}
        negative = {"bad", "invalid", "fail", "wrong", "broken", "incomplete", "vague"}

        a_pos = any(w in a.assessment.lower() for w in positive)
        a_neg = any(w in a.assessment.lower() for w in negative)
        b_pos = any(w in b.assessment.lower() for w in positive)
        b_neg = any(w in b.assessment.lower() for w in negative)

        if (a_pos and b_neg) or (a_neg and b_pos):
            return True

        return False

    def _resolve_pair(self, a: ModuleOpinion, b: ModuleOpinion,
                       context: str) -> Conflict:
        """Resolve a specific conflict between two modules."""
        conflict = Conflict(
            module_a=a, module_b=b,
            conflict_type="tension",
        )

        # Check for specific resolution rules
        pair_key = (a.module, b.module)
        reverse_key = (b.module, a.module)
        rule = _RESOLUTION_RULES.get(pair_key) or _RESOLUTION_RULES.get(reverse_key)

        if rule:
            conflict.resolution = rule["rule"]
            winner = rule["winner_fn"](a, b, context)
            if winner == "both":
                conflict.winner = "both (complementary)"
            else:
                conflict.winner = winner
        else:
            # Default: higher priority module wins, confidence breaks ties
            pri_a = _MODULE_PRIORITY.get(a.module, 5)
            pri_b = _MODULE_PRIORITY.get(b.module, 5)

            if pri_a > pri_b:
                conflict.winner = a.module
                conflict.resolution = f"{a.module} has higher priority ({pri_a} vs {pri_b})"
            elif pri_b > pri_a:
                conflict.winner = b.module
                conflict.resolution = f"{b.module} has higher priority ({pri_b} vs {pri_a})"
            else:
                # Same priority — higher confidence wins
                conflict.winner = a.module if a.confidence >= b.confidence else b.module
                conflict.resolution = f"Same priority, {conflict.winner} has higher confidence"

        return conflict
