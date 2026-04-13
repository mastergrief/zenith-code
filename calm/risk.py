"""
Auto-CALM Risk Assessment — structured "what could go wrong" analysis.

For any proposal, identifies risks, estimates likelihood and impact,
and suggests mitigations. Forces structured thinking about failure
modes instead of hoping everything works.

Usage:
    from calm.risk import RiskAssessor
    ra = RiskAssessor()
    result = ra.assess("Migrate production database from MySQL to PostgreSQL this weekend")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Risk:
    """A structured risk assessment."""
    description: str
    category: str           # "technical", "operational", "schedule", "security", "data", "people"
    likelihood: str = "medium"  # "low", "medium", "high"
    impact: str = "medium"      # "low", "medium", "high", "critical"
    mitigation: str = ""
    severity_score: int = 0     # 1-9 (likelihood × impact)


@dataclass
class RiskResult:
    """Risk assessment result."""
    risks: List[Risk] = field(default_factory=list)
    overall_risk: str = "unknown"   # "low", "moderate", "high", "critical"
    highest_risk: Optional[Risk] = None
    risk_count_by_severity: dict = field(default_factory=dict)

    def summary(self) -> str:
        if not self.risks:
            return "No risks identified"
        high = sum(1 for r in self.risks if r.severity_score >= 6)
        med = sum(1 for r in self.risks if 3 <= r.severity_score < 6)
        low = sum(1 for r in self.risks if r.severity_score < 3)
        return (f"{len(self.risks)} risks ({high} high, {med} medium, {low} low), "
                f"overall: {self.overall_risk}")

    @property
    def risk_matrix(self) -> str:
        """Format as a risk matrix."""
        lines = ["         Low Impact  Med Impact  High Impact"]
        for likelihood in ["high", "medium", "low"]:
            row_risks = {
                "low": [], "medium": [], "high": [], "critical": [],
            }
            for r in self.risks:
                if r.likelihood == likelihood:
                    row_risks[r.impact].append(r.description[:20])
            lbl = likelihood.ljust(8)
            low_r = ",".join(row_risks["low"][:1]) or "—"
            med_r = ",".join(row_risks["medium"][:1]) or "—"
            high_r = ",".join(row_risks.get("high", [])[:1] + row_risks.get("critical", [])[:1]) or "—"
            lines.append(f"  {lbl} {low_r:12s} {med_r:12s} {high_r}")
        return "\n".join(lines)


_SEVERITY_MATRIX = {
    ("low", "low"): 1, ("low", "medium"): 2, ("low", "high"): 3, ("low", "critical"): 4,
    ("medium", "low"): 2, ("medium", "medium"): 4, ("medium", "high"): 6, ("medium", "critical"): 7,
    ("high", "low"): 3, ("high", "medium"): 6, ("high", "high"): 8, ("high", "critical"): 9,
}

# Risk pattern detection
_RISK_PATTERNS = [
    # Data risks
    (re.compile(r'\b(?:migrat|move|transfer|convert)\b.*\b(?:data|database|db)\b', re.IGNORECASE),
     "data", "high", "high",
     "Data loss or corruption during migration",
     "Full backup before migration. Test with production data copy. Verify row counts and checksums."),
    (re.compile(r'\b(?:delet|drop|remov|purg)\b.*\b(?:data|table|column|index|database)\b', re.IGNORECASE),
     "data", "medium", "critical",
     "Accidental data deletion",
     "Backup first. Use soft deletes. Require confirmation for destructive operations."),
    # Operational risks
    (re.compile(r'\b(?:production|prod|live)\b.*\b(?:deploy|release|push|update|change)\b', re.IGNORECASE),
     "operational", "medium", "high",
     "Production deployment failure",
     "Deploy to staging first. Use blue-green or canary deployment. Have rollback plan."),
    (re.compile(r'\b(?:weekend|night|off.hours|maintenance window)\b', re.IGNORECASE),
     "operational", "medium", "medium",
     "Off-hours work with reduced team availability",
     "Ensure on-call coverage. Pre-stage all changes. Have escalation contacts."),
    (re.compile(r'\b(?:downtime|outage|unavailable|offline)\b', re.IGNORECASE),
     "operational", "high", "high",
     "Service downtime affecting users",
     "Minimize downtime window. Notify users. Have status page ready."),
    # Technical risks
    (re.compile(r'\b(?:rewrite|rebuild|rearchitect|redesign|from scratch)\b', re.IGNORECASE),
     "technical", "high", "high",
     "Scope creep and feature regression from rewrite",
     "Incremental migration preferred. Comprehensive test suite before starting."),
    (re.compile(r'\b(?:third.party|external|vendor|dependency|API)\b.*\b(?:change|update|deprecat)\b', re.IGNORECASE),
     "technical", "medium", "medium",
     "Third-party dependency breaking change",
     "Pin versions. Abstract behind interfaces. Monitor changelogs."),
    (re.compile(r'\b(?:concurren|parallel|async|thread|race condition)\b', re.IGNORECASE),
     "technical", "medium", "high",
     "Concurrency bugs (race conditions, deadlocks)",
     "Use proven concurrency primitives. Stress test under load. Review with concurrency expert."),
    # Schedule risks
    (re.compile(r'\b(?:deadline|due date|milestone|sprint|this week|today|tomorrow|ASAP|urgent)\b', re.IGNORECASE),
     "schedule", "high", "medium",
     "Time pressure leading to shortcuts",
     "Prioritize ruthlessly. Cut scope, not quality. Communicate timeline risks early."),
    # Security risks
    (re.compile(r'\b(?:auth|login|password|credential|token|session|permission|access)\b', re.IGNORECASE),
     "security", "medium", "high",
     "Authentication/authorization vulnerability",
     "Security review before deployment. Use proven auth libraries. Pen test."),
    # People risks
    (re.compile(r'\b(?:only (?:I|one person)|bus factor|single point|key person|knowledge silo)\b', re.IGNORECASE),
     "people", "medium", "high",
     "Key person dependency / bus factor",
     "Document decisions. Pair program. Cross-train team members."),
]


class RiskAssessor:
    """Structured risk assessment for proposals."""

    def assess(self, text: str) -> RiskResult:
        """Identify risks in a proposal or plan."""
        result = RiskResult()

        for pat, category, likelihood, impact, desc, mitigation in _RISK_PATTERNS:
            if pat.search(text):
                severity = _SEVERITY_MATRIX.get((likelihood, impact), 4)
                result.risks.append(Risk(
                    description=desc,
                    category=category,
                    likelihood=likelihood,
                    impact=impact,
                    mitigation=mitigation,
                    severity_score=severity,
                ))

        # Deduplicate by description
        seen = set()
        unique = []
        for r in result.risks:
            if r.description not in seen:
                seen.add(r.description)
                unique.append(r)
        result.risks = unique

        # Overall risk level
        if not result.risks:
            result.overall_risk = "low"
        else:
            max_severity = max(r.severity_score for r in result.risks)
            result.highest_risk = max(result.risks, key=lambda r: r.severity_score)
            if max_severity >= 8:
                result.overall_risk = "critical"
            elif max_severity >= 6:
                result.overall_risk = "high"
            elif max_severity >= 4:
                result.overall_risk = "moderate"
            else:
                result.overall_risk = "low"

        return result
