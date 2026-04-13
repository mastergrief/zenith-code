"""
Auto-CALM Metacognition — self-assessment of reasoning quality.

Integrates consistency, evidence, scope, and calibration into a single
"how good is this response?" assessment. The meta-cognitive layer that
sits above all other cognitive modules.

Produces a structured quality report for any model output.

Usage:
    from calm.metacognition import MetaCognition
    mc = MetaCognition()
    report = mc.assess(prompt, response)
    print(report.overall_quality)
    print(report.improvement_suggestions)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class QualityDimension:
    """One dimension of quality assessment."""
    name: str
    score: float       # 0-1
    details: str
    suggestions: List[str] = field(default_factory=list)


@dataclass
class MetaCognitiveReport:
    """Full quality report for a response."""
    dimensions: List[QualityDimension] = field(default_factory=list)
    overall_score: float = 0.0
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)

    @property
    def quality_label(self) -> str:
        if self.overall_score >= 0.8:
            return "high"
        if self.overall_score >= 0.6:
            return "moderate"
        if self.overall_score >= 0.4:
            return "low"
        return "poor"

    def summary(self) -> str:
        lines = [f"Quality: {self.quality_label} ({self.overall_score:.0%})"]
        for d in self.dimensions:
            marker = "+" if d.score >= 0.7 else "-" if d.score < 0.4 else "~"
            lines.append(f"  [{marker}] {d.name}: {d.score:.0%} — {d.details}")
        if self.improvement_suggestions:
            lines.append("Suggestions:")
            for s in self.improvement_suggestions[:3]:
                lines.append(f"  → {s}")
        return "\n".join(lines)


class MetaCognition:
    """Self-assessment of reasoning quality."""

    def assess(self, prompt: str, response: str,
               thinking: str = "") -> MetaCognitiveReport:
        """Assess the quality of a response across multiple dimensions."""
        report = MetaCognitiveReport()

        # 1. Scope precision
        from calm.scope import ScopeTracker
        st = ScopeTracker()
        scope_score, scope_label = st.score(response)
        scope_issues = st.check(response)
        report.dimensions.append(QualityDimension(
            name="Scope Precision",
            score=scope_score,
            details=scope_label,
            suggestions=[str(i) for i in scope_issues[:2]],
        ))

        # 2. Evidence quality
        from calm.evidence import EvidenceTracker
        et = EvidenceTracker()
        evidence = et.analyze_text(response)
        sourced = sum(1 for v in evidence.values() if v in ("single_source", "multi_source", "verified"))
        total = len(evidence) if evidence else 1
        evidence_score = sourced / total if total > 0 else 0.5
        report.dimensions.append(QualityDimension(
            name="Evidence Quality",
            score=evidence_score,
            details=f"{sourced}/{total} claims sourced",
            suggestions=["Add sources for unsupported claims"] if evidence_score < 0.5 else [],
        ))

        # 3. Assumption transparency
        from calm.assumptions import AssumptionDetector
        ad = AssumptionDetector()
        assumptions = ad.detect(response)
        high_risk = [a for a in assumptions if a.risk == "high"]
        assumption_score = max(0, 1.0 - len(high_risk) * 0.2)
        report.dimensions.append(QualityDimension(
            name="Assumption Transparency",
            score=assumption_score,
            details=f"{len(assumptions)} assumptions ({len(high_risk)} high-risk)",
            suggestions=[a.question for a in high_risk[:2]],
        ))

        # 4. Nuance / Qualification
        from calm.nuance import NuanceDetector
        nd = NuanceDetector()
        prompt_nuance = nd.analyze_prompt(prompt)
        if prompt_nuance.needs_qualification:
            response_nuance = nd.analyze_response(response)
            nuance_score = 0.8 if response_nuance.is_well_qualified else 0.3
            report.dimensions.append(QualityDimension(
                name="Nuance",
                score=nuance_score,
                details=response_nuance.summary,
                suggestions=[] if response_nuance.is_well_qualified else
                            ["Add structured branching (if X then Y, if Z then W)"],
            ))
        else:
            report.dimensions.append(QualityDimension(
                name="Nuance",
                score=0.9,
                details="Direct answer appropriate",
            ))

        # 5. Completeness (does response address the prompt?)
        from calm.decompose import Decomposer
        d = Decomposer()
        plan = d.decompose(prompt)
        if plan.steps:
            addressed = 0
            for step in plan.steps:
                # Check if any keywords from the sub-problem appear in the response
                keywords = set(step.question.lower().split()) - {
                    "what", "is", "the", "a", "an", "of", "how", "do", "you", "are",
                    "can", "does", "for", "to", "in", "and", "or", "this", "that",
                }
                if any(kw in response.lower() for kw in keywords if len(kw) > 3):
                    addressed += 1
            completeness = addressed / len(plan.steps)
        else:
            completeness = 0.7  # default if we can't decompose
        report.dimensions.append(QualityDimension(
            name="Completeness",
            score=completeness,
            details=f"{addressed if plan.steps else '?'}/{len(plan.steps) if plan.steps else '?'} aspects addressed",
        ))

        # 6. Chain soundness (if thinking is available)
        if thinking:
            from calm.chain_verify import ChainVerifier
            cv = ChainVerifier()
            chain = cv.extract_chain(thinking)
            if chain:
                cr = cv.verify_chain(chain)
                chain_score = 1.0 if cr.is_sound else max(0, 1.0 - cr.wrong_steps * 0.3)
                report.dimensions.append(QualityDimension(
                    name="Reasoning Chain",
                    score=chain_score,
                    details=cr.summary(),
                ))

        # Overall score = weighted average
        weights = {
            "Scope Precision": 0.15,
            "Evidence Quality": 0.2,
            "Assumption Transparency": 0.15,
            "Nuance": 0.15,
            "Completeness": 0.2,
            "Reasoning Chain": 0.15,
        }
        total_weight = sum(weights.get(d.name, 0.1) for d in report.dimensions)
        report.overall_score = sum(
            d.score * weights.get(d.name, 0.1)
            for d in report.dimensions
        ) / total_weight if total_weight > 0 else 0

        # Identify strengths and weaknesses
        for d in report.dimensions:
            if d.score >= 0.8:
                report.strengths.append(f"{d.name}: {d.details}")
            elif d.score < 0.5:
                report.weaknesses.append(f"{d.name}: {d.details}")
                report.improvement_suggestions.extend(d.suggestions)

        return report
