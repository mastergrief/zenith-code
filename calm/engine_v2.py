"""
CALM Engine v2 — full cognitive pipeline with self-healing quality loop.

The complete pipeline:
  1. PRE-ANALYZE: disambiguate, profile user, decompose, assess risk
  2. ENRICH: inject pre-analysis into system prompt
  3. PRECOMPUTE: inject verified backend facts
  4. GENERATE: model produces response
  5. VERIFY: check computational claims (Layer 1)
  6. ANALYZE: cognitive router runs relevant modules
  7. SELF-HEAL: if quality issues found, generate targeted correction
  8. RE-VERIFY: check corrected response

Usage:
    from calm.engine_v2 import CalmEngineV2
    engine = CalmEngineV2()
    result = engine.run("Should I use Redis or Memcached?")
    print(result.response)
    print(result.quality_report.summary())
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from calm.auto_calm import AutoCalmEngine, AutoCalmResult, AUTO_SYSTEM_PROMPT
from calm.router import CognitiveRouter, CognitiveReport
from calm.verify import Claim
from calm.adaptive import AdaptiveBudget
from calm.conversation import ConversationState


@dataclass
class QualityIssue:
    """A quality issue that can be fed back for correction."""
    module: str
    issue: str
    correction_prompt: str  # what to tell the model to fix


@dataclass
class EngineV2Result:
    """Full result from the v2 pipeline."""
    response: str = ""
    original_response: str = ""
    corrected: bool = False          # whether self-healing fired
    correction_reason: str = ""
    # From Auto-CALM
    claims_verified: int = 0
    claims_corrected: int = 0
    corrections: List[Claim] = field(default_factory=list)
    tok_per_sec: float = 0.0
    # From cognitive router
    quality_report: Optional[CognitiveReport] = None
    quality_before: float = 0.0      # quality score before self-heal
    quality_after: float = 0.0       # quality score after self-heal
    # Pre-analysis
    user_expertise: str = ""
    ambiguities_found: int = 0
    risks_found: int = 0
    # Adaptive thinking
    thinking_budget_used: int = 0
    thinking_tier: str = ""
    # Cross-turn insights
    cross_turn_insights: dict = field(default_factory=dict)
    # Timing
    total_ms: float = 0
    pre_analysis_ms: float = 0
    generation_ms: float = 0
    analysis_ms: float = 0
    correction_ms: float = 0

    def summary(self) -> str:
        parts = [f"quality={self.quality_after:.0%}"]
        if self.corrected:
            parts.append(f"self-healed ({self.quality_before:.0%}→{self.quality_after:.0%})")
        parts.append(f"{self.claims_verified} verified, {self.claims_corrected} corrected")
        parts.append(f"{self.total_ms:.0f}ms total")
        return ", ".join(parts)


# Quality threshold for triggering self-healing
_QUALITY_THRESHOLD = 0.75
# Maximum issues to include in correction prompt
_MAX_CORRECTION_ITEMS = 5
# Minimum severity to trigger self-heal
_MIN_ISSUES_FOR_HEAL = 3


class CalmEngineV2:
    """Full cognitive pipeline with self-healing quality loop."""

    def __init__(
        self,
        server: str = "http://localhost:8080",
        thinking_budget: int = 8192,
        max_tokens: int = 16384,
        quality_threshold: float = _QUALITY_THRESHOLD,
        self_heal: bool = True,
        max_heal_rounds: int = 1,
    ):
        self.server = server
        self.thinking_budget = thinking_budget
        self.max_tokens = max_tokens
        self.quality_threshold = quality_threshold
        self.self_heal = self_heal
        self.max_heal_rounds = max_heal_rounds

        self._calm = AutoCalmEngine(
            server=server,
            thinking_budget=thinking_budget,
            max_tokens=max_tokens,
        )
        self._router = CognitiveRouter()
        self._adaptive = AdaptiveBudget()
        self._conversation = ConversationState()

    def run(self, prompt: str, verbose: bool = False) -> EngineV2Result:
        """Run the full pipeline."""
        result = EngineV2Result()
        t_start = time.time()

        # === PHASE 1: PRE-ANALYZE ===
        t0 = time.time()
        pre_analysis = self._pre_analyze(prompt, verbose)
        result.pre_analysis_ms = (time.time() - t0) * 1000
        result.user_expertise = pre_analysis.get("expertise", "")
        result.ambiguities_found = pre_analysis.get("ambiguities", 0)
        result.risks_found = pre_analysis.get("risks", 0)

        # === PHASE 2: ENRICH SYSTEM PROMPT ===
        enriched_prompt = self._enrich_system_prompt(pre_analysis)

        # === PHASE 2.5: ADAPTIVE THINKING BUDGET ===
        from calm.precompute import precompute as _precompute
        precomputed = _precompute(prompt) if self._calm.precompute_enabled else {}
        budget_estimate = self._adaptive.estimate(prompt, precomputed, pre_analysis)
        adaptive_budget = budget_estimate.budget
        result.thinking_budget_used = adaptive_budget
        result.thinking_tier = budget_estimate.tier

        if verbose:
            print(f"[adaptive] {budget_estimate}")

        # === PHASE 3-5: GENERATE + VERIFY (via Auto-CALM) ===
        t0 = time.time()
        # Temporarily override the system prompt and thinking budget
        old_prompt = self._calm.system_prompt
        old_budget = self._calm.thinking_budget
        self._calm.system_prompt = enriched_prompt
        self._calm.thinking_budget = adaptive_budget
        calm_result = self._calm.run(prompt, verbose=verbose)
        self._calm.system_prompt = old_prompt
        self._calm.thinking_budget = old_budget
        result.generation_ms = (time.time() - t0) * 1000

        result.response = calm_result.response
        result.original_response = calm_result.original_response
        result.claims_verified = calm_result.claims_verified
        result.claims_corrected = calm_result.claims_corrected
        result.corrections = calm_result.corrections
        result.tok_per_sec = calm_result.tok_per_sec

        # === PHASE 6: COGNITIVE ANALYSIS ===
        t0 = time.time()
        report = self._router.analyze(prompt, result.response)
        result.quality_report = report
        result.quality_before = report.overall_quality
        result.quality_after = report.overall_quality
        result.analysis_ms = (time.time() - t0) * 1000

        if verbose:
            print(f"[router] {report.modules_run} modules, "
                  f"quality={report.overall_quality:.0%}, "
                  f"{report.total_issues} issues ({result.analysis_ms:.0f}ms)")

        # === PHASE 6.5: CROSS-TURN STATE ===
        insights = self._conversation.add_turn(
            prompt=prompt,
            response=result.response,
            quality_score=report.overall_quality,
            claims_verified=result.claims_verified,
            claims_corrected=result.claims_corrected,
            issues_found=report.total_issues,
        )
        result.cross_turn_insights = insights

        if verbose and insights:
            for key, value in insights.items():
                print(f"[cross-turn] {key}: {value}")

        # === PHASE 7: SELF-HEAL ===
        if (self.self_heal and
            report.overall_quality < self.quality_threshold and
            report.total_issues >= _MIN_ISSUES_FOR_HEAL):

            t0 = time.time()
            healed = self._self_heal(prompt, result.response, report, verbose)
            result.correction_ms = (time.time() - t0) * 1000

            if healed:
                result.response = healed["response"]
                result.corrected = True
                result.correction_reason = healed["reason"]
                result.quality_after = healed["quality_after"]

                if verbose:
                    print(f"[self-heal] quality {result.quality_before:.0%} → "
                          f"{result.quality_after:.0%}: {healed['reason']}")

        result.total_ms = (time.time() - t_start) * 1000
        return result

    def _pre_analyze(self, prompt: str, verbose: bool) -> Dict:
        """Phase 1: Pre-analyze the prompt before generating."""
        analysis = {}

        # Communication profiling
        from calm.communication import CommunicationAdapter
        ca = CommunicationAdapter()
        profile = ca.analyze_user(prompt)
        style = ca.recommend_style(profile)
        analysis["expertise"] = profile.expertise
        analysis["style"] = style

        # Disambiguation
        from calm.disambiguation import Disambiguator
        d = Disambiguator()
        disamb = d.check(prompt)
        analysis["ambiguities"] = len(disamb.ambiguities)
        analysis["clarifications"] = disamb.clarifying_questions[:3]

        # Decomposition
        from calm.decompose import Decomposer
        decomp = Decomposer()
        plan = decomp.decompose(prompt)
        analysis["sub_problems"] = [s.question for s in plan.steps[:5]]
        analysis["complexity"] = plan.complexity

        # Risk assessment
        from calm.risk import RiskAssessor
        ra = RiskAssessor()
        risk = ra.assess(prompt)
        analysis["risks"] = len(risk.risks)
        analysis["risk_level"] = risk.overall_risk
        analysis["risk_items"] = [r.description for r in risk.risks[:3]]

        if verbose:
            print(f"[pre] user={profile.expertise}, "
                  f"ambiguities={len(disamb.ambiguities)}, "
                  f"sub-problems={len(plan.steps)}, "
                  f"risk={risk.overall_risk}")

        return analysis

    def _enrich_system_prompt(self, pre_analysis: Dict) -> str:
        """Phase 2: Build an enriched system prompt from pre-analysis."""
        base = AUTO_SYSTEM_PROMPT
        additions = []

        # User expertise adaptation
        style = pre_analysis.get("style")
        if style:
            expertise = pre_analysis.get("expertise", "intermediate")
            if expertise == "expert":
                additions.append(
                    "The user is an expert. Be concise, use technical jargon, "
                    "skip basic explanations, focus on root causes and tradeoffs."
                )
            elif expertise == "beginner":
                additions.append(
                    "The user is a beginner. Define technical terms, use analogies, "
                    "include examples, explain step by step."
                )

        # Ambiguity warnings
        clarifications = pre_analysis.get("clarifications", [])
        if clarifications:
            additions.append(
                "Note: the question may be ambiguous. Address the most likely "
                "interpretation but mention alternatives: " +
                "; ".join(clarifications[:2])
            )

        # Decomposition guidance
        sub_problems = pre_analysis.get("sub_problems", [])
        if len(sub_problems) >= 3:
            additions.append(
                "This is a complex question. Address each aspect: " +
                "; ".join(sub_problems[:4])
            )

        # Risk awareness
        risk_items = pre_analysis.get("risk_items", [])
        if risk_items:
            additions.append(
                "Risks to mention: " + "; ".join(risk_items[:2])
            )

        # Quality instructions
        additions.append(
            "Be specific (avoid 'fast', 'easy', 'simple' without metrics). "
            "Qualify generalizations (avoid 'always', 'never' without evidence). "
            "If the answer depends on context, branch explicitly (if X then Y)."
        )

        if additions:
            return base + "\n\n" + "\n".join(additions)
        return base

    def _self_heal(self, prompt: str, response: str,
                    report: CognitiveReport, verbose: bool) -> Optional[Dict]:
        """Phase 7: Generate a correction based on cognitive analysis."""
        # Build a targeted correction prompt from the issues
        issues = self._extract_actionable_issues(report)
        if not issues:
            return None

        correction_parts = []
        for issue in issues[:_MAX_CORRECTION_ITEMS]:
            correction_parts.append(f"- {issue.correction_prompt}")

        correction_prompt = (
            "Your previous response had these quality issues:\n" +
            "\n".join(correction_parts) +
            "\n\nPlease provide an improved response that addresses these issues. "
            "Keep what was correct, fix what was flagged."
        )

        if verbose:
            print(f"[self-heal] {len(issues)} issues, regenerating...")
            for issue in issues[:3]:
                print(f"  fix: {issue.correction_prompt[:80]}")

        # Generate corrected response
        messages = [
            {"role": "system", "content": self._calm.system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
            {"role": "user", "content": correction_prompt},
        ]

        try:
            content, thinking, timings = self._calm._generate(messages)
        except Exception as e:
            if verbose:
                print(f"[self-heal] generation failed: {e}")
            return None

        # Verify the correction is actually better
        new_report = self._router.analyze(prompt, content)
        if new_report.overall_quality > report.overall_quality:
            return {
                "response": content,
                "reason": f"{len(issues)} issues fixed",
                "quality_after": new_report.overall_quality,
            }
        else:
            if verbose:
                print(f"[self-heal] correction not better "
                      f"({new_report.overall_quality:.0%} vs {report.overall_quality:.0%}), keeping original")
            return None

    def _extract_actionable_issues(self, report: CognitiveReport) -> List[QualityIssue]:
        """Extract issues that can be turned into correction prompts."""
        issues = []

        for r in report.results:
            if r.issues_found == 0:
                continue

            # Map module results to actionable corrections
            if r.module_name == "scope" and r.result:
                for scope_issue in r.result[:3]:
                    issues.append(QualityIssue(
                        module="scope",
                        issue=str(scope_issue),
                        correction_prompt=f"Replace '{scope_issue.trigger}' with a qualified statement: {scope_issue.suggestion}",
                    ))

            elif r.module_name == "precision" and r.result:
                for vague in r.result.vague_terms[:3]:
                    issues.append(QualityIssue(
                        module="precision",
                        issue=f"vague: {vague.term}",
                        correction_prompt=f"'{vague.term}' is vague. {vague.suggestion}",
                    ))

            elif r.module_name == "perspective" and r.result:
                for missing in r.result.missing[:2]:
                    from calm.perspective import PerspectiveChecker
                    pc = PerspectiveChecker()
                    questions = pc.suggest_questions([missing])
                    if questions:
                        issues.append(QualityIssue(
                            module="perspective",
                            issue=f"missing: {missing}",
                            correction_prompt=f"Missing {missing} perspective. {questions[0]}",
                        ))

            elif r.module_name == "completeness" and r.result:
                for unanswered in r.result.unanswered[:2]:
                    issues.append(QualityIssue(
                        module="completeness",
                        issue=f"unanswered: {unanswered}",
                        correction_prompt=f"You didn't address: {unanswered}",
                    ))

            elif r.module_name == "explanation" and r.result:
                if r.result.is_circular:
                    issues.append(QualityIssue(
                        module="explanation",
                        issue="circular",
                        correction_prompt="Your explanation is circular — explain the mechanism, not just restate the question.",
                    ))
                for jargon in r.result.jargon_undefined[:2]:
                    issues.append(QualityIssue(
                        module="explanation",
                        issue=f"undefined: {jargon}",
                        correction_prompt=f"You used '{jargon}' without defining it. Add a brief explanation.",
                    ))

            elif r.module_name == "density" and r.result:
                if r.result.filler_ratio > 0.3:
                    issues.append(QualityIssue(
                        module="density",
                        issue="filler",
                        correction_prompt="Remove filler phrases ('Sure!', 'Great question!', 'I hope this helps!'). Get to the point.",
                    ))

            elif r.module_name == "relevance" and r.result:
                if r.result.score < 0.5:
                    issues.append(QualityIssue(
                        module="relevance",
                        issue="off-topic",
                        correction_prompt="Your response doesn't directly address the question. Focus on what was asked.",
                    ))

            elif r.module_name == "disambiguation" and r.result:
                if r.result.is_ambiguous:
                    issues.append(QualityIssue(
                        module="disambiguation",
                        issue="ambiguous",
                        correction_prompt=f"The question is ambiguous. State your interpretation explicitly before answering.",
                    ))

        return issues
