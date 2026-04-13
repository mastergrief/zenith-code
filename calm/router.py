"""
Auto-CALM Cognitive Router — auto-selects and runs relevant modules.

Given a prompt and response, scores each cognitive module's relevance,
runs the top-N most relevant ones, aggregates results, and resolves
conflicts. This turns 33 standalone modules into a unified system.

Each module registers itself with trigger conditions. The router
matches conditions against the prompt/response and runs matches.

Usage:
    from calm.router import CognitiveRouter
    router = CognitiveRouter()
    report = router.analyze(prompt, response)
    print(report.summary())
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional, Any


@dataclass
class ModuleResult:
    """Result from one cognitive module."""
    module_name: str
    relevance: float        # 0-1, how relevant this module was
    result: Any = None       # the module's output
    summary: str = ""        # one-line summary
    issues_found: int = 0    # number of issues/warnings
    elapsed_ms: float = 0    # execution time


@dataclass
class CognitiveReport:
    """Aggregated report from all relevant modules."""
    prompt: str = ""
    modules_checked: int = 0
    modules_run: int = 0
    results: List[ModuleResult] = field(default_factory=list)
    overall_quality: float = 0.0
    total_issues: int = 0
    top_issues: List[str] = field(default_factory=list)
    elapsed_ms: float = 0

    def summary(self) -> str:
        lines = [
            f"Cognitive analysis: {self.modules_run}/{self.modules_checked} modules "
            f"({self.elapsed_ms:.0f}ms), quality={self.overall_quality:.0%}, "
            f"{self.total_issues} issues"
        ]
        for r in self.results:
            if r.issues_found > 0:
                marker = "!" if r.issues_found > 0 else " "
                lines.append(f"  [{marker}] {r.module_name}: {r.summary}")
            elif r.summary:
                lines.append(f"  [ ] {r.module_name}: {r.summary}")
        if self.top_issues:
            lines.append("Top issues:")
            for issue in self.top_issues[:5]:
                lines.append(f"  → {issue}")
        return "\n".join(lines)


@dataclass
class ModuleRegistration:
    """Registration info for a cognitive module."""
    name: str
    triggers: List[re.Pattern]         # when to consider running this module
    always_run: bool = False           # run on every prompt
    run_fn: Optional[Callable] = None  # function(prompt, response) → (summary, issues_count, result)
    category: str = ""                 # "verification", "reasoning", "quality", "meta"
    cost: str = "low"                  # "low", "medium", "high" — execution cost


class CognitiveRouter:
    """Routes prompts/responses to relevant cognitive modules."""

    def __init__(self, max_modules: int = 10):
        self._modules: List[ModuleRegistration] = []
        self._max_modules = max_modules
        self._register_all()

    def _register_all(self):
        """Register all cognitive modules with their triggers."""

        # === VERIFICATION LAYER (always run) ===

        self._register("chain_verify",
            triggers=[],
            always_run=False,
            category="verification",
            cost="low",
            run_fn=self._run_chain_verify,
            trigger_patterns=[
                r'step\s+\d|then|therefore|first.*second|calculate.*then',
            ],
        )

        self._register("consistency",
            triggers=[],
            always_run=False,  # needs conversation history
            category="verification",
            cost="low",
            run_fn=self._run_consistency,
            trigger_patterns=[],
        )

        self._register("logic",
            triggers=[],
            category="verification",
            cost="low",
            run_fn=self._run_logic,
            trigger_patterns=[
                r'therefore|thus|hence|so\s+\w+\s+(?:is|are|must)',
                r'all\s+\w+\s+are|no\s+\w+\s+are|if\s+.+then',
                r'because|since|implies|proves',
            ],
        )

        self._register("scope",
            triggers=[],
            always_run=True,  # always check for overgeneralization
            category="verification",
            cost="low",
            run_fn=self._run_scope,
        )

        # === REASONING LAYER ===

        self._register("decompose",
            triggers=[],
            category="reasoning",
            cost="low",
            run_fn=self._run_decompose,
            trigger_patterns=[
                r'how\s+(?:do|should|can|to)',
                r'compare|vs\.?|versus|difference between',
                r'migrate|optimize|debug|fix|design|build|implement',
            ],
        )

        self._register("causal",
            triggers=[],
            category="reasoning",
            cost="low",
            run_fn=self._run_causal,
            trigger_patterns=[
                r'depends? on|requires?|causes?|leads? to|because|if.*then',
                r'what (?:happens|breaks|changes) (?:if|when)',
                r'impact|effect|consequence|downstream',
            ],
        )

        self._register("assumptions",
            triggers=[],
            always_run=True,  # always check for hidden assumptions
            category="reasoning",
            cost="low",
            run_fn=self._run_assumptions,
        )

        self._register("temporal",
            triggers=[],
            category="reasoning",
            cost="low",
            run_fn=self._run_temporal,
            trigger_patterns=[
                r'before|after|first|then|next|step\s+\d',
                r'sequence|order|timeline|schedule|when',
            ],
        )

        self._register("hypothesis",
            triggers=[],
            category="reasoning",
            cost="low",
            run_fn=self._run_hypothesis,
            trigger_patterns=[
                r'error|bug|broken|fail|crash|slow|timeout|500',
                r'why\s+(?:does|is|did)|what\s+(?:caused|went wrong)',
                r'debug|diagnose|troubleshoot|investigate',
            ],
        )

        # === QUALITY LAYER ===

        self._register("nuance",
            triggers=[],
            category="quality",
            cost="low",
            run_fn=self._run_nuance,
            trigger_patterns=[
                r'(?:better|worse|faster|easier)\s+than',
                r'should I|which is|compare|vs',
                r'pros? and cons?|tradeoffs?|advantages?',
            ],
        )

        self._register("completeness",
            triggers=[],
            category="quality",
            cost="low",
            run_fn=self._run_completeness,
            trigger_patterns=[
                r'(?:,|and)\s+(?:and\s+)?(?:how|what|why|when)',
                r'\?\s*\w+.*\?',  # multiple questions
            ],
        )

        self._register("relevance",
            triggers=[],
            always_run=True,  # always check
            category="quality",
            cost="low",
            run_fn=self._run_relevance,
        )

        self._register("density",
            triggers=[],
            always_run=True,
            category="quality",
            cost="low",
            run_fn=self._run_density,
        )

        self._register("precision",
            triggers=[],
            always_run=True,
            category="quality",
            cost="low",
            run_fn=self._run_precision,
        )

        self._register("explanation",
            triggers=[],
            category="quality",
            cost="low",
            run_fn=self._run_explanation,
            trigger_patterns=[
                r'why|how does|explain|what causes',
            ],
        )

        # === META LAYER ===

        self._register("disambiguation",
            triggers=[],
            category="meta",
            cost="low",
            run_fn=self._run_disambiguation,
            trigger_patterns=[
                r'\b(?:table|index|key|node|port|model|service|token|driver|pool|link)\b',
                r'fix\s+(?:the|this)|improve\s+(?:the|this)|best\s+way',
            ],
        )

        self._register("perspective",
            triggers=[],
            category="meta",
            cost="low",
            run_fn=self._run_perspective,
            trigger_patterns=[
                r'should|design|architect|plan|decide|choose|strategy',
                r'deploy|release|build|implement|create',
            ],
        )

        self._register("risk",
            triggers=[],
            category="meta",
            cost="low",
            run_fn=self._run_risk,
            trigger_patterns=[
                r'deploy|release|migrate|production|prod',
                r'delete|drop|remove|rewrite|rebuild',
                r'security|auth|permission|credential',
            ],
        )

        self._register("communication",
            triggers=[],
            always_run=True,  # always profile the user
            category="meta",
            cost="low",
            run_fn=self._run_communication,
        )

    def _register(self, name: str, triggers: list, category: str,
                   cost: str, run_fn: Callable,
                   always_run: bool = False,
                   trigger_patterns: list = None):
        """Register a module with compiled trigger patterns."""
        compiled = [re.compile(p, re.IGNORECASE) for p in (trigger_patterns or [])]
        compiled.extend(triggers)
        self._modules.append(ModuleRegistration(
            name=name, triggers=compiled, always_run=always_run,
            run_fn=run_fn, category=category, cost=cost,
        ))

    def analyze(self, prompt: str, response: str,
                thinking: str = "") -> CognitiveReport:
        """Run relevant cognitive modules and produce aggregated report."""
        report = CognitiveReport(prompt=prompt)
        t_start = time.time()

        # Score relevance for each module
        scored = []
        for mod in self._modules:
            report.modules_checked += 1
            if mod.always_run:
                scored.append((mod, 1.0))
            else:
                relevance = self._score_relevance(mod, prompt, response)
                if relevance > 0.1:
                    scored.append((mod, relevance))

        # Sort by relevance, take top N
        scored.sort(key=lambda x: x[1], reverse=True)
        to_run = scored[:self._max_modules]

        # Run each module
        for mod, relevance in to_run:
            if mod.run_fn is None:
                continue
            t0 = time.time()
            try:
                summary, issues, result = mod.run_fn(prompt, response, thinking)
                elapsed = (time.time() - t0) * 1000
                mr = ModuleResult(
                    module_name=mod.name,
                    relevance=relevance,
                    result=result,
                    summary=summary,
                    issues_found=issues,
                    elapsed_ms=elapsed,
                )
                report.results.append(mr)
                report.modules_run += 1
                report.total_issues += issues
                if issues > 0:
                    report.top_issues.append(f"[{mod.name}] {summary}")
            except Exception as e:
                report.results.append(ModuleResult(
                    module_name=mod.name,
                    relevance=relevance,
                    summary=f"error: {e}",
                ))

        # Overall quality = weighted average of module scores
        if report.results:
            quality_scores = []
            for r in report.results:
                if r.issues_found == 0:
                    quality_scores.append(1.0)
                else:
                    quality_scores.append(max(0, 1.0 - r.issues_found * 0.15))
            report.overall_quality = sum(quality_scores) / len(quality_scores)

        report.elapsed_ms = (time.time() - t_start) * 1000
        return report

    def _score_relevance(self, mod: ModuleRegistration,
                          prompt: str, response: str) -> float:
        """Score how relevant a module is to this prompt/response."""
        if not mod.triggers:
            return 0.0
        text = prompt + " " + response
        hits = sum(1 for pat in mod.triggers if pat.search(text))
        return min(1.0, hits / max(len(mod.triggers), 1))

    # === Module runners ===
    # Each returns (summary: str, issues: int, result: Any)

    def _run_scope(self, prompt, response, thinking):
        from calm.scope import ScopeTracker
        st = ScopeTracker()
        issues = st.check(response)
        score, label = st.score(response)
        return f"{label} ({score:.0%})", len([i for i in issues if i.issue_type == "overgeneralization"]), issues

    def _run_assumptions(self, prompt, response, thinking):
        from calm.assumptions import AssumptionDetector
        ad = AssumptionDetector()
        assumptions = ad.detect(response)
        high = [a for a in assumptions if a.risk == "high"]
        return f"{len(assumptions)} assumptions ({len(high)} high-risk)", len(high), assumptions

    def _run_relevance(self, prompt, response, thinking):
        from calm.relevance import RelevanceChecker
        rc = RelevanceChecker()
        r = rc.check(prompt, response)
        issues = 1 if r.score < 0.5 else 0
        return f"{r.label} ({r.score:.0%})", issues, r

    def _run_density(self, prompt, response, thinking):
        from calm.density import DensityAnalyzer
        da = DensityAnalyzer()
        r = da.analyze(response)
        issues = 1 if r.filler_ratio > 0.3 else 0
        return f"{r.label} ({r.density_score:.0%})", issues, r

    def _run_precision(self, prompt, response, thinking):
        from calm.precision import PrecisionChecker
        pc = PrecisionChecker()
        r = pc.check(response)
        issues = len(r.vague_terms)
        return f"{r.label} ({r.precision_score:.0%}), {issues} vague terms", min(issues, 3), r

    def _run_communication(self, prompt, response, thinking):
        from calm.communication import CommunicationAdapter
        ca = CommunicationAdapter()
        profile = ca.analyze_user(prompt)
        style = ca.recommend_style(profile)
        return f"{profile.expertise} user → {style.verbosity} style", 0, (profile, style)

    def _run_disambiguation(self, prompt, response, thinking):
        from calm.disambiguation import Disambiguator
        d = Disambiguator()
        r = d.check(prompt)
        return r.summary(), len(r.ambiguities), r

    def _run_nuance(self, prompt, response, thinking):
        from calm.nuance import NuanceDetector
        nd = NuanceDetector()
        result = nd.qualify_check(prompt, response)
        issues = 1 if "NEEDS WORK" in result else 0
        return result[:60], issues, result

    def _run_completeness(self, prompt, response, thinking):
        from calm.completeness import CompletenessChecker
        cc = CompletenessChecker()
        r = cc.check(prompt, response)
        issues = len(r.unanswered)
        return r.summary(), issues, r

    def _run_explanation(self, prompt, response, thinking):
        from calm.explanation import ExplanationChecker
        ec = ExplanationChecker()
        r = ec.check(prompt, response)
        return r.summary(), len(r.issues), r

    def _run_logic(self, prompt, response, thinking):
        from calm.logic import LogicVerifier
        lv = LogicVerifier()
        r = lv.check_argument(response)
        issues = len(r.fallacies)
        return r.summary(), issues, r

    def _run_decompose(self, prompt, response, thinking):
        from calm.decompose import Decomposer
        d = Decomposer()
        plan = d.decompose(prompt)
        return plan.summary(), 0, plan

    def _run_causal(self, prompt, response, thinking):
        from calm.causal import CausalEngine
        ce = CausalEngine()
        count = ce.add_from_text(response)
        return f"{count} causal relationships extracted", 0, ce

    def _run_temporal(self, prompt, response, thinking):
        from calm.temporal import TemporalReasoner
        tr = TemporalReasoner()
        issues = tr.add(response)
        return f"{len(issues)} temporal issues", len(issues), tr

    def _run_hypothesis(self, prompt, response, thinking):
        from calm.hypothesis_gen import HypothesisEngine
        he = HypothesisEngine()
        r = he.generate(prompt)
        top = r.ranked[0].description[:50] if r.ranked else "none"
        return f"{len(r.hypotheses)} hypotheses, top: {top}", 0, r

    def _run_perspective(self, prompt, response, thinking):
        from calm.perspective import PerspectiveChecker
        pc = PerspectiveChecker()
        r = pc.full_check(response)
        issues = len(r.missing)
        return r.summary(), min(issues, 2), r

    def _run_risk(self, prompt, response, thinking):
        from calm.risk import RiskAssessor
        ra = RiskAssessor()
        r = ra.assess(prompt + " " + response)
        issues = sum(1 for risk in r.risks if risk.severity_score >= 6)
        return r.summary(), issues, r

    def _run_chain_verify(self, prompt, response, thinking):
        from calm.chain_verify import ChainVerifier
        cv = ChainVerifier()
        text = thinking if thinking else response
        chain = cv.extract_chain(text)
        if not chain:
            return "no reasoning chain found", 0, None
        r = cv.verify_chain(chain)
        issues = r.wrong_steps
        return r.summary(), issues, r

    def _run_consistency(self, prompt, response, thinking):
        from calm.consistency import ConsistencyTracker
        ct = ConsistencyTracker()
        contradictions = ct.add_claims(response)
        return f"{ct.claim_count} claims, {len(contradictions)} contradictions", len(contradictions), ct
