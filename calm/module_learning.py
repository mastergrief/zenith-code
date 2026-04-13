"""
Auto-CALM Module Learning — learn from cognitive module outputs over time.

When the router consistently finds the same issues (e.g. Gemma always
overgeneralizes on comparison questions), this module learns the pattern
and proactively injects prevention into the system prompt.

The feedback loop:
  modules detect issues → learning records patterns → system prompt adapts
  → fewer issues → learning updates → prompt evolves

Usage:
    from calm.module_learning import ModuleLearner
    ml = ModuleLearner()
    ml.record("scope", "overgeneralization", "comparison")
    ml.record("scope", "overgeneralization", "comparison")
    ml.record("scope", "overgeneralization", "comparison")
    prompt_additions = ml.suggest_prompt_additions("comparison")
    # → ["Avoid absolute terms. Qualify with 'in most cases' or 'typically'."]
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class IssueTrend:
    """A recurring issue pattern."""
    module: str
    issue_type: str
    context: str        # what kind of prompt triggers this
    frequency: int = 0
    last_seen_turn: int = 0
    prevention: str = ""  # prompt addition to prevent this


DEFAULT_DB_PATH = Path("calm/.module_learning.json")


class ModuleLearner:
    """Learns from cognitive module outputs and suggests prompt adaptations."""

    def __init__(self, db_path: Optional[Path] = None):
        self._trends: Dict[str, IssueTrend] = {}
        self._db_path = db_path or DEFAULT_DB_PATH
        self._turn = 0
        self._load()

    def _load(self):
        if not self._db_path.exists():
            return
        try:
            data = json.loads(self._db_path.read_text())
            for key, d in data.items():
                self._trends[key] = IssueTrend(**d)
        except (json.JSONDecodeError, TypeError):
            pass

    def _save(self):
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: {
            "module": t.module, "issue_type": t.issue_type,
            "context": t.context, "frequency": t.frequency,
            "last_seen_turn": t.last_seen_turn, "prevention": t.prevention,
        } for k, t in self._trends.items()}
        self._db_path.write_text(json.dumps(data, indent=2))

    def record(self, module: str, issue_type: str, context: str = "general"):
        """Record an issue occurrence."""
        self._turn += 1
        key = f"{module}:{issue_type}:{context}"

        if key not in self._trends:
            self._trends[key] = IssueTrend(
                module=module,
                issue_type=issue_type,
                context=context,
                prevention=self._generate_prevention(module, issue_type),
            )
        self._trends[key].frequency += 1
        self._trends[key].last_seen_turn = self._turn
        self._save()

    def record_from_report(self, report, prompt: str):
        """Record issues from a CognitiveReport."""
        context = self._detect_context(prompt)

        for r in report.results:
            if r.issues_found > 0:
                # Normalize issue_type: strip variable data (percentages, counts)
                # "well-scoped (95%)" → "well-scoped"
                # "3 vague terms" → "vague terms"
                # "checked 48 claims, 3 issues" → "checked claims, issues"
                issue_type = re.sub(r'\d+\.?\d*%?', '', r.summary[:40]).strip()
                issue_type = re.sub(r'\s+', ' ', issue_type)
                issue_type = re.sub(r'^\s*,\s*', '', issue_type)  # strip leading comma
                if issue_type:
                    self.record(r.module_name, issue_type, context)

    def suggest_prompt_additions(self, prompt: str) -> List[str]:
        """Suggest system prompt additions based on learned patterns."""
        context = self._detect_context(prompt)
        additions = []

        for key, trend in self._trends.items():
            # Only suggest for patterns seen 3+ times
            if trend.frequency < 3:
                continue
            # Only suggest if relevant to this context
            if trend.context == context or trend.context == "general":
                if trend.prevention:
                    additions.append(trend.prevention)

        # Deduplicate
        return list(dict.fromkeys(additions))

    def _detect_context(self, prompt: str) -> str:
        """Detect what kind of prompt this is."""
        prompt_lower = prompt.lower()
        if any(w in prompt_lower for w in ["compare", "vs", "versus", "difference", "better"]):
            return "comparison"
        if any(w in prompt_lower for w in ["debug", "fix", "error", "bug", "crash"]):
            return "debugging"
        if any(w in prompt_lower for w in ["explain", "how does", "why does", "what is"]):
            return "explanation"
        if any(w in prompt_lower for w in ["design", "architect", "build", "implement"]):
            return "design"
        if any(w in prompt_lower for w in ["deploy", "release", "migrate"]):
            return "operations"
        return "general"

    def _generate_prevention(self, module: str, issue_type: str) -> str:
        """Generate a prompt addition that prevents this issue."""
        preventions = {
            "scope": {
                "overgeneralization": "Avoid absolute terms ('always', 'never', 'best'). Qualify claims with conditions.",
                "missing_qualifier": "When comparing, specify the criteria (performance, cost, complexity, etc.).",
            },
            "precision": {
                "vague": "Be specific: replace 'fast' with latency numbers, 'scalable' with throughput targets.",
            },
            "density": {
                "filler": "Skip pleasantries and preambles. Start with the answer.",
            },
            "perspective": {
                "missing": "Consider multiple perspectives: engineering, operations, security, user experience.",
            },
            "completeness": {
                "incomplete": "Address every part of multi-part questions. Don't skip sub-questions.",
            },
            "explanation": {
                "circular": "Explain the mechanism, don't restate the question.",
                "undefined": "Define technical jargon when first used.",
            },
            "relevance": {
                "off-topic": "Answer the question directly before adding context or background.",
            },
            "confidence_check": {
                "overconfidence": "Hedge appropriately: use 'typically', 'in most cases', 'it depends on'. Avoid absolutes.",
                "absolute": "Replace 'always/never/impossible' with qualified statements.",
                "false_certainty": "Don't appeal to authority or obviousness. Provide evidence instead.",
            },
            "specificity": {
                "platitude": "Be actionable: instead of 'use caching', specify what to cache, which cache, and TTL strategy.",
                "generic": "Name specific tools, metrics, and thresholds instead of generic advice.",
                "hand_wave": "Specify the concrete steps, not just the high-level pattern.",
            },
            "factual_check": {
                "contradiction": "Double-check factual claims about tools, algorithms, and implementations before stating them.",
                "suspicious": "Qualify uncertain factual claims with 'typically' or 'in most implementations'.",
                "dynamic_cross_check": "Verify numeric claims (output sizes, port numbers, complexities) before stating them.",
            },
        }

        module_preventions = preventions.get(module, {})
        # Match by prefix of issue_type
        for key, prevention in module_preventions.items():
            if key in issue_type.lower():
                return prevention
        return ""

    @property
    def recurring_issues(self) -> List[IssueTrend]:
        """Issues that occur 3+ times."""
        return [t for t in self._trends.values() if t.frequency >= 3]

    def summary(self) -> str:
        """Summary of learned patterns."""
        total = len(self._trends)
        recurring = len(self.recurring_issues)
        return f"{total} patterns tracked, {recurring} recurring (3+ occurrences)"
