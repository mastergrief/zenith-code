"""
Auto-CALM Adaptive Thinking — dynamic thinking budget based on context.

When precompute already has the answer, the model doesn't need 8K tokens
to think. When the question is complex with multiple sub-problems, it
needs more. This module estimates the optimal thinking budget.

Saves time on easy questions, improves quality on hard ones.

Budget tiers:
  TRIVIAL (2048):  precompute has the full answer, just format it
  EASY (4096):     single factual question, answer is straightforward
  MEDIUM (8192):   comparison, explanation, moderate complexity
  HARD (16384):    multi-part, debugging, architecture decision
  DEEP (32768):    complex reasoning chain, novel problem, high stakes

Usage:
    from calm.adaptive import AdaptiveBudget
    ab = AdaptiveBudget()
    budget = ab.estimate("What is 17 * 23?", precomputed={"17 * 23": 391})
    # → 1024 (trivial — precompute has it)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class BudgetEstimate:
    """Estimated thinking budget with reasoning."""
    budget: int
    tier: str           # "trivial", "easy", "medium", "hard", "deep"
    reasoning: str
    confidence: float   # 0-1

    def __str__(self):
        return f"{self.tier} ({self.budget} tokens): {self.reasoning}"


_TIERS = {
    "trivial": 2048,
    "easy": 4096,
    "medium": 8192,
    "hard": 16384,
    "deep": 32768,
}

# Complexity signals
_COMPLEXITY_UP = [
    (re.compile(r'\b(?:compare|vs\.?|versus|difference between|pros and cons)\b', re.IGNORECASE), 1, "comparison"),
    (re.compile(r'\b(?:explain|how does|why does|what causes)\b', re.IGNORECASE), 1, "explanation"),
    (re.compile(r'\b(?:debug|fix|troubleshoot|diagnose|investigate)\b', re.IGNORECASE), 2, "debugging"),
    (re.compile(r'\b(?:design|architect|plan|strategy|roadmap)\b', re.IGNORECASE), 2, "design"),
    (re.compile(r'\b(?:migrate|rewrite|refactor|overhaul)\b', re.IGNORECASE), 2, "migration"),
    (re.compile(r'\b(?:trade.?off|balance|optimize|constraint)\b', re.IGNORECASE), 1, "tradeoff"),
    (re.compile(r'\b(?:security|vulnerability|threat|attack)\b', re.IGNORECASE), 1, "security"),
    (re.compile(r'\?.*\?', re.DOTALL), 1, "multi-question"),
    (re.compile(r'\b(?:and also|additionally|furthermore|as well as)\b', re.IGNORECASE), 1, "compound"),
    (re.compile(r'\b(?:step by step|in detail|thorough|comprehensive|exhaustive)\b', re.IGNORECASE), 1, "depth-request"),
]

_COMPLEXITY_DOWN = [
    (re.compile(r'^(?:what is|what are|how many|how much)\s+', re.IGNORECASE), -1, "simple-question"),
    (re.compile(r'^\w+\s*[\*\+\-\/]\s*\w+\s*[=?]', re.IGNORECASE), -2, "arithmetic"),
    (re.compile(r'^(?:convert|translate)\s+\d', re.IGNORECASE), -1, "conversion"),
    (re.compile(r'^(?:is|are|does|do|can|will)\s+', re.IGNORECASE), -1, "yes-no"),
]


class AdaptiveBudget:
    """Estimates optimal thinking budget based on prompt complexity."""

    def estimate(self, prompt: str, precomputed: Optional[Dict] = None,
                 pre_analysis: Optional[Dict] = None) -> BudgetEstimate:
        """Estimate the optimal thinking budget."""

        # Start at baseline
        complexity_score = 0
        reasons = []

        # Check precompute coverage
        if precomputed:
            # If precompute fully answers the question, minimal thinking needed
            prompt_words = set(re.findall(r'[a-z]+', prompt.lower()))
            question_words = prompt_words - {
                "what", "is", "the", "a", "an", "of", "how", "many", "much",
                "does", "do", "can", "will", "are", "in", "to", "for",
            }
            # Check if precomputed expressions cover the question
            precompute_coverage = 0
            for expr in precomputed:
                expr_words = set(re.findall(r'[a-z]+', expr.lower()))
                if question_words and expr_words & question_words:
                    precompute_coverage += 1

            if precompute_coverage >= len(precomputed) and len(precomputed) > 0:
                complexity_score -= 3
                reasons.append(f"precompute covers {precompute_coverage} expressions")

        # Check prompt complexity signals
        for pat, delta, reason in _COMPLEXITY_UP:
            if pat.search(prompt):
                complexity_score += delta
                reasons.append(reason)

        for pat, delta, reason in _COMPLEXITY_DOWN:
            if pat.search(prompt):
                complexity_score += delta
                reasons.append(reason)

        # Check pre-analysis if available
        if pre_analysis:
            sub_problems = len(pre_analysis.get("sub_problems", []))
            if sub_problems >= 5:
                complexity_score += 2
                reasons.append(f"{sub_problems} sub-problems")
            elif sub_problems >= 3:
                complexity_score += 1

            risk_level = pre_analysis.get("risk_level", "low")
            if risk_level in ("high", "critical"):
                complexity_score += 2
                reasons.append(f"risk={risk_level}")

            ambiguities = pre_analysis.get("ambiguities", 0)
            if ambiguities >= 2:
                complexity_score += 1
                reasons.append(f"{ambiguities} ambiguities")

        # Check prompt length (longer = more complex)
        word_count = len(prompt.split())
        if word_count > 100:
            complexity_score += 2
            reasons.append("long prompt")
        elif word_count > 50:
            complexity_score += 1

        # Map score to tier
        if complexity_score <= -2:
            tier = "trivial"
        elif complexity_score <= 0:
            tier = "easy"
        elif complexity_score <= 2:
            tier = "medium"
        elif complexity_score <= 4:
            tier = "hard"
        else:
            tier = "deep"

        budget = _TIERS[tier]
        confidence = min(1.0, 0.5 + abs(complexity_score) * 0.1)

        return BudgetEstimate(
            budget=budget,
            tier=tier,
            reasoning=", ".join(reasons) if reasons else "baseline",
            confidence=confidence,
        )
