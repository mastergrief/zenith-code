"""
CALM Hierarchical Router — decompose → route → compose.

Moves decomposition from post-hoc analysis to pre-generation routing.
Complex prompts get broken into sub-problems, computable ones answered
by backends, and the model gets a focused prompt with verified facts
pre-filled.

Simple prompts skip hierarchy entirely — no regression on easy questions.

Usage:
    from calm.hierarchical import HierarchicalRouter
    router = HierarchicalRouter()
    routing = router.route("Compare Redis vs PG. What is ACID?", precomputed)
    if routing:
        focused_prompt = routing.model_prompt
        # ... generate with focused prompt ...
        final = routing.compose(model_response)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from calm.decompose import Decomposer, DecompositionPlan, SubProblem
from calm.expression import safe_eval, ExpressionError


@dataclass
class RoutedStep:
    """A sub-problem with routing decision."""
    question: str
    source: str = "model"       # "backend", "precompute", "model"
    answer: Optional[str] = None
    backend_func: str = ""
    priority: int = 0
    depends_on: List[int] = field(default_factory=list)


@dataclass
class RoutingPlan:
    """Complete routing plan for a decomposed prompt."""
    original_prompt: str
    steps: List[RoutedStep] = field(default_factory=list)
    complexity: str = "simple"
    model_prompt: str = ""      # focused prompt for the model

    @property
    def has_model_steps(self) -> bool:
        return any(s.source == "model" for s in self.steps)

    @property
    def backend_answered(self) -> int:
        return sum(1 for s in self.steps if s.source in ("backend", "precompute"))

    @property
    def model_needed(self) -> int:
        return sum(1 for s in self.steps if s.source == "model")

    def compose(self, model_response: str) -> str:
        """Compose final response from backend answers + model response.

        The model response already addresses the focused questions.
        We DON'T restructure — the model's response is the primary output.
        Backend answers were injected as verified context, so the model
        already incorporated them. Just return the model's response.
        """
        return model_response

    def summary(self) -> str:
        total = len(self.steps)
        backend = self.backend_answered
        model = self.model_needed
        return (f"{self.complexity} → {total} sub-problems: "
                f"{backend} backend-answered, {model} need model")


class HierarchicalRouter:
    """Decompose prompts, route sub-problems, build focused model prompts."""

    def __init__(self):
        self._decomposer = Decomposer()

    def route(self, prompt: str, precomputed: Dict = None,
              verbose: bool = False) -> Optional[RoutingPlan]:
        """Decompose and route. Returns None for simple prompts (skip hierarchy)."""
        precomputed = precomputed or {}

        # 1. Decompose
        plan = self._decomposer.decompose(prompt)

        # 2. Simple prompts with ≤1 step AND no precomputed facts skip hierarchy
        if plan.complexity == "simple" and len(plan.steps) <= 1 and not precomputed:
            return None

        # Still skip if only 1 trivial step even with precomputed (precompute handles it)
        if len(plan.steps) <= 1:
            return None

        # 3. Route each sub-problem
        routing = RoutingPlan(
            original_prompt=prompt,
            complexity=plan.complexity,
        )

        for step in plan.steps:
            routed = RoutedStep(
                question=step.question,
                priority=step.priority,
                depends_on=step.depends_on,
            )

            # Try backend first
            if step.computable and step.suggested_backend:
                answer = self._try_backend(step)
                if answer is not None:
                    routed.source = "backend"
                    routed.answer = str(answer)
                    routed.backend_func = step.suggested_backend

            # Try precomputed facts
            if routed.source == "model" and precomputed:
                match = self._match_precomputed(step.question, precomputed)
                if match is not None:
                    routed.source = "precompute"
                    routed.answer = str(match)

            routing.steps.append(routed)

        # 4. Build focused prompt
        routing.model_prompt = self._build_focused_prompt(prompt, routing)

        if verbose:
            print(f"[hierarchy] {routing.summary()}")

        # 5. If everything was answered by backends, no need for hierarchy
        if not routing.has_model_steps:
            return None  # all answered — precompute path handles this

        return routing

    def _try_backend(self, step: SubProblem) -> Optional[object]:
        """Try to answer a sub-problem via backend function."""
        func_name = step.suggested_backend
        q = step.question.lower()

        # Direct function call patterns
        _DIRECT_PATTERNS = [
            (r'what (?:is|are) (?:the )?ACID', 'all_acid'),
            (r'what (?:is|are) (?:the )?CAP theorem', 'cap_theorem'),
            (r'what (?:is|are) (?:the )?SOLID', 'all_solid'),
            (r'(?:default )?port (?:for|of) (\w+)', 'protocol_info'),
            (r'complexity of ([\w\s]+)', 'ds_info'),
            (r'capital of ([\w\s]+)', 'country_capital'),
            (r'what (?:is|are) ([\w\s]+?) (?:sort|search)', 'sort_info'),
        ]

        # Try no-arg functions first
        for pattern, fn in _DIRECT_PATTERNS:
            m = re.search(pattern, q, re.IGNORECASE)
            if m:
                try:
                    if m.lastindex:  # has capture group
                        arg = m.group(1).strip()
                        result = safe_eval(f'{fn}("{arg}")')
                    else:
                        result = safe_eval(f'{fn}()')
                    if isinstance(result, dict) and "error" not in result:
                        return result
                    elif not isinstance(result, dict):
                        return result
                except ExpressionError:
                    pass

        # Generic: try the suggested backend function with extracted argument
        if func_name:
            m = re.search(r'what (?:is|are) (?:the )?([\w\s]+?)(?:\?|$)', q)
            if m:
                arg = m.group(1).strip()
                try:
                    result = safe_eval(f'{func_name}("{arg}")')
                    if isinstance(result, dict) and "error" not in result:
                        return result
                    elif not isinstance(result, dict):
                        return result
                except ExpressionError:
                    pass

        return None

    def _match_precomputed(self, question: str, precomputed: Dict) -> Optional[object]:
        """Check if any precomputed fact answers this sub-problem."""
        q_lower = question.lower()
        for expr, value in precomputed.items():
            # Check function name
            func_name = expr.split("(")[0] if "(" in expr else expr
            if func_name.lower() in q_lower:
                return value
            # Check arguments — e.g. service_port("PostgreSQL") matches "port for PostgreSQL"
            # Require BOTH the domain keyword AND the argument in the question
            arg_match = re.search(r'"([^"]+)"', expr)
            if arg_match:
                arg = arg_match.group(1).lower()
                # Map function prefixes to question keywords they should match
                domain_keywords = {
                    'port': ['port'], 'capital': ['capital'],
                    'currency': ['currency', 'decimal'], 'weight': ['weight', 'mass'],
                    'frequency': ['frequency', 'hz'], 'layer': ['layer', 'osi'],
                }
                fn_lower = func_name.lower()
                for fn_prefix, q_keywords in domain_keywords.items():
                    if fn_prefix in fn_lower and arg in q_lower:
                        if any(kw in q_lower for kw in q_keywords):
                            return value
        return None

    def _build_focused_prompt(self, original: str, routing: RoutingPlan) -> str:
        """Build a focused prompt for the model with backend answers pre-filled."""
        parts = [original]

        # Add verified answers as context
        verified = []
        for step in routing.steps:
            if step.answer:
                answer_str = step.answer
                if len(answer_str) > 200:
                    answer_str = answer_str[:200] + "..."
                verified.append(f"- {step.question} → {answer_str}")

        if verified:
            parts.append("\n\nThe following sub-questions have been verified by computation:")
            parts.extend(verified)
            parts.append("\nUse these verified facts in your response. Focus your reasoning on the remaining questions.")

        # List what the model needs to address
        model_steps = [s for s in routing.steps if s.source == "model"]
        if model_steps:
            parts.append("\nPlease specifically address:")
            for i, step in enumerate(model_steps, 1):
                parts.append(f"{i}. {step.question}")

        return "\n".join(parts)
