"""
Auto-CALM Decomposition — break complex problems into sub-problems.

The meta-skill: given a hard question, produce a list of easier questions
whose answers compose into the full answer. Each sub-problem can be
routed to the right backend or left for the model.

Two modes:
  1. Pattern-based: recognized problem types decompose deterministically
  2. Structural: detect complexity signals and suggest decomposition axes

Usage:
    from calm.decompose import Decomposer
    d = Decomposer()
    plan = d.decompose("Migrate the database from MySQL to PostgreSQL")
    for step in plan.steps:
        print(f"  {step.question} [backend: {step.suggested_backend}]")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from calm.expression import _FUNCTIONS


@dataclass
class SubProblem:
    """One sub-problem in a decomposition."""
    question: str                     # what to solve
    suggested_backend: str = ""       # which CALM backend could help
    depends_on: List[int] = field(default_factory=list)  # indices of prerequisite steps
    priority: int = 0                 # lower = do first
    computable: bool = False          # whether a backend can answer it


@dataclass
class DecompositionPlan:
    """A decomposed problem with ordered sub-problems."""
    original: str                     # original question
    steps: List[SubProblem] = field(default_factory=list)
    complexity: str = "simple"        # simple/moderate/complex
    axes: List[str] = field(default_factory=list)  # decomposition dimensions

    @property
    def computable_count(self) -> int:
        return sum(1 for s in self.steps if s.computable)

    def summary(self) -> str:
        total = len(self.steps)
        comp = self.computable_count
        return (f"{self.complexity} problem → {total} sub-problems "
                f"({comp} computable, {total - comp} need model reasoning)")


# Problem type patterns → decomposition templates.
# Each template produces a list of sub-problems.
_DECOMPOSITION_TEMPLATES = {
    "comparison": {
        "pattern": re.compile(
            r'(?:compare|difference|vs\.?|versus|better|which)\s+'
            r'(?:between\s+)?(\w[\w\s]*?)\s+(?:and|vs\.?|or|versus)\s+(\w[\w\s]*)',
            re.IGNORECASE
        ),
        "axes": ["definition", "strengths", "weaknesses", "use cases", "performance"],
        "template": lambda a, b: [
            SubProblem(f"What is {a}?", priority=0),
            SubProblem(f"What is {b}?", priority=0),
            SubProblem(f"What are the strengths of {a}?", priority=1, depends_on=[0]),
            SubProblem(f"What are the strengths of {b}?", priority=1, depends_on=[1]),
            SubProblem(f"When should you use {a} over {b}?", priority=2, depends_on=[2, 3]),
            SubProblem(f"When should you use {b} over {a}?", priority=2, depends_on=[2, 3]),
        ],
    },
    "migration": {
        "pattern": re.compile(
            r'(?:migrate|migration|move|convert|switch|transition)\s+'
            r'(?:from\s+)?(\w[\w\s]*?)\s+(?:to|into)\s+(\w[\w\s]*)',
            re.IGNORECASE
        ),
        "axes": ["compatibility", "data", "downtime", "testing", "rollback"],
        "template": lambda src, dst: [
            SubProblem(f"What are the key differences between {src} and {dst}?", priority=0),
            SubProblem(f"What data/schema changes are needed from {src} to {dst}?", priority=1, depends_on=[0]),
            SubProblem(f"What is the expected downtime for migrating from {src} to {dst}?", priority=1),
            SubProblem(f"How do you test the migration from {src} to {dst}?", priority=2, depends_on=[1]),
            SubProblem(f"What is the rollback plan if the {src} to {dst} migration fails?", priority=2),
        ],
    },
    "debugging": {
        "pattern": re.compile(
            r'(?:why|debug|fix|broken|error|bug|crash|fail|issue)\s+.{10,}',
            re.IGNORECASE
        ),
        "axes": ["reproduce", "isolate", "diagnose", "fix", "verify"],
        "template": lambda desc: [
            SubProblem(f"Can you reproduce the issue? What are the exact steps?", priority=0),
            SubProblem(f"What changed recently that could have caused this?", priority=0),
            SubProblem(f"What does the error message/stack trace tell us?", priority=1),
            SubProblem(f"What is the minimal code that triggers this?", priority=1, depends_on=[0]),
            SubProblem(f"What is the root cause?", priority=2, depends_on=[2, 3]),
            SubProblem(f"What is the fix?", priority=3, depends_on=[4]),
            SubProblem(f"How do you verify the fix doesn't break anything else?", priority=4, depends_on=[5]),
        ],
    },
    "optimization": {
        "pattern": re.compile(
            r'(?:optimize|speed up|improve|faster|slow|performance|bottleneck)\s+.{5,}',
            re.IGNORECASE
        ),
        "axes": ["measure", "profile", "identify", "fix", "verify"],
        "template": lambda desc: [
            SubProblem(f"What is the current performance baseline?", priority=0),
            SubProblem(f"Where is the bottleneck? (profile/measure)", priority=1, depends_on=[0]),
            SubProblem(f"What is the theoretical best-case performance?", priority=1),
            SubProblem(f"What are the candidate optimizations?", priority=2, depends_on=[1]),
            SubProblem(f"Which optimization has the best effort/impact ratio?", priority=3, depends_on=[3]),
            SubProblem(f"Implement and measure the optimization.", priority=4, depends_on=[4]),
            SubProblem(f"Verify correctness after optimization.", priority=5, depends_on=[5]),
        ],
    },
    "design": {
        "pattern": re.compile(
            r'(?:design|architect|build|implement|create|develop)\s+(?:a\s+)?(\w[\w\s]*)',
            re.IGNORECASE
        ),
        "axes": ["requirements", "constraints", "components", "interfaces", "tradeoffs"],
        "template": lambda thing: [
            SubProblem(f"What are the requirements for {thing}?", priority=0),
            SubProblem(f"What are the constraints (performance, cost, time)?", priority=0),
            SubProblem(f"What are the major components/modules?", priority=1, depends_on=[0, 1]),
            SubProblem(f"How do the components interact (interfaces/APIs)?", priority=2, depends_on=[2]),
            SubProblem(f"What are the key tradeoffs in this design?", priority=2, depends_on=[2]),
            SubProblem(f"What could go wrong? What are the risks?", priority=3, depends_on=[3, 4]),
        ],
    },
}

# Complexity signals — the more of these present, the harder the problem
_COMPLEXITY_SIGNALS = [
    re.compile(r'\b(?:and|also|additionally|furthermore|moreover)\b', re.IGNORECASE),
    re.compile(r'\b(?:but|however|although|while|whereas)\b', re.IGNORECASE),
    re.compile(r'\b(?:if|when|unless|assuming|given that)\b', re.IGNORECASE),
    re.compile(r'\b(?:multiple|several|many|various|different)\b', re.IGNORECASE),
    re.compile(r'\b(?:tradeoff|balance|compromise|versus)\b', re.IGNORECASE),
    re.compile(r'\?.*\?', re.DOTALL),  # multiple questions
]


class Decomposer:
    """Breaks complex problems into ordered sub-problems."""

    def decompose(self, prompt: str) -> DecompositionPlan:
        """Decompose a problem into sub-problems."""
        plan = DecompositionPlan(original=prompt)

        # Detect complexity
        signal_count = sum(1 for pat in _COMPLEXITY_SIGNALS if pat.search(prompt))
        if signal_count >= 4:
            plan.complexity = "complex"
        elif signal_count >= 2:
            plan.complexity = "moderate"
        else:
            plan.complexity = "simple"

        # Multi-question prompts: structural decomposition takes priority
        # over templates when there are 2+ explicit questions
        questions = re.findall(r'([^?.!]+\?)', prompt)
        if len(questions) >= 2:
            plan = self._structural_decompose(prompt, plan)
            if plan.steps:
                return plan

        # Single-question: try template-based decomposition
        for name, template_info in _DECOMPOSITION_TEMPLATES.items():
            m = template_info["pattern"].search(prompt)
            if m:
                plan.axes = template_info["axes"]
                groups = m.groups()
                if groups:
                    plan.steps = template_info["template"](*groups)
                else:
                    plan.steps = template_info["template"](prompt)
                self._tag_computable(plan)
                return plan

        # Structural decomposition for unrecognized patterns
        plan = self._structural_decompose(prompt, plan)
        return plan

    def _structural_decompose(self, prompt: str, plan: DecompositionPlan) -> DecompositionPlan:
        """Decompose based on structural analysis of the prompt."""
        # Multiple questions?
        questions = re.findall(r'([^?.]+\?)', prompt)
        if len(questions) >= 2:
            plan.axes = ["multi-question"]
            for i, q in enumerate(questions):
                plan.steps.append(SubProblem(q.strip(), priority=i))
            self._tag_computable(plan)
            return plan

        # "How do I X and Y?" → split on conjunctions
        parts = re.split(r'\b(?:and then|then|and also|and)\b', prompt, flags=re.IGNORECASE)
        if len(parts) >= 2:
            plan.axes = ["sequential"]
            for i, part in enumerate(parts):
                part = part.strip().rstrip('?.!')
                if len(part) > 10:
                    plan.steps.append(SubProblem(
                        f"{part}?", priority=i,
                        depends_on=[i-1] if i > 0 else [],
                    ))
            self._tag_computable(plan)
            return plan

        # Single complex question — suggest axes based on keywords
        plan.axes = self._detect_axes(prompt)
        if plan.axes:
            for i, axis in enumerate(plan.axes):
                plan.steps.append(SubProblem(
                    f"Considering the {axis} aspect: {prompt}",
                    priority=i,
                ))

        self._tag_computable(plan)
        return plan

    def _detect_axes(self, prompt: str) -> List[str]:
        """Detect natural decomposition axes from prompt content."""
        axes = []
        axis_keywords = {
            "performance": ["fast", "slow", "speed", "latency", "throughput", "scalab"],
            "security": ["secur", "vulnerab", "attack", "auth", "encrypt", "inject"],
            "cost": ["cost", "price", "budget", "expensive", "cheap", "free"],
            "maintainability": ["maintain", "refactor", "technical debt", "clean", "readable"],
            "compatibility": ["compat", "version", "backward", "legacy", "support"],
            "reliability": ["reliab", "uptime", "fault", "recover", "backup", "redundan"],
        }
        prompt_lower = prompt.lower()
        for axis, keywords in axis_keywords.items():
            if any(kw in prompt_lower for kw in keywords):
                axes.append(axis)
        return axes

    def _tag_computable(self, plan: DecompositionPlan):
        """Tag which sub-problems have matching CALM backends."""
        for step in plan.steps:
            q_lower = step.question.lower()
            # Check if any registered function name appears in the question
            for func_name in _FUNCTIONS:
                if '.' not in func_name and func_name in q_lower:
                    step.computable = True
                    step.suggested_backend = func_name
                    break

            # Check for known computable domains
            computable_keywords = {
                "complexity": "complexity_ops",
                "performance": "perf_ops",
                "security": "security_ops",
                "port": "port_kb",
                "license": "license_kb",
                "capital": "country_kb",
                "element": "elements_kb",
                "convert": "convert_ops",
            }
            for keyword, backend in computable_keywords.items():
                if keyword in q_lower:
                    step.computable = True
                    step.suggested_backend = backend
                    break

    def execution_order(self, plan: DecompositionPlan) -> List[List[int]]:
        """Return steps grouped by priority (parallelizable within a group)."""
        if not plan.steps:
            return []
        max_priority = max(s.priority for s in plan.steps)
        groups = []
        for p in range(max_priority + 1):
            group = [i for i, s in enumerate(plan.steps) if s.priority == p]
            if group:
                groups.append(group)
        return groups
