"""
Auto-CALM Judgment — structured evaluation for "is this good?" questions.

Instead of a vibes-based answer, decomposes into measurable criteria,
scores each independently, and produces a structured scorecard.
Backends verify what's measurable (complexity, security, Big-O).
Model reasons about what's not (readability, maintainability).

Usage:
    from calm.judgment import JudgmentEngine
    je = JudgmentEngine()
    result = je.evaluate(code, criteria=["complexity", "security", "readability"])
    print(result.scorecard)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from calm.expression import safe_eval, ExpressionError


@dataclass
class Criterion:
    """One evaluation criterion."""
    name: str
    score: Optional[float] = None  # 0-10, or None if qualitative
    verified: bool = False          # True if score came from a backend
    reasoning: str = ""             # why this score
    source: str = ""                # "backend" or "model"


@dataclass
class JudgmentResult:
    """Result of a structured evaluation."""
    criteria: List[Criterion] = field(default_factory=list)
    overall_score: Optional[float] = None
    summary: str = ""

    @property
    def scorecard(self) -> str:
        """Human-readable scorecard."""
        lines = []
        for c in self.criteria:
            marker = "V" if c.verified else "M"  # Verified vs Model-assessed
            score_str = f"{c.score:.1f}/10" if c.score is not None else "N/A"
            lines.append(f"  [{marker}] {c.name}: {score_str} — {c.reasoning}")
        if self.overall_score is not None:
            lines.append(f"  Overall: {self.overall_score:.1f}/10")
        return "\n".join(lines)


# Criteria → backend function mapping.
# If a criterion has a backend function, it's verified (not model-assessed).
_BACKEND_CRITERIA = {
    "complexity": {
        "func": "cyclomatic_complexity",
        "scorer": lambda v: max(0, 10 - v) if isinstance(v, (int, float)) else None,
        "description": "Cyclomatic complexity (lower is better)",
    },
    "security": {
        "func": "security.audit",
        "scorer": lambda v: 10.0 if "No issues" in str(v) else max(0, 10 - str(v).count("ISSUE") * 2),
        "description": "OWASP security audit",
    },
    "naming": {
        "func": "naming_quality",
        "scorer": lambda v: float(v) if isinstance(v, (int, float)) else None,
        "description": "Variable/function naming quality",
    },
    "test_coverage": {
        "func": "test_summary",
        "scorer": lambda v: None,  # parse from output
        "description": "Test coverage percentage",
    },
}

# Criteria that can only be model-assessed (no backend).
_MODEL_CRITERIA = {
    "readability": "How easy is the code to understand for a new developer?",
    "maintainability": "How easy would it be to modify or extend this code?",
    "correctness": "Does the code correctly implement the requirements?",
    "performance": "Is the code efficient for its use case?",
    "design": "Does the architecture follow good design principles?",
    "error_handling": "Are error cases handled appropriately?",
    "documentation": "Is the code adequately documented?",
}


class JudgmentEngine:
    """Structured evaluation engine."""

    def evaluate_code(self, code: str, criteria: Optional[List[str]] = None) -> JudgmentResult:
        """Evaluate code against criteria. Returns structured scorecard."""
        if criteria is None:
            criteria = ["complexity", "security", "readability", "maintainability"]

        result = JudgmentResult()

        for name in criteria:
            if name in _BACKEND_CRITERIA:
                criterion = self._evaluate_backend(name, code)
            elif name in _MODEL_CRITERIA:
                criterion = Criterion(
                    name=name, verified=False, source="model",
                    reasoning=_MODEL_CRITERIA[name],
                )
            else:
                criterion = Criterion(
                    name=name, verified=False, source="unknown",
                    reasoning=f"Unknown criterion: {name}",
                )
            result.criteria.append(criterion)

        # Overall = average of scored criteria
        scores = [c.score for c in result.criteria if c.score is not None]
        if scores:
            result.overall_score = sum(scores) / len(scores)

        return result

    def _evaluate_backend(self, name: str, code: str) -> Criterion:
        """Evaluate a criterion using a backend function."""
        info = _BACKEND_CRITERIA[name]
        criterion = Criterion(name=name, source="backend")

        try:
            # Call the backend function with the code
            value = safe_eval(f'{info["func"]}({repr(code)})')
            criterion.score = info["scorer"](value)
            criterion.verified = True
            criterion.reasoning = f"{info['description']}: {value}"
        except (ExpressionError, Exception) as e:
            criterion.reasoning = f"Could not evaluate: {e}"

        return criterion

    def compare(self, option_a: str, option_b: str,
                criteria: Optional[List[str]] = None) -> str:
        """Compare two options against criteria. Returns structured comparison."""
        result_a = self.evaluate_code(option_a, criteria)
        result_b = self.evaluate_code(option_b, criteria)

        lines = ["Option A vs Option B:"]
        for ca, cb in zip(result_a.criteria, result_b.criteria):
            sa = f"{ca.score:.1f}" if ca.score is not None else "?"
            sb = f"{cb.score:.1f}" if cb.score is not None else "?"
            winner = ""
            if ca.score is not None and cb.score is not None:
                if ca.score > cb.score:
                    winner = " ← A wins"
                elif cb.score > ca.score:
                    winner = " ← B wins"
                else:
                    winner = " ← tie"
            lines.append(f"  {ca.name}: A={sa} vs B={sb}{winner}")

        if result_a.overall_score is not None and result_b.overall_score is not None:
            lines.append(f"  Overall: A={result_a.overall_score:.1f} vs B={result_b.overall_score:.1f}")

        return "\n".join(lines)
