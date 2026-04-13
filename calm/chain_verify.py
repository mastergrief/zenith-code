"""
Auto-CALM Chain-of-Verification — multi-step reasoning verification.

When the model produces reasoning chains (A → B → C), this module:
1. Extracts intermediate steps from thinking/output
2. Verifies each step independently on CPU
3. Identifies the first wrong step (poison point)
4. Reports which downstream conclusions are tainted

The key insight: if step 2 of 5 is wrong, steps 3-5 are all suspect
regardless of whether they look correct. Finding the poison point
is more valuable than checking each step in isolation.

Usage:
    from calm.chain_verify import ChainVerifier
    cv = ChainVerifier()
    chain = cv.extract_chain(thinking_text)
    result = cv.verify_chain(chain)
    print(result.poison_step)  # first wrong step, or None
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from calm.expression import safe_eval, ExpressionError


@dataclass
class Step:
    """One step in a reasoning chain."""
    text: str           # original text of this step
    expression: str     # extracted computable expression (if any)
    claimed_value: str  # what the model claims the result is
    actual_value: object = None  # CPU-verified value
    verified: bool = False       # whether we could verify it
    correct: Optional[bool] = None  # True/False/None (unverifiable)
    step_number: int = 0


@dataclass
class ChainResult:
    """Result of verifying a reasoning chain."""
    steps: List[Step] = field(default_factory=list)
    poison_step: Optional[int] = None  # first wrong step (0-indexed)
    total_steps: int = 0
    verified_steps: int = 0
    correct_steps: int = 0
    wrong_steps: int = 0
    tainted_steps: int = 0  # steps after poison point

    @property
    def is_sound(self) -> bool:
        """Whether the entire chain is verified correct."""
        return self.wrong_steps == 0 and self.verified_steps > 0

    def summary(self) -> str:
        """Human-readable summary."""
        if self.total_steps == 0:
            return "no reasoning chain found"
        if self.is_sound:
            return f"chain sound: {self.correct_steps}/{self.total_steps} steps verified"
        return (f"chain broken at step {self.poison_step + 1}: "
                f"{self.correct_steps} correct, {self.wrong_steps} wrong, "
                f"{self.tainted_steps} tainted downstream")


# Patterns for extracting reasoning steps.
# These match common "therefore/so/thus" chains and numbered steps.
_STEP_PATTERNS = [
    # "Step 1: X = Y" or "1. X = Y"
    re.compile(r'(?:Step\s+)?(\d+)[.:]\s*(.+?)(?=(?:Step\s+)?\d+[.:]|\Z)', re.DOTALL),
    # "First, ... Then, ... Therefore, ..."
    re.compile(r'(?:First|Then|Next|Therefore|Thus|So|Hence|Finally)[,:]?\s*(.+?)(?=(?:First|Then|Next|Therefore|Thus|So|Hence|Finally)[,:]|\Z)', re.DOTALL | re.IGNORECASE),
]

# Patterns for extracting a computation from a step.
_COMPUTATION_RE = [
    # "X × Y = Z" or "X * Y = Z"
    re.compile(r'(\d[\d,\s]*(?:[\*×÷\+\-\/%\^]|\\times|\\cdot|\\div)[\d,\s\*×÷\+\-\/%\^\.]*\d)\s*[=≈]\s*([\-]?\d[\d,.]*)'),
    # "function(args) = result"
    re.compile(r'([a-z_]\w*\([^)]+\))\s*[=≈]\s*([\-]?\d[\d,.]*)'),
    # "X is Y" where X is a number and Y is a computed property
    re.compile(r'(\d[\d,]*)\s*(?:is|=)\s*([\-]?\d[\d,.]*)'),
]


class ChainVerifier:
    """Extracts and verifies multi-step reasoning chains."""

    def extract_chain(self, text: str) -> List[Step]:
        """Extract reasoning steps from text (thinking block or output)."""
        steps = []

        # Try numbered steps first (more structured)
        numbered = re.findall(
            r'(?:^|\n)\s*(?:Step\s+)?(\d+)[.:]\s*(.+?)(?=\n\s*(?:Step\s+)?\d+[.:]|\Z)',
            text, re.DOTALL
        )
        if len(numbered) >= 2:
            for num, content in numbered:
                step = self._parse_step(content.strip(), int(num) - 1)
                steps.append(step)
            return steps

        # Try transition-word chains
        transitions = re.split(
            r'\b(?:First|Then|Next|Therefore|Thus|So|Hence|Finally|Now|Since|Because|Given that)\b[,:]?\s*',
            text, flags=re.IGNORECASE
        )
        transitions = [t.strip() for t in transitions if t.strip() and len(t.strip()) > 10]
        if len(transitions) >= 2:
            for i, content in enumerate(transitions):
                step = self._parse_step(content, i)
                steps.append(step)
            return steps

        # Try sentence-level splitting for short chains
        sentences = re.split(r'[.!]\s+', text)
        computable = []
        for i, sent in enumerate(sentences):
            step = self._parse_step(sent.strip(), i)
            if step.expression:
                computable.append(step)
        if len(computable) >= 2:
            return computable

        return steps

    def _parse_step(self, text: str, index: int) -> Step:
        """Parse a single step, extracting any computable expression."""
        step = Step(text=text, expression="", claimed_value="", step_number=index)

        for pat in _COMPUTATION_RE:
            m = pat.search(text)
            if m:
                step.expression = self._normalize(m.group(1))
                step.claimed_value = m.group(2).replace(",", "")
                break

        return step

    def _normalize(self, expr: str) -> str:
        """Normalize expression for safe_eval."""
        expr = expr.strip()
        expr = expr.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/")
        expr = expr.replace("×", "*").replace("÷", "/")
        expr = expr.replace(",", "")
        return expr

    def verify_chain(self, steps: List[Step]) -> ChainResult:
        """Verify each step in a chain. Reports the poison point."""
        result = ChainResult(steps=steps, total_steps=len(steps))

        poison_found = False
        for i, step in enumerate(steps):
            if not step.expression:
                continue

            try:
                step.actual_value = safe_eval(step.expression)
                step.verified = True
                result.verified_steps += 1

                # Compare claimed vs actual
                if self._values_match(step.claimed_value, step.actual_value):
                    step.correct = True
                    result.correct_steps += 1
                else:
                    step.correct = False
                    result.wrong_steps += 1
                    if not poison_found:
                        result.poison_step = i
                        poison_found = True
            except ExpressionError:
                # Can't verify this step — skip
                pass

            if poison_found and i > result.poison_step:
                result.tainted_steps += 1

        return result

    def _values_match(self, claimed: str, actual: object) -> bool:
        """Check if claimed value matches actual, with tolerance."""
        try:
            claimed_num = float(claimed)
            actual_num = float(actual)
            # Exact match for integers
            if claimed_num == int(claimed_num) and actual_num == int(actual_num):
                return int(claimed_num) == int(actual_num)
            # Tolerance for floats
            if actual_num == 0:
                return abs(claimed_num) < 1e-6
            return abs(claimed_num - actual_num) / abs(actual_num) < 0.001
        except (ValueError, TypeError):
            return str(claimed).strip() == str(actual).strip()

    def verify_thinking(self, thinking: str) -> ChainResult:
        """Convenience: extract chain from thinking block and verify."""
        chain = self.extract_chain(thinking)
        return self.verify_chain(chain)
