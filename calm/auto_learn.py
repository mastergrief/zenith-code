"""
Auto-CALM self-learning — the system learns from its own corrections.

When Auto-CALM corrects a claim, that correction is logged. Over time,
the system builds a pattern database of what the model gets wrong, and
automatically precomputes those patterns in future sessions.

The learning loop:
  1. Model makes error → Auto-CALM corrects → log the pattern
  2. Next prompt, check if any logged patterns match → precompute
  3. Model sees precomputed fact → no error → no correction needed

This replaces LoRA for computational domains. Instead of training the
model to compute correctly, teach the ENGINE which computations need
precomputing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from calm.expression import safe_eval, ExpressionError


DEFAULT_DB = Path(".calm_training/auto/learned_patterns.jsonl")


class LearnedPattern:
    """A pattern learned from a correction."""
    def __init__(self, pattern_type: str, expression: str, frequency: int = 1):
        self.pattern_type = pattern_type  # "arithmetic", "bool", "function"
        self.expression = expression      # e.g. "is_prime(X)" where X is a placeholder
        self.frequency = frequency        # how often this pattern was corrected


class AutoLearner:
    """Tracks corrections and learns precompute patterns."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB
        self._patterns: Dict[str, LearnedPattern] = {}
        self._load()

    def _load(self):
        """Load learned patterns from disk."""
        if not self.db_path.exists():
            return
        with open(self.db_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                key = d["expression"]
                self._patterns[key] = LearnedPattern(
                    pattern_type=d["pattern_type"],
                    expression=d["expression"],
                    frequency=d.get("frequency", 1),
                )

    def _save(self):
        """Persist all patterns to disk."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "w") as f:
            for p in self._patterns.values():
                f.write(json.dumps({
                    "pattern_type": p.pattern_type,
                    "expression": p.expression,
                    "frequency": p.frequency,
                }) + "\n")

    def learn_from_correction(self, claim) -> None:
        """Learn a pattern from a correction. Extracts the general form."""
        if claim.correct or claim.actual_value is None:
            return

        expr = claim.expression
        # Generalize: replace specific numbers with pattern markers.
        # "17 * 23" → "N * M" (arithmetic pattern)
        # "is_prime(391)" → "is_prime(N)" (function pattern)

        general = self._generalize(expr)
        if general in self._patterns:
            self._patterns[general].frequency += 1
        else:
            ptype = self._classify(expr)
            self._patterns[general] = LearnedPattern(
                pattern_type=ptype,
                expression=general,
                frequency=1,
            )
        self._save()

    def suggest_precomputes(self, prompt: str) -> Dict[str, object]:
        """Suggest precomputations based on learned patterns.
        Returns {expression: computed_value} for patterns that match the prompt."""
        results = {}

        for key, pattern in self._patterns.items():
            # Only precompute patterns that have been wrong multiple times
            # or are high-frequency error patterns.
            if pattern.frequency < 1:
                continue

            # Try to instantiate the pattern with values from the prompt.
            expressions = self._instantiate(pattern, prompt)
            for expr in expressions:
                if expr not in results:
                    try:
                        val = safe_eval(expr)
                        results[expr] = val
                    except ExpressionError:
                        pass

        return results

    def _generalize(self, expr: str) -> str:
        """Generalize a specific expression to a pattern.
        '17 * 23' → 'N * M', 'is_prime(391)' → 'is_prime(N)'"""
        # Function calls: replace numeric args with N, M, etc.
        def _replace_args(m):
            func = m.group(1)
            args = m.group(2)
            placeholders = []
            for i, arg in enumerate(args.split(',')):
                arg = arg.strip()
                if re.match(r'^-?\d+\.?\d*$', arg):
                    placeholders.append(chr(78 + i))  # N, O, P...
                else:
                    placeholders.append(arg)
            return f"{func}({', '.join(placeholders)})"

        result = re.sub(r'(\w+)\(([^)]+)\)', _replace_args, expr)

        # Arithmetic: replace numbers with N, M.
        if result == expr:  # No function call found.
            parts = re.split(r'(\s*[\+\-\*\/\%\^]+\s*)', expr)
            placeholders = []
            p_idx = 0
            for part in parts:
                if re.match(r'^\s*-?\d+\.?\d*\s*$', part.strip()):
                    placeholders.append(chr(78 + p_idx))
                    p_idx += 1
                else:
                    placeholders.append(part)
            result = ''.join(placeholders)

        return result

    def _classify(self, expr: str) -> str:
        """Classify an expression type."""
        if '(' in expr:
            return "function"
        if any(op in expr for op in ['*', '/', '+', '-', '%', '**']):
            return "arithmetic"
        return "other"

    def _instantiate(self, pattern: LearnedPattern, prompt: str) -> List[str]:
        """Try to instantiate a pattern with values from the prompt."""
        results = []

        # Extract all numbers from the prompt.
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', prompt)
        if not numbers:
            return results

        expr_template = pattern.expression

        if pattern.pattern_type == "function":
            # is_prime(N) → is_prime(391) for each number in prompt.
            func_match = re.match(r'(\w+)\(([^)]*)\)', expr_template)
            if func_match:
                func = func_match.group(0)
                arg_count = len(func_match.group(2).split(','))
                if arg_count == 1 and 'N' in func:
                    for n in numbers:
                        results.append(func.replace('N', n))
                elif arg_count == 2 and len(numbers) >= 2:
                    for i in range(len(numbers)):
                        for j in range(len(numbers)):
                            if i != j:
                                inst = func.replace('N', numbers[i]).replace('O', numbers[j])
                                results.append(inst)

        elif pattern.pattern_type == "arithmetic":
            # N * M → 17 * 23 for each pair of numbers.
            if len(numbers) >= 2:
                ops = re.findall(r'[\+\-\*\/\%\^]+', expr_template)
                if ops:
                    for i in range(len(numbers)):
                        for j in range(len(numbers)):
                            if i != j:
                                expr = f"{numbers[i]} {ops[0].strip()} {numbers[j]}"
                                results.append(expr)

        return results[:10]  # Cap to avoid explosion.

    def stats(self) -> dict:
        """Return summary of learned patterns."""
        if not self._patterns:
            return {"total": 0}
        by_type = {}
        for p in self._patterns.values():
            by_type[p.pattern_type] = by_type.get(p.pattern_type, 0) + 1
        return {
            "total": len(self._patterns),
            "by_type": by_type,
            "top_patterns": sorted(
                [(p.expression, p.frequency) for p in self._patterns.values()],
                key=lambda x: -x[1],
            )[:10],
        }
