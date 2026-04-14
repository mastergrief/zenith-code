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


DEFAULT_DB = Path("calm/learned_patterns.jsonl")


class LearnedPattern:
    """A pattern learned from a correction.

    Two counters:
      - `frequency`: how often this pattern was LEARNED from (a correction
        with this shape was fed in).
      - `hits`: how often this pattern FIRED at inference time (matched a
        prompt and produced a precompute suggestion). Cold patterns
        (hits == 0 after many prompts) are pruning candidates.
    """
    def __init__(self, pattern_type: str, expression: str,
                 frequency: int = 1, hits: int = 0):
        self.pattern_type = pattern_type  # "arithmetic", "bool", "function"
        self.expression = expression      # e.g. "is_prime(N)"
        self.frequency = frequency
        self.hits = hits


class AutoLearner:
    """Tracks corrections and learns precompute patterns."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB
        self._patterns: Dict[str, LearnedPattern] = {}
        self._load()

    def _load(self):
        """Load learned patterns from disk. Legacy files without `hits`
        default to 0."""
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
                    hits=d.get("hits", 0),
                )

    def _save(self):
        """Persist all patterns to disk (including hit counters)."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "w") as f:
            for p in self._patterns.values():
                f.write(json.dumps({
                    "pattern_type": p.pattern_type,
                    "expression": p.expression,
                    "frequency": p.frequency,
                    "hits": p.hits,
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
        Returns {expression: computed_value} for patterns that match the prompt.

        Side effect: increments `hits` on every pattern that contributes
        at least one successful precompute. Hit counters are persisted.
        """
        results = {}
        fired_patterns: List[LearnedPattern] = []

        for key, pattern in self._patterns.items():
            # Only precompute patterns that have been wrong multiple times
            # or are high-frequency error patterns.
            if pattern.frequency < 1:
                continue

            # Try to instantiate the pattern with values from the prompt.
            pattern_contributed = False
            expressions = self._instantiate(pattern, prompt)
            for expr in expressions:
                if expr not in results:
                    # Guard: skip expressions with huge numbers that would
                    # hang (e.g. factorial(4532015112830366) from a CC number).
                    nums = re.findall(r'\d+', expr)
                    if any(int(n) > 10_000_000 for n in nums if len(n) < 20):
                        continue
                    try:
                        val = safe_eval(expr)
                        results[expr] = val
                        pattern_contributed = True
                    except ExpressionError:
                        pass
            if pattern_contributed:
                fired_patterns.append(pattern)

        if fired_patterns:
            for p in fired_patterns:
                p.hits += 1
            self._save()

        return results

    def prune_cold_patterns(self, min_hits: int = 1, min_frequency: int = 1) -> int:
        """Remove patterns that have never fired AND have only been
        learned once. Returns the number of patterns pruned.

        Default rules:
          - Patterns with `hits >= min_hits` are ALWAYS kept (they proved
            their worth at inference time).
          - Patterns with `hits == 0` are pruned unless `frequency >
            min_frequency` (the same error was seen multiple times, so
            it's worth keeping even if no hit yet).

        Caller controls thresholds. Callers that want aggressive pruning
        can pass `min_frequency=99` to cull anything that hasn't fired.
        """
        to_remove = [
            key for key, p in self._patterns.items()
            if p.hits < min_hits and p.frequency <= min_frequency
        ]
        for key in to_remove:
            del self._patterns[key]
        if to_remove:
            self._save()
        return len(to_remove)

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

    # Natural-language forms of each operator. Used to gate arithmetic
    # patterns so "what is 5 plus 7?" fires '+' patterns but not '*'.
    _OP_WORDS = {
        "+": ("plus", "added", "sum", "total", "more"),
        "-": ("minus", "subtract", "difference", "less", "fewer", "spent", "gave"),
        "*": ("times", "multiplied", "product", "each", "per"),
        "/": ("divided", "split", "over", "per"),
    }

    def _instantiate(self, pattern: LearnedPattern, prompt: str) -> List[str]:
        """Try to instantiate a pattern with values from the prompt.

        Shape gates prevent pattern pollution: a function pattern only
        fires if its function name (or a known alias) appears in the
        prompt; an arithmetic pattern only fires if the operator or a
        natural-language form appears. This replaces the old behavior
        where every pattern fired on every prompt with >=1 number,
        flooding precompute results with irrelevant suggestions.
        """
        results = []

        # Extract all numbers from the prompt.
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', prompt)
        if not numbers:
            return results

        expr_template = pattern.expression
        prompt_lower = prompt.lower()

        if pattern.pattern_type == "function":
            func_match = re.match(r'(\w+)\(([^)]*)\)', expr_template)
            if func_match:
                func = func_match.group(0)
                func_name = func_match.group(1)
                # Shape gate: function name (or a short alias) must appear
                # in the prompt. "factorial", "fact", "prime", "gcd", etc.
                if func_name.lower() not in prompt_lower:
                    # Aliases for common names so "is this prime" hits is_prime.
                    aliases = {
                        "is_prime": ("prime",),
                        "fibonacci": ("fib", "fibonacci"),
                        "factorial": ("factorial",),
                        "gcd": ("gcd",),
                        "lcm": ("lcm",),
                        "euler_totient": ("totient", "euler"),
                        "digital_root": ("digital root",),
                    }
                    hit_alias = any(a in prompt_lower
                                     for a in aliases.get(func_name, ()))
                    if not hit_alias:
                        return results
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
            if len(numbers) >= 2:
                ops = re.findall(r'[\+\-\*\/\%\^]+', expr_template)
                if ops:
                    op = ops[0].strip()
                    # Shape gate: operator (or NL form) must appear in prompt.
                    op_in_prompt = op in prompt
                    word_forms = self._OP_WORDS.get(op, ())
                    word_in_prompt = any(w in prompt_lower for w in word_forms)
                    if not (op_in_prompt or word_in_prompt):
                        return results
                    for i in range(len(numbers)):
                        for j in range(len(numbers)):
                            if i != j:
                                expr = f"{numbers[i]} {op} {numbers[j]}"
                                results.append(expr)

        return results[:10]  # Cap to avoid explosion.

    def stats(self) -> dict:
        """Return summary of learned patterns.

        Includes both learn-side counters (`frequency`) and inference-side
        counters (`hits`). A pattern with high frequency + 0 hits means
        the learner keeps seeing that error but the matcher never fires —
        probably a pattern-instantiation bug or prompt-shape mismatch.
        """
        if not self._patterns:
            return {"total": 0}
        by_type = {}
        cold = 0
        total_hits = 0
        for p in self._patterns.values():
            by_type[p.pattern_type] = by_type.get(p.pattern_type, 0) + 1
            if p.hits == 0:
                cold += 1
            total_hits += p.hits
        return {
            "total": len(self._patterns),
            "by_type": by_type,
            "total_hits": total_hits,
            "cold_patterns": cold,
            "top_patterns": sorted(
                [(p.expression, p.frequency, p.hits) for p in self._patterns.values()],
                key=lambda x: (-x[2], -x[1]),  # hits first, then frequency
            )[:10],
        }
