"""
Auto-CALM training data generator.

Every Auto-CALM correction is a free labeled training example.
This module captures corrections from all 3 layers and generates
training data in the distillation format (messages with <think> blocks).

Sub-collectors for each "computer":
  - MathCollector:  wrong arithmetic claims → correct reasoning examples
  - BoolCollector:  wrong primality/divisibility → correct examples
  - CodeCollector:  bug diagnosis + fix → coding reasoning examples
  - PrecomputeCollector: usage of verified facts → grounded examples

All output to .calm_training/auto/ as JSONL, compatible with the
distillation pipeline (agents/distill/).

Usage:
    from calm.auto_training import AutoTrainingCollector
    collector = AutoTrainingCollector()
    collector.collect_math(prompt, wrong_response, corrected_response, claims)
    collector.collect_code(file_path, diagnosis, before_source, after_source, test_results)
    print(collector.stats())
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


DEFAULT_DIR = Path(".calm_training/auto")

SYSTEM_PROMPT = "You are a helpful assistant"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_message(system: str, user: str, assistant: str) -> dict:
    """Build a training message in distillation format."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


class MathCollector:
    """Generate training examples from wrong arithmetic claims.

    When the model says '17 × 23 = 401' and Auto-CALM corrects to 391,
    this produces a training example where the assistant shows correct
    reasoning with a <think> block.
    """

    def __init__(self, output: Path):
        self.output = output / "math.jsonl"
        self.count = 0

    def collect(self, prompt: str, claims: list) -> int:
        """Generate examples from wrong claims. Returns count written."""
        written = 0
        for claim in claims:
            if claim.correct or claim.actual_value is None:
                continue
            # Build a training example that shows correct reasoning.
            think = (
                f"<think>\nI need to compute {claim.expression}.\n"
                f"Let me calculate carefully: {claim.expression} = {claim.actual_value}\n"
                f"I should verify: the result is {claim.actual_value}.\n</think>"
            )
            answer = (
                f"{think}\n\n"
                f"The answer is {claim.expression} = {claim.actual_value}."
            )
            example = _make_message(SYSTEM_PROMPT, prompt, answer)
            example["meta"] = {
                "source": "auto-calm-math",
                "timestamp": _now(),
                "wrong_value": claim.claimed_value,
                "correct_value": str(claim.actual_value),
                "expression": claim.expression,
            }
            self._append(example)
            written += 1

        self.count += written
        return written

    def _append(self, record: dict):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output, "a") as f:
            f.write(json.dumps(record) + "\n")


class BoolCollector:
    """Generate training examples from wrong boolean claims.

    When the model says '391 is prime' and Auto-CALM corrects it,
    this produces a training example with correct primality reasoning.
    """

    def __init__(self, output: Path):
        self.output = output / "bool.jsonl"
        self.count = 0

    def collect(self, prompt: str, claims: list) -> int:
        written = 0
        for claim in claims:
            if claim.correct or claim.actual_value is None:
                continue
            if not isinstance(claim.actual_value, bool):
                continue

            # Build reasoning for the boolean claim.
            if "is_prime" in claim.expression:
                think = self._prime_reasoning(claim)
            elif "is_perfect" in claim.expression:
                think = self._perfect_reasoning(claim)
            elif "%" in claim.expression:
                think = self._divisibility_reasoning(claim)
            else:
                continue

            answer = f"{think}\n\n{'Yes' if claim.actual_value else 'No'}."
            example = _make_message(SYSTEM_PROMPT, prompt, answer)
            example["meta"] = {
                "source": "auto-calm-bool",
                "timestamp": _now(),
                "expression": claim.expression,
                "correct": claim.actual_value,
            }
            self._append(example)
            written += 1

        self.count += written
        return written

    def _prime_reasoning(self, claim) -> str:
        import re
        m = re.search(r'is_prime\((\d+)\)', claim.expression)
        n = int(m.group(1)) if m else 0
        if claim.actual_value:
            return (
                f"<think>\nI need to check if {n} is prime.\n"
                f"Testing divisibility by primes up to sqrt({n}).\n"
                f"None divide evenly, so {n} is prime.\n</think>"
            )
        else:
            # Find a factor.
            from calm.expression import _factorize
            factors = _factorize(n) if n > 1 else []
            if factors:
                return (
                    f"<think>\nI need to check if {n} is prime.\n"
                    f"Testing divisibility: {n} = {factors[0]} × {n // factors[0]}\n"
                    f"Since {n} has factors other than 1 and itself, "
                    f"it is not prime.\n</think>"
                )
            return f"<think>\n{n} is not prime.\n</think>"

    def _perfect_reasoning(self, claim) -> str:
        import re
        m = re.search(r'is_perfect\((\d+)\)', claim.expression)
        n = int(m.group(1)) if m else 0
        from calm.expression import _divisors
        divs = _divisors(n) if n > 1 else []
        proper = divs[:-1] if divs else []
        s = sum(proper)
        return (
            f"<think>\nA perfect number equals the sum of its proper divisors.\n"
            f"Divisors of {n}: {proper}\n"
            f"Sum: {s}\n"
            f"{'Equals' if s == n else 'Does not equal'} {n}, so "
            f"{'it is' if claim.actual_value else 'it is not'} perfect.\n</think>"
        )

    def _divisibility_reasoning(self, claim) -> str:
        import re
        m = re.search(r'(\d+)\s*%\s*(\d+)', claim.expression)
        if not m:
            return f"<think>\n{claim.expression} = {claim.actual_value}\n</think>"
        n, d = int(m.group(1)), int(m.group(2))
        remainder = n % d
        return (
            f"<think>\n{n} ÷ {d} = {n // d} remainder {remainder}.\n"
            f"Since the remainder is {'0' if remainder == 0 else 'not 0'}, "
            f"{n} {'is' if remainder == 0 else 'is not'} divisible by {d}.\n</think>"
        )

    def _append(self, record: dict):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output, "a") as f:
            f.write(json.dumps(record) + "\n")


class CodeCollector:
    """Generate training examples from bug diagnosis + fix.

    When the model diagnoses bugs and the engine fixes them, this
    produces training examples that teach diagnosis + fix reasoning.
    """

    def __init__(self, output: Path):
        self.output = output / "code.jsonl"
        self.count = 0

    def collect(
        self,
        file_path: str,
        diagnosis: str,
        before_source: str,
        after_source: str,
        before_tests: str,
        after_tests: str,
    ) -> int:
        """Generate a training example from a successful bug fix."""
        import re

        # Only collect from successful fixes.
        before_count = int(m.group(1)) if (m := re.search(r'(\d+) passed', before_tests)) else 0
        after_count = int(m.group(1)) if (m := re.search(r'(\d+) passed', after_tests)) else 0
        if after_count <= before_count:
            return 0

        # Extract failure lines from test output.
        failures = [
            line.strip() for line in before_tests.splitlines()
            if line.strip().startswith("FAILED")
        ]

        # Build the training example.
        think = (
            f"<think>\nLet me analyze the failing tests:\n"
            + "\n".join(f"- {f}" for f in failures[:5]) + "\n\n"
            f"Looking at the code, I can identify these issues:\n"
            f"{diagnosis[:500]}\n\n"
            f"I'll fix each bug while keeping all existing functions.\n</think>"
        )

        prompt = (
            f"Here is a Python file with bugs:\n```python\n{before_source}\n```\n\n"
            f"These tests are failing:\n"
            + "\n".join(f"- {f}" for f in failures[:5])
            + "\n\nFix all the bugs."
        )

        answer = (
            f"{think}\n\n"
            f"Here's the fixed code:\n```python\n{after_source}\n```"
        )

        example = _make_message(SYSTEM_PROMPT, prompt, answer)
        example["meta"] = {
            "source": "auto-calm-code",
            "timestamp": _now(),
            "file": file_path,
            "before_passed": before_count,
            "after_passed": after_count,
            "bugs_fixed": after_count - before_count,
        }
        self._append(example)
        self.count += 1
        return 1

    def _append(self, record: dict):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output, "a") as f:
            f.write(json.dumps(record) + "\n")


class AutoTrainingCollector:
    """Unified collector that dispatches to sub-collectors."""

    def __init__(self, log_dir: Optional[Path] = None):
        self.dir = log_dir or DEFAULT_DIR
        self.math = MathCollector(self.dir)
        self.bool = BoolCollector(self.dir)
        self.code = CodeCollector(self.dir)

    def collect_from_verify(self, prompt: str, claims: list) -> int:
        """Collect from Layer 1/2 claim verification."""
        n = 0
        math_claims = [c for c in claims if not isinstance(c.actual_value, bool)]
        bool_claims = [c for c in claims if isinstance(c.actual_value, bool)]
        n += self.math.collect(prompt, math_claims)
        n += self.bool.collect(prompt, bool_claims)
        return n

    def collect_from_edit(
        self, file_path, diagnosis, before_src, after_src,
        before_tests, after_tests,
    ) -> int:
        """Collect from Layer 3 intent-to-edit."""
        return self.code.collect(
            file_path, diagnosis, before_src, after_src,
            before_tests, after_tests,
        )

    def stats(self) -> dict:
        """Count examples per sub-collector."""
        totals = {}
        for name, collector in [
            ("math", self.math), ("bool", self.bool), ("code", self.code),
        ]:
            path = collector.output
            if path.exists():
                with open(path) as f:
                    count = sum(1 for _ in f)
            else:
                count = 0
            totals[name] = count
        totals["total"] = sum(totals.values())
        return totals

    def export_merged(self, output_path: str = None) -> str:
        """Merge all sub-collector data into one JSONL file."""
        output = output_path or str(self.dir / "merged.jsonl")
        all_examples = []
        for collector in [self.math, self.bool, self.code]:
            if collector.output.exists():
                with open(collector.output) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            all_examples.append(json.loads(line))

        with open(output, "w") as f:
            for ex in all_examples:
                f.write(json.dumps(ex) + "\n")

        return f"{len(all_examples)} examples → {output}"
