"""
Auto-CALM — modular compute facade.

Thin facade that composes the three layers:
  Layer 1: verify.py   — claim extraction + verification
  Layer 2: precompute.py — precomputation + system prompt
  Layer 3: intent_edit.py — NL diagnosis → template fix → verify

Usage:
    from calm.auto_calm import AutoCalmEngine, IntentToEdit
    engine = AutoCalmEngine()
    result = engine.run("What is 17 * 23? Is it prime?")

    fixer = IntentToEdit()
    result = fixer.fix("app.py", "test_app.py", verbose=True)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

# Re-export from sub-modules for backward compatibility.
from calm.verify import AutoCalm, Claim, VerifyReport
from calm.precompute import build_system_prompt, precompute
from calm.intent_edit import IntentToEdit, EditResult, EditIntent
from calm.expression import safe_eval, ExpressionError


AUTO_SYSTEM_PROMPT = build_system_prompt()


@dataclass
class AutoCalmResult:
    """Result from the Auto-CALM engine."""
    response: str = ""
    original_response: str = ""
    claims_found: int = 0
    claims_corrected: int = 0
    claims_verified: int = 0
    thinking_chars: int = 0
    tok_per_sec: float = 0.0
    corrections: List[Claim] = field(default_factory=list)


class AutoCalmEngine:
    """
    Auto-CALM engine — modular compute facade.

    Composes: verify (Layer 1) + precompute (Layer 2) + self-learning.
    The model writes naturally, the engine verifies and corrects.
    """

    def __init__(
        self,
        server: str = "http://localhost:8080",
        system_prompt: str = AUTO_SYSTEM_PROMPT,
        max_tokens: int = 16384,
        thinking_budget: int = 32768,
        precompute_enabled: bool = True,
    ):
        self.server = server
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.thinking_budget = thinking_budget
        self.precompute_enabled = precompute_enabled
        self.verifier = AutoCalm()
        from calm.auto_learn import AutoLearner
        self.learner = AutoLearner()

    def run(self, prompt: str, verbose: bool = False) -> AutoCalmResult:
        """Run a prompt, verify claims, retry if wrong (max 1 retry)."""
        result = AutoCalmResult()

        # Layer 2: precompute from prompt + learned patterns.
        precomputed = precompute(prompt) if self.precompute_enabled else {}
        learned = self.learner.suggest_precomputes(prompt)
        if learned:
            precomputed.update(learned)
            if verbose:
                print(f"[learned] +{len(learned)} from error patterns")

        system = self.system_prompt
        if precomputed:
            facts = "; ".join(f"{k} = {v}" for k, v in precomputed.items())
            system += f"\n\nVerified facts: {facts}"
            if verbose:
                print(f"[precompute] {facts}")

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        # Generate response.
        content, thinking, timings = self._generate(messages)
        result.original_response = content
        result.thinking_chars = len(thinking)
        result.tok_per_sec = timings.get("predicted_per_second", 0)

        if verbose and thinking:
            print(f"[think] {len(thinking)} chars: {thinking[:200].replace(chr(10), ' ')}...")

        # Layer 1: verify inline claims.
        corrected, report = self.verifier.verify_and_correct(content)

        # Layer 2: verify answer against prompt expression.
        prompt_check = self._verify_prompt_answer(prompt, corrected)
        if prompt_check:
            report.claims.append(prompt_check)
            if prompt_check.correct:
                report.verified += 1
            else:
                report.corrections += 1

        # Retry on wrong prompt-level answer.
        if prompt_check and not prompt_check.correct:
            actual_str = self.verifier._format_value(prompt_check.actual_value)
            if verbose:
                print(f"[auto-calm] answer wrong: {prompt_check.expression}"
                      f" = {actual_str}, not {prompt_check.claimed_value}")
                print(f"[auto-calm] retrying with correction...")

            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": (
                f"Your computation has an error. "
                f"The correct value of {prompt_check.expression} is "
                f"{actual_str}. Please give a corrected answer."
            )})

            content2, thinking2, timings2 = self._generate(messages)
            result.thinking_chars += len(thinking2)
            if timings2.get("predicted_per_second", 0):
                result.tok_per_sec = timings2["predicted_per_second"]

            corrected2, report2 = self.verifier.verify_and_correct(content2)
            prompt_check2 = self._verify_prompt_answer(prompt, corrected2)

            if prompt_check2 and prompt_check2.correct:
                corrected, report = corrected2, report2
                report.claims.append(prompt_check2)
                report.verified += 1
                if verbose:
                    print(f"[auto-calm] retry succeeded")
            else:
                corrected += (
                    f"\n\n[Auto-CALM correction: {prompt_check.expression}"
                    f" = {actual_str}, not {prompt_check.claimed_value}]"
                )
                if verbose:
                    print(f"[auto-calm] retry failed, appending note")

        result.response = corrected
        result.claims_found = len(report.claims)
        result.claims_corrected = report.corrections
        result.claims_verified = report.verified
        result.corrections = [c for c in report.claims if not c.correct]

        if verbose:
            print(f"[auto-calm] {result.claims_found} claims: "
                  f"{result.claims_verified} OK, "
                  f"{result.claims_corrected} corrected, "
                  f"{report.unverifiable} unverifiable")
            for c in result.corrections:
                print(f"  FIX: {c.expression} = {c.claimed_value} → {c.actual_value}")

        # Training data + learning from corrections.
        if result.claims_corrected > 0:
            from calm.auto_training import AutoTrainingCollector
            tc = AutoTrainingCollector()
            n = tc.collect_from_verify(prompt, report.claims)
            if verbose and n:
                print(f"[training] +{n} examples generated")
            for c in report.claims:
                self.learner.learn_from_correction(c)
            if verbose:
                print(f"[learned] patterns: {self.learner.stats()['total']}")

        return result

    def _generate(self, messages):
        """Send chat completion. Returns (content, thinking, timings)."""
        import json
        import urllib.request

        payload = {
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.thinking_budget > 0:
            payload["enable_thinking"] = True
            payload["thinking_budget"] = self.thinking_budget

        req = urllib.request.Request(
            f"{self.server}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())

        choice = data["choices"][0]
        return (
            choice["message"].get("content", ""),
            choice["message"].get("reasoning_content", ""),
            data.get("timings", {}),
        )

    def _verify_prompt_answer(self, prompt: str, response: str) -> Optional[Claim]:
        """Cross-check the model's answer against prompt expression."""
        expr_patterns = [
            r'[Ww]hat is (.+?)[\?\.]',
            r'[Cc]ompute (.+?)[\?\.]',
            r'[Cc]alculate (.+?)[\?\.]',
        ]

        expr = None
        for pat in expr_patterns:
            m = re.search(pat, prompt)
            if m:
                expr = self.verifier._normalize_expr(m.group(1).strip())
                break

        if not expr:
            return None

        try:
            expected = safe_eval(expr)
        except ExpressionError:
            return None
        if expected is None:
            return None

        expected_strs = {str(expected)}
        if isinstance(expected, float) and expected == int(expected):
            expected_strs.add(str(int(expected)))
        if isinstance(expected, int):
            s = str(abs(expected))
            if len(s) > 3:
                formatted = ""
                for i, c in enumerate(reversed(s)):
                    if i > 0 and i % 3 == 0:
                        formatted = "," + formatted
                    formatted = c + formatted
                if expected < 0:
                    formatted = "-" + formatted
                expected_strs.add(formatted)

        response_clean = response.replace(",", "")
        found = any(
            es in response or es.replace(",", "") in response_clean
            for es in expected_strs
        )

        answer_m = re.search(
            r'(?:product|result|answer)\s+(?:is|=)\s+[\*]*(\d[\d,]*)',
            response, re.IGNORECASE,
        )
        claimed = answer_m.group(1).replace(",", "") if answer_m else "?"
        if claimed == "?":
            numbers = re.findall(r'\b(\d[\d,]*\d)\b', response)
            claimed = numbers[-1].replace(",", "") if numbers else "?"

        return Claim(
            original=f"[prompt: {expr}]", expression=expr,
            claimed_value=claimed, actual_value=expected,
            correct=found, span=(0, 0),
        )


# --- CLI entry points ---

def run_auto(prompt: str, verbose: bool = True, **kwargs) -> AutoCalmResult:
    """CLI convenience for auto-calm verification."""
    engine = AutoCalmEngine(**kwargs)
    result = engine.run(prompt, verbose=verbose)

    print(f"\n{'='*60}")
    if result.claims_corrected:
        print(f"CORRECTED Response:\n{result.response}")
        print(f"\nOriginal (wrong):\n{result.original_response}")
    else:
        print(f"Response:\n{result.response}")
    print(f"\nClaims:       {result.claims_found} found, "
          f"{result.claims_verified} OK, {result.claims_corrected} fixed")
    print(f"Thinking:     {result.thinking_chars} chars")
    print(f"Speed:        {result.tok_per_sec:.1f} tok/s")
    return result


def run_edit(file_path: str, test_path: str, verbose: bool = True) -> EditResult:
    """CLI convenience for intent-to-edit."""
    engine = IntentToEdit()
    result = engine.fix(file_path, test_path, verbose=verbose)

    print(f"\n{'='*60}")
    print(f"Before: {result.original_tests.splitlines()[-2] if result.original_tests else '?'}")
    print(f"After:  {result.final_tests.splitlines()[-2] if result.final_tests else '?'}")
    print(f"Edits:  {result.edits_applied}/{result.edits_attempted}")
    print(f"Result: {'SUCCESS' if result.success else 'NEEDS WORK'}")
    return result


if __name__ == "__main__":
    import sys
    prompt = sys.argv[1] if len(sys.argv) > 1 else (
        "What is 17 * 23? Is the result prime? "
        "What is its GCD with 782?"
    )
    run_auto(prompt)
