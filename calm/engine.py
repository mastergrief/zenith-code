"""
CALM v0.1 execution engine — closed-loop stream injection.

Sends a prompt to llama-server, stops at </calm>, processes the CALM
block via the interceptor (with TMR verification), injects results
back as continuation text, and resumes generation. Repeats until the
model finishes naturally (no more CALM blocks).

This closes the loop: the model emits <calm>push 17\npush 23\nmul -> <pending>\n</calm>,
the engine computes [391], injects "Result: [391]" after the block,
and the model sees it in subsequent tokens.

Usage:
    from calm.engine import CalmEngine
    engine = CalmEngine()
    result = engine.run("What is 17 * 23 + 42 * 19 - 100?")
    print(result.response)       # final text
    print(result.vm_outputs)     # all emitted values
    print(result.training_log)   # claim vs actual pairs
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional

import os
import time

from calm.interceptor import Event, EventType, Interceptor
from calm.training import TrainingCollector
from calm.verifier import make_verified_dispatcher

SERVER = "http://localhost:8080"

SYSTEM_PROMPT = """\
You have a compute engine. Embed <calm>...</calm> blocks for exact computation.

Write expressions, Python, or stack code. The engine handles all formats:
<calm>
17 * 23
</calm>
<calm>
next_prime(1000)
</calm>
<calm>
result = [p for p in range(2, 50) if is_prime(p)]
result
</calm>

Available functions:
  Arithmetic: +, -, *, /, //, %, ** (or ^)
  Math: sqrt, pow, abs, floor, ceil, log, log2, log10, pi, e, factorial
  Number theory: is_prime, next_prime, prev_prime, nth_prime, gcd, lcm,
    factorize, divisors, count_divisors, is_perfect, digit_sum, digital_root
  Sequences: fibonacci, collatz, collatz_length
  Algebra: solve_quadratic(a, b, c)
  Ranges: sum_range(a, b), product_range(a, b)
  Comparison: ==, !=, <, <=, >, >=, and, or
  Also: list comprehensions, variables, for loops, Python expressions

One expression per line. After </calm>, the engine shows results.
Use those results in your next step. Prefer <calm> over mental math."""


@dataclass
class EngineResult:
    response: str = ""                      # full assembled response text
    calm_blocks: int = 0                   # number of CALM blocks processed
    vm_outputs: List = field(default_factory=list)  # all emitted values
    training_log: List[dict] = field(default_factory=list)
    iterations: int = 0                    # generation rounds
    total_tokens: int = 0
    tok_per_sec: float = 0.0


class CalmEngine:
    """Closed-loop CALM execution engine."""

    def __init__(
        self,
        server: str = SERVER,
        system_prompt: str = SYSTEM_PROMPT,
        max_iterations: int = 10,
        max_tokens_per_turn: int = 8192,
        thinking_budget: int = 16384,
    ):
        self.server = server
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.max_tokens_per_turn = max_tokens_per_turn
        self.thinking_budget = thinking_budget
        self.dispatcher = make_verified_dispatcher()

    def run(self, prompt: str, verbose: bool = False) -> EngineResult:
        """
        Run the closed-loop engine.

        Hybrid approach:
        - If thinking_budget > 0: planning turn (thinking only, no CALM)
          followed by stop-mode execution turns
        - If thinking_budget = 0: pure stop-mode execution turns

        Stop-mode: halt generation at </calm>, process the block,
        inject real results, continue. Model never fabricates results.
        """
        result = EngineResult()
        assembled = ""
        interceptor = Interceptor(
            dispatcher=self.dispatcher, strict=False, persist_state=False,
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        # Optional planning phase with thinking.
        if self.thinking_budget > 0:
            plan_msgs = list(messages)
            plan_msgs[-1] = {
                "role": "user",
                "content": (
                    f"{prompt}\n\n"
                    f"Plan your approach step by step. "
                    f"Do NOT write <calm> blocks yet."
                ),
            }
            plan_content, thinking, timings, _ = self._generate(plan_msgs)
            if timings:
                result.tok_per_sec = timings.get("predicted_per_second", 0)
            if verbose and thinking:
                think_preview = thinking[:200] + "..." if len(thinking) > 200 else thinking
                print(f"[plan] {len(thinking)} chars: {think_preview}")
            # Feed the planning response back as context.
            if plan_content:
                messages.append({"role": "assistant", "content": plan_content})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Good plan. Now answer the original question: {prompt}\n"
                        f"Use <calm> blocks for all computation."
                    ),
                })

        # Execution loop: stop-mode, per-block injection.
        consecutive_errors = 0
        empty_retries = 0
        for i in range(self.max_iterations):
            result.iterations = i + 1

            content, thinking, timings, finish = self._generate(
                messages, stop=["</calm>"]
            )

            if timings:
                result.tok_per_sec = timings.get("predicted_per_second", 0)
                result.total_tokens += timings.get("predicted_n", 0)

            if thinking:
                interceptor.feed(thinking)

            # Empty response — model produced nothing. Retry (max 2).
            if not content.strip() and empty_retries < 2:
                empty_retries += 1
                if verbose:
                    print(f"[iter {i+1}] empty response — retry {empty_retries}")
                messages.append({
                    "role": "user",
                    "content": (
                        "Your response was empty. Please answer using "
                        "<calm>...</calm> to compute the result."
                    ),
                })
                continue

            has_calm = finish == "stop" and "<calm>" in content

            if has_calm:
                full_block = content + "</calm>"
                events = interceptor.feed(full_block)

                block_errors = [e for e in events if e.type == EventType.ERROR]

                # If line-by-line had errors, try the whole block as
                # Python (the model often writes multi-line Python).
                if block_errors:
                    import re
                    calm_match = re.search(r'<calm>(.*?)$', content, re.DOTALL)
                    if calm_match:
                        block_code = calm_match.group(1).strip()
                        # Strip comments and claim suffixes.
                        clean_lines = []
                        for ln in block_code.splitlines():
                            ln = ln.strip()
                            if ln.startswith("#") or ln.startswith("//") or ln.startswith("\\"):
                                continue
                            ln = re.sub(r'\s*->.*$', '', ln)
                            if ln and not ln.startswith("[engine:"):
                                clean_lines.append(ln)
                        if clean_lines:
                            from calm.sandbox import run_python as _run_py
                            sr = _run_py("\n".join(clean_lines), timeout=10.0)
                            if sr.ok and sr.value is not None:
                                # Python execution succeeded — override.
                                interceptor.state.stack.append(sr.value)
                                events = [
                                    Event(type=EventType.CALM_START),
                                    Event(
                                        type=EventType.EXECUTED,
                                        instruction="[python block]",
                                        actual_stack=list(interceptor.state.stack),
                                        text=f"python={sr.value}",
                                    ),
                                    Event(type=EventType.CALM_END),
                                ]
                                block_errors = []

                block_stack = list(interceptor.state.stack)
                block_output = list(interceptor.state.output)
                result.calm_blocks += 1
                result.vm_outputs = list(interceptor.state.output)

                injection = self._format_injection(
                    block_stack, block_output,
                    block_errors,
                    [e for e in events if e.type == EventType.DIVERGENCE],
                )

                has_errors = any(
                    e.type == EventType.ERROR for e in events
                )
                if has_errors:
                    consecutive_errors += 1
                else:
                    consecutive_errors = 0

                if verbose:
                    print(f"[iter {i+1}] CALM block {result.calm_blocks}"
                          + (f" ({consecutive_errors} errors)" if has_errors else ""))
                    print(f"  inject: {injection}")

                # Bail out if stuck in an error loop.
                if consecutive_errors >= 3:
                    if verbose:
                        print(f"[iter {i+1}] bailing out — 3 consecutive errors")
                    break

                assembled += full_block + "\n" + injection + "\n"
                messages.append({
                    "role": "assistant",
                    "content": full_block + "\n" + injection,
                })
                # Tailor continuation based on errors.
                if block_errors:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Engine result: {injection}\n"
                            f"Some lines had errors. Write simpler expressions, e.g.:\n"
                            f"<calm>\n"
                            f"result = 17 * 23\n"
                            f"is_prime(result)\n"
                            f"</calm>\n"
                            f"Use variables to chain steps. Continue answering."
                        ),
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Engine result: {injection} "
                            f"Continue answering the original question."
                        ),
                    })
            else:
                # Model didn't use CALM. Two paths:
                # 1. Post-verify: compute independently, check answer
                # 2. If wrong OR unverifiable: force a CALM retry
                if result.calm_blocks == 0 and content.strip() and i < self.max_iterations - 1:
                    correction = self._post_verify(prompt, content, verbose)
                    if correction:
                        # Wrong answer — force CALM.
                        if verbose:
                            print(f"[iter {i+1}] post-verify FAIL — forcing CALM")
                        messages.append({"role": "assistant", "content": content})
                        messages.append({"role": "user", "content": correction})
                        continue
                    elif correction is None and not self._can_verify(prompt):
                        # Can't verify — force CALM to be safe.
                        if verbose:
                            print(f"[iter {i+1}] unverifiable — forcing CALM")
                        messages.append({"role": "assistant", "content": content})
                        messages.append({
                            "role": "user",
                            "content": (
                                "Please verify your answer using <calm>...</calm> "
                                "to compute the result with the engine."
                            ),
                        })
                        continue

                assembled += content
                interceptor.feed(content)
                if verbose:
                    print(f"[iter {i+1}] done ({len(content)} chars)")
                break

        result.response = assembled
        result.training_log = interceptor.training_log

        if result.training_log:
            collector = TrainingCollector()
            n = collector.save(result, prompt=prompt)
            if verbose:
                stats = collector.stats()
                print(f"[training] +{n} entries (total: {stats['total']}, "
                      f"accuracy: {stats['accuracy']:.0f}%)")

        return result

    def _generate(self, messages, stop=None):
        """Send a chat completion request."""
        payload = {
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.max_tokens_per_turn,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop
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
        content = choice["message"].get("content", "")
        thinking = choice["message"].get("reasoning_content", "")
        timings = data.get("timings", {})
        finish = choice.get("finish_reason", "")
        return content, thinking, timings, finish

    def _post_verify(self, prompt: str, response: str, verbose: bool) -> Optional[str]:
        """
        Post-response verification. Tries to independently compute
        the answer from the prompt and checks if the model's response
        contains the correct result.

        Returns a correction message if the model was wrong, or None
        if the answer checks out (or can't be verified).
        """
        import re
        from calm.expression import safe_eval, ExpressionError
        from calm.sandbox import run_python

        # Try to extract computable expressions from the prompt.
        # Split on sentence boundaries, try each.
        expr_patterns = [
            r'[Ww]hat is (.+?)[\?\.]',
            r'[Cc]ompute (.+?)[\?\.]',
            r'[Cc]alculate (.+?)[\?\.]',
            r'[Ff]ind (.+?)[\?\.]',
            r'[Hh]ow (?:long|many|much) is (.+?)[\?\.]',
            r'[Hh]ow (?:long|many) .+? (?:of|from|starting from?) (.+?)[\?\.]',
            r'[Hh]ow long is (.+?)[\?\.]',
        ]

        expr = None
        for pat in expr_patterns:
            m = re.search(pat, prompt)
            if m:
                expr = m.group(1).strip()
                break

        if not expr:
            return None

        # Clean up the expression for evaluation.
        expr = (expr
            .replace('^', '**')
            .replace('×', '*')
            .replace('÷', '/')
            .replace(',', '')  # "1,024" → "1024"
        )

        # NL → expression translation for common phrasings.
        nl_rewrites = [
            (r'the (\d+)(?:st|nd|rd|th) [Ff]ibonacci number', r'fibonacci(\1)'),
            (r'the (\d+)(?:st|nd|rd|th) prime(?: number)?', r'nth_prime(\1)'),
            (r'the [Cc]ollatz sequence (?:starting |)from (\d+)', r'collatz(\1)'),
            (r'the length of the [Cc]ollatz .+ (\d+)', r'collatz_length(\1)'),
            (r'the digit sum of (\d+)', r'digit_sum(\1)'),
            (r'the digital root of (\d+)', r'digital_root(\1)'),
            (r'the smallest prime (?:greater than|>) (\d+)', r'next_prime(\1)'),
            (r'the (?:prime )?factors of (\d+)', r'factorize(\1)'),
            (r'the GCD of (\d+) and (\d+)', r'gcd(\1, \2)'),
            (r'the LCM of (\d+) and (\d+)', r'lcm(\1, \2)'),
        ]
        for pat, repl in nl_rewrites:
            new_expr = re.sub(pat, repl, expr)
            if new_expr != expr:
                expr = new_expr
                break

        # Try to compute it.
        computed = None
        try:
            computed = safe_eval(expr)
        except ExpressionError:
            sr = run_python(expr, timeout=5.0)
            if sr.ok and sr.value is not None:
                computed = sr.value

        if computed is None:
            return None  # Can't compute the answer independently.

        # Check if the computed answer appears in the response.
        # Build multiple string representations to match against.
        computed_strs = {str(computed)}
        if isinstance(computed, float) and computed == int(computed):
            computed_strs.add(str(int(computed)))
        if isinstance(computed, int):
            # Also check comma-formatted: 832040 → "832,040"
            s = str(computed)
            if len(s) > 3:
                formatted = ""
                for i, c in enumerate(reversed(s)):
                    if i > 0 and i % 3 == 0:
                        formatted = "," + formatted
                    formatted = c + formatted
                computed_strs.add(formatted)

        # Also strip commas from the response for matching.
        response_clean = response.replace(",", "")
        for cs in computed_strs:
            if cs in response or cs.replace(",", "") in response_clean:
                if verbose:
                    print(f"[verify] answer {cs} found in response — OK")
                return None

        # Model's answer doesn't contain the computed result.
        if verbose:
            print(f"[verify] expected {computed_str}, not found in response")
        return (
            f"Your answer appears incorrect. The compute engine says: "
            f"{expr} = {computed_str}. "
            f"Please use <calm>{expr}</calm> to verify and give the correct answer."
        )

    def _can_verify(self, prompt: str) -> bool:
        """Check if the prompt contains something we can compute independently."""
        import re
        # If the prompt asks for a computation, we should be able to verify.
        compute_signals = [
            r'\d+\s*[\+\-\*\/\^]',      # arithmetic operators
            r'[Ww]hat is',               # "What is X?"
            r'[Cc]ompute|[Cc]alculate',
            r'[Ff]ind.*\d',              # "Find X that..."
            r'[Ss]olve',
            r'prime|factor|gcd|lcm|fibonacci|collatz|digit|divisor|sqrt|log',
            r'[Hh]ow many|[Hh]ow long|[Hh]ow much',
        ]
        return any(re.search(p, prompt) for p in compute_signals)

    def _format_injection(self, stack, output, errors, divergences):
        """Format the injection text after a CALM block. Caps at 2000 chars."""
        parts = []
        if output:
            parts.append(f"output={self._truncate(output)}")
        if stack:
            parts.append(f"stack={self._truncate(stack)}")
        if not output and not stack:
            parts.append("stack=[]")
        if errors:
            err_msgs = [e.text for e in errors[:2]]
            parts.append(f"errors={err_msgs}")
        if divergences:
            parts.append("WARNING: TMR DIVERGENCE")
        result = f"[engine: {', '.join(parts)}]"
        return result[:2000]

    @staticmethod
    def _truncate(val, max_len=800):
        """Truncate large values (file contents, etc.) for injection."""
        # Deep truncate: shrink dict values that contain file content
        if isinstance(val, list):
            val = [CalmEngine._truncate_item(v) for v in val[-5:]]  # last 5 items only
        elif isinstance(val, dict):
            val = CalmEngine._truncate_item(val)
        s = str(val)
        if len(s) <= max_len:
            return s
        return s[:max_len] + "..."

    @staticmethod
    def _truncate_item(v, max_str=200):
        """Truncate individual items — strip file contents, long strings."""
        if isinstance(v, dict):
            out = {}
            for k, val in v.items():
                if k == "content" and isinstance(val, str) and len(val) > max_str:
                    out[k] = val[:max_str] + f"... ({len(val)} chars)"
                elif k == "output" and isinstance(val, str) and len(val) > max_str:
                    out[k] = val[:max_str] + "..."
                elif k == "lines" and isinstance(val, list) and len(val) > 10:
                    out[k] = val[:10] + [f"... ({len(val)} total)"]
                else:
                    out[k] = val
            return out
        if isinstance(v, str) and len(v) > max_str:
            return v[:max_str] + "..."
        return v


def run_engine(prompt: str, verbose: bool = True, **kwargs) -> EngineResult:
    """Convenience function for CLI usage."""
    engine = CalmEngine(**kwargs)
    result = engine.run(prompt, verbose=verbose)

    print(f"\n{'='*60}")
    print(f"Response:\n{result.response}")
    print(f"\nCALM blocks:  {result.calm_blocks}")
    print(f"VM outputs:   {result.vm_outputs}")
    print(f"Iterations:   {result.iterations}")
    print(f"Training log: {len(result.training_log)} entries")
    print(f"Speed:        {result.tok_per_sec:.1f} tok/s")
    return result


if __name__ == "__main__":
    import sys
    prompt = sys.argv[1] if len(sys.argv) > 1 else (
        "What is 17 * 23? Is the result prime? What is its GCD with 782?"
    )
    run_engine(prompt)
