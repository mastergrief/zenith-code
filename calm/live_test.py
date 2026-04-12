"""
CALM v0.1 live test — sends a math problem to llama-server with GBNF
grammar constraint, streams the response through the interceptor, and
reports execution events + final validation.

Usage:
    python3 -m calm.live_test
    python3 -m calm.live_test "What is 123 + 456?"
"""

from __future__ import annotations

import json
import sys
import urllib.request

from calm.grammar import generate_gbnf
from calm.interceptor import EventType, Interceptor

SERVER = "http://localhost:8080"

SYSTEM_PROMPT = """\
You are a calculator that solves math problems using CALM (Code for \
Append-only Lookup Machines). You MUST solve the problem by emitting \
a CALM program.

CALM is a stack-based language. Available instructions:
- push <number>: push a value onto the stack
- add, sub, mul, div, mod: pop two values, push result (second-from-top OP top)
- neg, abs: unary operations
- dup, drop, swap, over, rot: stack manipulation
- emit: pop and output a value
- halt: stop execution
- : name ... ; : define a reusable word

After each instruction, write ` -> [...]` showing the current stack state.
For example:
<calm>
push 17 -> [17]
push 23 -> [17, 23]
add -> [40]
emit -> []
halt
</calm>

Break complex operations into steps. Always emit the final answer."""


def send_calm_request(messages: list, grammar: str) -> tuple:
    """Send a chat completion request with CALM grammar and return full response."""
    payload = {
        "messages": messages,
        "grammar": grammar,
        "temperature": 0.0,
        "max_tokens": 512,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{SERVER}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())

    content = data["choices"][0]["message"]["content"]
    timings = data.get("timings", {})
    return content, timings


def process_and_print(content: str, ic: Interceptor) -> tuple:
    """Run content through interceptor, print events, return (mismatches, errors)."""
    events = ic.feed(content)

    for e in events:
        if e.type == EventType.CALM_START:
            print("  [START]")
        elif e.type == EventType.CALM_END:
            print("  [END]")
        elif e.type == EventType.EXECUTED:
            print(f"  EXEC  {e.instruction:20s} stack={e.actual_stack}")
        elif e.type == EventType.VALIDATED:
            print(f"  OK    {e.instruction:20s} claim={e.claimed_stack}")
        elif e.type == EventType.MISMATCH:
            print(f"  FAIL  {e.instruction:20s} claimed={e.claimed_stack} actual={e.actual_stack}")
        elif e.type == EventType.ERROR:
            print(f"  ERROR {e.text}")
        elif e.type == EventType.COMMENT:
            print(f"  //    {e.instruction}")

    mismatches = [e for e in events if e.type == EventType.MISMATCH]
    errors = [e for e in events if e.type == EventType.ERROR]
    executed = [e for e in events if e.type == EventType.EXECUTED]
    validated = [e for e in events if e.type == EventType.VALIDATED]

    print(f"  ({len(executed)} executed, {len(validated)} validated, "
          f"{len(mismatches)} mismatches, {len(errors)} errors)")

    return mismatches, errors


def run_live_test(prompt: str, max_retries: int = 2) -> bool:
    """Run a live CALM test with error-injection retries. Returns True if all claims validated."""
    grammar = generate_gbnf()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    print(f"Prompt: {prompt}")
    print(f"Grammar: {len(grammar)} chars")

    for attempt in range(1 + max_retries):
        print(f"\n--- Attempt {attempt + 1} ---")

        content, timings = send_calm_request(messages, grammar)

        print(f"Model:\n{content}")
        if timings:
            print(f"({timings.get('predicted_per_second', 0):.1f} tok/s)")

        print("\nEvents:")
        ic = Interceptor()
        mismatches, errors = process_and_print(content, ic)

        print(f"\nOutput: {ic.state.output}")
        print(f"Stack:  {ic.state.stack}")

        if not mismatches and not errors:
            print(f"\nRESULT: PASS (attempt {attempt + 1})")
            return True

        if attempt < max_retries:
            # Report only the FIRST mismatch or error — cascading errors
            # are noise that confuses the model.
            first_problem = (mismatches + errors)[0]
            correction = (
                f"{first_problem.text} "
                f"Please rewrite the entire CALM program with correct values."
            )
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": correction})
            print(f"\nInjecting: {correction[:120]}...")

    print(f"\nRESULT: FAIL after {1 + max_retries} attempts")
    return False


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "What is 17 * 23?"
    run_live_test(prompt)
