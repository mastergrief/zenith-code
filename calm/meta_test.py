"""
CALM v0.1 meta-orchestration test — Gemma reasons freely and
dispatches to CALM compute modules mid-thought.

Non-strict mode: mismatches are training signal, not errors.
The VM is always authoritative. Triple modular redundancy verifies
every backend dispatch across 3 independent implementations.

Usage:
    python3 -m calm.meta_test
    python3 -m calm.meta_test "What is 17 * 23 + 42 * 19 - 100?"
"""

from __future__ import annotations

import json
import sys
import urllib.request

from calm.backends import math_ops, string_ops, wasm_ops
from calm.interceptor import EventType, Interceptor
from calm.verifier import make_verified_dispatcher

SERVER = "http://localhost:8080"

# Minimal prompt — rely on structure, not instructions.
SYSTEM_PROMPT = """\
You have a compute engine. Embed <calm>...</calm> blocks for exact computation.

Stack-based: push values, call operations, results stay on stack.
  push <val>  add sub mul div mod  neg abs  dup drop swap over rot
  eq lt gt  emit  halt

Backends (CPU-native, always exact):
  math.sqrt .pow .floor .ceil .log .pi .is_prime .gcd .factorize
  str.len .upper .lower .contains .concat .regex_match .replace
  wasm.add .sub .mul .div .mod .pow .gcd .sqrt .floor .ceil

One block = one continuous stack. Write -> [state] or -> <pending> after ops.

<calm>
push 17
push 23
mul -> <pending>
emit
</calm>"""


def make_dispatcher():
    """Verified dispatcher with triple redundancy."""
    return make_verified_dispatcher()


def send_request(messages: list, thinking_budget: int = 16384) -> tuple:
    """Send a chat completion with thinking budget."""
    payload = {
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 2048,
        "stream": False,
    }
    # Enable thinking if the server supports it.
    if thinking_budget > 0:
        payload["enable_thinking"] = True
        payload["thinking_budget"] = thinking_budget

    req = urllib.request.Request(
        f"{SERVER}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())

    choice = data["choices"][0]["message"]
    content = choice.get("content", "")
    thinking = choice.get("reasoning_content", "")
    timings = data.get("timings", {})
    return content, thinking, timings


def process_events(events: list) -> dict:
    """Categorize and print events, return summary dict."""
    calm_blocks = 0
    mismatches, errors, validated = [], [], []

    for e in events:
        if e.type == EventType.CALM_START:
            calm_blocks += 1
            print(f"  [{calm_blocks}] START")
        elif e.type == EventType.CALM_END:
            print(f"  [{calm_blocks}] END")
        elif e.type == EventType.EXECUTED:
            print(f"  [{calm_blocks}] EXEC  {e.instruction:25s} stack={e.actual_stack}")
        elif e.type == EventType.VALIDATED:
            validated.append(e)
            print(f"  [{calm_blocks}] OK    {e.instruction:25s}")
        elif e.type == EventType.MISMATCH:
            mismatches.append(e)
            print(f"  [{calm_blocks}] MISS  {e.instruction:25s} said={e.claimed_stack} actual={e.actual_stack}")
        elif e.type == EventType.RESOLVED:
            validated.append(e)
            print(f"  [{calm_blocks}] DONE  {e.instruction:25s} {e.text}")
        elif e.type == EventType.VERIFIED:
            print(f"  [{calm_blocks}] TMR   {e.instruction:25s} {e.text}")
        elif e.type == EventType.DIVERGENCE:
            errors.append(e)
            print(f"  [{calm_blocks}] !!!   {e.instruction:25s} DIVERGENCE")
        elif e.type == EventType.ERROR:
            errors.append(e)
            print(f"  [{calm_blocks}] ERR   {e.text}")
        elif e.type == EventType.COMMENT:
            pass  # silent

    return {
        "blocks": calm_blocks,
        "validated": len(validated),
        "mismatches": len(mismatches),
        "errors": len(errors),
        "divergences": sum(1 for e in errors if e.type == EventType.DIVERGENCE),
    }


def run_meta_test(prompt: str) -> bool:
    """
    Run a meta-orchestration test. Non-strict: mismatches are logged
    as training signal but don't block execution. Only divergences
    (TMR disagreement) are real failures.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    print(f"Prompt: {prompt}\n")
    content, thinking, timings = send_request(messages)
    tps = timings.get("predicted_per_second", 0)

    if thinking:
        print(f"--- Thinking ({len(thinking)} chars) ---")
        # Show first/last 200 chars if long.
        if len(thinking) > 500:
            print(thinking[:250])
            print(f"  ... ({len(thinking) - 500} chars omitted) ...")
            print(thinking[-250:])
        else:
            print(thinking)
        print()

    print(f"--- Response ({tps:.1f} tok/s) ---")
    print(content)
    print()

    # Process both thinking and response through interceptor.
    ic = Interceptor(dispatcher=make_dispatcher(), strict=False)

    print("--- Thinking trace ---")
    thinking_events = ic.feed(thinking) if thinking else []
    t_summary = process_events(thinking_events) if thinking_events else {"blocks": 0}

    print("\n--- Response trace ---")
    response_events = ic.feed(content)
    r_summary = process_events(response_events)

    # Combined summary.
    total_blocks = t_summary.get("blocks", 0) + r_summary["blocks"]
    total_validated = t_summary.get("validated", 0) + r_summary.get("validated", 0)
    total_mismatches = t_summary.get("mismatches", 0) + r_summary.get("mismatches", 0)
    total_errors = t_summary.get("errors", 0) + r_summary.get("errors", 0)
    total_divergences = t_summary.get("divergences", 0) + r_summary.get("divergences", 0)

    print(f"\n--- Summary ---")
    print(f"CALM blocks:  {total_blocks}")
    print(f"Validated:    {total_validated}")
    print(f"Mismatches:   {total_mismatches} (training signal)")
    print(f"Errors:       {total_errors}")
    print(f"Divergences:  {total_divergences}")
    print(f"VM output:    {ic.state.output}")
    print(f"Training log: {len(ic.training_log)} entries")

    if total_blocks == 0:
        print("\nRESULT: NO_DISPATCH")
        return False

    if total_divergences > 0:
        print("\nRESULT: DIVERGENCE (TMR failure)")
        return False

    # In non-strict mode, mismatches don't count as failures.
    accuracy = (
        total_validated / (total_validated + total_mismatches) * 100
        if (total_validated + total_mismatches) > 0 else 100
    )
    print(f"Prediction accuracy: {accuracy:.0f}%")
    print(f"\nRESULT: PASS")
    return True


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else (
        "What is 17 * 23 + 42 * 19 - 100?"
    )
    run_meta_test(prompt)
