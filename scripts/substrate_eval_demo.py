"""Round-18 eval — SubstrateComputer on 20 realistic arithmetic prompts.

Answers the honest question: does the unified substrate reliably dispatch
common arithmetic phrasings to the compiled card and produce correct
verified answers?

20 prompts: 15 in-scope (fit dispatched_v4's operand ranges), 5 out-of-scope
(operands too large, unsupported ops, or not arithmetic). Expected
behaviour:
  * in-scope  → substrate returns the correct integer/bool
  * out-of-scope → substrate returns None (caller routes to Gemma)

If `localhost:8080` exposes a llama-server, a raw-Gemma baseline is run
for the same prompts as a comparison. Otherwise Gemma is skipped.
"""

from __future__ import annotations

import json
import math
import re
import time
import urllib.error
import urllib.request
from typing import Any, Optional, Union

from calm.llm_computer.substrate_compute import SubstrateComputer


LLAMA_URL = "http://localhost:8080/v1/chat/completions"


# 15 in-scope + 5 out-of-scope. Ground truth is a Python value or None.
PROMPTS: list[tuple[str, Optional[Union[int, bool]]]] = [
    # Addition (5)
    ("3 + 5", 8),
    ("what is 7 plus 8?", 15),
    ("0 + 0", 0),
    ("15 + 15", 30),
    ("2 added to 9", 11),
    # Multiplication (3)
    ("3 * 5", 15),
    ("what is 10 times 5?", 50),
    ("0 * 7", 0),
    # GCD (2)
    ("gcd(12, 15)", 3),
    ("gcd of 4 and 7", 1),
    # Factorial (3)
    ("5!", 120),
    ("factorial of 4", 24),
    ("0!", 1),
    # Prime (2)
    ("is 7 prime?", True),
    ("is 9 prime?", False),
    # --- Out of scope (5) — substrate should return None ---
    ("17 * 23", None),                       # operands too large
    ("2^10", None),                          # exponentiation unsupported
    ("what's the capital of France?", None),  # not arithmetic
    ("100 + 200", None),                     # operands too large
    ("sqrt(16)", None),                      # sqrt unsupported
]


def gemma_available() -> bool:
    try:
        req = urllib.request.Request(
            "http://localhost:8080/health",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def gemma_query(prompt: str, max_tokens: int = 64) -> Optional[str]:
    """Hit llama-server with a short math prompt. Returns the model's
    free-form answer text."""
    body = {
        "model": "gemma",
        "messages": [
            {"role": "system", "content":
             "Answer with just the numeric result (or 'yes'/'no' for prime "
             "questions). No explanations."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        LLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read().decode("utf-8"))
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"ERROR: {e}"


def parse_gemma_answer(text: str,
                       expected: Optional[Union[int, bool]]) -> Any:
    """Extract a comparable value from Gemma's free-form reply."""
    if text is None or text.startswith("ERROR:"):
        return text
    if isinstance(expected, bool) or expected in (None,):
        low = text.lower()
        if "yes" in low or "true" in low or "is prime" in low:
            return True
        if "no" in low or "false" in low or "not prime" in low:
            return False
    # Extract first integer
    m = re.search(r"-?\d+", text)
    if m:
        return int(m.group(0))
    return text


def main() -> None:
    comp = SubstrateComputer()
    gemma_on = gemma_available()

    if gemma_on:
        print("[eval] Gemma (llama-server @ :8080) is reachable — running both")
    else:
        print("[eval] Gemma unreachable — running substrate only")

    print(f"\n{'#':>3}  {'prompt':<34} {'truth':>8}  {'sub':>8}",
          end="")
    if gemma_on:
        print(f"  {'gemma':>14}", end="")
    print()
    print("-" * 90 if gemma_on else "-" * 64)

    sub_in = sub_in_ok = 0
    sub_out = sub_out_ok = 0
    gemma_in = gemma_in_ok = 0

    for i, (prompt, truth) in enumerate(PROMPTS, 1):
        t0 = time.time()
        sub = comp.query(prompt)
        sub_t = time.time() - t0
        in_scope = truth is not None

        sub_ok = (sub == truth)
        if in_scope:
            sub_in += 1
            if sub_ok:
                sub_in_ok += 1
        else:
            sub_out += 1
            if sub_ok:
                sub_out_ok += 1

        sub_mark = "✓" if sub_ok else "✗"

        line = (f"{i:>3}  {prompt:<34} {str(truth):>8}  "
                f"[{sub_mark}] {str(sub):>5}")

        if gemma_on:
            raw = gemma_query(prompt)
            gans = parse_gemma_answer(raw, truth)
            g_ok = (gans == truth) if in_scope else None
            g_mark = "✓" if g_ok else ("—" if g_ok is None else "✗")
            line += f"  [{g_mark}] {str(gans):>10}"
            if in_scope:
                gemma_in += 1
                if g_ok:
                    gemma_in_ok += 1

        print(line)

    print(f"\n[eval] substrate in-scope:  {sub_in_ok}/{sub_in} "
          f"({100 * sub_in_ok / max(sub_in, 1):.0f}%)")
    print(f"[eval] substrate out-of-scope (correctly None): "
          f"{sub_out_ok}/{sub_out} "
          f"({100 * sub_out_ok / max(sub_out, 1):.0f}%)")
    if gemma_on:
        print(f"[eval] gemma in-scope:      {gemma_in_ok}/{gemma_in} "
              f"({100 * gemma_in_ok / max(gemma_in, 1):.0f}%)")

    # Interpretation line
    all_in_scope_ok = sub_in_ok == sub_in
    all_out_scope_ok = sub_out_ok == sub_out
    status = "PASS" if (all_in_scope_ok and all_out_scope_ok) else "FAIL"
    print(f"\n[eval] substrate bridge: {status}")
    print("[eval] thesis — compiled cards + tokenizer bridge deliver "
          "verified answers on in-scope prompts AND refuse gracefully "
          "on out-of-scope:")
    print(f"[eval]   {'VALIDATED' if status == 'PASS' else 'NOT VALIDATED'}")


if __name__ == "__main__":
    main()
