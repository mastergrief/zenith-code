"""SubstrateComputer — text → answer via compiled dispatched_v4 card.

The minimum viable inference path for the unified substrate: parse common
arithmetic prompts with regex, dispatch to the compiled card's opcodes,
decode the argmax slot, return the answer.

Scope (matches dispatched_v4 exactly):
  * ADD / MUL:   a, b ∈ [0, 15]
  * GCD:         a, b ∈ [0, 15]
  * FACTORIAL:   n ∈ [0, 8]
  * IS_PRIME:    n ∈ [2, 15]

Out-of-scope queries (operands too large, operator unsupported) return
`None`. The caller can route those to Gemma or another backend — this
is the Brain+Cards pattern: card answers what it knows, brain handles
the rest.

Usage:
    >>> comp = SubstrateComputer()
    >>> comp.query("17 * 23")           # out of range
    None
    >>> comp.query("3 * 5")             # in range
    15
    >>> comp.query("gcd(12, 18)")
    6
    >>> comp.query("5!")
    120
    >>> comp.query("is 7 prime?")
    True
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from calm.llm_computer.programs.dispatched_v4 import (
    FACT_MAX_N, GCD_BASE, MUL_MAX_OPERAND, OPCODE_SHIFT, PRIME_MAX_N,
    PRIME_MIN_N, build_dispatched_v4, decode_output,
)


# Matches dispatched_v4 opcodes (0-indexed user-facing)
OP_GCD = 0
OP_FACT = 1
OP_PRIME = 2
OP_ADD = 3
OP_MUL = 4


# Regex patterns for common arithmetic phrasings. Tried in order —
# first match wins.
#
# Each pattern yields (opcode, a, b_or_None). b=None for unary ops.
_PATTERNS: list[Tuple[re.Pattern, int, str]] = [
    # Binary ops with explicit operator: "3 + 5", "12 * 4", "10 / 2", etc.
    (re.compile(r"^\s*(\d+)\s*\+\s*(\d+)\s*\??\s*$"), OP_ADD, "binary"),
    (re.compile(r"^\s*(\d+)\s*(?:\*|×|x|X)\s*(\d+)\s*\??\s*$"),
     OP_MUL, "binary"),

    # NL arithmetic: "3 plus 5", "what is 3 plus 5?"
    (re.compile(r"(?:what(?:'s|\s+is)\s+)?(\d+)\s+(?:plus|added\s+to)\s+(\d+)\s*\??",
                re.I), OP_ADD, "binary"),
    (re.compile(r"(?:what(?:'s|\s+is)\s+)?(\d+)\s+(?:times|multiplied\s+by)\s+(\d+)\s*\??",
                re.I), OP_MUL, "binary"),

    # GCD phrasings
    (re.compile(r"gcd\s*\(?\s*(\d+)\s*[,\s]\s*(\d+)\s*\)?", re.I),
     OP_GCD, "binary"),
    (re.compile(r"gcd\s+of\s+(\d+)\s+and\s+(\d+)", re.I),
     OP_GCD, "binary"),

    # Factorial
    (re.compile(r"^\s*(\d+)\s*!\s*$"), OP_FACT, "unary"),
    (re.compile(r"factorial\s*\(?\s*(\d+)\s*\)?", re.I), OP_FACT, "unary"),
    (re.compile(r"factorial\s+of\s+(\d+)", re.I), OP_FACT, "unary"),

    # Prime tests
    (re.compile(r"is\s+(\d+)\s+(?:a\s+)?prime\s*\??", re.I),
     OP_PRIME, "unary"),
    (re.compile(r"is_?prime\s*\(?\s*(\d+)\s*\)?", re.I), OP_PRIME, "unary"),
]


def parse_prompt(text: str) -> Optional[Tuple[int, int, int]]:
    """Try to extract (opcode, a, b) from free-form text.

    For unary ops (FACT, PRIME), b=0. Returns None if no pattern matches
    OR if operands fall outside the card's supported range.
    """
    for pattern, opcode, arity in _PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        try:
            a = int(m.group(1))
            b = int(m.group(2)) if arity == "binary" else 0
        except (ValueError, IndexError):
            continue

        # Range checks per op — card's vocab has hard limits.
        if opcode in (OP_GCD,):
            if not (0 <= a <= GCD_BASE - 1 and 0 <= b <= GCD_BASE - 1):
                return None
        elif opcode in (OP_ADD,):
            if not (0 <= a <= GCD_BASE - 1 and 0 <= b <= GCD_BASE - 1):
                return None
        elif opcode == OP_MUL:
            if not (0 <= a <= MUL_MAX_OPERAND and 0 <= b <= MUL_MAX_OPERAND):
                return None
        elif opcode == OP_FACT:
            if not (0 <= a <= FACT_MAX_N):
                return None
        elif opcode == OP_PRIME:
            if not (PRIME_MIN_N <= a <= PRIME_MAX_N):
                return None
        return opcode, a, b
    return None


class SubstrateComputer:
    """Wrapper around dispatched_v4: text in, answer out.

    Lazy-builds the card on first query. The 7M-param card is lightweight
    (~0.1s build) but we only build once.
    """

    def __init__(self, device: Union[str, torch.device] = "cpu"):
        self.device = torch.device(device)
        self._card = None

    def _ensure_card(self):
        if self._card is None:
            self._card = build_dispatched_v4()
            self._card.eval()
            self._card = self._card.to(self.device)

    def query(self, text: str) -> Optional[Union[int, bool]]:
        """Parse text, dispatch to card, return answer. None if out-of-scope."""
        parsed = parse_prompt(text)
        if parsed is None:
            return None
        opcode, a, b = parsed

        self._ensure_card()
        # dispatched_v4 expects pos-2 token = opcode + OPCODE_SHIFT
        x = torch.tensor([[a, b, opcode + OPCODE_SHIFT]],
                         dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self._card(x)[0, 2]
        slot = int(logits.argmax().item())
        return decode_output(opcode, slot)


if __name__ == "__main__":
    # Quick smoke test
    comp = SubstrateComputer()
    tests = [
        ("3 + 5", 8),
        ("7 * 9", 63),
        ("gcd(12, 15)", 3),
        ("5!", 120),
        ("is 7 prime?", True),
        ("is 9 prime?", False),
        ("what is 10 times 5?", 50),
        ("factorial of 6", 720),
        ("17 * 23", None),  # out of range
        ("hello world", None),  # not arithmetic
    ]
    for prompt, expected in tests:
        got = comp.query(prompt)
        mark = "✓" if got == expected else "✗"
        print(f"  [{mark}] {prompt!r:40} → {got!r:10} (expected {expected!r})")
