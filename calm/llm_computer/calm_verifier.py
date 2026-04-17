"""CALM-backed oracle for the substrate's learning loop.

Replaces the trivial hand-rolled Python verifier used in
`scripts/gemma_learning_loop_demo.py` (which only knew arithmetic:
`expected = (a + b) % 8`) with CALM's 1002-function registry via
`safe_eval`. A domain that has a CALM backend automatically has a
verifier; domains CALM can't evaluate fall through gracefully.

The verifier has two responsibilities:

  1. Translate an NL prompt to a CALM-evaluable expression (NL→expr)
  2. Evaluate the expression via safe_eval and return an integer
     answer in a bounded range (so it can be used as a KnowledgeStore
     value)

NL→expr is pattern-based for the MVP — small set of regex rules.
Eventually replaced by a Pointer Transducer (same thesis as the
CRLM split: PT learns the NL→structure translation, interpreter
handles evaluation). The key invariant is the contract: both today's
regex version and a future PT version return (expression_str | None).

The verifier composes with KnowledgeStore by being the source of
`correct_value` in `add_correction(key, correct_value)`. The key is
deterministic from the prompt (hash modulo max_key) so the same prompt
always maps to the same recall slot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from calm.expression import ExpressionError, safe_eval


# Regex rules: (NL pattern, CALM expression template). Order matters —
# first match wins. Each template uses \1, \2 backreferences.
NL_TO_EXPR_RULES: list[tuple[str, str]] = [
    # Arithmetic
    (r"(-?\d+)\s+plus\s+(-?\d+)", r"\1 + \2"),
    (r"(-?\d+)\s+minus\s+(-?\d+)", r"\1 - \2"),
    (r"(-?\d+)\s+times\s+(-?\d+)", r"\1 * \2"),
    (r"(-?\d+)\s+divided\s+by\s+(-?\d+)", r"\1 // \2"),
    (r"(-?\d+)\s+mod\s+(-?\d+)", r"\1 % \2"),
    (r"(-?\d+)\s*\+\s*(-?\d+)", r"\1 + \2"),
    (r"(-?\d+)\s*\*\s*(-?\d+)", r"\1 * \2"),
    # Number theory (single-arg)
    (r"is\s+(-?\d+)\s+prime", r"is_prime(\1)"),
    (r"factorial\s+of\s+(-?\d+)", r"factorial(\1)"),
    # Two-arg functions
    (r"gcd\s+of\s+(-?\d+)\s+and\s+(-?\d+)", r"gcd(\1, \2)"),
    (r"lcm\s+of\s+(-?\d+)\s+and\s+(-?\d+)", r"lcm(\1, \2)"),
]


def nl_to_expression(prompt: str) -> Optional[str]:
    """Translate an NL prompt into a CALM-evaluable expression. First
    matching rule wins. Returns None if no rule applies."""
    lowered = prompt.lower()
    for pattern, template in NL_TO_EXPR_RULES:
        if re.search(pattern, lowered):
            # Build the expression from the first match — drop the rest
            # of the prompt ("equals", question mark, etc.).
            m = re.search(pattern, lowered)
            return re.sub(pattern, template, m.group(0))
    return None


@dataclass
class CalmVerifier:
    """CALM-backed verifier. Calls safe_eval under the hood.

    max_value: expected-answer bound. Out-of-range answers return None
               (KnowledgeStore can't store them as compact keys anyway).
    """
    max_value: int = 64

    def verify(self, expression: str) -> Optional[int]:
        """Evaluate `expression` via CALM. Return an int in
        [0, max_value) or None if evaluation fails or the value is
        outside the storable range."""
        try:
            val = safe_eval(expression)
        except (ExpressionError, Exception):
            return None
        if isinstance(val, bool):
            val = int(val)
        if isinstance(val, int) and 0 <= val < self.max_value:
            return val
        return None

    def verify_nl(self, prompt: str) -> tuple[Optional[str], Optional[int]]:
        """End-to-end NL verification: translate to CALM expression,
        evaluate, return (expr_or_none, value_or_none)."""
        expr = nl_to_expression(prompt)
        if expr is None:
            return None, None
        return expr, self.verify(expr)


def make_key(prompt: str, max_key: int = 1024) -> int:
    """Deterministic key for KnowledgeStore lookup. Hash-based so
    repeated prompts map to the same recall slot."""
    # Python's hash() is per-process-salted; use a stable hash.
    import hashlib
    h = hashlib.sha1(prompt.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % max_key
