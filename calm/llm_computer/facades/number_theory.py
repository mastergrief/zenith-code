"""NumberTheoryFacade — R46.2-style tier-2 card for modular arithmetic,
GCD, and LCM.

Extends the decode-path compute-facade pattern proven by R46.2
(MultiStepReasoningFacade) and R22c (BaseConversionFacade) to a third
domain: number-theory operations Gemma gets wrong on non-trivial
operands.

Target failures (hypothesis — verified in probes):
  - `127 mod 13 = 10` style (Gemma often wrong at multi-digit modulo)
  - `GCD(48, 180) = 12` and harder cases like `GCD(391, 238) = 17`
  - `LCM(12, 18) = 36` and harder cases like `LCM(48, 180) = 720`

Exact compute via safe_eval (`gcd(a, b)`, `lcm(a, b)`, `a % b`).
Biases Gemma's natural decode to emit the correct decimal digit
sequence via step-through bias (R11/R46.2/R22c pattern).

Supported NL forms:
  - Modulo: "X mod Y", "X modulo Y", "X % Y", "remainder of X / Y",
            "what is X mod Y?"
  - GCD: "GCD of A and B", "greatest common divisor of A and B",
         "greatest common factor of A and B", "gcd(A, B)"
  - LCM: "LCM of A and B", "least common multiple of A and B",
         "lcm(A, B)"

Deliberately sidesteps the R22 retrieval-card install complexity: zero
VRAM, zero training, zero channel budget. Pure decode-time intervention.

Usage:
    facade = NumberTheoryFacade()
    facade.install(gemma, tokenizer)
    r = facade.solve("What is the GCD of 48 and 180?")
    # r.op='gcd', r.operands=(48, 180), r.value=12
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import torch

from calm.expression import ExpressionError, safe_eval


# Gemma 4 E4B BPE token IDs for single digits (shared with R11/R46.2/R22c).
_DIGIT_TO_GEMMA = {
    0: 236771, 1: 236770, 2: 236778, 3: 236800, 4: 236812,
    5: 236810, 6: 236825, 7: 236832, 8: 236828, 9: 236819,
}


@dataclass
class NumberTheoryResult:
    prompt: str
    op: Optional[str]               # 'mod', 'gcd', 'lcm'
    operands: Optional[tuple]       # (a, b)
    value: Optional[int]            # computed result
    generated: str
    parsed_answer: Optional[int]
    used_bias: bool


class NumberTheoryFacade:
    """Parse → safe_eval → step-through-digit-bias for modular
    arithmetic, GCD, LCM.

    Skeleton lifted from `BaseConversionFacade` (verified template
    via `compute_facades.md`). Only `parse`, `evaluate`, and the NL
    patterns differ.
    """

    DEFAULT_BOOST = 50.0
    DEFAULT_MAX_TOKENS = 40

    # Modulo: "127 mod 13", "127 % 13", "127 modulo 13",
    #         "remainder of 127 divided by 13", "remainder when 127
    #         is divided by 13"
    _MOD_RES = [
        re.compile(r"(-?\d+)\s*mod(?:ulo)?\s+(-?\d+)", re.IGNORECASE),
        re.compile(r"(-?\d+)\s*%\s*(-?\d+)"),
        # "remainder of X divided by Y" / "remainder when X is divided by Y"
        # / "remainder of X / Y"
        re.compile(
            r"remainder\s+(?:of|when)\s+(-?\d+)\s+(?:is\s+)?(?:divided\s+by|/)\s+(-?\d+)",
            re.IGNORECASE,
        ),
    ]
    # GCD: "GCD of 48 and 180", "gcd(48, 180)", "greatest common
    #      divisor of 48 and 180", "greatest common factor of ..."
    _GCD_RES = [
        re.compile(
            r"\b(?:gcd|greatest\s+common\s+(?:divisor|factor))\s*"
            r"(?:of\s+|\(\s*)?(-?\d+)\s*(?:,|and)\s*(-?\d+)\s*\)?",
            re.IGNORECASE,
        ),
    ]
    # LCM: "LCM of 12 and 18", "lcm(12, 18)", "least common multiple
    #      of 12 and 18"
    _LCM_RES = [
        re.compile(
            r"\b(?:lcm|least\s+common\s+multiple)\s*"
            r"(?:of\s+|\(\s*)?(-?\d+)\s*(?:,|and)\s*(-?\d+)\s*\)?",
            re.IGNORECASE,
        ),
    ]

    def __init__(
        self,
        boost: float = DEFAULT_BOOST,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        device: str = "cuda",
    ):
        self.boost = boost
        self.max_tokens = max_tokens
        self.device = device
        self._gemma = None
        self._tokenizer = None

    def install(self, gemma, tokenizer):
        self._gemma = gemma
        self._tokenizer = tokenizer

    def detach(self):
        self._gemma = None
        self._tokenizer = None

    def parse(self, prompt: str) -> tuple[Optional[str], Optional[tuple]]:
        """Returns (op, (a, b)) or (None, None) if no match. Precedence:
        mod > gcd > lcm (first-match)."""
        for pat in self._MOD_RES:
            m = pat.search(prompt)
            if m:
                try:
                    return "mod", (int(m.group(1)), int(m.group(2)))
                except ValueError:
                    pass
        for pat in self._GCD_RES:
            m = pat.search(prompt)
            if m:
                try:
                    return "gcd", (int(m.group(1)), int(m.group(2)))
                except ValueError:
                    pass
        for pat in self._LCM_RES:
            m = pat.search(prompt)
            if m:
                try:
                    return "lcm", (int(m.group(1)), int(m.group(2)))
                except ValueError:
                    pass
        return None, None

    def evaluate(self, op: str, operands: tuple) -> Optional[int]:
        try:
            a, b = operands
            if op == "mod":
                if b == 0:
                    return None
                val = safe_eval(f"{a} % {b}")
            elif op == "gcd":
                val = safe_eval(f"gcd({a}, {b})")
            elif op == "lcm":
                val = safe_eval(f"lcm({a}, {b})")
            else:
                return None
            if isinstance(val, bool):
                return None
            if isinstance(val, int):
                return val
            if isinstance(val, float) and val.is_integer():
                return int(val)
        except (ExpressionError, Exception):
            return None
        return None

    def solve(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        boost: Optional[float] = None,
        use_bias: bool = True,
    ) -> NumberTheoryResult:
        if self._gemma is None or self._tokenizer is None:
            raise RuntimeError("facade not installed — call install() first")
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        boost = boost if boost is not None else self.boost

        op, operands = self.parse(prompt)
        value = self.evaluate(op, operands) if (op and operands) else None

        digit_ids: list[int] = []
        if use_bias and value is not None:
            digit_ids = self._gemma_digit_tokens(value)

        fire_bias = bool(digit_ids)
        text = self._generate(prompt, digit_ids if fire_bias else [],
                              boost, max_tokens)
        parsed = self._parse_int(text)
        return NumberTheoryResult(
            prompt=prompt, op=op, operands=operands, value=value,
            generated=text, parsed_answer=parsed, used_bias=fire_bias,
        )

    # Gemma BPE id for the SentencePiece "▁" space-prefix token.
    _SPACE_TOKEN_ID = 236743

    def _gemma_digit_tokens(self, n: int) -> list[int]:
        """Encode integer as Gemma BPE, skipping BOS AND the leading
        `▁` (space-prefix) token if present. Rationale: `_generate`
        appends 'Answer: ' to the prompt so the prefill already ends
        with a space — biasing another space at step 0 wastes the
        boost slot and lets Gemma's strong natural `0` token win
        (logit ~57–66 on math prompts, +50 boost on `▁` can't flip)."""
        ids = self._tokenizer.encode(str(n))
        if ids and ids[0] == 2:  # BOS
            ids = ids[1:]
        if ids and ids[0] == self._SPACE_TOKEN_ID:
            ids = ids[1:]
        return ids

    @staticmethod
    def _parse_int(text: str) -> Optional[int]:
        """Extract the first integer, but truncate at first digit-run
        boundary to prevent Gemma's post-bias `0` loop from extending
        the match (e.g. 'facade emits "10", Gemma continues "0000..."'
        → naive regex returns 10**38 instead of 10).

        We also reject unreasonably long matches (> 12 digits) which
        typically indicate the post-bias loop has kicked in; fall back
        to the first 1-6 digit substring if present.
        """
        normalized = text.replace(",", "")
        # First digit run — limit 6 digits to catch the answer cleanly
        m = re.search(r"-?\d{1,12}", normalized)
        if not m:
            return None
        return int(m.group(0))

    def _generate(
        self,
        prompt: str,
        digit_token_ids: list[int],
        boost: float,
        max_tokens: int,
    ) -> str:
        """Step-through decode with optional digit bias. Same template
        as BaseConversionFacade._generate (R22c) and
        MultiStepReasoningFacade._generate (R46.2).
        """
        from calm.llm_computer.gemma_substrate import KVCache

        gemma = self._gemma
        tok = self._tokenizer
        if not prompt.rstrip().lower().endswith(("answer:", "= ")):
            prompt = prompt.rstrip() + " Answer: "
        ids = tok.encode(prompt)
        cache = KVCache(gemma.config.n_layers, device=self.device)
        gen = list(ids)
        digit_idx = 0 if digit_token_ids else -1

        # After the bias is exhausted, Gemma frequently continues emitting
        # digit tokens (scientific-notation-like "0" runs, ~40 more "0"s)
        # on math prompts — the first "Answer: 0" prior from this model
        # is very sticky. Cap the tail with a small `post_bias_budget` of
        # natural tokens and break on the first non-digit. That way we
        # capture the answer and no contamination.
        POST_BIAS_BUDGET = 4

        with torch.no_grad():
            logits = gemma.forward(
                torch.tensor([gen]), device=self.device,
                kv_cache=cache, start_pos=0,
            )
            if 0 <= digit_idx < len(digit_token_ids):
                logits[0, -1, digit_token_ids[digit_idx]] += boost
                digit_idx += 1
            nxt = int(logits[0, -1].argmax())
            gen.append(nxt)

            post_bias_steps = 0
            for _ in range(max_tokens - 1):
                if hasattr(tok, "EOS_ID") and nxt == tok.EOS_ID:
                    break
                # Post-bias truncation — stop at first non-digit or after
                # the small natural budget
                if digit_token_ids and digit_idx >= len(digit_token_ids):
                    post_bias_steps += 1
                    if post_bias_steps > POST_BIAS_BUDGET:
                        break
                logits = gemma.forward(
                    torch.tensor([[nxt]]), device=self.device,
                    kv_cache=cache, start_pos=len(gen) - 1,
                )
                if 0 <= digit_idx < len(digit_token_ids):
                    logits[0, -1, digit_token_ids[digit_idx]] += boost
                    digit_idx += 1
                nxt = int(logits[0, -1].argmax())
                gen.append(nxt)

        return tok.decode(gen[len(ids):]) if hasattr(tok, "decode") else ""
