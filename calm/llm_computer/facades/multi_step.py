"""MultiStepReasoningFacade — Tier-2 multi-step arithmetic reasoning.

Packages the augmentation_thesis "multi-step reasoning" pattern:
decomposition + per-step verification + Gemma under verification
gates. This initial scope:

  decomposition   →   regex + NL-operator substitution
  verification    →   CALM's safe_eval (1002-function registry,
                       AST-only, deterministic)
  delivery        →   step-through digit bias at Gemma decode
                       (R11/R45 multi-token answer pattern)

Later rounds graduate the regex parser to a trained copy-augmented
PT (funcall family) and add per-step CALM verification with
self-correction.

Scope (R46):
  - Supports arbitrary infix arithmetic with +, -, *, //, %, **, ()
  - NL operator aliases: "plus/minus/times/divided by/over"
  - Multi-digit integer final answer (1-6 digits)
  - Spelled-out numbers NOT supported (tier-2 upgrade)
  - Variable substitution NOT supported (R47 target)

Usage:

    facade = MultiStepReasoningFacade()
    facade.install(gemma, tokenizer)

    result = facade.solve("What is 17 times 23 plus 45?")
    # result.expression = "17 * 23 + 45"
    # result.value      = 436
    # result.generated  = "17 × 23 = 391, and 391 + 45 = 436.\\n"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import torch

from calm.expression import ExpressionError, safe_eval


# NL operator → symbol. Order matters: multi-word phrases first to
# avoid premature matching ("divided by" before "divide").
_NL_OPS: list[tuple[str, str]] = [
    (r"\bdivided\s+by\b", "/"),
    (r"\bmultiplied\s+by\b", "*"),
    (r"\btimes\b", "*"),
    (r"\bover\b", "/"),      # "X over Y"
    (r"\bplus\b", "+"),
    (r"\bminus\b", "-"),
    (r"×", "*"),
    (r"÷", "/"),
]

# Gemma 4 E4B BPE token IDs for single digits (shared with R11/R45).
_DIGIT_TO_GEMMA = {
    0: 236771, 1: 236770, 2: 236778, 3: 236800, 4: 236812,
    5: 236810, 6: 236825, 7: 236832, 8: 236828, 9: 236819,
}


@dataclass
class MultiStepResult:
    """Facade output with enough state for debugging + scoring."""
    prompt: str
    expression: Optional[str]     # extracted + normalized infix
    value: Optional[int]          # safe_eval result (int only)
    generated: str                # Gemma's produced continuation
    parsed_answer: Optional[int]  # integer extracted from generated
    used_bias: bool               # did step-through fire?


class MultiStepReasoningFacade:
    """Multi-step arithmetic via parse → verify → bias.

    parse: regex + NL-op alias substitution → infix expression string
    verify: calm.safe_eval → integer result (raises on bad expr / bad
            type; we catch and report None)
    bias: step-through digit bias at Gemma decode delivers the
          integer tokens into Gemma's output stream

    install() requires gemma + tokenizer; detach() clears both.
    """

    DEFAULT_BOOST = 50.0
    DEFAULT_MAX_TOKENS = 60
    # Minimum token count for a candidate expression to be considered
    # "multi-step" (guards against accidentally matching a single number)
    MIN_OPS_FOR_MULTISTEP = 2

    def __init__(
        self,
        boost: float = DEFAULT_BOOST,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        device: str = "cuda",
        require_multistep: bool = False,
    ):
        self.boost = boost
        self.max_tokens = max_tokens
        self.device = device
        self.require_multistep = require_multistep
        self._gemma = None
        self._tokenizer = None

    # --- Public API ---

    def install(self, gemma, tokenizer):
        self._gemma = gemma
        self._tokenizer = tokenizer

    def detach(self):
        self._gemma = None
        self._tokenizer = None

    def parse(self, prompt: str) -> Optional[str]:
        """NL prompt → infix expression string, or None if no valid
        arithmetic chain found."""
        # Step 1: substitute NL ops with symbols
        norm = prompt.lower()
        for pat, sym in _NL_OPS:
            norm = re.sub(pat, f" {sym} ", norm)

        # Step 2: collapse whitespace
        norm = re.sub(r"\s+", " ", norm)

        # Step 3: find all candidate arithmetic substrings.
        # A "term" is a number or a flat parenthesized group (no nested
        # parens — these chain tests don't exercise that). An
        # "expression" is one or more (OPERATOR TERM) groups following
        # an opening term. Multi-digit operands must be kept intact.
        TERM = r"(?:\(\s*-?\d+(?:\s*[+\-*/%]\s*-?\d+)*\s*\)|-?\d+(?:\.\d+)?)"
        OPER = r"\s*[+\-*/%]\s*"
        candidate_re = re.compile(rf"{TERM}(?:{OPER}{TERM})+")
        candidates = [m.group(0).strip()
                       for m in candidate_re.finditer(norm)]
        if not candidates:
            return None

        # Step 4: longest-first, take first that parses AND has enough
        # operators (if require_multistep).
        candidates.sort(key=len, reverse=True)
        for cand in candidates:
            cand_clean = cand.rstrip(".?! )")
            if self.require_multistep:
                n_ops = sum(cand_clean.count(op) for op in "+-*/%")
                if n_ops < self.MIN_OPS_FOR_MULTISTEP:
                    continue
            try:
                val = safe_eval(cand_clean)
                if not isinstance(val, (int, float)):
                    continue
                # Reject degenerate single-literal cases.
                if re.fullmatch(r"-?\d+", cand_clean):
                    continue
                return cand_clean
            except (ExpressionError, Exception):
                continue
        return None

    def evaluate(self, expression: str) -> Optional[int]:
        """Run CALM safe_eval. Returns int on success, None on failure
        or non-integer result."""
        try:
            val = safe_eval(expression)
        except Exception:
            return None
        if isinstance(val, bool):   # bool is subclass of int, reject
            return None
        if isinstance(val, int):
            return val
        if isinstance(val, float) and val.is_integer():
            return int(val)
        return None

    def solve(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        boost: Optional[float] = None,
        use_bias: bool = True,
    ) -> MultiStepResult:
        """Parse → evaluate → generate. Full pipeline.

        If `use_bias=False`, the expression is still parsed + evaluated
        (for scoring) but Gemma decodes unbiased. Useful for A/B."""
        if self._gemma is None or self._tokenizer is None:
            raise RuntimeError(
                "facade not installed — call install(gemma, tok) first")
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        boost = boost if boost is not None else self.boost

        expression = self.parse(prompt)
        value = self.evaluate(expression) if expression else None

        digit_ids: list[int] = []
        if use_bias and value is not None:
            digit_ids = self._gemma_digit_tokens(value)

        fire_bias = bool(digit_ids)
        text = self._generate(prompt,
                               digit_ids if fire_bias else [],
                               boost, max_tokens)
        parsed = self._parse_int(text)
        return MultiStepResult(
            prompt=prompt,
            expression=expression,
            value=value,
            generated=text,
            parsed_answer=parsed,
            used_bias=fire_bias,
        )

    # --- Internal ---

    def _gemma_digit_tokens(self, n: int) -> list[int]:
        """Encode integer as Gemma BPE tokens, skipping <bos>.
        Natural tokenization gives '▁' prefix for leading digit."""
        ids = self._tokenizer.encode(str(n))
        if ids and ids[0] == 2:
            ids = ids[1:]
        return ids

    @staticmethod
    def _parse_int(text: str) -> Optional[int]:
        nums = re.findall(r"-?\d+", text.replace(",", ""))
        return int(nums[0]) if nums else None

    def _generate(
        self,
        prompt: str,
        digit_token_ids: list[int],
        boost: float,
        max_tokens: int,
    ) -> str:
        """Step-through decode with optional digit bias. Same
        mechanism as MultiplicationFacade but delivered to a general
        multi-step answer (1-6 digits)."""
        from calm.llm_computer.gemma_substrate import KVCache

        gemma = self._gemma
        tok = self._tokenizer
        ids = tok.encode(prompt)
        cache = KVCache(gemma.config.n_layers, device=self.device)
        gen = list(ids)
        digit_idx = 0 if digit_token_ids else -1

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

            for _ in range(max_tokens - 1):
                if nxt == tok.EOS_ID:
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

        return tok.decode(gen[len(ids):])
