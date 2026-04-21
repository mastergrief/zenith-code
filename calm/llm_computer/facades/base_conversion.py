"""BaseConversionFacade — R46.2-style tier-2 card for hex/binary → decimal.

R46.2's MultiStepReasoningFacade handles +/-/*/% infix chains. This
facade applies the same parse → safe_eval → step-through-digit-bias
pattern to a NEW domain: base conversion.

Target failure: Gemma guesses wrong on non-trivial hex/binary to
decimal conversions (e.g. 0xDEADBEEF → 3735928559). Exact compute via
`int(X, 16)` / `int(X, 2)`. Biases Gemma's natural decode to emit the
correct digit sequence.

Supported formats:
  - "0x<HEX>" — Python hex literal
  - "0b<BIN>" — Python binary literal
  - "hex <HEX>" or "binary <BIN>" — NL form
  - Query pattern: "... in decimal" or "... as a decimal" or
                    "convert <X> to decimal"

Usage:
    facade = BaseConversionFacade()
    facade.install(gemma, tokenizer)
    r = facade.solve("What is 0xDEADBEEF in decimal?")
    # r.value = 3735928559
    # r.generated = "... 3735928559 ..."
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import torch


# Gemma 4 E4B BPE token IDs for single digits (shared with R11/R45/R46).
_DIGIT_TO_GEMMA = {
    0: 236771, 1: 236770, 2: 236778, 3: 236800, 4: 236812,
    5: 236810, 6: 236825, 7: 236832, 8: 236828, 9: 236819,
}


@dataclass
class BaseConversionResult:
    prompt: str
    source_value: Optional[str]   # the hex/binary literal extracted
    source_base: Optional[int]    # 16 or 2
    value: Optional[int]          # converted decimal
    generated: str
    parsed_answer: Optional[int]
    used_bias: bool


class BaseConversionFacade:
    """Parse NL hex/binary expression, compute decimal exactly, step-
    through bias Gemma output to emit the decimal digit sequence.
    """

    DEFAULT_BOOST = 50.0
    DEFAULT_MAX_TOKENS = 40

    # Ordered patterns — first match wins.
    _PATTERNS = [
        # 0xABCD, 0xDEADBEEF
        (re.compile(r"\b0x([0-9a-fA-F]+)\b"), 16),
        # 0b101011
        (re.compile(r"\b0b([01]+)\b"), 2),
        # "hex ABCD"
        (re.compile(r"\bhex\s+([0-9a-fA-F]+)\b", re.IGNORECASE), 16),
        # "binary 101011"
        (re.compile(r"\bbinary\s+([01]+)\b", re.IGNORECASE), 2),
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

    def parse(self, prompt: str) -> tuple[Optional[str], Optional[int]]:
        """Returns (literal_string, base) or (None, None) if no match."""
        # Require an "in decimal" / "as decimal" / "to decimal" signal
        # to avoid biasing random mentions of hex strings.
        if not re.search(r"\b(in|as|to)\s+decimal\b", prompt, re.IGNORECASE):
            return None, None
        for pat, base in self._PATTERNS:
            m = pat.search(prompt)
            if m:
                return m.group(1), base
        return None, None

    def evaluate(self, literal: str, base: int) -> Optional[int]:
        try:
            return int(literal, base)
        except (ValueError, TypeError):
            return None

    def solve(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        boost: Optional[float] = None,
        use_bias: bool = True,
    ) -> BaseConversionResult:
        if self._gemma is None or self._tokenizer is None:
            raise RuntimeError("facade not installed — call install() first")
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        boost = boost if boost is not None else self.boost

        literal, base = self.parse(prompt)
        value = self.evaluate(literal, base) if (literal and base) else None

        digit_ids: list[int] = []
        if use_bias and value is not None:
            digit_ids = self._gemma_digit_tokens(value)

        fire_bias = bool(digit_ids)
        text = self._generate(prompt, digit_ids if fire_bias else [],
                              boost, max_tokens)
        parsed = self._parse_int(text)
        return BaseConversionResult(
            prompt=prompt, source_value=literal, source_base=base,
            value=value, generated=text, parsed_answer=parsed,
            used_bias=fire_bias,
        )

    def _gemma_digit_tokens(self, n: int) -> list[int]:
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
        """Step-through decode with optional digit bias. Mirrors
        MultiStepReasoningFacade._generate (R46.2 pattern).

        Appends 'Answer: ' to the prompt if not already present so the
        first decode token is the answer's leading digit — lets the
        bias fire from step 0 without a marker-wait loop.
        """
        from calm.llm_computer.gemma_substrate import KVCache

        gemma = self._gemma
        tok = self._tokenizer
        if not prompt.rstrip().lower().endswith(("answer:", "decimal:", "= ")):
            prompt = prompt.rstrip() + " Answer: "
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
                if hasattr(tok, "EOS_ID") and nxt == tok.EOS_ID:
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
