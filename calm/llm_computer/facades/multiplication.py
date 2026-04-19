"""MultiplicationFacade — 2-digit × 2-digit multiplication on prod Gemma.

Packages Round 11's one-off `scripts/test_multiplier_facade.py` into a
reusable facade. Mirrors `MathAdditionFacade`'s API where possible, but
the install-mode differs: multi-digit answers (e.g. 17×23=391 = 4 Gemma
BPE tokens) can't use `VerificationHook` (single-token bias) or
token-embedding projection at position -1 (also single-token). The
step-through digit-bias decode is an autoregressive-loop mechanism,
not a forward-pass hook.

Usage:

    facade = MultiplicationFacade()
    facade.install(gemma_substrate, tokenizer)

    text = facade.generate(
        "what is 17 times 23? Answer with just the number.")
    # → "391" (Gemma's natural '401' or similar is overridden by
    #         verified digit chain from multiplier's 3390/3390 card)

    facade.detach()

Input constraint: multiplier.py is exhaustive on a·b < 1000
(MAX_PRODUCT=999 to fit on 8GB VRAM alongside Gemma). Operands outside
that range are not guaranteed. Wider ranges require digit decomposition
(tier-2 roadmap item).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import torch

from calm.llm_computer.programs.multiplier import build_multiplier


# Gemma 4 E4B BPE token IDs for single digits 0..9.
# Same mapping as MathAdditionFacade.
_DIGIT_TO_GEMMA = {
    0: 236771, 1: 236770, 2: 236778, 3: 236800, 4: 236812,
    5: 236810, 6: 236825, 7: 236832, 8: 236828, 9: 236819,
}

# Regex finds two positive integers for "a × b" patterns in English
# text. Matches "17 times 23", "17×23", "17 * 23", etc.
_OPERAND_RE = re.compile(
    r"(\d+)\s*(?:times|×|x|\*)\s*(\d+)",
    re.IGNORECASE,
)


@dataclass
class MultiplicationResult:
    """Result of a facade generate() call. Exposes internals for
    debugging and evaluation."""
    prompt: str
    operands: Optional[tuple[int, int]]   # None if parse failed
    verified_answer: Optional[int]        # None if no verify
    generated_text: str
    parsed_answer: Optional[int]          # answer extracted from text
    used_bias: bool                       # True if facade biased the decode


class MultiplicationFacade:
    """Step-through digit-bias multiplication for prod Gemma 4 E4B.

    One instance can be installed on multiple Gemmas (no per-gemma
    state kept except in generate()). Thread-safe as long as no two
    threads call generate() concurrently on the same instance.
    """

    DEFAULT_BOOST = 50.0
    DEFAULT_MAX_TOKENS = 60
    MAX_PRODUCT = 999  # multiplier's verified range

    def __init__(
        self,
        boost: float = DEFAULT_BOOST,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        device: str = "cuda",
    ):
        self.boost = boost
        self.max_tokens = max_tokens
        self.device = device
        self.multiplier = build_multiplier().to(device).eval()
        self._gemma = None
        self._tokenizer = None

    # --- Public API ---

    def install(self, gemma, tokenizer):
        """Attach facade to a Gemma substrate + tokenizer.

        Doesn't modify Gemma weights. The facade operates only during
        generate() via step-through biased decoding."""
        self._gemma = gemma
        self._tokenizer = tokenizer

    def detach(self):
        """Detach from Gemma. generate() will fail until install()."""
        self._gemma = None
        self._tokenizer = None

    def verify(self, a: int, b: int) -> Optional[int]:
        """Run compiled multiplier on (a, b). Returns verified product,
        or None if inputs are outside the verified range."""
        if a * b >= self.MAX_PRODUCT + 1 or a < 0 or b < 0:
            return None
        x = torch.tensor([[a, b]], device=self.device, dtype=torch.long)
        with torch.no_grad():
            return int(self.multiplier(x)[0, 1].argmax().item())

    def parse_operands(self, prompt: str) -> Optional[tuple[int, int]]:
        """Extract (a, b) from a multiplication prompt. Returns None if
        no recognizable a×b pattern."""
        m = _OPERAND_RE.search(prompt)
        if m is None:
            return None
        try:
            a, b = int(m.group(1)), int(m.group(2))
        except ValueError:
            return None
        return (a, b)

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        boost: Optional[float] = None,
    ) -> MultiplicationResult:
        """Generate a continuation with step-through digit bias.

        If prompt contains a recognizable a×b pattern AND a·b < 1000,
        runs multiplier, encodes answer as digit tokens, biases each
        decode step toward the expected digit. Otherwise runs plain
        greedy decode.

        Returns MultiplicationResult with full state for inspection.
        """
        if self._gemma is None or self._tokenizer is None:
            raise RuntimeError(
                "facade not installed — call install(gemma, tok) first")

        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        boost = boost if boost is not None else self.boost

        operands = self.parse_operands(prompt)
        verified = None
        digit_ids: list[int] = []
        if operands is not None:
            a, b = operands
            verified = self.verify(a, b)
            if verified is not None:
                digit_ids = self._gemma_digit_tokens(verified)

        use_bias = bool(digit_ids)
        text = self._generate(prompt, digit_ids if use_bias else [],
                               boost, max_tokens)
        parsed = self._parse_int(text)
        return MultiplicationResult(
            prompt=prompt,
            operands=operands,
            verified_answer=verified,
            generated_text=text,
            parsed_answer=parsed,
            used_bias=use_bias,
        )

    # --- Internal ---

    def _gemma_digit_tokens(self, n: int) -> list[int]:
        """Encode n as the sequence of Gemma tokens (skipping <bos>).
        Mirrors test_multiplier_facade.py's digits_as_gemma_tokens —
        gives us the '▁' prefix Gemma naturally emits before numbers."""
        s = str(n)
        ids = self._tokenizer.encode(s)
        if ids and ids[0] == 2:  # <bos>
            ids = ids[1:]
        return ids

    @staticmethod
    def _parse_int(text: str) -> Optional[int]:
        """Extract first integer from generated text (commas stripped)."""
        nums = re.findall(r"-?\d+", text.replace(",", ""))
        return int(nums[0]) if nums else None

    def _generate(
        self,
        prompt: str,
        digit_token_ids: list[int],
        boost: float,
        max_tokens: int,
    ) -> str:
        """Step-through decode. If digit_token_ids is empty, pure
        greedy decode. Otherwise bias next-token logit for each
        expected digit until the chain is done."""
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
