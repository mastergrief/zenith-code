"""MultiStepCompositionFacade — Tier-3 (a * b) + c composition on prod Gemma.

Round 50.7. Promotes R46's pragmatic MultiStepReasoningFacade path to a
substrate-hosted composition facade where the multiplication step runs
through the compiled `multiplier` card (3390/3390 exhaustive on
a*b < 1000) rather than Python `safe_eval`. The final `+ c` step uses
`safe_eval` because the existing compiled `adder` covers only
a, b ∈ [0, 99] and intermediate (a*b) can reach 999 — outside the
adder's verified range. Wider adder generation is deferred.

Pipeline:

    NL prompt ──► regex parse ──► (a, b, c) triple
                                      │
                              compiled multiplier card ──► a*b (verified)
                                      │
                              safe_eval(a*b + c) ──► final integer
                                      │
                              step-through digit bias ──► Gemma emits digits

Delivery is identical to R11/R46: biased autoregressive decode,
multi-token answer. The multiplier card is NOT installed into Gemma's
attention; it runs as a standalone verifier on CPU/GPU before
generation. This mirrors MultiplicationFacade's install model (no
CardSlots, no VerificationHook — the facade only touches Gemma during
generate()).

Usage:

    facade = MultiStepCompositionFacade()
    facade.install(gemma, tokenizer)

    result = facade.generate(
        "What is 17 times 23 plus 45? Answer: ")
    # result.operands         = (17, 23, 45)
    # result.multiplier_value = 391   (from compiled multiplier)
    # result.final_value      = 436   (from safe_eval)
    # result.generated_text   contains "436"

    facade.detach()

Scope (R50.7):
  - (a * b) + c and (a * b) - c where a * b < 1000, c ≥ 0
  - NL ops: times/×/*/x and plus/minus/+/-
  - Single final integer answer, delivered via step-through bias
  - Wider operand ranges require tier-2 adder regeneration (deferred)
  - Longer chains (a*b+c+d, etc) not in scope — use MultiStepReasoning
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import torch

from calm.expression import safe_eval
from calm.llm_computer.programs.multiplier import (
    MAX_PRODUCT, build_multiplier,
)


# (a, b, c) with explicit op between b and c. Captures sign of the
# final op so the facade distinguishes +c vs -c. NL aliases are
# handled by _normalize_prompt() before this regex runs.
_TRIPLE_RE = re.compile(
    r"(-?\d+)\s*\*\s*(-?\d+)\s*([+\-])\s*(-?\d+)"
)


@dataclass
class MultiStepCompositionResult:
    """Result of a generate() call. Exposes intermediate verified
    values for scoring and debugging."""
    prompt: str
    operands: Optional[tuple[int, int, int]]  # (a, b, c) or None
    op: Optional[str]                          # '+' or '-'
    multiplier_value: Optional[int]            # a * b from compiled card
    final_value: Optional[int]                 # (a*b) op c from safe_eval
    generated_text: str
    parsed_answer: Optional[int]               # first int in generated
    used_bias: bool                            # True iff bias fired
    substrate_native: bool                     # True iff multiplier ran


class MultiStepCompositionFacade:
    """(a * b) + c composition with compiled multiplier + safe_eval.

    One instance can be installed on multiple Gemmas; generate() keeps
    no per-call state beyond what's returned in the result. The
    multiplier card is built once at __init__ and lives on self.device.

    The `substrate_native` flag on each result records whether the
    multiplication step was served by the compiled card (True) or fell
    back to safe_eval because operands were out of the multiplier's
    verified range (False). Use it to measure substrate-native coverage
    as the card library grows.
    """

    DEFAULT_BOOST = 50.0
    DEFAULT_MAX_TOKENS = 60
    MAX_PRODUCT = MAX_PRODUCT  # 999, from multiplier.py
    MAX_C = 10**6              # guardrail for safe_eval (sane c only)

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
        """Attach to a Gemma substrate + tokenizer. Doesn't modify
        Gemma weights. The facade only biases during generate()."""
        self._gemma = gemma
        self._tokenizer = tokenizer

    def detach(self):
        """Detach from Gemma. generate() fails until install()."""
        self._gemma = None
        self._tokenizer = None

    def parse(self, prompt: str) -> Optional[tuple[int, int, str, int]]:
        """NL prompt → (a, b, op, c) or None if no recognizable
        composition pattern."""
        norm = self._normalize_prompt(prompt)
        m = _TRIPLE_RE.search(norm)
        if m is None:
            return None
        try:
            a = int(m.group(1))
            b = int(m.group(2))
            op = m.group(3)
            c = int(m.group(4))
        except ValueError:
            return None
        return (a, b, op, c)

    def verify_product(self, a: int, b: int) -> Optional[int]:
        """Run compiled multiplier on (a, b). Returns verified product,
        or None if the pair is outside the card's verified range."""
        if a < 0 or b < 0:
            return None
        if a * b > self.MAX_PRODUCT:
            return None
        x = torch.tensor([[a, b]], device=self.device, dtype=torch.long)
        with torch.no_grad():
            return int(self.multiplier(x)[0, 1].argmax().item())

    def compute_final(self, product: int, op: str, c: int) -> Optional[int]:
        """Compute (product) op (c) via safe_eval. Returns int or None
        on failure or non-integer result."""
        if op not in ("+", "-"):
            return None
        if abs(c) > self.MAX_C:
            return None
        expr = f"({product}) {op} ({c})"
        try:
            val = safe_eval(expr)
        except Exception:
            return None
        if isinstance(val, bool):
            return None
        if isinstance(val, int):
            return val
        if isinstance(val, float) and val.is_integer():
            return int(val)
        return None

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        boost: Optional[float] = None,
        use_bias: bool = True,
    ) -> MultiStepCompositionResult:
        """Full pipeline: parse → verify product via multiplier →
        compute final via safe_eval → biased decode.

        If parse fails or any step produces None, falls back to plain
        greedy decode (no bias). The result always carries the prompt
        and whatever intermediate values succeeded.
        """
        if self._gemma is None or self._tokenizer is None:
            raise RuntimeError(
                "facade not installed — call install(gemma, tok) first")
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        boost = boost if boost is not None else self.boost

        parsed = self.parse(prompt)
        operands = None
        op = None
        product = None
        final = None
        substrate_native = False

        if parsed is not None:
            a, b, op, c = parsed
            operands = (a, b, c)
            product = self.verify_product(a, b)
            if product is not None:
                substrate_native = True
                final = self.compute_final(product, op, c)

        digit_ids: list[int] = []
        if use_bias and final is not None:
            digit_ids = self._gemma_digit_tokens(final)

        fire_bias = bool(digit_ids)
        text = self._generate(
            prompt, digit_ids if fire_bias else [], boost, max_tokens)
        parsed_answer = self._parse_int(text)

        return MultiStepCompositionResult(
            prompt=prompt,
            operands=operands,
            op=op,
            multiplier_value=product,
            final_value=final,
            generated_text=text,
            parsed_answer=parsed_answer,
            used_bias=fire_bias,
            substrate_native=substrate_native,
        )

    # --- Internal ---

    @staticmethod
    def _normalize_prompt(prompt: str) -> str:
        """Substitute NL operator aliases with symbols so the triple
        regex can match. Order matters: multi-word phrases first."""
        norm = prompt.lower()
        # Multiplication aliases
        norm = re.sub(r"\bmultiplied\s+by\b", " * ", norm)
        norm = re.sub(r"\btimes\b", " * ", norm)
        norm = norm.replace("×", " * ")
        norm = re.sub(r"(?<=\d)\s*x\s*(?=\d)", " * ", norm)
        # Addition / subtraction aliases
        norm = re.sub(r"\bplus\b", " + ", norm)
        norm = re.sub(r"\bminus\b", " - ", norm)
        # Collapse whitespace
        norm = re.sub(r"\s+", " ", norm)
        return norm

    def _gemma_digit_tokens(self, n: int) -> list[int]:
        """Encode integer as Gemma BPE token sequence, skipping <bos>."""
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
        """Step-through decode with optional digit bias. Identical
        mechanism to MultiplicationFacade / MultiStepReasoningFacade."""
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
