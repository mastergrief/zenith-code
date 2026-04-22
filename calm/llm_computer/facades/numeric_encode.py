"""NumericEncodeFacade — decode-path facade for int → hex/binary/octal.

Complement of BaseConversionFacade (which goes hex/binary → decimal).
Generalizes the R46.2/R22c/R53a skeleton to a TEXT answer where each
emitted token can be a letter (hex) or digit. First multi-char-alphabet
decode-path facade after Icd10RecallFacade (diagnoses), but with a
compact fixed vocab — so this is more reliable than ICD-10 text recall.

Supported NL forms:
  - "What is N in hex?" / "N in binary" / "N in octal"
  - "Convert N to hex/binary/octal"
  - "Express N as hex/binary/octal"

Answer format: unprefixed (hex=`FF`, binary=`1010`, octal=`777`) since
the prompt's "in hex" already qualifies the base. Prefixed variant
(0xFF) can be selected via `prefixed=True`.

Usage:
    f = NumericEncodeFacade()
    f.install(gemma, tokenizer)
    r = f.solve("What is 255 in hex?")
    # r.source=255, r.target_base=16, r.encoded='FF'
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class NumericEncodeResult:
    prompt: str
    source: Optional[int]
    target_base: Optional[int]    # 16, 2, 8
    encoded: Optional[str]
    generated: str
    used_bias: bool


class NumericEncodeFacade:

    DEFAULT_BOOST = 50.0
    DEFAULT_MAX_TOKENS = 20

    _BASE_NAMES = {
        "hex": 16, "hexadecimal": 16,
        "binary": 2, "bin": 2,
        "octal": 8, "oct": 8,
    }

    # Parse patterns — each yields (source_int, target_base)
    _PATTERNS = [
        # "N in hex / binary / octal" or "N in hexadecimal"
        re.compile(
            r"\b(-?\d+)\s+in\s+(hex|hexadecimal|binary|bin|octal|oct)\b",
            re.IGNORECASE,
        ),
        # "convert N to hex"
        re.compile(
            r"convert\s+(-?\d+)\s+to\s+(hex|hexadecimal|binary|bin|octal|oct)\b",
            re.IGNORECASE,
        ),
        # "express N as hex" / "express N in hex"
        re.compile(
            r"express\s+(-?\d+)\s+(?:as|in)\s+(hex|hexadecimal|binary|bin|octal|oct)\b",
            re.IGNORECASE,
        ),
        # "N as hex / binary / octal"
        re.compile(
            r"\b(-?\d+)\s+as\s+(hex|hexadecimal|binary|bin|octal|oct)\b",
            re.IGNORECASE,
        ),
    ]

    def __init__(
        self,
        boost: float = DEFAULT_BOOST,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        device: str = "cuda",
        prefixed: bool = False,
    ):
        self.boost = boost
        self.max_tokens = max_tokens
        self.device = device
        self.prefixed = prefixed
        self._gemma = None
        self._tokenizer = None

    def install(self, gemma, tokenizer):
        self._gemma = gemma
        self._tokenizer = tokenizer

    def detach(self):
        self._gemma = None
        self._tokenizer = None

    def parse(self, prompt: str) -> tuple[Optional[int], Optional[int]]:
        """Returns (source_int, target_base) or (None, None)."""
        for pat in self._PATTERNS:
            m = pat.search(prompt)
            if m:
                try:
                    source = int(m.group(1))
                except ValueError:
                    continue
                base_name = m.group(2).lower()
                base = self._BASE_NAMES.get(base_name)
                if base is not None:
                    return source, base
        return None, None

    def evaluate(self, source: int, base: int) -> Optional[str]:
        if base == 16:
            s = format(source, "X") if source >= 0 else "-" + format(-source, "X")
            return f"0x{s}" if self.prefixed else s
        if base == 2:
            s = format(source, "b") if source >= 0 else "-" + format(-source, "b")
            return f"0b{s}" if self.prefixed else s
        if base == 8:
            s = format(source, "o") if source >= 0 else "-" + format(-source, "o")
            return f"0o{s}" if self.prefixed else s
        return None

    def solve(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        boost: Optional[float] = None,
        use_bias: bool = True,
    ) -> NumericEncodeResult:
        if self._gemma is None or self._tokenizer is None:
            raise RuntimeError("facade not installed — call install() first")
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        boost = boost if boost is not None else self.boost

        source, base = self.parse(prompt)
        encoded = self.evaluate(source, base) if (source is not None and base) else None

        bias_ids: list[int] = []
        if use_bias and encoded is not None:
            bias_ids = self._encoded_to_gemma_tokens(encoded)

        fire_bias = bool(bias_ids)
        text = self._generate(prompt, bias_ids if fire_bias else [],
                              boost, max_tokens)
        return NumericEncodeResult(
            prompt=prompt, source=source, target_base=base,
            encoded=encoded, generated=text, used_bias=fire_bias,
        )

    _SPACE_TOKEN_ID = 236743

    def _encoded_to_gemma_tokens(self, s: str) -> list[int]:
        """Encode the answer string as Gemma BPE, stripping BOS + leading ▁."""
        ids = self._tokenizer.encode(s)
        if ids and ids[0] == 2:  # BOS
            ids = ids[1:]
        if ids and ids[0] == self._SPACE_TOKEN_ID:
            ids = ids[1:]
        return ids

    def _generate(
        self,
        prompt: str,
        bias_token_ids: list[int],
        boost: float,
        max_tokens: int,
    ) -> str:
        from calm.llm_computer.gemma_substrate import KVCache

        gemma = self._gemma
        tok = self._tokenizer
        if not prompt.rstrip().lower().endswith(("answer:", "= ")):
            prompt = prompt.rstrip() + " Answer: "
        ids = tok.encode(prompt)
        cache = KVCache(gemma.config.n_layers, device=self.device)
        gen = list(ids)
        bias_idx = 0 if bias_token_ids else -1

        # Same post-bias truncation as NumberTheoryFacade — Gemma sticks
        # on the last-emitted char pattern when the bias ends (e.g. after
        # "FF" it may emit more F's). Cap at 4 natural tokens.
        POST_BIAS_BUDGET = 4

        with torch.no_grad():
            logits = gemma.forward(
                torch.tensor([gen]), device=self.device,
                kv_cache=cache, start_pos=0,
            )
            if 0 <= bias_idx < len(bias_token_ids):
                logits[0, -1, bias_token_ids[bias_idx]] += boost
                bias_idx += 1
            nxt = int(logits[0, -1].argmax())
            gen.append(nxt)

            post_bias_steps = 0
            for _ in range(max_tokens - 1):
                if hasattr(tok, "EOS_ID") and nxt == tok.EOS_ID:
                    break
                if bias_token_ids and bias_idx >= len(bias_token_ids):
                    post_bias_steps += 1
                    if post_bias_steps > POST_BIAS_BUDGET:
                        break
                logits = gemma.forward(
                    torch.tensor([[nxt]]), device=self.device,
                    kv_cache=cache, start_pos=len(gen) - 1,
                )
                if 0 <= bias_idx < len(bias_token_ids):
                    logits[0, -1, bias_token_ids[bias_idx]] += boost
                    bias_idx += 1
                nxt = int(logits[0, -1].argmax())
                gen.append(nxt)

        return tok.decode(gen[len(ids):]) if hasattr(tok, "decode") else ""
