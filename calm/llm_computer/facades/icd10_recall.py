"""Icd10RecallFacade — tier-3 compute facade for ICD-10 code lookup.

Per augmentation_thesis.md §"Customer verticals = card decks" +
tracing_roadmap.md §"Tier-3 validation: ICD-10 recall card", ICD-10
lookup is the canonical tier-3 demo: Gemma has ZERO reliable prior
for specific code → diagnosis mappings on rare codes and hallucinates
plausible-but-wrong answers.

Decode-path approach (simpler than CardSlot):
  parse:    extract ICD-10 code from NL prompt ("What is ICD-10 code
            J45.909?" / "What does K21.9 mean?" / bare "E11.9")
  evaluate: JSON dict lookup (72,748 codes from smog1210 2022 CMS dump)
  deliver:  multi-token step-through bias at Gemma decode — same
            mechanism as R11/R46.2/R22c but emitting the diagnosis
            TEXT tokens instead of digit tokens

Zero VRAM overhead, zero training, zero channel budget. Composes with
any other compute facade without interference.

Usage:
    facade = Icd10RecallFacade()
    facade.load_db(Path(".cache/icd10/icd10cm_codes_2022.json"))
    facade.install(gemma, tokenizer)
    r = facade.solve("What is ICD-10 code E11.9?")
    # r.code='E119', r.diagnosis='Type 2 diabetes...', r.used_bias=True

Target result: 50/50 on Gemma-fail corpus per roadmap, ≥20% absolute
lift with zero regressions.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class Icd10Result:
    prompt: str
    code: Optional[str]             # e.g. 'E119' (no dot, DB-normalized)
    code_raw: Optional[str]         # the user's form e.g. 'E11.9'
    diagnosis: Optional[str]        # looked-up text, or None if miss
    generated: str
    used_bias: bool


class Icd10RecallFacade:
    """NL ICD-10 lookup via decode-path bias. R46.2/R22c skeleton
    extended to text (not integer) answers."""

    DEFAULT_BOOST = 50.0
    DEFAULT_MAX_TOKENS = 80   # diagnoses are longer than digit answers

    # Code pattern — ICD-10 codes are 1 letter + 2 digits optional
    # + optional "." + up to 4 more alphanumeric chars. e.g. J45.909,
    # E11.9, S72.001A, I10, M54.5
    _CODE_RE = re.compile(
        r"\b([A-TV-Z])(\d{2})(?:\.(\d{1,4}[A-Z]?))?\b"
    )

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
        self._db: dict[str, str] = {}

    def load_db(self, path: Path | str) -> int:
        """Load JSON dict {code_nodot: diagnosis_text}."""
        path = Path(path)
        with path.open() as f:
            self._db = json.load(f)
        return len(self._db)

    def add_codes(self, pairs: dict[str, str]) -> None:
        """Merge extra codes (useful for tests or future curation)."""
        for k, v in pairs.items():
            self._db[k.replace(".", "")] = v

    def install(self, gemma, tokenizer):
        self._gemma = gemma
        self._tokenizer = tokenizer

    def detach(self):
        self._gemma = None
        self._tokenizer = None

    def parse(self, prompt: str) -> tuple[Optional[str], Optional[str]]:
        """Returns (code_raw, code_normalized) or (None, None). Only
        fires if a well-known ICD-10 lookup phrase AND a code literal
        both appear — avoids biasing on accidental code-like strings.
        """
        # Gate: require an explicit ICD-10 lookup signal
        if not re.search(
            r"\b(?:ICD[-\s]?10|diagnosis|diagnostic\s+code|medical\s+code)\b",
            prompt, re.IGNORECASE,
        ):
            return None, None
        m = self._CODE_RE.search(prompt)
        if not m:
            return None, None
        raw = "".join(g for g in m.groups() if g is not None)
        # Raw form like 'E119'; user form like 'E11.9'
        # User's literal is in the full match
        user_form = m.group(0)
        return user_form, raw

    def evaluate(self, code_normalized: str) -> Optional[str]:
        return self._db.get(code_normalized)

    def solve(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        boost: Optional[float] = None,
        use_bias: bool = True,
    ) -> Icd10Result:
        if self._gemma is None or self._tokenizer is None:
            raise RuntimeError("facade not installed — call install() first")
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        boost = boost if boost is not None else self.boost

        user_form, normalized = self.parse(prompt)
        diagnosis = self.evaluate(normalized) if normalized else None

        bias_ids: list[int] = []
        if use_bias and diagnosis is not None:
            bias_ids = self._diagnosis_to_gemma_tokens(diagnosis)

        fire_bias = bool(bias_ids)
        text = self._generate(prompt, bias_ids if fire_bias else [],
                              boost, max_tokens)
        return Icd10Result(
            prompt=prompt, code_raw=user_form, code=normalized,
            diagnosis=diagnosis, generated=text, used_bias=fire_bias,
        )

    def _diagnosis_to_gemma_tokens(self, text: str) -> list[int]:
        """Tokenize the diagnosis text as Gemma BPE. Skip BOS."""
        ids = self._tokenizer.encode(text)
        if ids and ids[0] == 2:  # BOS
            ids = ids[1:]
        return ids

    def _generate(
        self,
        prompt: str,
        bias_token_ids: list[int],
        boost: float,
        max_tokens: int,
    ) -> str:
        """Multi-token step-through bias at Gemma decode.

        Same template as MultiStepReasoningFacade._generate (R46.2) and
        BaseConversionFacade._generate (R22c), generalized from digit
        tokens to arbitrary Gemma BPE tokens (diagnosis text).

        Appends 'Answer: ' if prompt doesn't already end with a colon /
        fragment so first decode step fires the first diagnosis token.
        """
        from calm.llm_computer.gemma_substrate import KVCache

        gemma = self._gemma
        tok = self._tokenizer
        if not prompt.rstrip().lower().endswith(
            ("answer:", "diagnosis:", "means:", "is:")
        ):
            prompt = prompt.rstrip() + " Diagnosis: "
        ids = tok.encode(prompt)
        cache = KVCache(gemma.config.n_layers, device=self.device)
        gen = list(ids)
        bias_idx = 0 if bias_token_ids else -1

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

            for _ in range(max_tokens - 1):
                if hasattr(tok, "EOS_ID") and nxt == tok.EOS_ID:
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
