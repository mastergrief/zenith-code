"""PlannerFacade — orchestrates 4 tier-2/3 compute facades behind one
NL entry point, per `tracing_roadmap.md` §"Planner card" (Option A MVP).

Design goal per handoff: given an NL task, dispatch to the right
facade (math / base conversion / number theory / ICD-10 recall). One
prompt → one correct answer, regardless of which domain it lives in.

This is the Option-A MVP: a pure decode-path auto-dispatcher with a
priority chain. Option C (compiled planner card with channel-as-
register state) is follow-on work once Gemma's output is empirically
the bottleneck.

Dispatch order (first match wins; each facade's parse() is the gate):
  1. Icd10RecallFacade    — presence of 'ICD-10' phrase AND a code
  2. BaseConversionFacade — hex/binary + 'in/as/to decimal' phrase
  3. NumberTheoryFacade   — mod / gcd / lcm keywords
  4. MultiStepReasoningFacade — infix arithmetic (catch-all)

When no facade parses, pass-through to Gemma natural decode.

Why ordered: ICD-10 codes like J45.909 could superficially look like
decimal numbers if parsed as arithmetic; run the ICD-10 parser first.
Base conversion requires specific "in decimal" phrasing, so it's
after ICD-10 but before generic numeric parsers. Number theory
(mod/gcd/lcm) is specific-keyword gated. MultiStep is the catch-all
for infix math — and happens to also handle single-op cases.

Usage:
    planner = PlannerFacade()
    planner.load_icd10_db(Path('.cache/icd10/icd10cm_codes_2022.json'))
    planner.install(gemma, tokenizer)
    r = planner.solve("What is the diagnosis for ICD-10 code E11.9?")
    # r.facade = 'icd10', r.text = 'Type 2 diabetes mellitus...'
    r = planner.solve("What is the GCD of 48 and 180?")
    # r.facade = 'number_theory', r.text = '12'
    r = planner.solve("What is 17 × 23?")
    # r.facade = 'multi_step', r.text = '391'
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from calm.llm_computer.facades.base_conversion import BaseConversionFacade
from calm.llm_computer.facades.icd10_recall import Icd10RecallFacade
from calm.llm_computer.facades.multi_step import MultiStepReasoningFacade
from calm.llm_computer.facades.number_theory import NumberTheoryFacade
from calm.llm_computer.facades.numeric_encode import NumericEncodeFacade


@dataclass
class PlannerResult:
    prompt: str
    facade: Optional[str]      # 'icd10' | 'base_conv' | 'number_theory' | 'multi_step' | 'numeric_encode' | 'chain:<A>→<B>' | None
    used_bias: bool
    generated: str             # raw facade output OR pass-through Gemma
    parsed_value: Optional[object] = None  # facade-specific (int for math, text for ICD-10)
    chain_steps: Optional[list] = None     # for chain: [(facade, value), ...]


# "… in hex/binary/octal/decimal" chain suffix detector.
_CHAIN_SUFFIX_RE = re.compile(
    r"\bin\s+(hex|hexadecimal|binary|bin|octal|oct)\b",
    re.IGNORECASE,
)


class PlannerFacade:
    """Top-level dispatch facade. Minimal orchestration — single-facade
    routing, no chaining between facades yet. Fixed priority order,
    first matcher wins."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._gemma = None
        self._tokenizer = None
        self.icd10 = Icd10RecallFacade(device=device)
        self.base_conv = BaseConversionFacade(device=device)
        self.number_theory = NumberTheoryFacade(device=device)
        self.multi_step = MultiStepReasoningFacade(device=device)
        self.numeric_encode = NumericEncodeFacade(device=device)

    def load_icd10_db(self, path: Path | str) -> int:
        return self.icd10.load_db(path)

    def install(self, gemma, tokenizer):
        self._gemma = gemma
        self._tokenizer = tokenizer
        self.icd10.install(gemma, tokenizer)
        self.base_conv.install(gemma, tokenizer)
        self.number_theory.install(gemma, tokenizer)
        self.multi_step.install(gemma, tokenizer)
        self.numeric_encode.install(gemma, tokenizer)

    def detach(self):
        self.icd10.detach()
        self.base_conv.detach()
        self.number_theory.detach()
        self.multi_step.detach()
        self.numeric_encode.detach()
        self._gemma = None
        self._tokenizer = None

    def _chain_detect(self, prompt: str) -> Optional[tuple[str, int]]:
        """Detect 'X in hex|binary|octal' chain where X is a sub-query.
        Returns (primary_prompt, target_base) or None.

        Rule: ONLY a chain when the suffix is 'in <base>' AND there's
        no explicit decimal literal that the existing BaseConversion
        facade would handle (so 0xFF stays with base_conv). Strips
        the suffix and trailing punctuation so the primary classifier
        can reparse the remainder cleanly.
        """
        m = _CHAIN_SUFFIX_RE.search(prompt)
        if not m:
            return None
        # Base conv handles "0xFF in decimal" already — don't chain.
        if re.search(r"\b0[xb]", prompt, re.IGNORECASE):
            return None
        base_map = {
            "hex": 16, "hexadecimal": 16,
            "binary": 2, "bin": 2,
            "octal": 8, "oct": 8,
        }
        base = base_map[m.group(1).lower()]
        # The primary sub-prompt is everything before "in <base>".
        primary = prompt[:m.start()].rstrip(" ,?")
        # Append a question-shaped continuation so the primary classifier
        # matches its regex; most of those patterns don't require a ?
        # at the end but the math regex is position-aware.
        if not primary:
            return None
        primary = primary.rstrip() + "?"
        return primary, base

    def classify(self, prompt: str) -> Optional[str]:
        """Returns the tag of the first facade that would parse the
        prompt, or None for pass-through. Pure, no inference.

        Chain case ('X in hex'): returns 'chain:<primary>→encode<base>'
        where <primary> is one of the single-facade tags.
        """
        # Chain detection first — if it fires, the primary must also
        # parse for the chain to be valid; else fall back to single-facade.
        chain = self._chain_detect(prompt)
        if chain:
            primary_prompt, target_base = chain
            primary_tag = self._classify_single(primary_prompt)
            if primary_tag is not None and primary_tag != "icd10":
                return f"chain:{primary_tag}→encode{target_base}"

        return self._classify_single(prompt)

    def _classify_single(self, prompt: str) -> Optional[str]:
        """Single-facade classifier (no chain)."""
        if self.icd10.parse(prompt) != (None, None):
            return "icd10"
        if self.base_conv.parse(prompt) != (None, None):
            return "base_conv"
        if self.numeric_encode.parse(prompt) != (None, None):
            return "numeric_encode"
        if self.number_theory.parse(prompt) != (None, None):
            return "number_theory"
        if self.multi_step.parse(prompt) is not None:
            return "multi_step"
        return None

    def solve(self, prompt: str, *, use_bias: bool = True) -> PlannerResult:
        if self._gemma is None or self._tokenizer is None:
            raise RuntimeError("planner not installed — call install() first")

        tag = self.classify(prompt)

        # --- Chain: primary facade → numeric_encode ---
        # Tag format: "chain:<primary>→encode<base>"
        if tag and tag.startswith("chain:"):
            _, payload = tag.split(":", 1)
            primary_tag, encode_part = payload.split("→")
            target_base = int(encode_part.replace("encode", ""))

            # Run primary on the stripped prompt — same logic as
            # _chain_detect to rebuild primary_prompt
            chain = self._chain_detect(prompt)
            assert chain is not None
            primary_prompt, _ = chain

            primary_value: Optional[int] = None
            steps = []
            if primary_tag == "number_theory":
                pr = self.number_theory.solve(primary_prompt, use_bias=False)
                primary_value = pr.value
            elif primary_tag == "multi_step":
                pr = self.multi_step.solve(primary_prompt, use_bias=False)
                primary_value = pr.value
            elif primary_tag == "base_conv":
                pr = self.base_conv.solve(primary_prompt, use_bias=False)
                primary_value = pr.value
            elif primary_tag == "numeric_encode":
                pr = self.numeric_encode.solve(primary_prompt, use_bias=False)
                # numeric_encode already produces a string — re-decode
                # to int for the second-stage encode.
                primary_value = pr.source
            steps.append((primary_tag, primary_value))

            if primary_value is None:
                # Primary didn't eval — fall back to single-facade
                tag = self._classify_single(prompt)
            else:
                # Synthesize an encode prompt using the primary's value.
                # We want Gemma to emit the encoded form of primary_value.
                base_name = {16: "hex", 2: "binary", 8: "octal"}[target_base]
                encode_prompt = f"What is {primary_value} in {base_name}?"
                er = self.numeric_encode.solve(encode_prompt, use_bias=use_bias)
                steps.append(("numeric_encode", er.encoded))
                return PlannerResult(
                    prompt=prompt, facade=tag, used_bias=er.used_bias,
                    generated=er.generated, parsed_value=er.encoded,
                    chain_steps=steps,
                )

        if tag == "icd10":
            r = self.icd10.solve(prompt, use_bias=use_bias)
            return PlannerResult(
                prompt=prompt, facade="icd10", used_bias=r.used_bias,
                generated=r.generated, parsed_value=r.diagnosis,
            )
        if tag == "base_conv":
            r = self.base_conv.solve(prompt, use_bias=use_bias)
            return PlannerResult(
                prompt=prompt, facade="base_conv", used_bias=r.used_bias,
                generated=r.generated, parsed_value=r.value,
            )
        if tag == "numeric_encode":
            r = self.numeric_encode.solve(prompt, use_bias=use_bias)
            return PlannerResult(
                prompt=prompt, facade="numeric_encode", used_bias=r.used_bias,
                generated=r.generated, parsed_value=r.encoded,
            )
        if tag == "number_theory":
            r = self.number_theory.solve(prompt, use_bias=use_bias)
            return PlannerResult(
                prompt=prompt, facade="number_theory", used_bias=r.used_bias,
                generated=r.generated, parsed_value=r.value,
            )
        if tag == "multi_step":
            r = self.multi_step.solve(prompt, use_bias=use_bias)
            return PlannerResult(
                prompt=prompt, facade="multi_step", used_bias=r.used_bias,
                generated=r.generated, parsed_value=r.value,
            )

        # Pass-through — no facade engaged
        from calm.llm_computer.gemma_substrate import KVCache
        import torch
        gemma = self._gemma
        tok = self._tokenizer
        ids = tok.encode(prompt)
        cache = KVCache(gemma.config.n_layers, device=self.device)
        with torch.no_grad():
            logits = gemma.forward(
                torch.tensor([ids]), device=self.device,
                kv_cache=cache, start_pos=0,
            )
            gen = list(ids)
            nxt = int(logits[0, -1].argmax())
            gen.append(nxt)
            for _ in range(60):
                if hasattr(tok, "EOS_ID") and nxt == tok.EOS_ID:
                    break
                logits = gemma.forward(
                    torch.tensor([[nxt]]), device=self.device,
                    kv_cache=cache, start_pos=len(gen) - 1,
                )
                nxt = int(logits[0, -1].argmax())
                gen.append(nxt)
        text = tok.decode(gen[len(ids):]) if hasattr(tok, "decode") else ""
        return PlannerResult(
            prompt=prompt, facade=None, used_bias=False,
            generated=text, parsed_value=None,
        )
