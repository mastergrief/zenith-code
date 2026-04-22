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


# Auto-facade registry — list of (tag, module, class_name) triples.
# Registered by default in PlannerFacade.__init__ via register_default_auto_facades().
# Each auto-facade follows the multi_step / number_theory contract:
#   - parse(prompt) → Optional[tuple] (operands) or None
#   - solve(prompt, use_bias=bool) → dataclass with .value, .used_bias, .generated
# Order matters: earlier facades match first (before multi_step catch-all).
DEFAULT_AUTO_FACADES = [
    # (tag, module_name, class_name)
    ("factorial",    "factorial_auto",    "FactorialFacade"),
    ("fibonacci",    "fibonacci_auto",    "FibonacciFacade"),
    ("combinations", "combinations_auto", "CombinationsFacade"),
    ("permutations", "permutations_auto", "PermutationsFacade"),
    ("power",        "power_auto",        "PowerFacade"),
    ("next_prime",   "next_prime_auto",   "NextPrimeFacade"),
    ("days_between", "days_between_auto", "DaysBetweenFacade"),
    ("is_prime",     "is_prime_auto",     "IsPrimeFacade"),
]


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

# N-step chain: arithmetic-op intermediate step.
# Matches: "multiply by N" / "times N" / "plus N" / "minus N" /
#          "divided by N" / "add N" / "subtract N" / "* N" / "+ N" etc.
# Single capture group = signed operand.
_CHAIN_OP_RE = re.compile(
    r"\b(?:multiplied?\s+by|multiply\s+by|times|\*)\s+(-?\d+)"      # ×N
    r"|\b(?:divided?\s+by|divide\s+by|over|/)\s+(-?\d+)"             # ÷N
    r"|\b(?:plus|add(?:ed)?(?:\s+by)?|\+)\s+(-?\d+)"                 # +N
    r"|\b(?:minus|subtract(?:ed)?(?:\s+by)?|\-)\s+(-?\d+)",          # −N
    re.IGNORECASE,
)
# Op-category order (regex alternation above). Maps group idx → (op, sign)
# group 1 = *, 2 = /, 3 = +, 4 = -
_CHAIN_OP_ORDER = ["*", "/", "+", "-"]


class PlannerFacade:
    """Top-level dispatch facade. Minimal orchestration — single-facade
    routing, no chaining between facades yet. Fixed priority order,
    first matcher wins."""

    def __init__(self, device: str = "cuda", register_auto: bool = True):
        self.device = device
        self._gemma = None
        self._tokenizer = None
        self.icd10 = Icd10RecallFacade(device=device)
        self.base_conv = BaseConversionFacade(device=device)
        self.number_theory = NumberTheoryFacade(device=device)
        self.multi_step = MultiStepReasoningFacade(device=device)
        self.numeric_encode = NumericEncodeFacade(device=device)
        # Auto-facades: list of (tag, instance). Dispatched between
        # numeric_encode and number_theory in priority (tighter regexes).
        self.auto_facades: list[tuple[str, object]] = []
        if register_auto:
            self.register_default_auto_facades()

    def register_default_auto_facades(self) -> int:
        """Register the DEFAULT_AUTO_FACADES list (factorial, fibonacci,
        combinations, permutations, power, next_prime). Silently skips
        any module that fails to import (e.g. if the .py file hasn't
        been generated yet). Returns count of successfully registered."""
        import importlib
        count = 0
        for tag, module_name, class_name in DEFAULT_AUTO_FACADES:
            try:
                mod = importlib.import_module(
                    f"calm.llm_computer.facades.{module_name}"
                )
                cls = getattr(mod, class_name)
                self.register_auto_facade(tag, cls(device=self.device))
                count += 1
            except (ImportError, AttributeError):
                continue
        return count

    def register_auto_facade(self, tag: str, facade: object) -> None:
        """Register a generated facade under a unique tag. Facade must
        implement parse(prompt) → Optional[tuple] and solve(prompt,
        use_bias: bool) → object with .value / .used_bias / .generated.
        """
        if any(t == tag for t, _ in self.auto_facades):
            raise ValueError(f"auto-facade tag {tag!r} already registered")
        self.auto_facades.append((tag, facade))

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
        for _tag, fac in self.auto_facades:
            fac.install(gemma, tokenizer)

    def detach(self):
        self.icd10.detach()
        self.base_conv.detach()
        self.number_theory.detach()
        self.multi_step.detach()
        self.numeric_encode.detach()
        for _tag, fac in self.auto_facades:
            fac.detach()
        self._gemma = None
        self._tokenizer = None

    @staticmethod
    def _split_chain_steps(prompt: str) -> list[str]:
        """Split a prompt into ordered chain steps on ", then " / " then ".
        Returns list of segment strings. Single-segment (no connective)
        returns [prompt] unchanged."""
        # Split on "then" connectives; preserve order
        parts = re.split(
            r",?\s*\bthen\b\s*,?|,(?=\s*(?:multiply|times|plus|add|minus|subtract|divided?|\*|\+|\-|/|in\s+(?:hex|binary|octal)))",
            prompt,
            flags=re.IGNORECASE,
        )
        return [p.strip(" ,?") for p in parts if p.strip(" ,?")]

    @staticmethod
    def _parse_chain_op(segment: str) -> Optional[tuple[str, int]]:
        """Extract (op, operand) from a chain step like 'multiply by 3'.
        Returns None if segment is not a chain-op step."""
        m = _CHAIN_OP_RE.search(segment)
        if not m:
            return None
        for idx, op in enumerate(_CHAIN_OP_ORDER, start=1):
            if m.group(idx) is not None:
                return (op, int(m.group(idx)))
        return None

    @staticmethod
    def _apply_chain_op(value: int, op: str, operand: int) -> Optional[int]:
        if op == "*":
            return value * operand
        if op == "/":
            if operand == 0:
                return None
            q, r = divmod(value, operand)
            return q if r == 0 else None  # integer division only
        if op == "+":
            return value + operand
        if op == "-":
            return value - operand
        return None

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
        """Single-facade classifier (no chain). Auto-facades checked
        BEFORE multi_step catch-all — their regexes are tighter so they
        won't match infix-math prompts incorrectly.
        """
        if self.icd10.parse(prompt) != (None, None):
            return "icd10"
        if self.base_conv.parse(prompt) != (None, None):
            return "base_conv"
        if self.numeric_encode.parse(prompt) != (None, None):
            return "numeric_encode"
        if self.number_theory.parse(prompt) != (None, None):
            return "number_theory"
        # Auto-facades (factorial, fibonacci, combinations, etc.) —
        # checked before multi_step because they have narrower regexes.
        for tag, fac in self.auto_facades:
            if fac.parse(prompt) is not None:
                return f"auto:{tag}"
        if self.multi_step.parse(prompt) is not None:
            return "multi_step"
        return None

    def _primary_solve(self, primary_prompt: str) -> tuple[Optional[str], Optional[int]]:
        """Run classified single-facade solve (no bias) on a primary
        sub-prompt. Returns (facade_tag, integer_value) or (None, None)
        if no facade parses. Used by N-step chain dispatch.
        """
        tag = self._classify_single(primary_prompt)
        if tag is None:
            return None, None
        if tag == "number_theory":
            pr = self.number_theory.solve(primary_prompt, use_bias=False)
            return tag, pr.value
        if tag == "multi_step":
            pr = self.multi_step.solve(primary_prompt, use_bias=False)
            return tag, pr.value
        if tag == "base_conv":
            pr = self.base_conv.solve(primary_prompt, use_bias=False)
            return tag, pr.value
        if tag == "numeric_encode":
            pr = self.numeric_encode.solve(primary_prompt, use_bias=False)
            return tag, pr.source
        if tag.startswith("auto:"):
            sub = tag.split(":", 1)[1]
            for t, fac in self.auto_facades:
                if t == sub:
                    pr = fac.solve(primary_prompt, use_bias=False)
                    return tag, getattr(pr, "value", None)
        return None, None

    def _nstep_chain_dispatch(
        self, prompt: str, use_bias: bool
    ) -> Optional[PlannerResult]:
        """N-step chain parser. Splits on 'then' / ',' connectives into
        ordered steps. Step 0 is a primary facade call; subsequent steps
        are either arithmetic ops on the running value ('multiply by 3')
        or a final numeric-encode ('in hex').

        Returns PlannerResult when a chain is detected AND every step
        resolves; None otherwise (caller falls back to single-facade /
        2-step chain / pass-through).
        """
        segments = self._split_chain_steps(prompt)
        if len(segments) < 2:
            return None

        primary_prompt = segments[0]
        # Hint the primary regexes by shaping the segment as a question.
        primary_shaped = primary_prompt.rstrip(" ,?") + "?"
        primary_tag, value = self._primary_solve(primary_shaped)
        if primary_tag is None or value is None:
            return None

        steps: list = [(primary_tag, value)]
        final_encoded: Optional[str] = None
        final_generated: str = ""
        final_used_bias = False
        final_base: Optional[int] = None

        for seg in segments[1:]:
            # In-base suffix (must be last step)
            mb = _CHAIN_SUFFIX_RE.search(seg)
            if mb:
                base_map = {"hex": 16, "hexadecimal": 16,
                            "binary": 2, "bin": 2,
                            "octal": 8, "oct": 8}
                final_base = base_map[mb.group(1).lower()]
                # Issue encode via numeric_encode facade using running value.
                base_name = {16: "hex", 2: "binary", 8: "octal"}[final_base]
                encode_prompt = f"What is {value} in {base_name}?"
                er = self.numeric_encode.solve(encode_prompt, use_bias=use_bias)
                final_encoded = er.encoded
                final_generated = er.generated
                final_used_bias = er.used_bias
                steps.append(("numeric_encode", er.encoded))
                break
            # Arithmetic op step
            op_pair = self._parse_chain_op(seg)
            if op_pair is None:
                return None  # unrecognized step → abort chain
            op, operand = op_pair
            nv = self._apply_chain_op(value, op, operand)
            if nv is None:
                return None
            value = nv
            steps.append((f"arith:{op}{operand}", value))

        # If the final step was not a numeric encode, emit the integer
        # value via multi_step facade so bias fires on the answer digits.
        # multi_step needs a parseable expression — use `value + 0` so
        # the parser fires and the digit bias anchors on `value`. Plain
        # "What is {value}?" produces no parse → no bias → Gemma emits
        # EOS turn token immediately. See r70d corpus for the failure mode.
        if final_encoded is None:
            emit_prompt = f"What is {value} + 0?"
            er = self.multi_step.solve(emit_prompt, use_bias=use_bias)
            final_generated = er.generated
            final_used_bias = er.used_bias

        # Facade tag encodes the full chain: "chain:A→…→encodeB" or
        # "chain:A→arith:*3→…"
        tag_str = "chain:" + "→".join(
            t if not t.startswith("auto:") else t for t, _ in steps
        )
        return PlannerResult(
            prompt=prompt,
            facade=tag_str,
            used_bias=final_used_bias,
            generated=final_generated,
            parsed_value=(final_encoded if final_encoded is not None else value),
            chain_steps=steps,
        )

    def solve(self, prompt: str, *, use_bias: bool = True) -> PlannerResult:
        if self._gemma is None or self._tokenizer is None:
            raise RuntimeError("planner not installed — call install() first")

        # Try N-step chain dispatch FIRST. It handles 3+ step chains
        # connected by 'then' or comma connectives. If it fires, return
        # directly. If it misses (single-step / wrong shape), fall
        # through to the 2-step chain + single-facade path.
        nstep = self._nstep_chain_dispatch(prompt, use_bias)
        if nstep is not None:
            return nstep

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

        # Auto-facade dispatch ("auto:<tag>") ------------------------------
        if tag and tag.startswith("auto:"):
            tag_name = tag.split(":", 1)[1]
            for t, fac in self.auto_facades:
                if t == tag_name:
                    r = fac.solve(prompt, use_bias=use_bias)
                    return PlannerResult(
                        prompt=prompt, facade=f"auto:{t}",
                        used_bias=getattr(r, "used_bias", False),
                        generated=getattr(r, "generated", ""),
                        parsed_value=getattr(r, "value", None),
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
