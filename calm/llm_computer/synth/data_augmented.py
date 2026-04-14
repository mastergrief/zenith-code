"""Augmented Family A data generator — adds library templates to the pool.

Wraps `SynthFamilyAGenerator` and expands the template pool with any
templates loaded from the persistent library (or passed in directly).
Used for self-distillation: take what the Discoverer has learned via
the `!correct` loop, fold it back into synth-A's training distribution,
retrain, and measure whether synth now autonomously discovers those
templates.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

from calm.llm_computer.synth.data import (
    SynthFamilyADataset, SynthSample, _eval, _TEMPLATES,
)
from calm.llm_computer.synth.library import Library


class AugmentedSynthGenerator:
    """Family A generator extended with user-taught / library templates.

    extra_templates: list of template strings like 'a / 2', 'a * a'.
    The generator draws uniformly from `_TEMPLATES` ∪ extra_templates.
    """

    def __init__(self, seed: int = 42, n_examples: int = 3,
                 const_range=(1, 9), var_range=(1, 9),
                 extra_templates: Optional[List[str]] = None):
        self._rng = random.Random(seed)
        self._n_examples = n_examples
        self._const_range = const_range
        self._var_range = var_range
        self._templates = list(_TEMPLATES) + list(extra_templates or [])

    @classmethod
    def from_library(cls, library: Library, **kwargs):
        """Convenience: build a generator whose extra templates come from
        every entry in the library."""
        extras = [entry.expression for entry in library]
        return cls(extra_templates=extras, **kwargs)

    def _instantiate(self, template: str) -> str:
        if "C" in template:
            c = self._rng.randint(*self._const_range)
            return template.replace("C", str(c))
        return template

    def _draw_pair(self) -> Tuple[int, int]:
        return (self._rng.randint(*self._var_range),
                self._rng.randint(*self._var_range))

    def _safe_draw(self, expr: str) -> Tuple[int, int, int]:
        for _ in range(50):
            a, b = self._draw_pair()
            try:
                out = _eval(expr, a, b)
            except Exception:
                continue
            if out is None:
                continue
            # Allow fractional outputs for e.g. 'a / 2' when a is even only;
            # reject non-integer results to keep target vocab integer.
            if isinstance(out, float) and out != int(out):
                continue
            out = int(out) if isinstance(out, float) else out
            if abs(out) < 1000:
                return a, b, out
        raise RuntimeError(f"can't sample for {expr!r}")

    def generate(self, n: int) -> List[SynthSample]:
        out: List[SynthSample] = []
        for _ in range(n):
            tmpl = self._rng.choice(self._templates)
            expr = self._instantiate(tmpl)
            examples: List[Tuple[int, int, int]] = []
            used: set = set()
            attempts = 0
            while len(examples) < self._n_examples and attempts < 200:
                attempts += 1
                try:
                    t = self._safe_draw(expr)
                except RuntimeError:
                    break
                if (t[0], t[1]) in used:
                    continue
                used.add((t[0], t[1]))
                examples.append(t)
            if len(examples) < self._n_examples:
                continue
            try:
                qa, qb, qout = self._safe_draw(expr)
            except RuntimeError:
                continue
            while (qa, qb) in used:
                qa, qb, qout = self._safe_draw(expr)
            out.append(SynthSample(template=expr, examples=examples,
                                    query_a=qa, query_b=qb, query_out=qout))
        return out
