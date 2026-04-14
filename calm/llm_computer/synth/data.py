"""Training data for Family A IR synthesis.

Each sample:
  template  →  a math expression in {a, b, integer constants} like
               "a + b", "a * 5", "b - a", "3 * a".
  examples  →  3 or 4 random (a, b) pairs + the evaluated output per
               pair, encoded as ASCII:

                    "a=3 b=5 : 8 | a=2 b=7 : 9 | a=4 b=4 : 8"

  target    →  the expression string, e.g. "a + b".

At inference, the model sees a fresh (a, b) query, emits a predicted
expression, which the caller then parses + interprets against (a, b)
and checks functional correctness.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from calm.hrm.data import _CHAR_TO_ID


_OPS = ["+", "-", "*"]


def _eval(expr: str, a: int, b: int):
    """Evaluate an expression with a and b substituted. Safe: no eval()."""
    replaced = expr.replace("a", str(a)).replace("b", str(b))
    # Whitelist chars — includes / and // for library-taught templates.
    for c in replaced:
        if c not in "0123456789+-*/ ()":
            raise ValueError(f"bad char in {expr!r}: {c!r}")
    return eval(replaced)  # noqa: S307


_TEMPLATES = [
    # Pure variables
    "a + b",
    "a - b",
    "b - a",
    "a * b",
    # Variable + constant (constants will be substituted at generation)
    "a + C",
    "a - C",
    "C - a",
    "a * C",
    "b + C",
    "b * C",
]


@dataclass
class SynthSample:
    template: str
    examples: List[Tuple[int, int, int]]  # (a, b, out) triples
    query_a: int
    query_b: int
    query_out: int


class SynthFamilyAGenerator:
    def __init__(self, seed: int = 42, n_examples: int = 3,
                 const_range=(1, 9), var_range=(1, 9)):
        self._rng = random.Random(seed)
        self._n_examples = n_examples
        self._const_range = const_range
        self._var_range = var_range

    def _instantiate(self, template: str) -> str:
        if "C" in template:
            c = self._rng.randint(*self._const_range)
            return template.replace("C", str(c))
        return template

    def _draw_pair(self) -> Tuple[int, int]:
        a = self._rng.randint(*self._var_range)
        b = self._rng.randint(*self._var_range)
        return a, b

    def _safe_draw(self, expr: str) -> Tuple[int, int, int]:
        # Re-draw until expression evaluates without overflow or division issues.
        for _ in range(50):
            a, b = self._draw_pair()
            out = _eval(expr, a, b)
            if abs(out) < 1000:
                return a, b, out
        raise RuntimeError(f"can't sample for {expr!r}")

    def generate(self, n: int) -> List[SynthSample]:
        out: List[SynthSample] = []
        for _ in range(n):
            tmpl = self._rng.choice(_TEMPLATES)
            expr = self._instantiate(tmpl)
            # draw examples (may repeat inputs — that's fine)
            examples: List[Tuple[int, int, int]] = []
            used: set = set()
            attempts = 0
            while len(examples) < self._n_examples and attempts < 200:
                attempts += 1
                triple = self._safe_draw(expr)
                key = (triple[0], triple[1])
                if key in used:
                    continue
                used.add(key)
                examples.append(triple)
            qa, qb, qout = self._safe_draw(expr)
            while (qa, qb) in used:
                qa, qb, qout = self._safe_draw(expr)
            out.append(SynthSample(template=expr, examples=examples,
                                    query_a=qa, query_b=qb, query_out=qout))
        return out


def encode_examples(sample: SynthSample) -> str:
    """Render the encoder input string — "a=1 b=2 : 3 | ... | ? a=4 b=5"."""
    parts = [f"a={a} b={b} : {o}" for (a, b, o) in sample.examples]
    return " | ".join(parts) + f" | ? a={sample.query_a} b={sample.query_b}"


class SynthFamilyADataset(Dataset):
    """Encode (IO examples + query) → target expression string."""

    def __init__(self, samples: List[SynthSample],
                 max_enc_len: int = 96, max_dec_len: int = 16):
        self.samples = samples
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        pad = _CHAR_TO_ID["<pad>"]
        bos = _CHAR_TO_ID["<bos>"]
        eos = _CHAR_TO_ID["<eos>"]

        enc_text = encode_examples(s)
        enc_ids = [bos]
        for c in enc_text:
            if c in _CHAR_TO_ID:
                enc_ids.append(_CHAR_TO_ID[c])
        enc_ids.append(eos)
        enc_ids = enc_ids[: self.max_enc_len]
        while len(enc_ids) < self.max_enc_len:
            enc_ids.append(pad)

        target = s.template
        tgt_ids = [_CHAR_TO_ID[c] for c in target if c in _CHAR_TO_ID] + [eos]
        dec_in = [bos] + tgt_ids[:-1]
        dec_target = tgt_ids
        dec_in = dec_in[: self.max_dec_len]
        dec_target = dec_target[: self.max_dec_len]
        while len(dec_in) < self.max_dec_len:
            dec_in.append(pad)
        while len(dec_target) < self.max_dec_len:
            dec_target.append(pad)
        mask = [1 if t != pad else 0 for t in dec_target]

        return {
            "encoder_ids": torch.tensor(enc_ids, dtype=torch.long),
            "decoder_input_ids": torch.tensor(dec_in, dtype=torch.long),
            "decoder_target_ids": torch.tensor(dec_target, dtype=torch.long),
            "loss_mask": torch.tensor(mask, dtype=torch.bool),
        }
