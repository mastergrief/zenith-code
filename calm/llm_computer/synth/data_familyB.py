"""Family B IR synth data — adds Delegate nodes (gcd, factorial, etc.).

Family B extends Family A with single-argument and binary backend calls:
  - factorial(a)        — single arg
  - is_prime(a)         — single arg
  - fibonacci(a)        — single arg
  - gcd(a, b)           — binary
  - lcm(a, b)           — binary

Goal: ≥50% functional-correctness on held-out IO pairs. If this passes,
we've shown the synth model can emit not just arithmetic but also
named-function calls that route to compiled/Python backends via the
gate-graph IR's Delegate node.

Input format same as Family A:
    "a=3 b=5 : 15 | a=7 b=2 : 14 | ... | ? a=4 b=6"
Target: an expression string like "a * b", "factorial(a)", "gcd(a, b)".
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from calm.hrm.data import _CHAR_TO_ID


_ARITHMETIC_TEMPLATES = [
    # Family A base
    "a + b", "a - b", "b - a", "a * b",
    "a + C", "a - C", "C - a", "a * C",
    "b + C", "b * C",
]

# Family B — backend-delegated templates. Each entry is a (template, compute_fn).
def _fact(a, b): return math.factorial(a) if 0 <= a <= 8 else None
def _isprime(a, b):
    if a < 2: return 0
    if a < 4: return 1
    if a % 2 == 0: return 0
    for d in range(3, int(a ** 0.5) + 1, 2):
        if a % d == 0: return 0
    return 1
def _fib(a, b):
    if a < 0 or a > 20: return None
    f, g = 0, 1
    for _ in range(a): f, g = g, f + g
    return f
def _gcd(a, b): return math.gcd(a, b)
def _lcm(a, b): return (a * b) // math.gcd(a, b) if math.gcd(a, b) > 0 else 0


_BACKEND_TEMPLATES = [
    ("factorial(a)",  _fact),
    ("is_prime(a)",   _isprime),
    ("fibonacci(a)",  _fib),
    ("gcd(a, b)",     _gcd),
    ("lcm(a, b)",     _lcm),
]


@dataclass
class SynthBSample:
    template: str
    examples: List[Tuple[int, int, int]]
    query_a: int
    query_b: int
    query_out: int


def _eval_arith(expr: str, a: int, b: int) -> int:
    r = expr.replace("a", str(a)).replace("b", str(b))
    for c in r:
        if c not in "0123456789+-* ()":
            raise ValueError(c)
    return eval(r)  # noqa: S307


class SynthFamilyBGenerator:
    def __init__(self, seed: int = 42, n_examples: int = 3,
                 const_range=(1, 9), var_range=(1, 10),
                 mix_ratio: float = 0.4):
        """mix_ratio: fraction of samples drawn from backend-delegated
        templates (the rest are arithmetic, like Family A)."""
        self._rng = random.Random(seed)
        self._n_examples = n_examples
        self._const_range = const_range
        self._var_range = var_range
        self._mix = mix_ratio

    def _instantiate_arith(self, template: str) -> Tuple[str, callable]:
        if "C" in template:
            c = self._rng.randint(*self._const_range)
            expr = template.replace("C", str(c))
        else:
            expr = template
        return expr, lambda a, b: _eval_arith(expr, a, b)

    def _safe_draw_arith(self, compute):
        for _ in range(50):
            a = self._rng.randint(*self._var_range)
            b = self._rng.randint(*self._var_range)
            try:
                out = compute(a, b)
            except Exception:
                continue
            if out is None:
                continue
            if abs(out) < 10000:
                return a, b, out
        return None

    def _safe_draw_backend(self, template, compute):
        for _ in range(50):
            # For single-arg templates, b is irrelevant but still present.
            if template.startswith("factorial"):
                a = self._rng.randint(0, 8); b = 0
            elif template.startswith("fibonacci"):
                a = self._rng.randint(0, 15); b = 0
            elif template.startswith("is_prime"):
                a = self._rng.randint(2, 30); b = 0
            else:
                a = self._rng.randint(*self._var_range)
                b = self._rng.randint(*self._var_range)
            try:
                out = compute(a, b)
            except Exception:
                continue
            if out is None:
                continue
            if abs(out) < 100000:
                return a, b, out
        return None

    def generate(self, n: int) -> List[SynthBSample]:
        out: List[SynthBSample] = []
        for _ in range(n):
            if self._rng.random() < self._mix:
                template, compute = self._rng.choice(_BACKEND_TEMPLATES)
                draw = lambda: self._safe_draw_backend(template, compute)
            else:
                tmpl = self._rng.choice(_ARITHMETIC_TEMPLATES)
                template, compute = self._instantiate_arith(tmpl)
                draw = lambda: self._safe_draw_arith(compute)

            examples: List[Tuple[int, int, int]] = []
            used: set = set()
            attempts = 0
            while len(examples) < self._n_examples and attempts < 200:
                attempts += 1
                t = draw()
                if t is None:
                    continue
                key = (t[0], t[1])
                if key in used:
                    continue
                used.add(key)
                examples.append(t)
            if len(examples) < self._n_examples:
                continue
            q = draw()
            if q is None:
                continue
            out.append(SynthBSample(template=template, examples=examples,
                                     query_a=q[0], query_b=q[1], query_out=q[2]))
        return out


def encode_examples(sample: SynthBSample) -> str:
    parts = [f"a={a} b={b} : {o}" for (a, b, o) in sample.examples]
    return " | ".join(parts) + f" | ? a={sample.query_a} b={sample.query_b}"


class SynthFamilyBDataset(Dataset):
    def __init__(self, samples: List[SynthBSample],
                 max_enc_len: int = 96, max_dec_len: int = 20):
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
        enc_ids = [bos] + [_CHAR_TO_ID[c] for c in enc_text if c in _CHAR_TO_ID] + [eos]
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
            dec_in.append(pad); dec_target.append(pad)
        mask = [1 if t != pad else 0 for t in dec_target]

        return {
            "encoder_ids": torch.tensor(enc_ids, dtype=torch.long),
            "decoder_input_ids": torch.tensor(dec_in, dtype=torch.long),
            "decoder_target_ids": torch.tensor(dec_target, dtype=torch.long),
            "loss_mask": torch.tensor(mask, dtype=torch.bool),
        }
