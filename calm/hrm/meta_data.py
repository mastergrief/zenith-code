"""Meta-learning data — few-shot-prefixed encoder input.

Round 2 of the L3-L6 roadmap (Layer 3: in-context schema induction).

Training sample shape:
    <bos> EX1_in <sep> EX1_out <sep>
          EX2_in <sep> EX2_out <sep>
          EX3_in <sep> EX3_out <sep>
          QUERY_in <eos>                  → encoder
    <bos> QUERY_expression = <eos>        → decoder (target)

Each sample is drawn from a single format. A *batch* contains many
samples, each from a random format. The model sees 3 demonstrations +
the query together and must emit the query's math expression.

TRAIN_FORMATS is a subset of the 15 available 2-operand formats;
TEST_FORMATS is the complement and is held out entirely (never seen
during training, only via in-context examples at inference). Meta-
learning works iff the HRM at inference on TEST_FORMATS can induce
the format pattern from its 3 examples and produce the correct
expression for the 4th query.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from calm.hrm.data import _CHAR_TO_ID
from calm.hrm.extended_data import _GENERATORS as EXT_GENERATORS
from calm.hrm.extended2_data import _GENERATORS as EXT2_GENERATORS


# Combined catalog: format-name → callable(rng) that returns (input_str, expr_str).
# All sources expose `.input` and `.expression` attributes on the returned dataclass.
FormatFn = Callable[[random.Random], Tuple[str, str]]


def _wrap(gen_fn):
    """Wrap an ExtFormatProblem / Ext2Problem generator into a (input, expr) fn."""
    def _f(rng: random.Random) -> Tuple[str, str]:
        p = gen_fn(rng)
        return p.input, p.expression
    return _f


FORMATS: Dict[str, FormatFn] = {}
for name, fn in EXT_GENERATORS.items():
    FORMATS[name] = _wrap(fn)
for name, fn in EXT2_GENERATORS.items():
    if name == "three_op":
        continue  # 3-operand target, drop for uniform 2-operand pool
    FORMATS[name] = _wrap(fn)


# Train / held-out split. 5 test formats, 10 train formats.
TEST_FORMATS = ("eq_var", "possessive", "verb_by", "question_first", "when_then")
TRAIN_FORMATS = tuple(n for n in FORMATS if n not in TEST_FORMATS)

assert set(TRAIN_FORMATS).isdisjoint(TEST_FORMATS)
assert len(TRAIN_FORMATS) + len(TEST_FORMATS) == len(FORMATS)


@dataclass
class MetaSample:
    """A single few-shot-prefixed problem."""
    format: str
    examples: List[Tuple[str, str]]  # [(in_1, out_1), (in_2, out_2), (in_3, out_3)]
    query_in: str
    query_expr: str


def _eval_expr(expr: str):
    """Reject any expression the interpreter can't evaluate — safety net."""
    from calm.expression import ExpressionError, safe_eval
    try:
        ans = safe_eval(expr)
        if isinstance(ans, float) and ans == int(ans):
            ans = int(ans)
        return ans
    except (ExpressionError, OverflowError):
        return None


def _draw_one(rng: random.Random, fmt: str, used: set) -> Tuple[str, str]:
    """Draw a unique (input, expression) pair from format `fmt`."""
    for _ in range(100):
        inp, expr = FORMATS[fmt](rng)
        if (inp, expr) in used:
            continue
        if _eval_expr(expr) is None:
            continue
        return inp, expr
    raise RuntimeError(f"format {fmt} exhausted with {len(used)} used samples")


class MetaGenerator:
    """Generates meta-learning samples. Pick from `formats` (default = TRAIN)."""

    def __init__(self, seed: int = 42, formats: Tuple[str, ...] = TRAIN_FORMATS):
        self._rng = random.Random(seed)
        self._formats = formats

    def generate(self, n: int) -> List[MetaSample]:
        samples: List[MetaSample] = []
        for _ in range(n):
            fmt = self._rng.choice(self._formats)
            used: set = set()
            examples = []
            for _i in range(3):
                inp, expr = _draw_one(self._rng, fmt, used)
                used.add((inp, expr))
                examples.append((inp, expr))
            q_in, q_expr = _draw_one(self._rng, fmt, used)
            samples.append(MetaSample(format=fmt, examples=examples,
                                       query_in=q_in, query_expr=q_expr))
        return samples


class MetaDataset(Dataset):
    """Encode prefix + query; target = query_expr + '='."""

    def __init__(self, samples: List[MetaSample],
                 max_enc_len: int = 512, max_dec_len: int = 28):
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
        sep = _CHAR_TO_ID["<sep>"]

        def _chars(text: str) -> List[int]:
            return [_CHAR_TO_ID[c] for c in text if c in _CHAR_TO_ID]

        enc_ids = [bos]
        for ex_in, ex_out in s.examples:
            enc_ids += _chars(ex_in) + [sep] + _chars(ex_out) + [sep]
        enc_ids += _chars(s.query_in) + [eos]
        enc_ids = enc_ids[: self.max_enc_len]
        while len(enc_ids) < self.max_enc_len:
            enc_ids.append(pad)

        target_str = s.query_expr + "="
        tgt_ids = _chars(target_str) + [eos]
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
