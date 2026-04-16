"""NL → math expression training data for integration #3.

Generates `(natural language question, math expression)` pairs. The
HRM learns to translate the NL encoder input into a math expression
decoder target; at inference time the expression goes through the
existing `parse_expression` + `interpret` pipeline to recompute the
value analytically.

Same tokenizer and special tokens as `data.py` — letters are already
in the char vocab. Only the encoder input is new (NL uses spaces and
lowercase letters more heavily than math expressions).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from calm.expression import ExpressionError, safe_eval
from calm.hrm.data import _CHAR_TO_ID


@dataclass
class NLMathProblem:
    """A natural-language math question with its expression answer."""
    question: str
    expression: str
    answer: str


# Template bank. Each entry is (NL template, expression template, operand
# range). Operand ranges cap at 2 digits so NL strings stay short enough
# to fit in max_enc ≈ 48. Uses only operators HRM's existing structure-
# only pipeline already handles via `parse_expression`.
_TEMPLATES: List[Tuple[str, str, int]] = [
    ("what is {a} plus {b}",       "{a} + {b}",     999),
    ("what is {a} minus {b}",      "{a} - {b}",     999),
    ("what is {a} times {b}",      "{a} * {b}",     999),
    ("sum of {a} and {b}",         "{a} + {b}",     999),
    ("product of {a} and {b}",     "{a} * {b}",     999),
    ("{a} added to {b}",           "{a} + {b}",     999),
    ("{a} multiplied by {b}",      "{a} * {b}",     999),
    ("difference of {a} and {b}",  "{a} - {b}",     999),
    ("{a} more than {b}",          "{b} + {a}",     999),  # swapped
    ("{a} less than {b}",          "{b} - {a}",     999),  # swapped
    ("is {a} prime",               "is_prime({a})", 999),
    ("gcd of {a} and {b}",         "gcd({a}, {b})", 999),
    ("factorial of {a}",           "factorial({a})", 10),
]


class NLMathDataGenerator:
    """Generate (NL question, expression, answer) triples from templates."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def _sample_operand(self, max_val: int) -> int:
        """Sample with uniform digit-length coverage.

        Equal probability across digit-length buckets: [1-9], [10-99],
        [100-max_val]. Ensures every digit length is well-represented.
        """
        if max_val <= 9:
            return self._rng.randint(1, max_val)
        buckets = [(1, 9)]
        if max_val >= 10:
            buckets.append((10, min(99, max_val)))
        if max_val >= 100:
            buckets.append((100, max_val))
        lo, hi = self._rng.choice(buckets)
        return self._rng.randint(lo, hi)

    def generate(self, n: int = 2000) -> List[NLMathProblem]:
        problems: List[NLMathProblem] = []
        while len(problems) < n:
            nl_tmpl, expr_tmpl, max_val = self._rng.choice(_TEMPLATES)
            vals = {
                "a": self._sample_operand(max_val),
                "b": self._sample_operand(max_val),
            }
            question = nl_tmpl.format(**vals)
            expression = expr_tmpl.format(**vals)
            try:
                ans = safe_eval(expression)
                if isinstance(ans, float) and ans == int(ans):
                    ans = int(ans)
                problems.append(NLMathProblem(
                    question=question,
                    expression=expression,
                    answer=str(ans),
                ))
            except (ExpressionError, OverflowError):
                continue
        return problems


class NLMathSeq2SeqDataset(Dataset):
    """Encoder input = NL question; decoder target = math expression + `=` + <eos>.

    Structure-only loss semantics: decoder learns to emit the expression
    and a `=` terminator. Values are recomputed by LLM-Computer at
    inference time, so the decoder never has to compute — only transduce.
    """

    def __init__(self, problems: List[NLMathProblem], max_enc_len: int = 48,
                 max_dec_len: int = 24):
        self.problems = problems
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len

    def __len__(self) -> int:
        return len(self.problems)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        p = self.problems[idx]
        pad_id = _CHAR_TO_ID["<pad>"]
        bos_id = _CHAR_TO_ID["<bos>"]
        eos_id = _CHAR_TO_ID["<eos>"]

        # Encoder: <bos> question <eos> pad...
        enc_ids = [bos_id] + [_CHAR_TO_ID[c] for c in p.question if c in _CHAR_TO_ID] + [eos_id]
        enc_ids = enc_ids[: self.max_enc_len]
        while len(enc_ids) < self.max_enc_len:
            enc_ids.append(pad_id)

        # Decoder target: expression + `=` + <eos>
        target_str = p.expression + "="
        tgt_ids = [_CHAR_TO_ID[c] for c in target_str if c in _CHAR_TO_ID] + [eos_id]

        dec_in = [bos_id] + tgt_ids[:-1]
        dec_target = tgt_ids
        dec_in = dec_in[: self.max_dec_len]
        dec_target = dec_target[: self.max_dec_len]
        while len(dec_in) < self.max_dec_len:
            dec_in.append(pad_id)
        while len(dec_target) < self.max_dec_len:
            dec_target.append(pad_id)

        mask = [1 if t != pad_id else 0 for t in dec_target]

        return {
            "encoder_ids": torch.tensor(enc_ids, dtype=torch.long),
            "decoder_input_ids": torch.tensor(dec_in, dtype=torch.long),
            "decoder_target_ids": torch.tensor(dec_target, dtype=torch.long),
            "loss_mask": torch.tensor(mask, dtype=torch.bool),
        }
