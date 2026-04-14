"""10-format multi-task HRM training data — Experiment 2 (distribution probe).

Pools the original 4 domains (math, nl-template, word, gsm) with the 6
extended formats (code_var, prefix_op, distractor, units, let_bound,
eq_complete). Tests the hypothesis: *format diversity in training
teaches format-invariance, lifting OOD on held-out format variations*.

Same HRM architecture (`h=32`, 48K params) and same target language
(math expression) as the 4-domain multi-task training — only the input
distribution grows.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from calm.hrm.data import MathDataGenerator, _CHAR_TO_ID
from calm.hrm.extended_data import ExtendedFormatGenerator
from calm.hrm.gsm_data import GSMDataGenerator
from calm.hrm.nl_data import NLMathDataGenerator
from calm.hrm.word_data import WordProblemGenerator


@dataclass
class Multi10Problem:
    source: str
    input: str
    expression: str


class Multi10Generator:
    """Sample uniformly from 10 input formats."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._math = MathDataGenerator(seed=seed)
        self._nl = NLMathDataGenerator(seed=seed + 1)
        self._word = WordProblemGenerator(seed=seed + 2)
        self._gsm = GSMDataGenerator(seed=seed + 3)
        self._ext = ExtendedFormatGenerator(seed=seed + 4)

    def generate(self, n: int = 10000) -> List[Multi10Problem]:
        """Balanced ~n/10 from each format.

        Original 4 domains: `n/10` samples each (4 × n/10 = 4n/10 total).
        Extended 6 formats: `n/10` samples each — pulled as a batch of
        6 × n/10 from ExtendedFormatGenerator then tagged per source.
        """
        per = n // 10
        problems: List[Multi10Problem] = []

        for p in self._math.generate(per):
            problems.append(Multi10Problem("math", p.expression, p.expression))
        for p in self._nl.generate(per):
            problems.append(Multi10Problem("nl", p.question, p.expression))
        for p in self._word.generate(per):
            problems.append(Multi10Problem("word", p.problem, p.expression))
        for p in self._gsm.generate(per):
            problems.append(Multi10Problem("gsm", p.problem, p.expression))
        # Extended formats come balanced within themselves (per/6 × 6 categories).
        # To get per × 6 total, ask for 6 * per. ExtendedFormatGenerator uses
        # n // 6 per format, so 6 * per // 6 = per per format.
        for ep in self._ext.generate(6 * per):
            problems.append(Multi10Problem(ep.source, ep.input, ep.expression))

        self._rng.shuffle(problems)
        return problems[:n]


class Multi10Dataset(Dataset):
    """Encoder: input (any of 10 formats); decoder target: expression + `=` + eos."""

    def __init__(self, problems: List[Multi10Problem],
                 max_enc_len: int = 128, max_dec_len: int = 28):
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

        enc_ids = ([bos_id] +
                   [_CHAR_TO_ID[c] for c in p.input if c in _CHAR_TO_ID] +
                   [eos_id])
        enc_ids = enc_ids[: self.max_enc_len]
        while len(enc_ids) < self.max_enc_len:
            enc_ids.append(pad_id)

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
