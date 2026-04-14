"""20-format multi-task HRM training data — Experiment 2b.

Pools all 10 formats from multi10 (original 4 domains + 6 extended) plus
10 more extended-2 formats. Tests whether distribution scaling continues
to lift OOD past the 50% ceiling observed at 10 formats.

Same HRM architecture (`h=32`, 48K params). Only input distribution
grows.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from calm.hrm.data import MathDataGenerator, _CHAR_TO_ID
from calm.hrm.extended_data import ExtendedFormatGenerator
from calm.hrm.extended2_data import Extended2FormatGenerator
from calm.hrm.gsm_data import GSMDataGenerator
from calm.hrm.nl_data import NLMathDataGenerator
from calm.hrm.word_data import WordProblemGenerator


@dataclass
class Multi20Problem:
    source: str
    input: str
    expression: str


class Multi20Generator:
    """Samples from 20 input formats uniformly."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._math = MathDataGenerator(seed=seed)
        self._nl = NLMathDataGenerator(seed=seed + 1)
        self._word = WordProblemGenerator(seed=seed + 2)
        self._gsm = GSMDataGenerator(seed=seed + 3)
        self._ext = ExtendedFormatGenerator(seed=seed + 4)
        self._ext2 = Extended2FormatGenerator(seed=seed + 5)

    def generate(self, n: int = 20000) -> List[Multi20Problem]:
        per = n // 20
        probs: List[Multi20Problem] = []

        for p in self._math.generate(per):
            probs.append(Multi20Problem("math", p.expression, p.expression))
        for p in self._nl.generate(per):
            probs.append(Multi20Problem("nl", p.question, p.expression))
        for p in self._word.generate(per):
            probs.append(Multi20Problem("word", p.problem, p.expression))
        for p in self._gsm.generate(per):
            probs.append(Multi20Problem("gsm", p.problem, p.expression))
        for ep in self._ext.generate(6 * per):
            probs.append(Multi20Problem(ep.source, ep.input, ep.expression))
        for ep2 in self._ext2.generate(10 * per):
            probs.append(Multi20Problem(ep2.source, ep2.input, ep2.expression))

        self._rng.shuffle(probs)
        return probs[:n]


class Multi20Dataset(Dataset):
    """Encoder: input (any of 20 formats); decoder target: expression + `=` + eos."""

    def __init__(self, problems: List[Multi20Problem],
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
