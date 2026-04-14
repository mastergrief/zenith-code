"""Multi-task HRM data — pools math/NL/word/GSM generators.

A single 48K HRM trained on this dataset has to learn to recognize
WHICH input language it's seeing (math expression echo vs NL question
vs multi-sentence word problem vs long-narrative GSM) and emit the
right structured output for each. The target language is the same for
all four (`math expression + = + <eos>`) — only the input format varies.

This is the Vector-2 phase-1 probe: does the same-size HRM that
handles each domain at 93-100% individually hold up when it has to
switch between them on the fly?
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from calm.hrm.data import (
    MathDataGenerator, _CHAR_TO_ID,
)
from calm.hrm.gsm_data import GSMDataGenerator
from calm.hrm.nl_data import NLMathDataGenerator
from calm.hrm.word_data import WordProblemGenerator


@dataclass
class MultiTaskProblem:
    """Unified shape — encoder input is the problem (in whatever domain's
    native form), decoder target is the math expression."""
    source: str        # "math" | "nl" | "word" | "gsm" — for telemetry
    input: str         # what the HRM encodes
    expression: str    # what the HRM decoder emits


class MultiTaskGenerator:
    """Samples from all four domains with balanced proportions."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        # Seed each sub-generator with distinct seeds so they don't
        # accidentally coincide.
        self._math = MathDataGenerator(seed=seed)
        self._nl = NLMathDataGenerator(seed=seed + 1)
        self._word = WordProblemGenerator(seed=seed + 2)
        self._gsm = GSMDataGenerator(seed=seed + 3)

    def generate(self, n: int = 2000) -> List[MultiTaskProblem]:
        """Balanced mix: n/4 from each source."""
        per = n // 4
        problems: List[MultiTaskProblem] = []

        # Math: encoder input = the expression itself (math-echo task).
        for p in self._math.generate(per):
            problems.append(MultiTaskProblem(
                source="math", input=p.expression, expression=p.expression,
            ))
        # NL templates.
        for p in self._nl.generate(per):
            problems.append(MultiTaskProblem(
                source="nl", input=p.question, expression=p.expression,
            ))
        # Word problems.
        for p in self._word.generate(per):
            problems.append(MultiTaskProblem(
                source="word", input=p.problem, expression=p.expression,
            ))
        # GSM-style.
        for p in self._gsm.generate(n - len(problems)):
            problems.append(MultiTaskProblem(
                source="gsm", input=p.problem, expression=p.expression,
            ))

        self._rng.shuffle(problems)
        return problems


class MultiTaskDataset(Dataset):
    """Encoder: problem (any domain); decoder target: expression + '=' + eos."""

    def __init__(self, problems: List[MultiTaskProblem],
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
