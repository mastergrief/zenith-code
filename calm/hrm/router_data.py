"""Router training data — balanced (query, domain_label) samples.

Round 3 of the L3-L6 roadmap. The router is a tiny classifier that
labels an incoming query with the sub-specialist that should handle it:

  label  name      → specialist checkpoint
  -----  --------    ---------------------------------------
    0    math        calm/hrm/checkpoints/math_structure_best.pt
    1    nl          calm/hrm/checkpoints/nl_math_structure_best.pt
    2    word        calm/hrm/checkpoints/word_problem_best.pt
    3    gsm         calm/hrm/checkpoints/gsm_best.pt
    4    meta        calm/hrm/checkpoints/meta_best.pt (few-shot-prefixed)

Meta samples use the same encoder format as MetaDataset (3 <sep>-
separated demonstrations + query), so the router can distinguish them
purely by counting <sep> tokens — but letting the router LEARN that
via an HRM encoder still exercises the full routing pipeline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from calm.hrm.data import MathDataGenerator, _CHAR_TO_ID
from calm.hrm.gsm_data import GSMDataGenerator
from calm.hrm.meta_data import MetaGenerator, TEST_FORMATS, TRAIN_FORMATS
from calm.hrm.nl_data import NLMathDataGenerator
from calm.hrm.word_data import WordProblemGenerator


LABELS = ("math", "nl", "word", "gsm", "meta")
N_LABELS = len(LABELS)
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}


@dataclass
class RouterSample:
    text: str
    label_id: int


def _make_meta_text(sample) -> str:
    """Stringify a MetaSample into encoder-ready text (with <sep> markers).

    We represent <sep> as a literal '\\x01' so the tokenizer emits the sep
    token ID. Callers join this with other text by going through the same
    tokenize path — never pass it through `<sep>` as the multi-char string.
    """
    # Use a real control character that maps 1:1 to the <sep> token id.
    sep = "\x01"
    parts = []
    for ex_in, ex_out in sample.examples:
        parts.append(ex_in + sep + ex_out)
    body = sep.join(parts)
    return body + sep + sample.query_in


class RouterGenerator:
    """Sample ~n/N_LABELS problems per label and return a shuffled list."""

    def __init__(self, seed: int = 42,
                 meta_formats_for_training=TRAIN_FORMATS,
                 use_test_formats_for_meta: bool = False):
        self._rng = random.Random(seed)
        self._math = MathDataGenerator(seed=seed)
        self._nl = NLMathDataGenerator(seed=seed + 1)
        self._word = WordProblemGenerator(seed=seed + 2)
        self._gsm = GSMDataGenerator(seed=seed + 3)
        meta_pool = TEST_FORMATS if use_test_formats_for_meta else meta_formats_for_training
        self._meta = MetaGenerator(seed=seed + 4, formats=meta_pool)

    def generate(self, n: int = 5000) -> List[RouterSample]:
        per = n // N_LABELS
        samples: List[RouterSample] = []
        for p in self._math.generate(per):
            samples.append(RouterSample(text=p.expression, label_id=LABEL_TO_ID["math"]))
        for p in self._nl.generate(per):
            samples.append(RouterSample(text=p.question, label_id=LABEL_TO_ID["nl"]))
        for p in self._word.generate(per):
            samples.append(RouterSample(text=p.problem, label_id=LABEL_TO_ID["word"]))
        for p in self._gsm.generate(per):
            samples.append(RouterSample(text=p.problem, label_id=LABEL_TO_ID["gsm"]))
        for ms in self._meta.generate(per):
            samples.append(RouterSample(text=_make_meta_text(ms),
                                         label_id=LABEL_TO_ID["meta"]))
        self._rng.shuffle(samples)
        return samples


class RouterDataset(Dataset):
    """Encode each query text into fixed-length token ids + label."""

    def __init__(self, samples: List[RouterSample], max_len: int = 384):
        self.samples = samples
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        pad = _CHAR_TO_ID["<pad>"]
        bos = _CHAR_TO_ID["<bos>"]
        eos = _CHAR_TO_ID["<eos>"]
        sep = _CHAR_TO_ID["<sep>"]

        ids = [bos]
        for ch in s.text:
            if ch == "\x01":
                ids.append(sep)
            elif ch in _CHAR_TO_ID:
                ids.append(_CHAR_TO_ID[ch])
        ids.append(eos)
        ids = ids[: self.max_len]
        while len(ids) < self.max_len:
            ids.append(pad)

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(s.label_id, dtype=torch.long),
        }
