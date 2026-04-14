"""Pipeline tests for the L3 meta-learning dataset."""

from __future__ import annotations

import torch

from calm.hrm.data import _CHAR_TO_ID
from calm.hrm.meta_data import (
    FORMATS, MetaDataset, MetaGenerator, TEST_FORMATS, TRAIN_FORMATS,
)


def test_train_and_test_are_disjoint():
    assert set(TRAIN_FORMATS).isdisjoint(TEST_FORMATS)
    assert len(TRAIN_FORMATS) >= 8
    assert len(TEST_FORMATS) >= 3
    assert set(TRAIN_FORMATS) | set(TEST_FORMATS) == set(FORMATS.keys())


def test_sample_shape():
    gen = MetaGenerator(seed=7)
    samples = gen.generate(5)
    for s in samples:
        assert s.format in TRAIN_FORMATS
        assert len(s.examples) == 3
        for ex_in, ex_out in s.examples:
            assert isinstance(ex_in, str) and len(ex_in) > 0
            assert isinstance(ex_out, str) and len(ex_out) > 0
        assert s.query_in and s.query_expr


def test_test_pool_does_not_leak_into_train():
    gen = MetaGenerator(seed=11, formats=TRAIN_FORMATS)
    for s in gen.generate(50):
        assert s.format not in TEST_FORMATS


def test_dataset_encoding_contains_separators():
    gen = MetaGenerator(seed=3)
    ds = MetaDataset(gen.generate(3), max_enc_len=384, max_dec_len=28)
    sep = _CHAR_TO_ID["<sep>"]
    bos = _CHAR_TO_ID["<bos>"]
    eos = _CHAR_TO_ID["<eos>"]
    enc = ds[0]["encoder_ids"]
    # Three demos × 2 separators each = 6 sep tokens (in-out divider + post-out divider).
    sep_count = int((enc == sep).sum().item())
    assert sep_count == 6, sep_count
    assert int(enc[0].item()) == bos
    # Encoder ends with eos before padding
    non_pad = enc[enc != 0]
    assert int(non_pad[-1].item()) == eos


def test_test_format_generator():
    """TEST_FORMATS should also be generable, used only at eval."""
    gen = MetaGenerator(seed=5, formats=TEST_FORMATS)
    samples = gen.generate(10)
    fmts_seen = {s.format for s in samples}
    assert fmts_seen.issubset(set(TEST_FORMATS))
    assert len(fmts_seen) >= 2   # at least some variety from 10 samples
