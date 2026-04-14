"""Tests for Family A IR synth data pipeline + functional validator."""

from __future__ import annotations

from calm.llm_computer.synth.data import (
    SynthFamilyADataset, SynthFamilyAGenerator, _eval, encode_examples,
)
from calm.llm_computer.synth.infer import functional_correct


def test_generator_templates_cover_all():
    gen = SynthFamilyAGenerator(seed=3)
    samples = gen.generate(200)
    # At least the 4 pure-variable templates and a couple of constant templates
    # should appear across 200 draws.
    kinds = {s.template for s in samples}
    assert len(kinds) >= 6


def test_generator_examples_consistent():
    gen = SynthFamilyAGenerator(seed=7)
    samples = gen.generate(50)
    for s in samples:
        for a, b, out in s.examples:
            assert _eval(s.template, a, b) == out
        assert _eval(s.template, s.query_a, s.query_b) == s.query_out


def test_dataset_shape():
    gen = SynthFamilyAGenerator(seed=11)
    ds = SynthFamilyADataset(gen.generate(10), max_enc_len=96, max_dec_len=16)
    item = ds[0]
    assert item["encoder_ids"].shape == (96,)
    assert item["decoder_target_ids"].shape == (16,)


def test_functional_correct_matches_template():
    """Oracle test: passing the true template always passes the validator."""
    gen = SynthFamilyAGenerator(seed=17)
    for s in gen.generate(20):
        assert functional_correct(s.template, s), (s.template, s.query_a,
                                                     s.query_b, s.query_out)


def test_encode_examples_contains_query():
    gen = SynthFamilyAGenerator(seed=29)
    s = gen.generate(1)[0]
    text = encode_examples(s)
    assert f"a={s.query_a}" in text
    assert f"b={s.query_b}" in text
    assert "?" in text
