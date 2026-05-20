"""Tokenizer-level gates for the locked S0b2 GSM8k contract.

Covers:
- vocab built from train+val only (test is OOV check, not source)
- normalizer v2 applied to question + target
- reserved-extras included in declared vocab
- hard-fail on OOV at corpus-coverage check
- checkpoint round-trip via vocab_as_list + from_metadata
- mismatched normalizer_version refuses to construct
"""
from __future__ import annotations

import pytest

from calm.llm_computer.gsm8k_tokenizer import (
    NORMALIZER_VERSION,
    SPECIAL_TOKENS,
    Gsm8kTokenizer,
    _WHITESPACE_NORMALIZE,
    normalize_text,
)


def _row(q: str, expected: int) -> dict:
    return {"question": q, "expected": expected}


def test_normalizer_maps_smart_punctuation_and_whitespace():
    src = "It’s “hot” — 50 %, isn't it​?"
    out = normalize_text(src)
    # smart quotes → ASCII
    assert "’" not in out and "“" not in out and "”" not in out
    assert "'" in out and '"' in out
    # em dash → hyphen
    assert "—" not in out and "-" in out
    # non-breaking space → space
    assert " " not in out
    # zero-width space → dropped
    assert "​" not in out
    # retained punctuation
    assert "%" in out and "?" in out


def test_normalize_text_exact_output_regression():
    """Exact-output regression per codex audit `1779314284912`. Pins every
    codepoint by explicit \\uXXXX escape so a tool-rendering pass that
    silently collapses \\u00a0 (NBSP) -> U+0020 — which would map every
    ASCII space to \\n via the \\u2028 -> \\n rule — fails loudly here
    rather than at train time.
    """
    # The load-bearing one: ASCII space MUST remain a space.
    assert normalize_text("a b") == "a b"
    # NBSP -> ASCII space.
    assert normalize_text("a\u00a0b") == "a b"
    # Zero-width space -> drop.
    assert normalize_text("a\u200bb") == "ab"
    # Line separator -> newline.
    assert normalize_text("a\u2028b") == "a\nb"
    # Paragraph separator -> newline.
    assert normalize_text("a\u2029b") == "a\nb"
    # Smart quotes: left+right single and left+right double.
    assert normalize_text("\u2018x\u2019") == "'x'"
    assert normalize_text("\u201cx\u201d") == '"x"'
    # En + em dashes.
    assert normalize_text("a\u2013b\u2014c") == "a-b-c"
    # Mixed: a real-shape phrase exercising every rule + ASCII pass-through.
    assert normalize_text("It\u2019s \u201chot\u201d \u2014 50%, isn't it\u200b?") \
        == "It's \"hot\" - 50%, isn't it?"


def test_whitespace_normalize_keys_have_expected_codepoints():
    """Catches the tool-rendering failure class where unicode dict keys
    silently collapse to ASCII space. If any key reads as U+0020, this
    asserts immediately rather than poisoning the trained model.
    """
    expected = {0x00A0, 0x200B, 0x2028, 0x2029}
    actual = {ord(k) for k in _WHITESPACE_NORMALIZE}
    assert actual == expected, (
        f"_WHITESPACE_NORMALIZE keys have wrong codepoints: "
        f"expected {sorted(hex(c) for c in expected)}, "
        f"got {sorted(hex(c) for c in actual)}"
    )


def test_from_corpus_includes_reserved_extras():
    rows = [_row("If Alex has 17 apples and 23 oranges, what is the total?", 40)]
    tok = Gsm8kTokenizer.from_corpus(rows)
    # Reserved-extras: '#' not in this corpus but MUST be in vocab.
    assert "#" in tok.char_to_id


def test_special_tokens_get_lowest_ids():
    rows = [_row("17 + 23 = ?", 40)]
    tok = Gsm8kTokenizer.from_corpus(rows)
    for i, sp in enumerate(SPECIAL_TOKENS):
        assert tok.char_to_id[sp] == i


def test_encode_example_shape_and_sep_position():
    rows = [_row("17 + 23?", 40)]
    tok = Gsm8kTokenizer.from_corpus(rows)
    ids, sep_pos = tok.encode_example("17 + 23?", 40)
    # bos + question (8 chars) + sep + "40" (2 chars) + eos = 13 tokens
    assert len(ids) == 1 + 8 + 1 + 2 + 1
    assert ids[0] == tok.bos_id
    assert ids[sep_pos] == tok.sep_id
    assert ids[-1] == tok.eos_id
    # Target span starts at sep_pos + 1
    target_span = ids[sep_pos + 1:-1]
    assert tok.decode(target_span, stop_at_eos=False) == "40"


def test_encode_round_trip_preserves_normalized_text():
    rows = [_row("She had 10 dollars; she spent 3.", 7)]
    tok = Gsm8kTokenizer.from_corpus(rows)
    text = "She had 10 dollars; she spent 3."
    ids = tok.encode(text)
    decoded = "".join(tok.id_to_char[i] for i in ids)
    assert decoded == normalize_text(text)


def test_assert_corpus_covered_passes_for_train_corpus():
    rows = [_row("Alex has 17 apples.", 17), _row("Bob has 23.", 23)]
    tok = Gsm8kTokenizer.from_corpus(rows)
    tok.assert_corpus_covered(rows)  # must not raise


def test_assert_corpus_covered_hard_fails_on_oov():
    train = [_row("17 + 23?", 40)]
    test_with_oov = [_row("What is 17 € plus 23?", 40)]  # € is OOV here
    tok = Gsm8kTokenizer.from_corpus(train)
    with pytest.raises(ValueError, match="OOV"):
        tok.assert_corpus_covered(test_with_oov, label="test")


def test_checkpoint_round_trip_via_vocab_as_list():
    rows = [_row("If x = 17, what is 2x?", 34)]
    tok = Gsm8kTokenizer.from_corpus(rows)
    vocab_list = tok.vocab_as_list()
    # Persistence shape: list[str], id == index.
    assert isinstance(vocab_list, list) and all(isinstance(t, str) for t in vocab_list)
    assert vocab_list[tok.bos_id] == "<bos>"
    tok2 = Gsm8kTokenizer.from_metadata(vocab_list, NORMALIZER_VERSION)
    assert tok2.char_to_id == tok.char_to_id
    assert tok2.vocab_size == tok.vocab_size


def test_from_metadata_refuses_mismatched_normalizer_version():
    rows = [_row("17 + 23?", 40)]
    tok = Gsm8kTokenizer.from_corpus(rows)
    vocab_list = tok.vocab_as_list()
    with pytest.raises(ValueError, match="normalizer_version"):
        Gsm8kTokenizer.from_metadata(vocab_list, "v_old")
