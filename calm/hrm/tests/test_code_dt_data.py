"""Tests for `code_dt_data.py` — extraction + paraphrase augmentation."""
import pytest

from calm.hrm.code_dt_data import (
    CODE_VOCAB_SIZE,
    CodeProblem,
    _clean_prob,
    _extract_skeleton,
    _paraphrase_augment,
    code_detokenize,
    code_tokenize,
)


def test_vocab_contains_colon():
    """Required for function headers."""
    from calm.hrm.code_dt_data import _CODE_CHAR_TO_ID
    assert ":" in _CODE_CHAR_TO_ID


def test_vocab_size():
    assert CODE_VOCAB_SIZE == 81   # 4 specials + 77 chars


def test_tokenize_detokenize_roundtrip():
    text = "def FN(arr, n):"
    ids = code_tokenize(text)
    out = code_detokenize(ids)
    assert out == text


def test_clean_prob_accepts_normal_problem():
    prob = "Write a function to find the longest chain which can be formed"
    cleaned = _clean_prob(prob)
    assert cleaned == prob


def test_clean_prob_rejects_too_short():
    assert _clean_prob("short") is None


def test_clean_prob_collapses_whitespace():
    prob = "Write   a\n\nfunction  to do  stuff here now."
    cleaned = _clean_prob(prob)
    assert "  " not in cleaned


def test_clean_prob_drops_unicode():
    prob = "Write a function to convert emoji 🔥 to ASCII equivalents."
    cleaned = _clean_prob(prob)
    assert "🔥" not in cleaned


def test_extract_skeleton_uses_placeholder():
    sol = "```python\ndef max_chain_length(arr, n):\n    return 0\n```"
    result = _extract_skeleton(sol)
    assert result is not None
    fn_name, skeleton = result
    assert fn_name == "max_chain_length"
    assert skeleton == "def FN(arr, n):"
    assert "max_chain_length" not in skeleton


def test_extract_skeleton_prefers_top_level():
    """MBPP pattern: helper class method, target fn last."""
    sol = """
class Helper:
    def __init__(self, a):
        self.a = a

def target_fn(x, y):
    return x + y
"""
    result = _extract_skeleton(sol)
    assert result is not None
    fn_name, skeleton = result
    assert fn_name == "target_fn"
    assert skeleton == "def FN(x, y):"


def test_extract_skeleton_custom_placeholder():
    sol = "def foo(x):\n    pass"
    result = _extract_skeleton(sol, placeholder="F")
    assert result is not None
    _, skeleton = result
    assert skeleton == "def F(x):"


def test_paraphrase_augment_expands():
    pairs = [
        CodeProblem(
            question="Write a function to compute x.",
            expression="def FN(x):",
        ),
    ]
    augmented = _paraphrase_augment(pairs, factor=4, seed=0)
    # Original + up to 3 paraphrases
    assert len(augmented) >= 2
    assert len(augmented) <= 4
    # All variants have the same skeleton
    assert all(p.expression == "def FN(x):" for p in augmented)
    # Distinct question strings
    qs = {p.question for p in augmented}
    assert len(qs) >= 2


def test_paraphrase_preserves_original():
    """The original prompt always survives augmentation."""
    pairs = [
        CodeProblem(
            question="Write a function to square x.",
            expression="def FN(x):",
        ),
    ]
    augmented = _paraphrase_augment(pairs, factor=5, seed=0)
    questions = [p.question for p in augmented]
    assert "Write a function to square x." in questions


def test_paraphrase_no_match_returns_single():
    """If no template prefix matches, only the original is emitted."""
    pairs = [
        CodeProblem(
            question="Utterly arbitrary prefix that no template matches here now.",
            expression="def FN():",
        ),
    ]
    augmented = _paraphrase_augment(pairs, factor=5, seed=0)
    # No template matched → only the original passes through
    assert len(augmented) == 1
    assert augmented[0].question == pairs[0].question


def test_paraphrase_factor_1_noop():
    pairs = [
        CodeProblem(
            question="Write a function to do something small.",
            expression="def FN():",
        ),
    ]
    augmented = _paraphrase_augment(pairs, factor=1, seed=0)
    # factor=1 means "only the original" — factor-1 = 0 extra
    assert len(augmented) == 1
