"""Tests for DT skeleton repair — regex rewrites on DT decode output."""
import pytest

from calm.hrm.dt_skeleton_repair import repair_skeleton, _is_valid, _valid_args


# --- Validity helpers ---

def test_valid_args_empty():
    assert _valid_args("") is True


def test_valid_args_single():
    assert _valid_args("n") is True


def test_valid_args_multi():
    assert _valid_args("a, b") is True
    assert _valid_args("a,b") is True


def test_valid_args_self():
    assert _valid_args("self") is True


def test_valid_args_varargs():
    assert _valid_args("*args") is True
    assert _valid_args("**kwargs") is True


def test_valid_args_bad():
    assert _valid_args(",") is False
    assert _valid_args("1x") is False  # can't start with digit
    assert _valid_args("a, , b") is False


# --- Full-skeleton validity ---

def test_is_valid_canonical():
    assert _is_valid("def FN(n):")
    assert _is_valid("def FN():")
    assert _is_valid("def FN(a, b):")
    assert _is_valid("def FN(self):")


def test_is_valid_rejects_malformed():
    assert not _is_valid("d FN(n):")
    assert not _is_valid("def FN(n")
    assert not _is_valid("def(n):")
    assert not _is_valid("FN(n):")


# --- Repair rules: each malformation from R1/R2 eval output ---

def test_already_valid_is_noop():
    assert repair_skeleton("def FN(n):") == "def FN(n):"
    assert repair_skeleton("def FN():") == "def FN():"


def test_repair_missing_ef_in_def():
    """R1 case: 'd FN(n):' → 'def FN(n):'"""
    assert repair_skeleton("d FN(n):") == "def FN(n):"


def test_repair_missing_fn_wrapper_one_arg():
    """R2 case: 'def m, n):' → 'def FN(m, n):'"""
    assert repair_skeleton("def m, n):") == "def FN(m, n):"


def test_repair_unclosed_paren_before_colon():
    """R2 case: 'def FN(sel:' → 'def FN(sel):'"""
    assert repair_skeleton("def FN(sel:") == "def FN(sel):"
    assert repair_skeleton("def FN(dlel:") == "def FN(dlel):"


def test_repair_trailing_comma_before_colon():
    """R2 case: 'def FN(x,:' → 'def FN(x):'"""
    assert repair_skeleton("def FN(x,:") == "def FN(x):"
    assert repair_skeleton("def FN(x,,:") == "def FN(x):"


def test_repair_unrepairable_returns_original():
    """No amount of cleanup produces a valid skeleton."""
    assert repair_skeleton("def):") == "def):"  # too much missing
    # Random gibberish
    assert repair_skeleton("abc xyz") == "abc xyz"


def test_repair_does_not_break_known_rare_classes():
    """Preserve valid rare classes."""
    for skel in ["def FN(s):", "def FN(self):", "def FN(x):",
                 "def FN(xs):", "def FN(a,b):", "def FN(*args):",
                 "def FN(**kwargs):"]:
        assert repair_skeleton(skel) == skel


def test_repair_strips_whitespace():
    assert repair_skeleton("  def FN(n):  ") == "def FN(n):"


def test_repair_does_not_produce_invalid():
    """Invariant: repair_skeleton either returns a valid skeleton or
    returns input unchanged."""
    test_inputs = [
        "d FN(n):", "def m, n):", "def FN(sel:", "def FN(x,:",
        "def):", "", "abc", "def", "def FN",
    ]
    for inp in test_inputs:
        out = repair_skeleton(inp)
        if out != inp and out != inp.strip():
            assert _is_valid(out), (
                f"Repair produced invalid output: {inp!r} → {out!r}"
            )


def test_repair_handles_multi_arg_unclosed():
    assert repair_skeleton("def FN(a, b:") == "def FN(a, b):"


def test_repair_handles_truncated_trailing_comma():
    assert repair_skeleton("def FN(a, b, :") == "def FN(a, b):"
