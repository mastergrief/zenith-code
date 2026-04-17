"""Unit tests for CalmVerifier — exercise the CALM-as-oracle contract."""

import pytest

from calm.llm_computer.calm_verifier import (
    CalmVerifier, make_key, nl_to_expression,
)


# ---------- NL→expression translation ----------

@pytest.mark.parametrize("prompt,expected_expr", [
    ("2 plus 3", "2 + 3"),
    ("10 times 4", "10 * 4"),
    ("15 minus 7", "15 - 7"),
    ("20 divided by 4", "20 // 4"),
    ("7 mod 3", "7 % 3"),
    ("is 5 prime", "is_prime(5)"),
    ("Is 391 prime?", "is_prime(391)"),
    ("factorial of 4", "factorial(4)"),
    ("gcd of 12 and 18", "gcd(12, 18)"),
    ("lcm of 4 and 6", "lcm(4, 6)"),
    # Raw operator form
    ("17 * 23", "17 * 23"),
])
def test_nl_to_expression_translates(prompt, expected_expr):
    assert nl_to_expression(prompt) == expected_expr


def test_nl_to_expression_unmatched_returns_none():
    assert nl_to_expression("What is love?") is None
    assert nl_to_expression("Tell me a joke") is None


# ---------- CalmVerifier.verify (raw expression) ----------

def test_verify_arithmetic():
    v = CalmVerifier(max_value=64)
    assert v.verify("2 + 3") == 5
    assert v.verify("7 * 6") == 42
    assert v.verify("18 // 3") == 6


def test_verify_booleans_as_ints():
    v = CalmVerifier(max_value=64)
    assert v.verify("is_prime(7)") == 1
    assert v.verify("is_prime(391)") == 0


def test_verify_number_theory():
    v = CalmVerifier(max_value=64)
    assert v.verify("gcd(12, 18)") == 6
    assert v.verify("lcm(4, 6)") == 12
    assert v.verify("factorial(4)") == 24


def test_verify_out_of_range_returns_none():
    v = CalmVerifier(max_value=8)
    assert v.verify("2 + 3") == 5
    assert v.verify("5 + 5") is None  # 10 >= 8
    assert v.verify("factorial(4)") is None  # 24 >= 8


def test_verify_syntax_error_returns_none():
    v = CalmVerifier(max_value=64)
    assert v.verify("not a real expression") is None
    assert v.verify("1 / 0") is None


# ---------- CalmVerifier.verify_nl (NL → expr → value) ----------

@pytest.mark.parametrize("prompt,expected_value", [
    ("What is 3 plus 4?", 7),
    ("5 times 6", 30),
    ("Is 7 prime?", 1),
    ("Is 391 prime?", 0),
    ("gcd of 12 and 18", 6),
    ("factorial of 3", 6),
])
def test_verify_nl_end_to_end(prompt, expected_value):
    v = CalmVerifier(max_value=64)
    expr, value = v.verify_nl(prompt)
    assert value == expected_value, \
        f"prompt={prompt!r} expr={expr!r} got={value!r}"


def test_verify_nl_unknown_domain():
    v = CalmVerifier(max_value=64)
    expr, value = v.verify_nl("What's the weather?")
    assert expr is None
    assert value is None


# ---------- Key hashing ----------

def test_make_key_deterministic():
    k1 = make_key("2 plus 3")
    k2 = make_key("2 plus 3")
    assert k1 == k2


def test_make_key_distinct():
    k1 = make_key("2 plus 3")
    k2 = make_key("3 plus 2")
    # Collisions are possible in principle but vanishingly unlikely
    # for these two prompts with a 1024-wide hash space.
    assert k1 != k2


def test_make_key_bounded():
    for i in range(100):
        k = make_key(f"test prompt {i}", max_key=64)
        assert 0 <= k < 64
