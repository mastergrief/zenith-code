"""CPU exactness fence for `_truncate_toward_zero_division`.

Guards the float64 truncation path against >2^53 precision loss while proving
behavior preservation on the admitted decay domain (acc ∈ [-127,127], etc.).
"""
from __future__ import annotations

import math

import pytest

from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    _truncate_toward_zero_division,
)


def _exact_oracle(numerator: int, denominator: int) -> int:
    n = int(numerator)
    d = int(denominator)
    if d <= 0:
        raise ValueError("denominator must be > 0")
    q = abs(n) // d
    return q if n >= 0 else -q


def _legacy_float_trunc(numerator: int, denominator: int) -> int:
    if int(denominator) <= 0:
        raise ValueError("denominator must be > 0")
    return int(math.trunc(float(int(numerator)) / float(int(denominator))))


def test_denominator_guard_raises() -> None:
    with pytest.raises(ValueError, match="denominator must be > 0"):
        _truncate_toward_zero_division(1, 0)
    with pytest.raises(ValueError, match="denominator must be > 0"):
        _truncate_toward_zero_division(1, -1)


def test_exhaustive_admitted_domain_zero_mismatches() -> None:
    mismatches = 0
    legacy_mismatches = 0
    cases = 0
    for acc in range(-127, 128):
        for num in range(0, 257):
            for den in range(1, 257):
                numerator = acc * num
                got = _truncate_toward_zero_division(numerator, den)
                want = _exact_oracle(numerator, den)
                legacy = _legacy_float_trunc(numerator, den)
                cases += 1
                if got != want:
                    mismatches += 1
                    if mismatches <= 5:
                        pytest.fail(
                            f"oracle mismatch at acc={acc} num={num} den={den}: "
                            f"got={got} want={want}"
                        )
                if legacy != got:
                    legacy_mismatches += 1
                    if legacy_mismatches <= 5:
                        pytest.fail(
                            f"legacy mismatch at acc={acc} num={num} den={den}: "
                            f"got={got} legacy={legacy}"
                        )
    assert cases == 16_776_960
    assert mismatches == 0
    assert legacy_mismatches == 0


def test_above_float53_regression_vectors_match_oracle_not_legacy() -> None:
    vectors = [
        (2**53 + 1, 3, 3002399751580331),
        (3 * 2**53 + 5, 3, 9007199254740993),
        (10**17 + 1, 7, 14285714285714285),
    ]
    for numerator, denominator, exact in vectors:
        assert _exact_oracle(numerator, denominator) == exact
        assert _truncate_toward_zero_division(numerator, denominator) == exact
        legacy = _legacy_float_trunc(numerator, denominator)
        assert legacy != exact, (
            f"legacy unexpectedly exact for ({numerator},{denominator})"
        )


def test_true_negative_control_legacy_and_exact_agree() -> None:
    numerator = 2**53 + 3
    denominator = 7
    exact = _exact_oracle(numerator, denominator)
    legacy = _legacy_float_trunc(numerator, denominator)
    got = _truncate_toward_zero_division(numerator, denominator)
    assert exact == legacy
    assert got == exact


def test_negative_numerator_trunc_toward_zero() -> None:
    cases = [
        (-31, 32, 0),
        (-32, 32, -1),
        (-63, 32, -1),
        (-64, 32, -2),
        (-127 * 31, 32, -123),
        (-1, 2, 0),
        (-3, 2, -1),
        (-5, 3, -1),
        (-100, 7, -14),
    ]
    for numerator, denominator, want in cases:
        assert _exact_oracle(numerator, denominator) == want
        assert _truncate_toward_zero_division(numerator, denominator) == want
