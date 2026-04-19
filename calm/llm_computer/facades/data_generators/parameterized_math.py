"""ParameterizedMathGenerator — heavy parameter sweeps of math primitives.

Each template function yields many verified variants by sweeping over
a parameter axis (divisor, base, exponent, modulus, ...). All
solutions are sandbox-executable and deterministic.

Examples:
  - is_divisible_by_K for K ∈ primes and small composites (2-97)
  - multiples_up_to_N for K ∈ {3, 5, 7, 11, 13}
  - power_base_exp for base ∈ {2, 3, 5, 7} × exp ∈ {2, 3, 4, 5}
  - count_of_primes_below_N
  - gcd_with(K) for small K
"""

from __future__ import annotations

import random
from typing import List, Tuple

from calm.llm_computer.facades.data_generators import register_generator
from calm.llm_computer.facades.data_generators.base import (
    DomainDataGenerator,
    VerifiedExample,
)


# Primes up to 100 for divisibility sweep
_PRIMES_100 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
               53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
_SMALL_COMPOSITES = [4, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30]


def _sweep_is_divisible() -> List[VerifiedExample]:
    """is_divisible_by_K(n) for K over primes + small composites."""
    out: List[VerifiedExample] = []
    for k in _PRIMES_100 + _SMALL_COMPOSITES:
        fn = f"is_divisible_by_{k}"
        # Test set: zero, exact multiples positive + negative, near-miss, k-1
        test_cases = [
            (0, True),
            (k, True),
            (k * 2, True),
            (k * 3, True),
            (-k * 4, True),
            (1, k == 1),
            (k - 1, k == 1),
            (k + 1, False),
        ]
        out.append(VerifiedExample(
            problem=(
                f"Write a Python function `{fn}(n)` that returns True if "
                f"n is divisible by {k}, False otherwise. Handle negative n correctly."
            ),
            signature=f"def {fn}(n):",
            solution=f"def {fn}(n):\n    return n % {k} == 0\n",
            test_cases=test_cases,
            reasoning="",
            algorithm=f"integer modulo by {k}",
            complexity="O(1)",
            edge_cases=["zero is divisible", "negatives work with %", "k=1 makes all ints divisible"],
            category="param_math_divisibility",
            generator_name="param_math",
        ))
    return out


def _sweep_multiples_up_to_n() -> List[VerifiedExample]:
    """multiples_of_K_below(limit) for K ∈ small primes and composites."""
    out: List[VerifiedExample] = []
    for k in [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 20]:
        fn = f"multiples_of_{k}_below"
        # Test with various limits
        def _gen_tests(k):
            tests = []
            for lim in [0, 1, k, k + 1, 3 * k + 1, 50, 100]:
                expected = [i for i in range(k, lim, k)]
                tests.append((lim, expected))
            return tests
        out.append(VerifiedExample(
            problem=(
                f"Write a Python function `{fn}(limit)` that returns a list "
                f"of positive multiples of {k} strictly less than limit. "
                f"Empty list if limit <= {k}."
            ),
            signature=f"def {fn}(limit):",
            solution=(
                f"def {fn}(limit):\n"
                f"    return list(range({k}, limit, {k}))\n"
            ),
            test_cases=_gen_tests(k),
            reasoning="",
            algorithm=f"range(step={k}) iteration",
            complexity="O(limit / k)",
            edge_cases=["limit <= k returns empty", "strictly less than (< not <=)"],
            category="param_math_multiples",
            generator_name="param_math",
        ))
    return out


def _sweep_power_base_exp() -> List[VerifiedExample]:
    """power_of_base_B(exp) for small B using repeated squaring."""
    out: List[VerifiedExample] = []
    for b in [2, 3, 5, 7, 10]:
        fn = f"power_of_{b}"
        test_cases: List[Tuple] = []
        for e in [0, 1, 2, 3, 5, 10]:
            test_cases.append((e, b ** e))
        out.append(VerifiedExample(
            problem=(
                f"Write a Python function `{fn}(exp)` that returns {b}^exp "
                f"for non-negative exp, using binary exponentiation "
                f"(repeated squaring). Do not use the ** operator."
            ),
            signature=f"def {fn}(exp):",
            solution=(
                f"def {fn}(exp):\n"
                f"    if exp < 0:\n"
                f"        raise ValueError('exp must be non-negative')\n"
                f"    base = {b}\n"
                f"    result = 1\n"
                f"    while exp:\n"
                f"        if exp & 1:\n"
                f"            result *= base\n"
                f"        base *= base\n"
                f"        exp >>= 1\n"
                f"    return result\n"
            ),
            test_cases=test_cases,
            reasoning="",
            algorithm="binary exponentiation (repeated squaring)",
            complexity="O(log exp)",
            edge_cases=["exp = 0 returns 1", "exp = 1 returns base", "negative exp raises"],
            category="param_math_power",
            generator_name="param_math",
        ))
    return out


def _sweep_gcd_with() -> List[VerifiedExample]:
    """gcd_with_K(n) — specialize GCD for a fixed K."""
    import math
    out: List[VerifiedExample] = []
    for k in [2, 3, 4, 6, 7, 8, 10, 12, 15, 16, 18, 24, 30, 60, 120]:
        fn = f"gcd_with_{k}"
        test_cases: List[Tuple] = []
        for n in [0, 1, k, k * 2, k + 1, 100, 256, 999]:
            test_cases.append((n, math.gcd(n, k)))
        out.append(VerifiedExample(
            problem=(
                f"Write a Python function `{fn}(n)` that returns gcd(n, {k}). "
                f"Use the Euclidean algorithm; handle zero correctly."
            ),
            signature=f"def {fn}(n):",
            solution=(
                f"def {fn}(n):\n"
                f"    a, b = abs(n), {k}\n"
                f"    while b:\n"
                f"        a, b = b, a % b\n"
                f"    return a\n"
            ),
            test_cases=test_cases,
            reasoning="",
            algorithm=f"Euclidean gcd specialized for {k}",
            complexity="O(log min(n, k))",
            edge_cases=["gcd(0, k) = k", "negative n (use abs)"],
            category="param_math_gcd",
            generator_name="param_math",
        ))
    return out


def _sweep_sum_digits_base() -> List[VerifiedExample]:
    """digit_sum_base_B(n) — digit sum in different bases."""
    out: List[VerifiedExample] = []
    for b in [2, 8, 10, 16]:
        fn = f"digit_sum_base_{b}"
        def _gen_tests(b):
            tests = []
            for n in [0, 1, b, b * b, 100, 255, 1023]:
                s = 0
                x = n
                while x:
                    s += x % b
                    x //= b
                tests.append((n, s))
            return tests
        out.append(VerifiedExample(
            problem=(
                f"Write a Python function `{fn}(n)` that returns the sum of "
                f"digits of a non-negative integer n when written in base {b}. "
                f"Example: {fn}(255) interprets 255 in base {b} and sums digits."
            ),
            signature=f"def {fn}(n):",
            solution=(
                f"def {fn}(n):\n"
                f"    total = 0\n"
                f"    while n:\n"
                f"        total += n % {b}\n"
                f"        n //= {b}\n"
                f"    return total\n"
            ),
            test_cases=_gen_tests(b),
            reasoning="",
            algorithm=f"mod-{b} digit extraction",
            complexity=f"O(log_{b} n)",
            edge_cases=["n = 0 returns 0", f"base {b} digits are 0..{b-1}"],
            category="param_math_digits",
            generator_name="param_math",
        ))
    return out


def _sweep_count_occurrences_char() -> List[VerifiedExample]:
    """count_X_in_string(s) for various X values."""
    out: List[VerifiedExample] = []
    for ch, name in [
        ('a', 'a'), ('e', 'e'), ('o', 'o'), ('z', 'z'),
        (' ', 'spaces'), ('.', 'periods'), (',', 'commas'),
        ('0', 'zeros'), ('1', 'ones'),
    ]:
        fn = f"count_{name}"
        def _gen_tests(ch):
            tests = []
            for s in ["", ch, ch * 5, "hello world", "abc" + ch + "def", "no match here"]:
                tests.append((s, s.count(ch)))
            return tests
        out.append(VerifiedExample(
            problem=(
                f"Write a Python function `{fn}(s)` that returns the number "
                f"of {name!r} characters in s. Case-sensitive."
            ),
            signature=f"def {fn}(s):",
            solution=(
                f"def {fn}(s):\n"
                f"    return s.count({ch!r})\n"
            ),
            test_cases=_gen_tests(ch),
            reasoning="",
            algorithm=f"str.count({ch!r})",
            complexity="O(n)",
            edge_cases=["empty string", "no matches", "target char repeated"],
            category="param_math_count_char",
            generator_name="param_math",
        ))
    return out


def _all_sweeps() -> List[VerifiedExample]:
    out: List[VerifiedExample] = []
    out.extend(_sweep_is_divisible())
    out.extend(_sweep_multiples_up_to_n())
    out.extend(_sweep_power_base_exp())
    out.extend(_sweep_gcd_with())
    out.extend(_sweep_sum_digits_base())
    out.extend(_sweep_count_occurrences_char())
    return out


class ParameterizedMathGenerator(DomainDataGenerator):
    """Heavy parameter sweeps of math primitives. Produces many
    verified variants by iterating over divisors / bases / exponents /
    character targets. All solutions are small + deterministic + fully
    sandbox-executable."""

    name = "param_math"

    def __init__(self, rng=None):
        super().__init__(rng)
        self._sweeps = _all_sweeps()

    def generate_raw(self, n: int) -> List[VerifiedExample]:
        self.rng.shuffle(self._sweeps)
        return self._sweeps[:n]


register_generator("param_math", ParameterizedMathGenerator)
