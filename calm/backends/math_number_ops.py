"""
CALM Number operations backend — floor/ceil, rounding, number bases, numeric properties.

Models approximate rounding, confuse floor vs truncate. Pure computation.
"""

from __future__ import annotations

import math


def floor(x: float) -> int:
    """Floor: largest integer ≤ x."""
    return math.floor(float(x))


def ceil(x: float) -> int:
    """Ceiling: smallest integer ≥ x."""
    return math.ceil(float(x))


def truncate(x: float) -> int:
    """Truncate toward zero (not same as floor for negatives)."""
    return int(float(x))


def round_half_up(x: float, decimals: int = 0) -> float:
    """Round half up (school rounding). 0.5 → 1, -0.5 → 0."""
    d = int(decimals)
    mult = 10 ** d
    return math.floor(float(x) * mult + 0.5) / mult


def round_half_even(x: float, decimals: int = 0) -> float:
    """Round half to even (banker's rounding). Python's default round()."""
    return round(float(x), int(decimals))


def significant_figures(x: float, n: int) -> float:
    """Round to n significant figures."""
    x = float(x)
    if x == 0:
        return 0.0
    n = int(n)
    return round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))


def is_integer(x: float) -> bool:
    """Whether x is an integer (no fractional part)."""
    return float(x) == int(float(x))


def sign(x: float) -> int:
    """Sign of x: -1, 0, or 1."""
    x = float(x)
    if x > 0: return 1
    if x < 0: return -1
    return 0


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp x to [lo, hi]."""
    return max(float(lo), min(float(hi), float(x)))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation: a + (b - a) × t."""
    return round(float(a) + (float(b) - float(a)) * float(t), 6)


def inverse_lerp(a: float, b: float, value: float) -> float:
    """Inverse lerp: t such that lerp(a, b, t) = value."""
    a, b = float(a), float(b)
    if a == b:
        return 0.0
    return round((float(value) - a) / (b - a), 6)


def map_range(x: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """Map x from [in_min, in_max] to [out_min, out_max]."""
    t = inverse_lerp(in_min, in_max, x)
    return lerp(out_min, out_max, t)


def is_power_of_two(n: int) -> bool:
    """Whether n is a power of 2."""
    n = int(n)
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_two(n: int) -> int:
    """Smallest power of 2 ≥ n."""
    n = int(n)
    if n <= 0:
        return 1
    if is_power_of_two(n):
        return n
    return 1 << (n - 1).bit_length()


def log_base(x: float, base: float) -> float:
    """Logarithm of x in arbitrary base."""
    return round(math.log(float(x)) / math.log(float(base)), 6)


def geometric_mean(values: list) -> float:
    """Geometric mean of a list of positive numbers."""
    vals = [float(v) for v in values]
    if any(v <= 0 for v in vals):
        return -1.0
    return round(math.exp(sum(math.log(v) for v in vals) / len(vals)), 6)


def harmonic_mean(values: list) -> float:
    """Harmonic mean: n / Σ(1/xi)."""
    vals = [float(v) for v in values]
    if any(v == 0 for v in vals):
        return 0.0
    return round(len(vals) / sum(1.0 / v for v in vals), 6)


def weighted_average(values: list, weights: list) -> float:
    """Weighted average: Σ(vi × wi) / Σ(wi)."""
    vals = [float(v) for v in values]
    wts = [float(w) for w in weights]
    total_weight = sum(wts)
    if total_weight == 0:
        return 0.0
    return round(sum(v * w for v, w in zip(vals, wts)) / total_weight, 6)


def number_of_digits(n: int) -> int:
    """Count digits in an integer."""
    n = abs(int(n))
    if n == 0:
        return 1
    return int(math.log10(n)) + 1


def sum_of_digits(n: int) -> int:
    """Sum of digits of an integer."""
    return sum(int(d) for d in str(abs(int(n))))


def reverse_number(n: int) -> int:
    """Reverse digits of an integer."""
    n = int(n)
    neg = n < 0
    result = int(str(abs(n))[::-1])
    return -result if neg else result


def is_armstrong(n: int) -> bool:
    """Whether n is an Armstrong/narcissistic number (sum of digits^len = n)."""
    n = int(n)
    digits = str(abs(n))
    k = len(digits)
    return sum(int(d) ** k for d in digits) == abs(n)


MATH_NUMBER_FUNCTIONS = {
    "floor": floor,
    "ceil": ceil,
    "truncate": truncate,
    "round_half_up": round_half_up,
    "round_half_even": round_half_even,
    "significant_figures": significant_figures,
    "is_integer": is_integer,
    "sign": sign,
    "clamp": clamp,
    "lerp": lerp,
    "inverse_lerp": inverse_lerp,
    "map_range": map_range,
    "is_power_of_two": is_power_of_two,
    "next_power_of_two": next_power_of_two,
    "log_base": log_base,
    "geometric_mean": geometric_mean,
    "harmonic_mean": harmonic_mean,
    "weighted_average": weighted_average,
    "number_of_digits": number_of_digits,
    "sum_of_digits": sum_of_digits,
    "reverse_number": reverse_number,
    "is_armstrong": is_armstrong,
}

MATH_NUMBER_NL_PATTERNS = [
    (r'floor\s+(?:of\s+)?([-\d.]+)', 'floor({0})'),
    (r'ceil(?:ing)?\s+(?:of\s+)?([-\d.]+)', 'ceil({0})'),
    (r'round\s+([-\d.]+)\s+to\s+(\d+)\s+(?:decimal|sig)', 'round_half_up({0}, {1})'),
    (r'(?:is)\s+(\d+)\s+(?:a\s+)?power\s+of\s+(?:two|2)', 'is_power_of_two({0})'),
    (r'next\s+power\s+of\s+(?:two|2)\s+(?:after|>=|above)\s+(\d+)', 'next_power_of_two({0})'),
    (r'log\s+base\s+(\d+)\s+(?:of\s+)?([\d.]+)', 'log_base({1}, {0})'),
    (r'(?:geometric|harmonic)\s+mean\s+(?:of\s+)?\[([-\d.,\s]+)\]', None),
    (r'sum\s+(?:of\s+)?digits?\s+(?:of|in)\s+(\d+)', 'sum_of_digits({0})'),
    (r'(?:is)\s+(\d+)\s+(?:an?\s+)?(?:armstrong|narcissistic)', 'is_armstrong({0})'),
    (r'(?:how many|number of|count)\s+digits?\s+(?:in|of)\s+(\d+)', 'number_of_digits({0})'),
    (r'weighted\s+average', None),
]
