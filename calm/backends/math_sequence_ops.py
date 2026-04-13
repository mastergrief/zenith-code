"""
CALM Math sequences backend — arithmetic/geometric series, triangular, pentagonal, etc.

Models botch series sums, confuse formulas, miscalculate terms.
"""

from __future__ import annotations

import math


def arithmetic_sum(a1: float, d: float, n: int) -> float:
    """Sum of arithmetic series: n/2 × (2a1 + (n-1)d)."""
    a, delta, count = float(a1), float(d), int(n)
    return round(count / 2 * (2 * a + (count - 1) * delta), 4)


def arithmetic_nth(a1: float, d: float, n: int) -> float:
    """Nth term of arithmetic sequence: a1 + (n-1)d."""
    return round(float(a1) + (int(n) - 1) * float(d), 4)


def geometric_sum(a1: float, r: float, n: int) -> float:
    """Sum of geometric series: a1(1 - r^n) / (1 - r)."""
    a, ratio, count = float(a1), float(r), int(n)
    if ratio == 1:
        return a * count
    return round(a * (1 - ratio ** count) / (1 - ratio), 4)


def geometric_nth(a1: float, r: float, n: int) -> float:
    """Nth term of geometric sequence: a1 × r^(n-1)."""
    return round(float(a1) * float(r) ** (int(n) - 1), 4)


def geometric_infinite_sum(a1: float, r: float) -> float:
    """Sum of infinite geometric series: a1 / (1 - r). Only valid for |r| < 1."""
    ratio = float(r)
    if abs(ratio) >= 1:
        return float('inf')
    return round(float(a1) / (1 - ratio), 6)


def triangular(n: int) -> int:
    """Nth triangular number: n(n+1)/2."""
    n = int(n)
    return n * (n + 1) // 2


def is_triangular(n: int) -> bool:
    """Whether n is a triangular number."""
    n = int(n)
    if n < 0:
        return False
    # n = k(k+1)/2 → k = (-1 + sqrt(1+8n))/2
    k = (-1 + math.sqrt(1 + 8 * n)) / 2
    return k == int(k)


def pentagonal(n: int) -> int:
    """Nth pentagonal number: n(3n-1)/2."""
    n = int(n)
    return n * (3 * n - 1) // 2


def hexagonal(n: int) -> int:
    """Nth hexagonal number: n(2n-1)."""
    n = int(n)
    return n * (2 * n - 1)


def sum_of_squares(n: int) -> int:
    """Sum of squares 1² + 2² + ... + n² = n(n+1)(2n+1)/6."""
    n = int(n)
    return n * (n + 1) * (2 * n + 1) // 6


def sum_of_cubes(n: int) -> int:
    """Sum of cubes 1³ + 2³ + ... + n³ = [n(n+1)/2]²."""
    n = int(n)
    return (n * (n + 1) // 2) ** 2


def harmonic(n: int) -> float:
    """Nth harmonic number: 1 + 1/2 + 1/3 + ... + 1/n."""
    return round(sum(1.0 / i for i in range(1, int(n) + 1)), 6)


def sum_natural(n: int) -> int:
    """Sum of first n natural numbers: n(n+1)/2."""
    n = int(n)
    return n * (n + 1) // 2


MATH_SEQUENCE_FUNCTIONS = {
    "arithmetic_sum": arithmetic_sum,
    "arithmetic_nth": arithmetic_nth,
    "geometric_sum": geometric_sum,
    "geometric_nth": geometric_nth,
    "geometric_infinite_sum": geometric_infinite_sum,
    "triangular": triangular,
    "is_triangular": is_triangular,
    "pentagonal": pentagonal,
    "hexagonal": hexagonal,
    "sum_of_squares": sum_of_squares,
    "sum_of_cubes": sum_of_cubes,
    "harmonic": harmonic,
    "sum_natural": sum_natural,
}

MATH_SEQUENCE_NL_PATTERNS = [
    (r'sum\s+(?:of\s+)?(?:the\s+)?(?:first\s+)?(\d+)\s+(?:natural\s+)?(?:numbers?|integers?)', 'sum_natural({0})'),
    (r'(\d+)(?:th|st|nd|rd)\s+triangular\s+number', 'triangular({0})'),
    (r'(?:is)\s+(\d+)\s+(?:a\s+)?triangular\s+number', 'is_triangular({0})'),
    (r'sum\s+(?:of\s+)?squares?\s+(?:of\s+)?(?:first\s+)?(\d+)', 'sum_of_squares({0})'),
    (r'sum\s+(?:of\s+)?cubes?\s+(?:of\s+)?(?:first\s+)?(\d+)', 'sum_of_cubes({0})'),
    (r'(\d+)(?:th|st|nd|rd)\s+harmonic\s+number', 'harmonic({0})'),
    (r'arithmetic\s+(?:series?\s+)?sum.*?a[_1]?\s*=\s*([\d.]+).*?d\s*=\s*([\d.]+).*?n\s*=\s*(\d+)', 'arithmetic_sum({0}, {1}, {2})'),
    (r'geometric\s+(?:series?\s+)?sum.*?a[_1]?\s*=\s*([\d.]+).*?r\s*=\s*([\d.]+).*?n\s*=\s*(\d+)', 'geometric_sum({0}, {1}, {2})'),
    (r'infinite\s+geometric\s+(?:series?\s+)?sum.*?a\s*=\s*([\d.]+).*?r\s*=\s*([\d.]+)', 'geometric_infinite_sum({0}, {1})'),
]
