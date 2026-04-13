"""
CALM Ratio/Fraction backend — simplify, convert, compare.

Models round fractions wrong and mess up percentage conversions.
"""

from __future__ import annotations

import math


def simplify_fraction(numerator: int, denominator: int) -> str:
    """Simplify a fraction to lowest terms."""
    n, d = int(numerator), int(denominator)
    if d == 0:
        return "undefined (division by zero)"
    g = math.gcd(abs(n), abs(d))
    n, d = n // g, d // g
    if d < 0:
        n, d = -n, -d
    if d == 1:
        return str(n)
    return f"{n}/{d}"


def fraction_to_decimal(numerator: int, denominator: int) -> float:
    """Convert fraction to decimal."""
    d = int(denominator)
    if d == 0:
        return float('inf')
    return round(int(numerator) / d, 10)


def decimal_to_fraction(decimal: float) -> str:
    """Convert decimal to simplified fraction."""
    d = float(decimal)
    # Handle simple cases
    if d == int(d):
        return str(int(d))
    # Find denominator by multiplying until we get an integer
    for denom in range(2, 10001):
        numer = d * denom
        if abs(numer - round(numer)) < 1e-9:
            return simplify_fraction(round(numer), denom)
    return f"~{d}"


def percent_to_decimal(percent: float) -> float:
    """Convert percentage to decimal (50% → 0.5)."""
    return float(percent) / 100


def decimal_to_percent(decimal: float) -> float:
    """Convert decimal to percentage (0.5 → 50%)."""
    return round(float(decimal) * 100, 4)


def fraction_to_percent(numerator: int, denominator: int) -> float:
    """Convert fraction to percentage."""
    d = int(denominator)
    if d == 0:
        return float('inf')
    return round(int(numerator) / d * 100, 4)


def percent_change(old_value: float, new_value: float) -> float:
    """Percentage change from old to new."""
    old = float(old_value)
    if old == 0:
        return float('inf')
    return round((float(new_value) - old) / abs(old) * 100, 4)


def ratio_simplify(a: int, b: int) -> str:
    """Simplify a ratio (e.g., 6:4 → 3:2)."""
    a, b = int(a), int(b)
    g = math.gcd(abs(a), abs(b))
    return f"{a // g}:{b // g}"


def proportion_solve(a: float, b: float, c: float) -> float:
    """Solve proportion a/b = c/x. Returns x."""
    a_val, b_val, c_val = float(a), float(b), float(c)
    if a_val == 0:
        return float('inf')
    return round(b_val * c_val / a_val, 6)


RATIO_FUNCTIONS = {
    "simplify_fraction": simplify_fraction,
    "fraction_to_decimal": fraction_to_decimal,
    "decimal_to_fraction": decimal_to_fraction,
    "percent_to_decimal": percent_to_decimal,
    "decimal_to_percent": decimal_to_percent,
    "fraction_to_percent": fraction_to_percent,
    "percent_change": percent_change,
    "ratio_simplify": ratio_simplify,
    "proportion_solve": proportion_solve,
}

RATIO_NL_PATTERNS = [
    (r'simplify\s+(\d+)/(\d+)', 'simplify_fraction({0}, {1})'),
    (r'(\d+)/(\d+)\s+(?:as|to|in)\s+(?:a\s+)?(?:decimal|percent)', 'fraction_to_decimal({0}, {1})'),
    (r'(?:percent|percentage)\s+change\s+(?:from\s+)?([\d.]+)\s+to\s+([\d.]+)', 'percent_change({0}, {1})'),
    (r'simplify.*?ratio\s+(\d+)\s*:\s*(\d+)', 'ratio_simplify({0}, {1})'),
    (r'([\d.]+)%\s+(?:as|to|in)\s+(?:a\s+)?decimal', 'percent_to_decimal({0})'),
]
