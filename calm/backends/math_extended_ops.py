"""
CALM extended math backend — linear algebra, calculus, number bases.

Sub-backend for math_ops: covers domains where 4B models commonly
hallucinate — matrix operations, numerical calculus, base conversion.

Functions: matrix ops, numerical derivatives/integrals, base conversion,
modular arithmetic, combinatorial identities.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Union


# ---------------------------------------------------------------------------
# Number base conversion
# ---------------------------------------------------------------------------

def to_base(n: int, base: int) -> str:
    """Convert integer to string in given base (2-36)."""
    n, base = int(n), int(base)
    if base < 2 or base > 36:
        return "error: base must be 2-36"
    if n == 0:
        return "0"
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    negative = n < 0
    n = abs(n)
    result = []
    while n:
        result.append(digits[n % base])
        n //= base
    if negative:
        result.append('-')
    return ''.join(reversed(result))


def from_base(s: str, base: int) -> int:
    """Convert string in given base to integer."""
    return int(str(s), int(base))


def to_binary(n: int) -> str:
    """Convert to binary string."""
    return bin(int(n))[2:]


def to_hex(n: int) -> str:
    """Convert to hex string."""
    return hex(int(n))[2:]


def to_octal(n: int) -> str:
    """Convert to octal string."""
    return oct(int(n))[2:]


# ---------------------------------------------------------------------------
# Modular arithmetic
# ---------------------------------------------------------------------------

def mod_pow(base: int, exp: int, mod: int) -> int:
    """Modular exponentiation: base^exp mod mod."""
    return pow(int(base), int(exp), int(mod))


def mod_inverse(a: int, m: int) -> int:
    """Modular multiplicative inverse: a^(-1) mod m."""
    a, m = int(a), int(m)
    g = math.gcd(a, m)
    if g != 1:
        return -1  # No inverse exists.
    return pow(a, -1, m)


def chinese_remainder(remainders: list, moduli: list) -> int:
    """Chinese Remainder Theorem: find x such that x ≡ r_i (mod m_i).
    Example: chinese_remainder([2, 3, 2], [3, 5, 7]) → 23"""
    if len(remainders) != len(moduli):
        return -1
    N = 1
    for m in moduli:
        N *= m
    result = 0
    for r, m in zip(remainders, moduli):
        Ni = N // m
        yi = mod_inverse(Ni, m)
        if yi == -1:
            return -1
        result += r * Ni * yi
    return result % N


# ---------------------------------------------------------------------------
# Matrix operations (lists of lists)
# ---------------------------------------------------------------------------

def matrix_multiply(a: list, b: list) -> list:
    """Multiply two matrices (as lists of lists)."""
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])
    if cols_a != rows_b:
        return [["error: incompatible dimensions"]]
    result = [[0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    return result


def matrix_determinant(m: list) -> float:
    """Determinant of a square matrix."""
    n = len(m)
    if n == 1:
        return m[0][0]
    if n == 2:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]
    det = 0
    for j in range(n):
        minor = [row[:j] + row[j+1:] for row in m[1:]]
        det += ((-1) ** j) * m[0][j] * matrix_determinant(minor)
    return det


def matrix_transpose(m: list) -> list:
    """Transpose a matrix."""
    return [list(row) for row in zip(*m)]


def dot_product(a: list, b: list) -> float:
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))


def cross_product(a: list, b: list) -> list:
    """Cross product of two 3D vectors."""
    if len(a) != 3 or len(b) != 3:
        return ["error: need 3D vectors"]
    return [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ]


# ---------------------------------------------------------------------------
# Numerical calculus (approximations)
# ---------------------------------------------------------------------------

def numerical_derivative(expr_str: str, x: float, h: float = 1e-8) -> float:
    """Numerical derivative of an expression at x.
    Uses central difference: f'(x) ≈ (f(x+h) - f(x-h)) / 2h.

    Example: numerical_derivative("x**2", 3) → 6.0
    """
    from calm.expression import safe_eval, _FUNCTIONS
    fns = dict(_FUNCTIONS)
    fns["x"] = float(x) + float(h)
    f_plus = safe_eval(expr_str, fns)
    fns["x"] = float(x) - float(h)
    f_minus = safe_eval(expr_str, fns)
    return round((f_plus - f_minus) / (2 * float(h)), 8)


def numerical_integral(expr_str: str, a: float, b: float, n: int = 1000) -> float:
    """Numerical integral using Simpson's rule.
    Example: numerical_integral("x**2", 0, 1) → 0.333...
    """
    from calm.expression import safe_eval, _FUNCTIONS
    a, b, n = float(a), float(b), int(n)
    if n % 2 == 1:
        n += 1
    h = (b - a) / n

    def f(x_val):
        fns = dict(_FUNCTIONS)
        fns["x"] = x_val
        return safe_eval(expr_str, fns)

    total = f(a) + f(b)
    for i in range(1, n):
        x = a + i * h
        total += 4 * f(x) if i % 2 == 1 else 2 * f(x)
    return round(total * h / 3, 8)


MATH_EXTENDED_FUNCTIONS = {
    "to_base": to_base,
    "from_base": from_base,
    "to_binary": to_binary,
    "to_hex": to_hex,
    "to_octal": to_octal,
    "mod_pow": mod_pow,
    "mod_inverse": mod_inverse,
    "chinese_remainder": chinese_remainder,
    "matrix_multiply": matrix_multiply,
    "matrix_determinant": matrix_determinant,
    "matrix_transpose": matrix_transpose,
    "dot_product": dot_product,
    "cross_product": cross_product,
    "numerical_derivative": numerical_derivative,
    "numerical_integral": numerical_integral,
}
