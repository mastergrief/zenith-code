"""
CALM Calculus backend — limits, derivatives, integrals, Taylor series.

Models approximate calculus. Numerical methods give exact-enough answers.
"""

from __future__ import annotations

import math


def limit(f_str: str, x_val: float, h: float = 1e-10) -> float:
    """Numerical limit of expression as x approaches x_val.
    f_str should use 'x' as variable. E.g. limit('(x**2-1)/(x-1)', 1)."""
    # Evaluate from both sides
    from calm.expression import safe_eval
    try:
        left = safe_eval(f_str.replace('x', f'({float(x_val) - h})'))
        right = safe_eval(f_str.replace('x', f'({float(x_val) + h})'))
        if abs(left - right) < 1e-6:
            return round((left + right) / 2, 6)
        return round((left + right) / 2, 6)  # average even if slightly different
    except Exception:
        return float('nan')


def derivative(f_str: str, x_val: float, h: float = 1e-8) -> float:
    """Numerical derivative f'(x) using central difference.
    f_str should use 'x'. E.g. derivative('x**2', 3) → 6.0."""
    from calm.expression import safe_eval
    x = float(x_val)
    try:
        f_plus = safe_eval(f_str.replace('x', f'({x + h})'))
        f_minus = safe_eval(f_str.replace('x', f'({x - h})'))
        return round((f_plus - f_minus) / (2 * h), 6)
    except Exception:
        return float('nan')


def second_derivative(f_str: str, x_val: float, h: float = 1e-5) -> float:
    """Numerical second derivative f''(x)."""
    from calm.expression import safe_eval
    x = float(x_val)
    try:
        f_plus = safe_eval(f_str.replace('x', f'({x + h})'))
        f_0 = safe_eval(f_str.replace('x', f'({x})'))
        f_minus = safe_eval(f_str.replace('x', f'({x - h})'))
        return round((f_plus - 2 * f_0 + f_minus) / (h ** 2), 4)
    except Exception:
        return float('nan')


def integral(f_str: str, a: float, b: float, n: int = 1000) -> float:
    """Numerical definite integral using Simpson's rule.
    E.g. integral('x**2', 0, 1) → 0.333333."""
    from calm.expression import safe_eval
    a, b, n = float(a), float(b), max(int(n), 2)
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    try:
        total = safe_eval(f_str.replace('x', f'({a})')) + safe_eval(f_str.replace('x', f'({b})'))
        for i in range(1, n):
            xi = a + i * h
            coeff = 4 if i % 2 == 1 else 2
            total += coeff * safe_eval(f_str.replace('x', f'({xi})'))
        return round(total * h / 3, 6)
    except Exception:
        return float('nan')


def taylor_exp(x: float, terms: int = 10) -> float:
    """Taylor series for e^x: sum(x^n/n!, n=0..terms)."""
    x = float(x)
    result = 0.0
    for n in range(int(terms)):
        result += x ** n / math.factorial(n)
    return round(result, 6)


def taylor_sin(x: float, terms: int = 10) -> float:
    """Taylor series for sin(x): sum((-1)^n × x^(2n+1) / (2n+1)!, n=0..terms)."""
    x = float(x)
    result = 0.0
    for n in range(int(terms)):
        result += ((-1) ** n) * (x ** (2 * n + 1)) / math.factorial(2 * n + 1)
    return round(result, 6)


def taylor_cos(x: float, terms: int = 10) -> float:
    """Taylor series for cos(x): sum((-1)^n × x^(2n) / (2n)!, n=0..terms)."""
    x = float(x)
    result = 0.0
    for n in range(int(terms)):
        result += ((-1) ** n) * (x ** (2 * n)) / math.factorial(2 * n)
    return round(result, 6)


def taylor_ln1px(x: float, terms: int = 20) -> float:
    """Taylor series for ln(1+x): sum((-1)^(n+1) × x^n / n, n=1..terms). |x| ≤ 1."""
    x = float(x)
    if abs(x) > 1:
        return float('nan')
    result = 0.0
    for n in range(1, int(terms) + 1):
        result += ((-1) ** (n + 1)) * (x ** n) / n
    return round(result, 6)


def riemann_sum(f_str: str, a: float, b: float, n: int = 100, method: str = "midpoint") -> float:
    """Riemann sum approximation of integral. Methods: left, right, midpoint."""
    from calm.expression import safe_eval
    a, b, n = float(a), float(b), int(n)
    dx = (b - a) / n
    total = 0.0
    for i in range(n):
        if method == "left":
            xi = a + i * dx
        elif method == "right":
            xi = a + (i + 1) * dx
        else:  # midpoint
            xi = a + (i + 0.5) * dx
        total += safe_eval(f_str.replace('x', f'({xi})'))
    return round(total * dx, 6)


def is_increasing(f_str: str, a: float, b: float, samples: int = 10) -> bool:
    """Test if f is increasing on [a,b] by sampling."""
    from calm.expression import safe_eval
    a, b = float(a), float(b)
    prev = None
    for i in range(int(samples) + 1):
        xi = a + (b - a) * i / samples
        val = safe_eval(f_str.replace('x', f'({xi})'))
        if prev is not None and val < prev - 1e-10:
            return False
        prev = val
    return True


def critical_points(f_str: str, a: float, b: float, samples: int = 100) -> list:
    """Find approximate critical points where f'(x) ≈ 0 on [a,b]."""
    a, b = float(a), float(b)
    points = []
    for i in range(int(samples)):
        xi = a + (b - a) * i / samples
        d = derivative(f_str, xi)
        if abs(d) < 0.01:
            points.append(round(xi, 4))
    # Deduplicate nearby points
    if not points:
        return points
    deduped = [points[0]]
    for p in points[1:]:
        if abs(p - deduped[-1]) > 0.1:
            deduped.append(p)
    return deduped


CALCULUS_FUNCTIONS = {
    "limit": limit,
    "derivative": derivative,
    "second_derivative": second_derivative,
    "integral": integral,
    "taylor_exp": taylor_exp,
    "taylor_sin": taylor_sin,
    "taylor_cos": taylor_cos,
    "taylor_ln1px": taylor_ln1px,
    "riemann_sum": riemann_sum,
    "is_increasing": is_increasing,
    "critical_points": critical_points,
}

CALCULUS_NL_PATTERNS = [
    (r'derivative\s+of\s+(.+?)\s+at\s+x\s*=\s*([-\d.]+)', 'derivative("{0}", {1})'),
    (r'integral\s+(?:of\s+)?(.+?)\s+from\s+([-\d.]+)\s+to\s+([-\d.]+)', 'integral("{0}", {1}, {2})'),
    (r'limit\s+(?:of\s+)?(.+?)\s+as\s+x\s*(?:→|->|approaches?)\s*([-\d.]+)', 'limit("{0}", {1})'),
    (r'taylor\s+(?:series\s+)?(?:of\s+)?e\^([-\d.]+)', 'taylor_exp({0})'),
    (r'taylor\s+(?:series\s+)?(?:of\s+)?sin\(([-\d.]+)\)', 'taylor_sin({0})'),
    (r'taylor\s+(?:series\s+)?(?:of\s+)?cos\(([-\d.]+)\)', 'taylor_cos({0})'),
]
