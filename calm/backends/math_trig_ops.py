"""
CALM Trigonometry backend — trig functions, identities, inverse trig.

Models approximate trig values. Pure computation.
"""

from __future__ import annotations

import math


def sin_deg(degrees: float) -> float:
    """Sine of angle in degrees."""
    return round(math.sin(math.radians(float(degrees))), 6)


def cos_deg(degrees: float) -> float:
    """Cosine of angle in degrees."""
    return round(math.cos(math.radians(float(degrees))), 6)


def tan_deg(degrees: float) -> float:
    """Tangent of angle in degrees."""
    return round(math.tan(math.radians(float(degrees))), 6)


def asin_deg(value: float) -> float:
    """Inverse sine, result in degrees."""
    return round(math.degrees(math.asin(float(value))), 6)


def acos_deg(value: float) -> float:
    """Inverse cosine, result in degrees."""
    return round(math.degrees(math.acos(float(value))), 6)


def atan_deg(value: float) -> float:
    """Inverse tangent, result in degrees."""
    return round(math.degrees(math.atan(float(value))), 6)


def atan2_deg(y: float, x: float) -> float:
    """Two-argument arctangent, result in degrees."""
    return round(math.degrees(math.atan2(float(y), float(x))), 6)


def csc(degrees: float) -> float:
    """Cosecant (1/sin) of angle in degrees."""
    s = math.sin(math.radians(float(degrees)))
    if abs(s) < 1e-15:
        return float('inf')
    return round(1.0 / s, 6)


def sec(degrees: float) -> float:
    """Secant (1/cos) of angle in degrees."""
    c = math.cos(math.radians(float(degrees)))
    if abs(c) < 1e-15:
        return float('inf')
    return round(1.0 / c, 6)


def cot(degrees: float) -> float:
    """Cotangent (1/tan) of angle in degrees."""
    t = math.tan(math.radians(float(degrees)))
    if abs(t) < 1e-15:
        return float('inf')
    return round(1.0 / t, 6)


def sinh_val(x: float) -> float:
    """Hyperbolic sine."""
    return round(math.sinh(float(x)), 6)


def cosh_val(x: float) -> float:
    """Hyperbolic cosine."""
    return round(math.cosh(float(x)), 6)


def tanh_val(x: float) -> float:
    """Hyperbolic tangent."""
    return round(math.tanh(float(x)), 6)


def degrees_to_radians(degrees: float) -> float:
    """Convert degrees to radians."""
    return round(math.radians(float(degrees)), 6)


def radians_to_degrees(radians: float) -> float:
    """Convert radians to degrees."""
    return round(math.degrees(float(radians)), 6)


def law_of_cosines(a: float, b: float, C_deg: float) -> float:
    """Side c from law of cosines: c² = a² + b² - 2ab·cos(C)."""
    a, b = float(a), float(b)
    C = math.radians(float(C_deg))
    return round(math.sqrt(a**2 + b**2 - 2*a*b*math.cos(C)), 6)


def law_of_sines_angle(a: float, A_deg: float, b: float) -> float:
    """Find angle B from law of sines: sin(B)/b = sin(A)/a. Returns degrees."""
    A = math.radians(float(A_deg))
    ratio = float(b) * math.sin(A) / float(a)
    if abs(ratio) > 1:
        return -1.0  # no solution
    return round(math.degrees(math.asin(ratio)), 6)


def angle_sum_sin(a_deg: float, b_deg: float) -> float:
    """sin(A+B) = sin(A)cos(B) + cos(A)sin(B)."""
    a, b = math.radians(float(a_deg)), math.radians(float(b_deg))
    return round(math.sin(a)*math.cos(b) + math.cos(a)*math.sin(b), 6)


def angle_sum_cos(a_deg: float, b_deg: float) -> float:
    """cos(A+B) = cos(A)cos(B) - sin(A)sin(B)."""
    a, b = math.radians(float(a_deg)), math.radians(float(b_deg))
    return round(math.cos(a)*math.cos(b) - math.sin(a)*math.sin(b), 6)


def double_angle_sin(a_deg: float) -> float:
    """sin(2A) = 2·sin(A)·cos(A)."""
    a = math.radians(float(a_deg))
    return round(2 * math.sin(a) * math.cos(a), 6)


def double_angle_cos(a_deg: float) -> float:
    """cos(2A) = cos²(A) - sin²(A)."""
    a = math.radians(float(a_deg))
    return round(math.cos(a)**2 - math.sin(a)**2, 6)


MATH_TRIG_FUNCTIONS = {
    "sin_deg": sin_deg,
    "cos_deg": cos_deg,
    "tan_deg": tan_deg,
    "asin_deg": asin_deg,
    "acos_deg": acos_deg,
    "atan_deg": atan_deg,
    "atan2_deg": atan2_deg,
    "csc": csc,
    "sec": sec,
    "cot": cot,
    "sinh_val": sinh_val,
    "cosh_val": cosh_val,
    "tanh_val": tanh_val,
    "degrees_to_radians": degrees_to_radians,
    "radians_to_degrees": radians_to_degrees,
    "law_of_cosines": law_of_cosines,
    "law_of_sines_angle": law_of_sines_angle,
    "angle_sum_sin": angle_sum_sin,
    "angle_sum_cos": angle_sum_cos,
    "double_angle_sin": double_angle_sin,
    "double_angle_cos": double_angle_cos,
}

MATH_TRIG_NL_PATTERNS = [
    (r'sin\s*\(?\s*([\d.]+)\s*(?:°|degrees?)\s*\)?', 'sin_deg({0})'),
    (r'cos\s*\(?\s*([\d.]+)\s*(?:°|degrees?)\s*\)?', 'cos_deg({0})'),
    (r'tan\s*\(?\s*([\d.]+)\s*(?:°|degrees?)\s*\)?', 'tan_deg({0})'),
    (r'arcsin\s*\(?\s*([-\d.]+)\s*\)?', 'asin_deg({0})'),
    (r'arccos\s*\(?\s*([-\d.]+)\s*\)?', 'acos_deg({0})'),
    (r'arctan\s*\(?\s*([-\d.]+)\s*\)?', 'atan_deg({0})'),
    (r'convert\s+([\d.]+)\s*(?:°|degrees?)\s+to\s+radians', 'degrees_to_radians({0})'),
    (r'convert\s+([\d.]+)\s*rad(?:ians?)?\s+to\s+degrees', 'radians_to_degrees({0})'),
    (r'law\s+of\s+cosines.*?a\s*=\s*([\d.]+).*?b\s*=\s*([\d.]+).*?C\s*=\s*([\d.]+)', 'law_of_cosines({0}, {1}, {2})'),
]
