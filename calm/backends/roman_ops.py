"""
CALM Roman numeral backend — convert and validate.

Models reliably fail at MCMXCIV (1994) and similar subtractive forms.
"""

from __future__ import annotations

import re

_TO_ROMAN = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]

_FROM_ROMAN = {
    "I": 1, "V": 5, "X": 10, "L": 50,
    "C": 100, "D": 500, "M": 1000,
}

_VALID_ROMAN = re.compile(
    r'^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'
)


def to_roman(n: int) -> str:
    """Convert integer to Roman numeral (1-3999)."""
    n = int(n)
    if n < 1 or n > 3999:
        return f"out of range: {n} (valid: 1-3999)"
    result = []
    for value, numeral in _TO_ROMAN:
        while n >= value:
            result.append(numeral)
            n -= value
    return "".join(result)


def from_roman(s: str) -> int:
    """Convert Roman numeral to integer."""
    s = str(s).strip().upper()
    if not s:
        return -1
    total = 0
    prev = 0
    for ch in reversed(s):
        val = _FROM_ROMAN.get(ch, 0)
        if val == 0:
            return -1
        if val < prev:
            total -= val
        else:
            total += val
        prev = val
    return total


def roman_validate(s: str) -> bool:
    """Check if a string is a valid Roman numeral."""
    s = str(s).strip().upper()
    if not s:
        return False
    if not _VALID_ROMAN.match(s):
        return False
    return from_roman(s) > 0


ROMAN_FUNCTIONS = {
    "to_roman": to_roman,
    "from_roman": from_roman,
    "roman_validate": roman_validate,
}
