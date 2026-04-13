"""
CALM Base conversion backend — binary, octal, hex, arbitrary base.

Models botch base conversions constantly. Pure int/format.
"""

from __future__ import annotations

_DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"


def to_binary(n: int) -> str:
    """Convert integer to binary string."""
    n = int(n)
    if n < 0:
        return "-" + bin(-n)[2:]
    return bin(n)[2:]


def to_octal(n: int) -> str:
    """Convert integer to octal string."""
    n = int(n)
    if n < 0:
        return "-" + oct(-n)[2:]
    return oct(n)[2:]


def to_hex(n: int) -> str:
    """Convert integer to hexadecimal string."""
    n = int(n)
    if n < 0:
        return "-" + hex(-n)[2:]
    return hex(n)[2:]


def from_binary(s: str) -> int:
    """Convert binary string to integer."""
    return int(str(s).strip(), 2)


def from_octal(s: str) -> int:
    """Convert octal string to integer."""
    return int(str(s).strip(), 8)


def from_hex(s: str) -> int:
    """Convert hex string to integer."""
    s = str(s).strip().lstrip("0x").lstrip("0X")
    return int(s, 16) if s else 0


def to_base(n: int, base: int) -> str:
    """Convert integer to arbitrary base (2-36)."""
    n, base = int(n), int(base)
    if base < 2 or base > 36:
        return f"base must be 2-36, got {base}"
    if n == 0:
        return "0"
    negative = n < 0
    n = abs(n)
    result = []
    while n > 0:
        result.append(_DIGITS[n % base])
        n //= base
    if negative:
        result.append("-")
    return "".join(reversed(result))


def from_base(s: str, base: int) -> int:
    """Convert string in arbitrary base (2-36) to integer."""
    return int(str(s).strip(), int(base))


def base_convert(s: str, from_base: int, to_base_num: int) -> str:
    """Convert a number string from one base to another."""
    try:
        n = int(str(s).strip(), int(from_base))
        return to_base(n, int(to_base_num))
    except ValueError as e:
        return f"error: {e}"


BASECONV_FUNCTIONS = {
    "to_binary": to_binary,
    "to_octal": to_octal,
    "to_hex": to_hex,
    "from_binary": from_binary,
    "from_octal": from_octal,
    "from_hex": from_hex,
    "to_base": to_base,
    "from_base": from_base,
    "base_convert": base_convert,
}
