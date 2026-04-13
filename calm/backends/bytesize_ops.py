"""
CALM Byte size backend — human-readable formatting, parsing, IEC vs SI.

Models confuse 1024 vs 1000 (MiB vs MB) constantly. Pure math.
"""

from __future__ import annotations

import re

_SI_UNITS = ["B", "KB", "MB", "GB", "TB", "PB", "EB"]
_IEC_UNITS = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"]

_PARSE_MAP = {
    "b": 1, "byte": 1, "bytes": 1,
    "kb": 1000, "kib": 1024, "kilobyte": 1000, "kilobytes": 1000,
    "mb": 1000**2, "mib": 1024**2, "megabyte": 1000**2, "megabytes": 1000**2,
    "gb": 1000**3, "gib": 1024**3, "gigabyte": 1000**3, "gigabytes": 1000**3,
    "tb": 1000**4, "tib": 1024**4, "terabyte": 1000**4, "terabytes": 1000**4,
    "pb": 1000**5, "pib": 1024**5, "petabyte": 1000**5, "petabytes": 1000**5,
    "eb": 1000**6, "eib": 1024**6, "exabyte": 1000**6, "exabytes": 1000**6,
}


def bytes_format_si(n: int) -> str:
    """Format bytes as human-readable with SI units (1 KB = 1000 B)."""
    n = int(n)
    if n < 0:
        return f"-{bytes_format_si(-n)}"
    if n < 1000:
        return f"{n} B"
    for i, unit in enumerate(_SI_UNITS[1:], 1):
        threshold = 1000 ** i
        if n < 1000 ** (i + 1) or i == len(_SI_UNITS) - 1:
            val = n / threshold
            return f"{val:.2f} {unit}" if val != int(val) else f"{int(val)} {unit}"
    return f"{n} B"


def bytes_format_iec(n: int) -> str:
    """Format bytes as human-readable with IEC units (1 KiB = 1024 B)."""
    n = int(n)
    if n < 0:
        return f"-{bytes_format_iec(-n)}"
    if n < 1024:
        return f"{n} B"
    for i, unit in enumerate(_IEC_UNITS[1:], 1):
        threshold = 1024 ** i
        if n < 1024 ** (i + 1) or i == len(_IEC_UNITS) - 1:
            val = n / threshold
            return f"{val:.2f} {unit}" if val != int(val) else f"{int(val)} {unit}"
    return f"{n} B"


def bytes_parse(s: str) -> int:
    """Parse human-readable byte string to bytes (e.g. '10 MB' → 10000000)."""
    s = str(s).strip()
    m = re.match(r'^([\d.]+)\s*([a-zA-Z]+)$', s)
    if not m:
        try:
            return int(float(s))
        except ValueError:
            return -1
    val = float(m.group(1))
    unit = m.group(2).lower()
    multiplier = _PARSE_MAP.get(unit, -1)
    if multiplier == -1:
        return -1
    return int(val * multiplier)


def bytes_convert(n: int, from_unit: str, to_unit: str) -> str:
    """Convert between byte units (e.g. MB to GiB)."""
    from_mult = _PARSE_MAP.get(from_unit.lower(), -1)
    to_mult = _PARSE_MAP.get(to_unit.lower(), -1)
    if from_mult == -1:
        return f"unknown unit: {from_unit}"
    if to_mult == -1:
        return f"unknown unit: {to_unit}"
    total_bytes = int(n) * from_mult
    result = total_bytes / to_mult
    if result == int(result):
        return f"{int(result)} {to_unit}"
    return f"{result:.4f} {to_unit}"


def bytes_diff_si_iec(n: int) -> str:
    """Show the difference between SI and IEC interpretation of a byte count."""
    n = int(n)
    return f"SI: {bytes_format_si(n)}, IEC: {bytes_format_iec(n)}"


def kb_to_bytes(n: int) -> int:
    """Convert KB (SI, 1000) to bytes."""
    return int(n) * 1000


def kib_to_bytes(n: int) -> int:
    """Convert KiB (IEC, 1024) to bytes."""
    return int(n) * 1024


BYTESIZE_NL_PATTERNS = [
    (r"(?:how (?:many|much)|convert)\s+(\d+)\s*(KB|MB|GB|TB|KiB|MiB|GiB|TiB)\s+(?:to|in)\s+(bytes|B|KB|MB|GB|TB|KiB|MiB|GiB|TiB)", 'bytes_convert({0}, "{1}", "{2}")'),
    (r"(?:what is|how big is)\s+(\d+)\s*(?:bytes|B)\s+in\s+(?:human|readable)", 'bytes_format_si({0})'),
    (r"(?:difference|diff)\s+(?:between\s+)?(?:MB|MiB|SI|IEC)", 'bytes_diff_si_iec(1000000)'),
]

BYTESIZE_FUNCTIONS = {
    "bytes_format_si": bytes_format_si,
    "bytes_format_iec": bytes_format_iec,
    "bytes_parse": bytes_parse,
    "bytes_convert": bytes_convert,
    "bytes_diff_si_iec": bytes_diff_si_iec,
    "kb_to_bytes": kb_to_bytes,
    "kib_to_bytes": kib_to_bytes,
}
