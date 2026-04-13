"""
CALM Formatting backend — number formatting, currency formatting, date formatting.

Models hallucinate formatting rules. Pure string manipulation.
"""

from __future__ import annotations

import math


def format_number(n: float, decimals: int = 2, separator: str = ",") -> str:
    """Format number with thousands separator and decimal places."""
    parts = f"{float(n):,.{int(decimals)}f}"
    if separator != ",":
        parts = parts.replace(",", separator)
    return parts


def format_currency(amount: float, symbol: str = "$", decimals: int = 2) -> str:
    """Format as currency: $1,234.56."""
    return f"{symbol}{format_number(amount, decimals)}"


def format_percent(value: float, decimals: int = 1) -> str:
    """Format as percentage: 0.156 → '15.6%'."""
    return f"{round(float(value) * 100, int(decimals)):.{int(decimals)}f}%"


def format_bytes(n: float, binary: bool = True) -> str:
    """Format bytes as human-readable. binary=True: KiB/MiB/GiB, False: KB/MB/GB."""
    n = float(n)
    if n < 0:
        return f"-{format_bytes(-n, binary)}"
    base = 1024 if binary else 1000
    suffixes = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"] if binary else ["B", "KB", "MB", "GB", "TB", "PB"]
    if n < base:
        return f"{n:.0f} {suffixes[0]}"
    for i, suffix in enumerate(suffixes[1:], 1):
        unit = base ** i
        if n < unit * base or i == len(suffixes) - 1:
            return f"{n / unit:.2f} {suffix}"
    return f"{n:.0f} B"


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    s = float(seconds)
    if s < 0:
        return f"-{format_duration(-s)}"
    if s < 1:
        return f"{s * 1000:.0f}ms"
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        m = int(s // 60)
        sec = s % 60
        return f"{m}m {sec:.0f}s"
    if s < 86400:
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        return f"{h}h {m}m"
    d = int(s // 86400)
    h = int((s % 86400) // 3600)
    return f"{d}d {h}h"


def format_ordinal(n: int) -> str:
    """Format integer as ordinal: 1→1st, 2→2nd, 3→3rd, 4→4th."""
    n = int(n)
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    suffixes = {1: "st", 2: "nd", 3: "rd"}
    return f"{n}{suffixes.get(n % 10, 'th')}"


def format_roman(n: int) -> str:
    """Convert integer to Roman numeral (1-3999)."""
    n = int(n)
    if n < 1 or n > 3999:
        return "out of range (1-3999)"
    vals = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'),
            (90, 'XC'), (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]
    result = ""
    for value, numeral in vals:
        while n >= value:
            result += numeral
            n -= value
    return result


def format_scientific(n: float, decimals: int = 2) -> str:
    """Format in scientific notation: 1234 → '1.23e+03'."""
    return f"{float(n):.{int(decimals)}e}"


def format_engineering(n: float) -> str:
    """Format in engineering notation (exponent multiple of 3)."""
    if n == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(float(n)))))
    eng_exp = (exp // 3) * 3
    mantissa = float(n) / (10 ** eng_exp)
    return f"{mantissa:.3f}e{eng_exp:+d}"


def format_hex(n: int, width: int = 0) -> str:
    """Format integer as hex with optional width: format_hex(255, 4) → '00ff'."""
    return f"{int(n):0{int(width)}x}"


def format_binary(n: int, width: int = 0) -> str:
    """Format integer as binary with optional width."""
    return f"{int(n):0{int(width)}b}"


def format_phone_us(digits: str) -> str:
    """Format 10 digits as US phone: (555) 123-4567."""
    d = ''.join(c for c in str(digits) if c.isdigit())
    if len(d) == 11 and d[0] == '1':
        d = d[1:]
    if len(d) != 10:
        return f"invalid: need 10 digits, got {len(d)}"
    return f"({d[:3]}) {d[3:6]}-{d[6:]}"


def format_ssn(digits: str) -> str:
    """Format 9 digits as SSN: XXX-XX-XXXX."""
    d = ''.join(c for c in str(digits) if c.isdigit())
    if len(d) != 9:
        return f"invalid: need 9 digits, got {len(d)}"
    return f"{d[:3]}-{d[3:5]}-{d[5:]}"


def format_credit_card(digits: str) -> str:
    """Format 16 digits as credit card: XXXX XXXX XXXX XXXX."""
    d = ''.join(c for c in str(digits) if c.isdigit())
    if len(d) != 16:
        return f"invalid: need 16 digits, got {len(d)}"
    return f"{d[:4]} {d[4:8]} {d[8:12]} {d[12:]}"


def pluralize(word: str, count: int) -> str:
    """Basic English pluralization."""
    w = str(word)
    n = int(count)
    if n == 1:
        return w
    if w.endswith(('s', 'sh', 'ch', 'x', 'z')):
        return w + 'es'
    if w.endswith('y') and w[-2] not in 'aeiou':
        return w[:-1] + 'ies'
    if w.endswith('f'):
        return w[:-1] + 'ves'
    if w.endswith('fe'):
        return w[:-2] + 'ves'
    return w + 's'


def truncate_text(text: str, max_len: int, suffix: str = "...") -> str:
    """Truncate text to max_len, adding suffix if truncated."""
    t = str(text)
    m = int(max_len)
    if len(t) <= m:
        return t
    return t[:m - len(suffix)] + suffix


def slug(text: str) -> str:
    """Convert text to URL slug: 'Hello World!' → 'hello-world'."""
    import re
    t = str(text).lower().strip()
    t = re.sub(r'[^\w\s-]', '', t)
    t = re.sub(r'[\s_]+', '-', t)
    return t.strip('-')


FORMAT_FUNCTIONS = {
    "format_number": format_number,
    "format_currency": format_currency,
    "format_percent": format_percent,
    "format_bytes": format_bytes,
    "format_duration": format_duration,
    "format_ordinal": format_ordinal,
    "format_roman": format_roman,
    "format_scientific": format_scientific,
    "format_engineering": format_engineering,
    "format_hex": format_hex,
    "format_binary": format_binary,
    "format_phone_us": format_phone_us,
    "format_ssn": format_ssn,
    "format_credit_card": format_credit_card,
    "pluralize": pluralize,
    "truncate_text": truncate_text,
    "slug": slug,
}

FORMAT_NL_PATTERNS = [
    (r'format\s+([\d.]+)\s+(?:as\s+)?(?:currency|dollars?|money)', 'format_currency({0})'),
    (r'format\s+([\d.]+)\s+(?:as\s+)?(?:percent|percentage)', 'format_percent({0})'),
    (r'format\s+([\d.]+)\s+bytes?\s+(?:as\s+)?human', 'format_bytes({0})'),
    (r'format\s+([\d.]+)\s+seconds?\s+(?:as\s+)?(?:duration|human)', 'format_duration({0})'),
    (r'(\d+)\s+(?:as|to|in)\s+(?:ordinal|1st|2nd|3rd)', 'format_ordinal({0})'),
    (r'(\d+)\s+(?:as|to|in)\s+roman', 'format_roman({0})'),
    (r'slug(?:ify)?\s+["\'](.+?)["\']', 'slug("{0}")'),
]
