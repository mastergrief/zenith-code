"""
CALM ASCII knowledge backend — control chars, escape sequences, code points.

Models mix up \\r vs \\n, get code points wrong, confuse DEL vs BS.
Stable forever — ASCII hasn't changed since 1963.
"""

from __future__ import annotations

_DATA_VERSION = "1963"  # ASCII is eternal

_CONTROL_CHARS = {
    0: ("NUL", "\\0", "Null"),
    1: ("SOH", "\\x01", "Start of Heading"),
    2: ("STX", "\\x02", "Start of Text"),
    3: ("ETX", "\\x03", "End of Text (Ctrl+C)"),
    4: ("EOT", "\\x04", "End of Transmission (Ctrl+D)"),
    7: ("BEL", "\\a", "Bell"),
    8: ("BS", "\\b", "Backspace"),
    9: ("HT", "\\t", "Horizontal Tab"),
    10: ("LF", "\\n", "Line Feed (newline)"),
    11: ("VT", "\\v", "Vertical Tab"),
    12: ("FF", "\\f", "Form Feed"),
    13: ("CR", "\\r", "Carriage Return"),
    27: ("ESC", "\\x1b", "Escape"),
    32: ("SP", " ", "Space"),
    127: ("DEL", "\\x7f", "Delete"),
}

_ESCAPE_SEQUENCES = {
    "\\n": (10, "LF", "Line Feed — moves cursor to next line"),
    "\\r": (13, "CR", "Carriage Return — moves cursor to start of line"),
    "\\t": (9, "HT", "Horizontal Tab"),
    "\\0": (0, "NUL", "Null character"),
    "\\a": (7, "BEL", "Bell / alert"),
    "\\b": (8, "BS", "Backspace"),
    "\\f": (12, "FF", "Form Feed"),
    "\\v": (11, "VT", "Vertical Tab"),
    "\\\\": (92, "BACKSLASH", "Literal backslash"),
    "\\'": (39, "APOSTROPHE", "Single quote"),
    '\\"': (34, "QUOTE", "Double quote"),
}

# Line ending conventions
_LINE_ENDINGS = {
    "unix": "\\n (LF, 0x0A)",
    "linux": "\\n (LF, 0x0A)",
    "macos": "\\n (LF, 0x0A) — classic Mac OS used \\r (CR)",
    "windows": "\\r\\n (CRLF, 0x0D 0x0A)",
    "dos": "\\r\\n (CRLF, 0x0D 0x0A)",
    "http": "\\r\\n (CRLF) — required by HTTP spec",
    "oldmac": "\\r (CR, 0x0D) — pre-OS X only",
}


def ascii_code(char: str) -> int:
    """ASCII code point for a character."""
    return ord(str(char)[0]) if char else -1


def ascii_char(code: int) -> str:
    """Character for an ASCII code point."""
    code = int(code)
    if 0 <= code <= 127:
        if code in _CONTROL_CHARS:
            name, esc, desc = _CONTROL_CHARS[code]
            return f"{name} ({esc}) — {desc}"
        return chr(code)
    return f"not ASCII: {code} (0-127 only)"


def ascii_control(name: str) -> str:
    """Look up a control character by name (NUL, LF, CR, TAB, ESC, etc.)."""
    name = str(name).strip().upper()
    aliases = {
        "NULL": 0, "NUL": 0, "TAB": 9, "HT": 9,
        "NEWLINE": 10, "LF": 10, "LINEFEED": 10,
        "CR": 13, "RETURN": 13, "CARRIAGERETURN": 13,
        "ESCAPE": 27, "ESC": 27, "SPACE": 32, "SP": 32,
        "DELETE": 127, "DEL": 127, "BACKSPACE": 8, "BS": 8,
        "BELL": 7, "BEL": 7, "FORMFEED": 12, "FF": 12,
    }
    code = aliases.get(name.replace(" ", ""))
    if code is not None and code in _CONTROL_CHARS:
        cname, esc, desc = _CONTROL_CHARS[code]
        return f"{code} (0x{code:02X}) {cname} {esc} — {desc}"
    return f"unknown control char: {name}"


def ascii_escape(seq: str) -> str:
    """Explain an escape sequence (\\n, \\r, \\t, etc.)."""
    seq = str(seq).strip()
    if not seq.startswith("\\"):
        seq = "\\" + seq
    data = _ESCAPE_SEQUENCES.get(seq)
    if data:
        code, name, desc = data
        return f"{seq} = {code} (0x{code:02X}) {name} — {desc}"
    return f"unknown escape: {seq}"


def ascii_line_ending(platform: str) -> str:
    """Line ending convention for a platform (unix, windows, http, etc.)."""
    return _LINE_ENDINGS.get(str(platform).strip().lower(), f"unknown platform: {platform}")


def ascii_printable_range() -> str:
    """The printable ASCII range."""
    return "32-126 (space through tilde ~)"


def ascii_diff_cr_lf() -> str:
    """Explain the difference between \\r and \\n."""
    return ("\\r (CR, 13, 0x0D): Carriage Return — moves cursor to start of current line. "
            "\\n (LF, 10, 0x0A): Line Feed — moves cursor to next line. "
            "Windows uses \\r\\n (CRLF), Unix/Mac use \\n (LF only).")


ASCII_FUNCTIONS = {
    "ascii_code": ascii_code,
    "ascii_char": ascii_char,
    "ascii_control": ascii_control,
    "ascii_escape": ascii_escape,
    "ascii_line_ending": ascii_line_ending,
    "ascii_printable_range": ascii_printable_range,
    "ascii_diff_cr_lf": ascii_diff_cr_lf,
}
