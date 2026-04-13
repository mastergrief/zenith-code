"""
CALM Unicode backend — codepoint lookup, category, normalization, confusables.

Models hallucinate Unicode properties. Pure stdlib unicodedata module.
"""

from __future__ import annotations

import unicodedata


def unicode_name(char: str) -> str:
    """Unicode name of a character."""
    try:
        return unicodedata.name(char[0])
    except (ValueError, IndexError):
        return "unknown"


def unicode_category(char: str) -> str:
    """Unicode general category (e.g. 'Lu' = uppercase letter)."""
    categories = {
        "Lu": "Uppercase Letter", "Ll": "Lowercase Letter",
        "Lt": "Titlecase Letter", "Lm": "Modifier Letter",
        "Lo": "Other Letter", "Mn": "Nonspacing Mark",
        "Mc": "Spacing Mark", "Me": "Enclosing Mark",
        "Nd": "Decimal Number", "Nl": "Letter Number",
        "No": "Other Number", "Pc": "Connector Punctuation",
        "Pd": "Dash Punctuation", "Ps": "Open Punctuation",
        "Pe": "Close Punctuation", "Pi": "Initial Punctuation",
        "Pf": "Final Punctuation", "Po": "Other Punctuation",
        "Sm": "Math Symbol", "Sc": "Currency Symbol",
        "Sk": "Modifier Symbol", "So": "Other Symbol",
        "Zs": "Space Separator", "Zl": "Line Separator",
        "Zp": "Paragraph Separator", "Cc": "Control",
        "Cf": "Format", "Cs": "Surrogate", "Co": "Private Use",
        "Cn": "Unassigned",
    }
    try:
        cat = unicodedata.category(char[0])
        desc = categories.get(cat, cat)
        return f"{cat} ({desc})"
    except IndexError:
        return "empty"


def unicode_codepoint(char: str) -> str:
    """Unicode codepoint in U+XXXX format."""
    try:
        cp = ord(char[0])
        return f"U+{cp:04X}"
    except IndexError:
        return "empty"


def unicode_normalize(text: str, form: str = "NFC") -> str:
    """Normalize Unicode text (NFC, NFD, NFKC, NFKD)."""
    form = form.upper()
    if form not in ("NFC", "NFD", "NFKC", "NFKD"):
        return f"invalid form: {form} (use NFC/NFD/NFKC/NFKD)"
    return unicodedata.normalize(form, text)


def unicode_is_normalized(text: str, form: str = "NFC") -> bool:
    """Check if text is already in the given normal form."""
    form = form.upper()
    if form not in ("NFC", "NFD", "NFKC", "NFKD"):
        return False
    return unicodedata.is_normalized(form, text)


# Common confusable pairs (homoglyphs used in phishing/spoofing)
_CONFUSABLES = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y",
    "х": "x", "ѕ": "s", "і": "i", "ј": "j", "ԁ": "d", "ɡ": "g",
    "ɩ": "l", "ν": "v", "ω": "w", "Α": "A", "Β": "B", "Ε": "E",
    "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N",
    "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    "ℯ": "e", "ℊ": "g", "ℎ": "h", "ℓ": "l", "ℕ": "N",
    "ℤ": "Z", "ℝ": "R", "ℚ": "Q",
    "\u200b": "(ZWS)", "\u200c": "(ZWNJ)", "\u200d": "(ZWJ)",
    "\ufeff": "(BOM)", "\u00a0": "(NBSP)",
}


def unicode_confusables(text: str) -> list:
    """Detect confusable/homoglyph characters in text."""
    found = []
    for i, ch in enumerate(text):
        if ch in _CONFUSABLES:
            found.append({
                "pos": i,
                "char": ch,
                "looks_like": _CONFUSABLES[ch],
                "codepoint": f"U+{ord(ch):04X}",
                "name": unicodedata.name(ch, "unknown"),
            })
    return found


def unicode_string_width(text: str) -> int:
    """Estimated display width (CJK = 2, most others = 1)."""
    width = 0
    for ch in text:
        cat = unicodedata.east_asian_width(ch)
        if cat in ("W", "F"):
            width += 2
        elif cat == "Na" or cat == "H" or cat == "N":
            width += 1
        else:
            width += 1
    return width


UNICODE_FUNCTIONS = {
    "unicode_name": unicode_name,
    "unicode_category": unicode_category,
    "unicode_codepoint": unicode_codepoint,
    "unicode_normalize": unicode_normalize,
    "unicode_is_normalized": unicode_is_normalized,
    "unicode_confusables": unicode_confusables,
    "unicode_string_width": unicode_string_width,
}

UNICODE_NL_PATTERNS = [
    (r'(?:codepoint|code point|unicode)\s+(?:of|for)\s+["\']?(.)["\']?', 'unicode_codepoint("{0}")'),
    (r'(?:what is|character at)\s+U\+([0-9A-Fa-f]{4,6})', None),
    (r'(?:category|type)\s+(?:of\s+)?unicode\s+(?:character\s+)?["\']?(.)["\']?', 'unicode_category("{0}")'),
]
