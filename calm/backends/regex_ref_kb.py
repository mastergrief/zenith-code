"""
CALM Regex Reference knowledge backend — common patterns and syntax.

Models hallucinate regex syntax. Lookup table for common patterns.
"""

from __future__ import annotations

_DATA_VERSION = "2025-01"

_COMMON_PATTERNS = {
    "email": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
    "url": r'^https?://[^\s/$.?#].[^\s]*$',
    "ipv4": r'^(\d{1,3}\.){3}\d{1,3}$',
    "ipv6": r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$',
    "phone_us": r'^\+?1?\d{10}$',
    "phone_intl": r'^\+\d{1,3}\d{4,14}$',
    "date_iso": r'^\d{4}-\d{2}-\d{2}$',
    "date_us": r'^\d{2}/\d{2}/\d{4}$',
    "time_24h": r'^([01]\d|2[0-3]):[0-5]\d(:[0-5]\d)?$',
    "hex_color": r'^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$',
    "uuid": r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$',
    "semver": r'^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?(\+[a-zA-Z0-9.]+)?$',
    "slug": r'^[a-z0-9]+(-[a-z0-9]+)*$',
    "username": r'^[a-zA-Z0-9_]{3,20}$',
    "password_strong": r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$',
    "credit_card": r'^\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}$',
    "ssn": r'^\d{3}-\d{2}-\d{4}$',
    "zip_us": r'^\d{5}(-\d{4})?$',
    "mac_address": r'^([0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}$',
    "domain": r'^([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}$',
}

_SYNTAX_REFERENCE = {
    ".": "any character except newline",
    "\\d": "digit [0-9]",
    "\\D": "non-digit",
    "\\w": "word character [a-zA-Z0-9_]",
    "\\W": "non-word character",
    "\\s": "whitespace",
    "\\S": "non-whitespace",
    "\\b": "word boundary",
    "^": "start of string/line",
    "$": "end of string/line",
    "*": "0 or more (greedy)",
    "+": "1 or more (greedy)",
    "?": "0 or 1 (optional)",
    "*?": "0 or more (lazy)",
    "+?": "1 or more (lazy)",
    "{n}": "exactly n times",
    "{n,m}": "between n and m times",
    "[abc]": "character class (a, b, or c)",
    "[^abc]": "negated class (not a, b, or c)",
    "(...)": "capturing group",
    "(?:...)": "non-capturing group",
    "(?=...)": "positive lookahead",
    "(?!...)": "negative lookahead",
    "(?<=...)": "positive lookbehind",
    "(?<!...)": "negative lookbehind",
    "|": "alternation (OR)",
}


def regex_for(name: str) -> str:
    """Get a common regex pattern by name (email, url, ipv4, etc.)."""
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    return _COMMON_PATTERNS.get(key, f"unknown pattern: {name}")


def regex_syntax(symbol: str) -> str:
    """Explain a regex syntax element."""
    return _SYNTAX_REFERENCE.get(symbol.strip(), f"unknown syntax: {symbol}")


def regex_list_patterns() -> list:
    """List all available common regex patterns."""
    return sorted(_COMMON_PATTERNS.keys())


def regex_list_syntax() -> list:
    """List all regex syntax elements with descriptions."""
    return [f"{sym}: {desc}" for sym, desc in _SYNTAX_REFERENCE.items()]


REGEX_REF_FUNCTIONS = {
    "regex_for": regex_for,
    "regex_syntax": regex_syntax,
    "regex_list_patterns": regex_list_patterns,
    "regex_list_syntax": regex_list_syntax,
}

REGEX_REF_NL_PATTERNS = [
    (r'regex\s+(?:for|to match|pattern for)\s+(\w[\w\s]*)', 'regex_for("{0}")'),
    (r'(?:what does|explain)\s+([\\][dwsb.+*?])\s+mean\s+in\s+regex', 'regex_syntax("{0}")'),
]
