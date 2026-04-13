"""
CALM Regex common patterns knowledge backend — email, URL, IP, phone, date patterns.

Models hallucinate regex patterns. These are tested, working patterns.
Extends regex_ref_kb with more patterns and use-case-specific matchers.
"""

from __future__ import annotations

import re

_DATA_VERSION = "2025-01"

_COMMON_PATTERNS = {
    "email": {
        "pattern": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        "description": "Basic email validation (RFC 5322 subset)",
        "matches": ["user@example.com", "user.name+tag@domain.co.uk"],
        "rejects": ["@domain.com", "user@", "user@.com"],
        "note": "Not fully RFC 5322 compliant — use a library for production",
    },
    "url": {
        "pattern": r'https?://(?:www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_+.~#?&/=]*)',
        "description": "HTTP/HTTPS URL",
        "matches": ["https://example.com", "http://sub.domain.co.uk/path?q=1"],
    },
    "ipv4": {
        "pattern": r'^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$',
        "description": "IPv4 address (0.0.0.0 to 255.255.255.255)",
        "matches": ["192.168.1.1", "10.0.0.1", "255.255.255.255"],
        "rejects": ["256.1.1.1", "1.2.3.4.5"],
    },
    "uuid": {
        "pattern": r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$',
        "description": "UUID (any version)",
        "matches": ["550e8400-e29b-41d4-a716-446655440000"],
    },
    "phone_us": {
        "pattern": r'^(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$',
        "description": "US phone number (with optional +1 and formatting)",
        "matches": ["+1 (555) 123-4567", "555-123-4567", "5551234567"],
    },
    "phone_international": {
        "pattern": r'^\+\d{1,3}[-.\s]?\d{1,14}(?:[-.\s]\d+)*$',
        "description": "International phone number (E.164 flexible)",
    },
    "date_iso": {
        "pattern": r'^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$',
        "description": "ISO 8601 date (YYYY-MM-DD)",
        "matches": ["2025-01-15", "2024-12-31"],
        "rejects": ["2025-13-01", "2025-00-15"],
    },
    "date_us": {
        "pattern": r'^(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/\d{4}$',
        "description": "US date format (MM/DD/YYYY)",
    },
    "time_24h": {
        "pattern": r'^(?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d)?$',
        "description": "24-hour time (HH:MM or HH:MM:SS)",
        "matches": ["23:59", "00:00:00", "14:30:15"],
    },
    "hex_color": {
        "pattern": r'^#(?:[0-9a-fA-F]{3}){1,2}$',
        "description": "Hex color code (#RGB or #RRGGBB)",
        "matches": ["#fff", "#FF5733", "#a1b2c3"],
    },
    "credit_card": {
        "pattern": r'^\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}$',
        "description": "Credit card number (16 digits, optional separators)",
        "note": "Structural only — use Luhn for validation",
    },
    "slug": {
        "pattern": r'^[a-z0-9]+(?:-[a-z0-9]+)*$',
        "description": "URL-safe slug (lowercase, hyphens)",
        "matches": ["hello-world", "my-post-123"],
    },
    "semver": {
        "pattern": r'^\d+\.\d+\.\d+(?:-[a-zA-Z0-9.]+)?(?:\+[a-zA-Z0-9.]+)?$',
        "description": "Semantic version (MAJOR.MINOR.PATCH with optional pre-release)",
        "matches": ["1.0.0", "2.3.4-beta.1", "1.0.0+build.123"],
    },
    "jwt": {
        "pattern": r'^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$',
        "description": "JWT token (three base64url segments separated by dots)",
    },
    "mac_address": {
        "pattern": r'^(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}$',
        "description": "MAC address (colon or hyphen separated)",
        "matches": ["00:1A:2B:3C:4D:5E", "00-1A-2B-3C-4D-5E"],
    },
    "strong_password": {
        "pattern": r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^a-zA-Z\d]).{8,}$',
        "description": "Strong password (8+ chars, upper, lower, digit, special)",
        "note": "Length > complexity. NIST recommends 12+ chars, no complexity rules.",
    },
    "html_tag": {
        "pattern": r'<([a-zA-Z][a-zA-Z0-9]*)\b[^>]*>.*?</\1>',
        "description": "Simple HTML tag pair (NOT a parser — use for simple extraction only)",
        "note": "Don't parse HTML with regex in production. Use a proper parser.",
    },
}


def get_pattern(name: str) -> dict:
    """Get a tested regex pattern by name."""
    key = str(name).lower().strip().replace(" ", "_").replace("-", "_")
    entry = _COMMON_PATTERNS.get(key)
    if not entry:
        # Fuzzy search
        for k, v in _COMMON_PATTERNS.items():
            if key in k or k in key:
                return {"name": k, **v}
        return {"error": f"Unknown pattern: {name}", "valid": sorted(_COMMON_PATTERNS.keys())}
    return {"name": key, **entry}


def test_pattern(name: str, text: str) -> bool:
    """Test a named pattern against a string."""
    info = get_pattern(name)
    if "error" in info:
        return False
    return bool(re.match(info["pattern"], str(text)))


def list_patterns() -> list[str]:
    """List all available regex patterns."""
    return sorted(_COMMON_PATTERNS.keys())


def pattern_for_extension(ext: str) -> str:
    """Get a regex pattern to match a file extension."""
    e = str(ext).lower().strip().lstrip(".")
    return rf'\.{e}$'


REGEX_COMMON_FUNCTIONS = {
    "get_pattern": get_pattern,
    "test_pattern": test_pattern,
    "list_patterns": list_patterns,
    "pattern_for_extension": pattern_for_extension,
}

REGEX_COMMON_NL_PATTERNS = [
    (r'(?:regex|pattern)\s+(?:for|to match|to validate)\s+(?:an?\s+)?(email|url|ip|ipv4|uuid|phone|date|time|hex color|credit card|slug|semver|jwt|mac address|password)', 'get_pattern("{0}")'),
    (r'(?:is|does)\s+["\'](.+?)["\']\s+(?:match|valid)\s+(?:as\s+)?(?:an?\s+)?(email|url|ip|phone|date|uuid)', 'test_pattern("{1}", "{0}")'),
]
