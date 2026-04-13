"""
CALM Validation backend — input validation, format checking, data sanitization.

Models approximate validation rules. Pure regex + computation.
"""

from __future__ import annotations

import re


def is_valid_email(email: str) -> bool:
    """Basic email format validation."""
    return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', str(email)))


def is_valid_url(url: str) -> bool:
    """Basic URL format validation (http/https)."""
    return bool(re.match(r'^https?://[^\s<>"{}|\\^`\[\]]+$', str(url)))


def is_valid_ipv4(ip: str) -> bool:
    """IPv4 address validation."""
    parts = str(ip).split('.')
    if len(parts) != 4:
        return False
    return all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)


def is_valid_ipv6(ip: str) -> bool:
    """IPv6 address validation (simplified)."""
    return bool(re.match(r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$|^::$|^([0-9a-fA-F]{1,4}:)*:([0-9a-fA-F]{1,4}:)*[0-9a-fA-F]{1,4}$', str(ip)))


def is_valid_uuid(uuid_str: str) -> bool:
    """UUID v4 format validation."""
    return bool(re.match(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$', str(uuid_str)))


def is_valid_hex_color(color: str) -> bool:
    """Hex color code validation (#RGB or #RRGGBB)."""
    return bool(re.match(r'^#(?:[0-9a-fA-F]{3}){1,2}$', str(color)))


def is_valid_credit_card(number: str) -> bool:
    """Credit card number validation using Luhn algorithm."""
    digits = ''.join(c for c in str(number) if c.isdigit())
    if len(digits) < 13 or len(digits) > 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def is_valid_isbn10(isbn: str) -> bool:
    """ISBN-10 validation (checksum)."""
    digits = ''.join(c for c in str(isbn) if c.isdigit() or c == 'X')
    if len(digits) != 10:
        return False
    total = 0
    for i, c in enumerate(digits):
        val = 10 if c == 'X' else int(c)
        total += val * (10 - i)
    return total % 11 == 0


def is_valid_isbn13(isbn: str) -> bool:
    """ISBN-13 validation (checksum)."""
    digits = ''.join(c for c in str(isbn) if c.isdigit())
    if len(digits) != 13:
        return False
    total = sum(int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(digits))
    return total % 10 == 0


def is_valid_ean(code: str) -> bool:
    """EAN-13 barcode validation."""
    return is_valid_isbn13(code)


def is_valid_mac(mac: str) -> bool:
    """MAC address validation."""
    return bool(re.match(r'^(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}$', str(mac)))


def is_valid_semver(version: str) -> bool:
    """Semantic version validation."""
    return bool(re.match(r'^\d+\.\d+\.\d+(?:-[a-zA-Z0-9.]+)?(?:\+[a-zA-Z0-9.]+)?$', str(version)))


def is_valid_date(date_str: str) -> bool:
    """ISO 8601 date validation (YYYY-MM-DD)."""
    import datetime
    try:
        datetime.date.fromisoformat(str(date_str))
        return True
    except ValueError:
        return False


def is_valid_json(text: str) -> bool:
    """JSON format validation."""
    import json
    try:
        json.loads(str(text))
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def is_valid_base64(text: str) -> bool:
    """Base64 format validation."""
    import base64
    try:
        decoded = base64.b64decode(str(text), validate=True)
        return base64.b64encode(decoded).decode() == str(text)
    except Exception:
        return False


def sanitize_html(text: str) -> str:
    """Strip HTML tags from text."""
    return re.sub(r'<[^>]+>', '', str(text))


def sanitize_sql(text: str) -> str:
    """Escape single quotes for SQL (NOT a substitute for parameterized queries)."""
    return str(text).replace("'", "''")


def sanitize_filename(name: str) -> str:
    """Remove unsafe characters from a filename."""
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', str(name))


def is_strong_password(password: str) -> dict:
    """Check password strength requirements."""
    pw = str(password)
    checks = {
        "min_length_8": len(pw) >= 8,
        "has_uppercase": bool(re.search(r'[A-Z]', pw)),
        "has_lowercase": bool(re.search(r'[a-z]', pw)),
        "has_digit": bool(re.search(r'\d', pw)),
        "has_special": bool(re.search(r'[^a-zA-Z\d]', pw)),
        "min_length_12": len(pw) >= 12,
    }
    checks["passed"] = all(v for k, v in checks.items() if k != "min_length_12")
    checks["strong"] = checks["passed"] and checks["min_length_12"]
    return checks


VALIDATION_FUNCTIONS = {
    "is_valid_email": is_valid_email,
    "is_valid_url": is_valid_url,
    "is_valid_ipv4": is_valid_ipv4,
    "is_valid_ipv6": is_valid_ipv6,
    "is_valid_uuid": is_valid_uuid,
    "is_valid_hex_color": is_valid_hex_color,
    "is_valid_credit_card": is_valid_credit_card,
    "is_valid_isbn10": is_valid_isbn10,
    "is_valid_isbn13": is_valid_isbn13,
    "is_valid_ean": is_valid_ean,
    "is_valid_mac": is_valid_mac,
    "is_valid_semver": is_valid_semver,
    "is_valid_date": is_valid_date,
    "is_valid_json": is_valid_json,
    "is_valid_base64": is_valid_base64,
    "sanitize_html": sanitize_html,
    "sanitize_sql": sanitize_sql,
    "sanitize_filename": sanitize_filename,
    "is_strong_password": is_strong_password,
}

VALIDATION_NL_PATTERNS = [
    (r'(?:is|validate)\s+["\']?(.+?)["\']?\s+(?:a\s+)?valid\s+(email|url|ip|ipv4|ipv6|uuid|mac|semver|date|json|base64)', None),
    (r'(?:is)\s+["\']?(\d[\d\s-]+)["\']?\s+(?:a\s+)?valid\s+(?:credit card|ISBN)', None),
    (r'(?:sanitize|strip|clean)\s+(?:HTML|html)\s+(?:from|in|tags)', None),
    (r'(?:check|validate)\s+password\s+strength', None),
]
