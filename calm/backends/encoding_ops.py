"""
CALM encoding/hashing backend — verified data transformations.

The model writes "the base64 of 'hello' is aGVsbG8=" — the engine
verifies by encoding on CPU.

Functions: base64, hex, URL encoding, hashing (md5, sha1, sha256).
"""

from __future__ import annotations

import base64
import hashlib
import urllib.parse


def base64_encode(text: str) -> str:
    """Encode text to base64."""
    return base64.b64encode(text.encode()).decode()


def base64_decode(text: str) -> str:
    """Decode base64 to text."""
    try:
        return base64.b64decode(text.encode()).decode()
    except Exception as e:
        return f"decode error: {e}"


def hex_encode(text: str) -> str:
    """Encode text to hex."""
    return text.encode().hex()


def hex_decode(text: str) -> str:
    """Decode hex to text."""
    try:
        return bytes.fromhex(text).decode()
    except Exception as e:
        return f"decode error: {e}"


def url_encode(text: str) -> str:
    """URL-encode text."""
    return urllib.parse.quote(text, safe="")


def url_decode(text: str) -> str:
    """URL-decode text."""
    return urllib.parse.unquote(text)


def md5(text: str) -> str:
    """MD5 hash of text."""
    return hashlib.md5(text.encode()).hexdigest()


def sha1(text: str) -> str:
    """SHA-1 hash of text."""
    return hashlib.sha1(text.encode()).hexdigest()


def sha256(text: str) -> str:
    """SHA-256 hash of text."""
    return hashlib.sha256(text.encode()).hexdigest()


def char_code(char: str) -> int:
    """Unicode code point of a character."""
    return ord(char[0]) if char else 0


def from_char_code(code: int) -> str:
    """Character from Unicode code point."""
    return chr(int(code))


def byte_length(text: str) -> int:
    """UTF-8 byte length of text."""
    return len(text.encode("utf-8"))


ENCODING_NL_PATTERNS = [
    (r'base64\s+(?:encode|encoding)\s+(?:of\s+)?["\']([^"\']+)["\']', 'base64_encode("{0}")'),
    (r'(?:SHA-?256|sha256)\s+(?:hash|digest)\s+(?:of\s+)?["\']([^"\']+)["\']', 'sha256("{0}")'),
    (r'(?:MD5|md5)\s+(?:hash|digest)\s+(?:of\s+)?["\']([^"\']+)["\']', 'md5("{0}")'),
    (r'(?:ASCII|ascii)\s+(?:code|value)\s+(?:of|for)\s+["\']?(\w)["\']?', 'char_code("{0}")'),
    (r'(?:UTF-?8|utf-?8)\s+(?:byte\s+)?length\s+(?:of\s+)?["\']([^"\']+)["\']', 'byte_length("{0}")'),
]

ENCODING_FUNCTIONS = {
    "base64_encode": base64_encode,
    "base64_decode": base64_decode,
    "hex_encode": hex_encode,
    "hex_decode": hex_decode,
    "url_encode": url_encode,
    "url_decode": url_decode,
    "md5": md5,
    "sha1": sha1,
    "sha256": sha256,
    "char_code": char_code,
    "from_char_code": from_char_code,
    "byte_length": byte_length,
}
