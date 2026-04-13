"""
CALM Cryptography operations backend — hashing, encoding, key derivation.

Actual hash/encoding computations using stdlib. No secrets generated.
"""

from __future__ import annotations

import hashlib
import base64
import hmac


def md5_hash(text: str) -> str:
    """MD5 hash of text (hex digest). NOT for security — checksums only."""
    return hashlib.md5(str(text).encode()).hexdigest()


def sha1_hash(text: str) -> str:
    """SHA-1 hash of text (hex digest). NOT for security — legacy only."""
    return hashlib.sha1(str(text).encode()).hexdigest()


def sha256_hash(text: str) -> str:
    """SHA-256 hash of text (hex digest)."""
    return hashlib.sha256(str(text).encode()).hexdigest()


def sha512_hash(text: str) -> str:
    """SHA-512 hash of text (hex digest)."""
    return hashlib.sha512(str(text).encode()).hexdigest()


def base64_encode(text: str) -> str:
    """Base64 encode text."""
    return base64.b64encode(str(text).encode()).decode()


def base64_decode(encoded: str) -> str:
    """Base64 decode text."""
    try:
        return base64.b64decode(str(encoded)).decode()
    except Exception as e:
        return f"error: {e}"


def base64url_encode(text: str) -> str:
    """URL-safe base64 encode (used in JWT)."""
    return base64.urlsafe_b64encode(str(text).encode()).decode().rstrip('=')


def base64url_decode(encoded: str) -> str:
    """URL-safe base64 decode."""
    try:
        padded = str(encoded) + '=' * (4 - len(encoded) % 4)
        return base64.urlsafe_b64decode(padded).decode()
    except Exception as e:
        return f"error: {e}"


def hex_encode(text: str) -> str:
    """Hex encode text."""
    return str(text).encode().hex()


def hex_decode(hex_str: str) -> str:
    """Hex decode to text."""
    try:
        return bytes.fromhex(str(hex_str)).decode()
    except Exception as e:
        return f"error: {e}"


def hmac_sha256(key: str, message: str) -> str:
    """HMAC-SHA256 of message with key (hex digest)."""
    return hmac.new(str(key).encode(), str(message).encode(), hashlib.sha256).hexdigest()


def hash_compare(text: str) -> dict:
    """Compute MD5, SHA-1, SHA-256, SHA-512 of text for comparison."""
    t = str(text).encode()
    return {
        "md5": hashlib.md5(t).hexdigest(),
        "sha1": hashlib.sha1(t).hexdigest(),
        "sha256": hashlib.sha256(t).hexdigest(),
        "sha512": hashlib.sha512(t).hexdigest(),
    }


def rot13(text: str) -> str:
    """ROT13 cipher (Caesar cipher with shift 13)."""
    result = []
    for c in str(text):
        if 'a' <= c <= 'z':
            result.append(chr((ord(c) - ord('a') + 13) % 26 + ord('a')))
        elif 'A' <= c <= 'Z':
            result.append(chr((ord(c) - ord('A') + 13) % 26 + ord('A')))
        else:
            result.append(c)
    return ''.join(result)


def caesar_cipher(text: str, shift: int) -> str:
    """Caesar cipher with arbitrary shift."""
    s = int(shift) % 26
    result = []
    for c in str(text):
        if 'a' <= c <= 'z':
            result.append(chr((ord(c) - ord('a') + s) % 26 + ord('a')))
        elif 'A' <= c <= 'Z':
            result.append(chr((ord(c) - ord('A') + s) % 26 + ord('A')))
        else:
            result.append(c)
    return ''.join(result)


def xor_strings(s1: str, s2: str) -> str:
    """XOR two equal-length hex strings."""
    try:
        b1 = bytes.fromhex(str(s1))
        b2 = bytes.fromhex(str(s2))
        if len(b1) != len(b2):
            return "error: different lengths"
        return bytes(a ^ b for a, b in zip(b1, b2)).hex()
    except Exception as e:
        return f"error: {e}"


def entropy_bits(password: str) -> float:
    """Estimate entropy of a password in bits."""
    import math
    import re
    pw = str(password)
    charset = 0
    if re.search(r'[a-z]', pw): charset += 26
    if re.search(r'[A-Z]', pw): charset += 26
    if re.search(r'\d', pw): charset += 10
    if re.search(r'[^a-zA-Z\d]', pw): charset += 32
    if charset == 0:
        return 0.0
    return round(len(pw) * math.log2(charset), 1)


CRYPTO_FUNCTIONS = {
    "md5_hash": md5_hash,
    "sha1_hash": sha1_hash,
    "sha256_hash": sha256_hash,
    "sha512_hash": sha512_hash,
    "base64_encode": base64_encode,
    "base64_decode": base64_decode,
    "base64url_encode": base64url_encode,
    "base64url_decode": base64url_decode,
    "hex_encode": hex_encode,
    "hex_decode": hex_decode,
    "hmac_sha256": hmac_sha256,
    "hash_compare": hash_compare,
    "rot13": rot13,
    "caesar_cipher": caesar_cipher,
    "xor_strings": xor_strings,
    "entropy_bits": entropy_bits,
}

CRYPTO_NL_PATTERNS = [
    (r'(?:SHA-?256|sha256)\s+(?:hash\s+)?(?:of|for)\s+["\']?(.+?)["\']?$', 'sha256_hash("{0}")'),
    (r'(?:MD5|md5)\s+(?:hash\s+)?(?:of|for)\s+["\']?(.+?)["\']?$', 'md5_hash("{0}")'),
    (r'base64\s+(?:encode|encoding)\s+(?:of|for)\s+["\']?(.+?)["\']?$', 'base64_encode("{0}")'),
    (r'base64\s+decode\s+["\']?(.+?)["\']?$', 'base64_decode("{0}")'),
    (r'rot13\s+(?:of|for)\s+["\']?(.+?)["\']?$', 'rot13("{0}")'),
    (r'(?:entropy|bits of entropy)\s+(?:of|for|in)\s+["\']?(\S+)["\']?', 'entropy_bits("{0}")'),
]
