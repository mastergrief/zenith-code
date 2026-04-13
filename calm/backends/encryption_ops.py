"""
CALM Encryption/security reference backend — hash comparison, key sizes, password strength.

Models confuse bcrypt vs argon2, hallucinate key sizes, give wrong hash lengths.
No actual crypto operations — just reference data and strength analysis.
"""

from __future__ import annotations

import math
import re


_HASH_ALGORITHMS = {
    "md5": {"output_bits": 128, "output_hex": 32, "secure": False, "use": "checksums only, NOT passwords", "collision": "trivial (2^18)"},
    "sha1": {"output_bits": 160, "output_hex": 40, "secure": False, "use": "legacy git, NOT passwords", "collision": "feasible (SHAttered, 2017)"},
    "sha256": {"output_bits": 256, "output_hex": 64, "secure": True, "use": "data integrity, HMAC, blockchain", "collision": "infeasible"},
    "sha384": {"output_bits": 384, "output_hex": 96, "secure": True, "use": "TLS, certificates", "collision": "infeasible"},
    "sha512": {"output_bits": 512, "output_hex": 128, "secure": True, "use": "high-security integrity", "collision": "infeasible"},
    "sha3-256": {"output_bits": 256, "output_hex": 64, "secure": True, "use": "post-quantum consideration", "collision": "infeasible"},
    "blake2b": {"output_bits": 512, "output_hex": 128, "secure": True, "use": "fast hashing, file integrity", "collision": "infeasible"},
    "blake3": {"output_bits": 256, "output_hex": 64, "secure": True, "use": "fastest secure hash, parallelizable", "collision": "infeasible"},
    "bcrypt": {"output_bits": 184, "output_hex": "60-char encoded", "secure": True, "use": "password hashing", "type": "adaptive KDF", "max_input": 72},
    "argon2": {"output_bits": "variable", "output_hex": "variable", "secure": True, "use": "password hashing (recommended)", "type": "memory-hard KDF", "variants": ["argon2d", "argon2i", "argon2id"]},
    "scrypt": {"output_bits": "variable", "output_hex": "variable", "secure": True, "use": "password hashing (memory-hard)", "type": "memory-hard KDF"},
    "pbkdf2": {"output_bits": "variable", "output_hex": "variable", "secure": True, "use": "password hashing (legacy, use argon2)", "type": "iterative KDF", "min_iterations": 600000},
}

_KEY_SIZES = {
    "aes-128": {"bits": 128, "security_level": 128, "type": "symmetric", "status": "secure"},
    "aes-192": {"bits": 192, "security_level": 192, "type": "symmetric", "status": "secure"},
    "aes-256": {"bits": 256, "security_level": 256, "type": "symmetric", "status": "secure, post-quantum safe"},
    "rsa-2048": {"bits": 2048, "security_level": 112, "type": "asymmetric", "status": "minimum acceptable"},
    "rsa-3072": {"bits": 3072, "security_level": 128, "type": "asymmetric", "status": "recommended"},
    "rsa-4096": {"bits": 4096, "security_level": 140, "type": "asymmetric", "status": "high security"},
    "ed25519": {"bits": 256, "security_level": 128, "type": "asymmetric (EdDSA)", "status": "recommended for signatures"},
    "x25519": {"bits": 256, "security_level": 128, "type": "key exchange (ECDH)", "status": "recommended"},
    "p-256": {"bits": 256, "security_level": 128, "type": "asymmetric (ECDSA)", "status": "NIST standard"},
    "p-384": {"bits": 384, "security_level": 192, "type": "asymmetric (ECDSA)", "status": "NIST standard"},
    "chacha20": {"bits": 256, "security_level": 256, "type": "symmetric stream", "status": "secure, mobile-friendly"},
}


def hash_info(algorithm: str) -> dict:
    """Get details about a hash algorithm: output size, security status, use case."""
    key = str(algorithm).lower().strip().replace("-", "").replace("_", "")
    # Normalize common variants
    for k, v in _HASH_ALGORITHMS.items():
        if key == k.replace("-", "").replace("_", ""):
            return {"algorithm": k, **v}
    return {"error": f"Unknown algorithm: {algorithm}", "valid": list(_HASH_ALGORITHMS.keys())}


def key_size_info(algorithm: str) -> dict:
    """Get key size and security level for a cipher/algorithm."""
    key = str(algorithm).lower().strip().replace("_", "-")
    entry = _KEY_SIZES.get(key)
    if not entry:
        return {"error": f"Unknown: {algorithm}", "valid": list(_KEY_SIZES.keys())}
    return {"algorithm": key, **entry}


def password_strength(password: str) -> dict:
    """Estimate password strength: entropy, crack time, weaknesses."""
    pw = str(password)
    charset = 0
    has_lower = bool(re.search(r'[a-z]', pw))
    has_upper = bool(re.search(r'[A-Z]', pw))
    has_digit = bool(re.search(r'\d', pw))
    has_special = bool(re.search(r'[^a-zA-Z0-9]', pw))

    if has_lower: charset += 26
    if has_upper: charset += 26
    if has_digit: charset += 10
    if has_special: charset += 32

    entropy = round(len(pw) * math.log2(max(charset, 1)), 1) if charset > 0 else 0

    weaknesses = []
    if len(pw) < 8: weaknesses.append("too short (< 8 chars)")
    if len(pw) < 12: weaknesses.append("should be 12+ chars")
    if not has_upper: weaknesses.append("no uppercase")
    if not has_lower: weaknesses.append("no lowercase")
    if not has_digit: weaknesses.append("no digits")
    if not has_special: weaknesses.append("no special chars")
    if pw.lower() in ('password', '123456', 'qwerty', 'admin', 'letmein'):
        weaknesses.append("common password")
        entropy = 0

    if entropy >= 80: strength = "strong"
    elif entropy >= 60: strength = "good"
    elif entropy >= 40: strength = "moderate"
    elif entropy >= 20: strength = "weak"
    else: strength = "very weak"

    # Rough crack time at 10B guesses/sec (offline GPU attack)
    if entropy > 0:
        seconds = 2 ** entropy / 1e10
        if seconds < 1: crack_time = "instant"
        elif seconds < 60: crack_time = f"{seconds:.0f} seconds"
        elif seconds < 3600: crack_time = f"{seconds/60:.0f} minutes"
        elif seconds < 86400: crack_time = f"{seconds/3600:.0f} hours"
        elif seconds < 31536000: crack_time = f"{seconds/86400:.0f} days"
        elif seconds < 31536000 * 1000: crack_time = f"{seconds/31536000:.0f} years"
        else: crack_time = f"{seconds/31536000:.1e} years"
    else:
        crack_time = "instant"

    return {
        "length": len(pw),
        "entropy_bits": entropy,
        "strength": strength,
        "crack_time_10B_guesses_sec": crack_time,
        "charset_size": charset,
        "weaknesses": weaknesses,
    }


def hash_output_length(algorithm: str) -> int:
    """Output length in hex characters for a hash algorithm."""
    info = hash_info(algorithm)
    if "error" in info:
        return -1
    val = info.get("output_hex")
    return val if isinstance(val, int) else -1


def is_hash_secure(algorithm: str) -> bool:
    """Whether a hash algorithm is currently considered secure."""
    info = hash_info(algorithm)
    return info.get("secure", False)


def compare_hashes(alg1: str, alg2: str) -> dict:
    """Compare two hash algorithms."""
    h1 = hash_info(alg1)
    h2 = hash_info(alg2)
    if "error" in h1 or "error" in h2:
        return {"error": "Unknown algorithm", "h1": h1, "h2": h2}
    return {
        "algorithm_1": {k: v for k, v in h1.items()},
        "algorithm_2": {k: v for k, v in h2.items()},
        "recommendation": h1.get("use", "") if h1.get("secure") else h2.get("use", ""),
    }


def bcrypt_vs_argon2() -> dict:
    """Compare bcrypt and argon2 for password hashing."""
    return {
        "bcrypt": {"max_password": "72 bytes", "memory_hard": False, "gpu_resistant": "moderate", "standard": "legacy, widely deployed"},
        "argon2id": {"max_password": "unlimited", "memory_hard": True, "gpu_resistant": "strong", "standard": "OWASP recommended, RFC 9106"},
        "recommendation": "Use argon2id for new applications. bcrypt is acceptable for existing systems.",
        "min_params": {"bcrypt_cost": 12, "argon2_memory": "64 MiB", "argon2_iterations": 3, "argon2_parallelism": 1},
    }


ENCRYPTION_FUNCTIONS = {
    "hash_info": hash_info,
    "key_size_info": key_size_info,
    "password_strength": password_strength,
    "hash_output_length": hash_output_length,
    "is_hash_secure": is_hash_secure,
    "compare_hashes": compare_hashes,
    "bcrypt_vs_argon2": bcrypt_vs_argon2,
}

ENCRYPTION_NL_PATTERNS = [
    (r'(?:what is|explain|info about)\s+(\w+)\s+(?:hash|hashing)', 'hash_info("{0}")'),
    (r'(?:is)\s+(\w+)\s+(?:hash\s+)?(?:secure|safe|broken)', 'is_hash_secure("{0}")'),
    (r'(?:output|length|size)\s+(?:of\s+)?(\w+)\s+hash', 'hash_output_length("{0}")'),
    (r'(?:compare|difference|vs)\s+(?:between\s+)?bcrypt\s+(?:and|vs)\s+argon2', 'bcrypt_vs_argon2()'),
    (r'(?:compare|difference|vs)\s+(?:between\s+)?(\w+)\s+(?:and|vs)\s+(\w+)\s+hash', 'compare_hashes("{0}", "{1}")'),
    (r'(?:key size|security level)\s+(?:of|for)\s+([\w-]+)', 'key_size_info("{0}")'),
    (r'(?:password\s+)?strength\s+(?:of|for)\s+["\']?(\S+)["\']?', 'password_strength("{0}")'),
    (r'(?:how\s+)?(?:strong|secure)\s+(?:is\s+)?(?:the\s+)?password\s+["\']?(\S+)["\']?', 'password_strength("{0}")'),
]
