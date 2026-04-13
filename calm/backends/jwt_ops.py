"""
CALM JWT backend — decode, inspect, validate structure.

Models hallucinate JWT fields and structure. Pure base64+json decode
(no signature verification — that requires keys).
"""

from __future__ import annotations

import base64
import json
import time


def _b64url_decode(s: str) -> bytes:
    """Decode base64url (JWT variant) with padding fix."""
    s = s.replace("-", "+").replace("_", "/")
    pad = 4 - len(s) % 4
    if pad != 4:
        s += "=" * pad
    return base64.b64decode(s)


def jwt_decode_header(token: str) -> str:
    """Decode JWT header (first segment). Returns JSON string."""
    try:
        parts = token.strip().split(".")
        if len(parts) != 3:
            return f"invalid JWT: expected 3 parts, got {len(parts)}"
        raw = _b64url_decode(parts[0])
        return json.dumps(json.loads(raw), indent=2)
    except Exception as e:
        return f"decode error: {e}"


def jwt_decode_payload(token: str) -> str:
    """Decode JWT payload (second segment). Returns JSON string."""
    try:
        parts = token.strip().split(".")
        if len(parts) != 3:
            return f"invalid JWT: expected 3 parts, got {len(parts)}"
        raw = _b64url_decode(parts[1])
        return json.dumps(json.loads(raw), indent=2)
    except Exception as e:
        return f"decode error: {e}"


def jwt_claims(token: str) -> dict:
    """Extract all claims from JWT payload as a dict."""
    try:
        parts = token.strip().split(".")
        if len(parts) != 3:
            return {"error": f"invalid JWT: expected 3 parts, got {len(parts)}"}
        raw = _b64url_decode(parts[1])
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e)}


def jwt_algorithm(token: str) -> str:
    """Extract the signing algorithm from JWT header."""
    try:
        parts = token.strip().split(".")
        header = json.loads(_b64url_decode(parts[0]))
        return header.get("alg", "none")
    except Exception as e:
        return f"error: {e}"


def jwt_is_expired(token: str) -> str:
    """Check if JWT is expired based on 'exp' claim. Returns status string."""
    try:
        parts = token.strip().split(".")
        payload = json.loads(_b64url_decode(parts[1]))
        exp = payload.get("exp")
        if exp is None:
            return "no exp claim"
        now = int(time.time())
        if now > int(exp):
            return f"expired ({now - int(exp)}s ago)"
        return f"valid ({int(exp) - now}s remaining)"
    except Exception as e:
        return f"error: {e}"


def jwt_validate_structure(token: str) -> str:
    """Validate JWT structure (3 base64url segments, valid JSON header/payload)."""
    try:
        parts = token.strip().split(".")
        if len(parts) != 3:
            return f"invalid: expected 3 parts, got {len(parts)}"
        header = json.loads(_b64url_decode(parts[0]))
        if not isinstance(header, dict):
            return "invalid: header is not a JSON object"
        if "alg" not in header:
            return "invalid: header missing 'alg' field"
        payload = json.loads(_b64url_decode(parts[1]))
        if not isinstance(payload, dict):
            return "invalid: payload is not a JSON object"
        return f"valid (alg={header.get('alg')}, {len(payload)} claims)"
    except Exception as e:
        return f"invalid: {e}"


def jwt_part_count(token: str) -> int:
    """Count the number of dot-separated parts in a JWT."""
    return len(token.strip().split("."))


JWT_FUNCTIONS = {
    "jwt_decode_header": jwt_decode_header,
    "jwt_decode_payload": jwt_decode_payload,
    "jwt_claims": jwt_claims,
    "jwt_algorithm": jwt_algorithm,
    "jwt_is_expired": jwt_is_expired,
    "jwt_validate_structure": jwt_validate_structure,
    "jwt_part_count": jwt_part_count,
}
