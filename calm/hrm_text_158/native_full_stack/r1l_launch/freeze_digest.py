"""CONTENT_DIGEST for R1-L freeze/fixture member tables.

Digest rule (stdlib only — not canonical-JSON hashing):
  sha256(concat over sorted(basename_utf8 + b'\\0' + raw_32byte_digest))
"""
from __future__ import annotations

import hashlib
from typing import Mapping


def content_digest_from_members(members: Mapping[str, str]) -> str:
    """Return hex digest for ``{basename: sha256_hex}`` member map."""
    parts: list[bytes] = []
    for name in sorted(members.keys()):
        digest_hex = members[name]
        raw = bytes.fromhex(digest_hex)
        if len(raw) != 32:
            raise ValueError(f"member digest must be 32 raw bytes: {name!r} got {len(raw)}")
        parts.append(name.encode("utf-8") + b"\0" + raw)
    return hashlib.sha256(b"".join(parts)).hexdigest()


def content_digest_from_member_records(
    members: Mapping[str, Mapping[str, object]],
    *,
    sha_key: str = "sha256",
) -> str:
    """Digest from records that each carry a sha256 hex field."""
    flat = {str(k): str(v[sha_key]) for k, v in members.items()}
    return content_digest_from_members(flat)
