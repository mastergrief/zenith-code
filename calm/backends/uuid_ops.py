"""
CALM UUID backend — generate, validate, parse, compare.

Models hallucinate UUID formats. Pure stdlib uuid module.
"""

from __future__ import annotations

import uuid
import time


def uuid_v4() -> str:
    """Generate a random UUID v4."""
    return str(uuid.uuid4())


def uuid_v1() -> str:
    """Generate a time-based UUID v1."""
    return str(uuid.uuid1())


def uuid_validate(s: str) -> bool:
    """Check if a string is a valid UUID."""
    try:
        uuid.UUID(s)
        return True
    except (ValueError, AttributeError):
        return False


def uuid_version(s: str) -> int:
    """Extract the version number from a UUID."""
    try:
        return uuid.UUID(s).version
    except (ValueError, AttributeError):
        return -1


def uuid_variant(s: str) -> str:
    """Extract the variant of a UUID."""
    try:
        v = uuid.UUID(s).variant
        if v == uuid.RFC_4122:
            return "RFC 4122"
        elif v == uuid.RESERVED_NCS:
            return "NCS"
        elif v == uuid.RESERVED_MICROSOFT:
            return "Microsoft"
        elif v == uuid.RESERVED_FUTURE:
            return "Future"
        return str(v)
    except (ValueError, AttributeError):
        return "invalid"


def uuid_from_string(name: str, namespace: str = "dns") -> str:
    """Generate a deterministic UUID v5 from a name and namespace."""
    ns_map = {
        "dns": uuid.NAMESPACE_DNS,
        "url": uuid.NAMESPACE_URL,
        "oid": uuid.NAMESPACE_OID,
        "x500": uuid.NAMESPACE_X500,
    }
    ns = ns_map.get(namespace.lower(), uuid.NAMESPACE_DNS)
    return str(uuid.uuid5(ns, name))


def uuid_timestamp_v1(s: str) -> str:
    """Extract timestamp from a UUID v1 (returns ISO 8601 or error)."""
    try:
        u = uuid.UUID(s)
        if u.version != 1:
            return f"not a v1 UUID (version={u.version})"
        # UUID v1 timestamp is 100-ns intervals since 1582-10-15
        import datetime
        ts = (u.time - 0x01B21DD213814000) / 1e7
        dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
        return dt.isoformat()
    except (ValueError, AttributeError, OSError) as e:
        return f"error: {e}"


def uuid_nil() -> str:
    """Return the nil UUID (all zeros)."""
    return "00000000-0000-0000-0000-000000000000"


UUID_FUNCTIONS = {
    "uuid_v4": uuid_v4,
    "uuid_v1": uuid_v1,
    "uuid_validate": uuid_validate,
    "uuid_version": uuid_version,
    "uuid_variant": uuid_variant,
    "uuid_from_string": uuid_from_string,
    "uuid_timestamp_v1": uuid_timestamp_v1,
    "uuid_nil": uuid_nil,
}
