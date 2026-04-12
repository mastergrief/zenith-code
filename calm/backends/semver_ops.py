"""
CALM semantic versioning backend — verified version operations.

Models get "is 2.0 compatible with ^1.5?" wrong constantly.
This backend parses, compares, and checks compatibility deterministically.

Functions: parse, compare, satisfies, bump, is_compatible, sort_versions.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple


def _parse(version: str) -> Tuple[int, int, int, str]:
    """Parse a semver string into (major, minor, patch, prerelease)."""
    v = str(version).strip().lstrip("v")
    m = re.match(r'^(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:-(.+))?$', v)
    if not m:
        return (0, 0, 0, "")
    return (
        int(m.group(1)),
        int(m.group(2) or 0),
        int(m.group(3) or 0),
        m.group(4) or "",
    )


def semver_parse(version: str) -> dict:
    """Parse a version string into components.
    Example: semver_parse("1.2.3-beta.1")
    → {major: 1, minor: 2, patch: 3, prerelease: "beta.1"}"""
    major, minor, patch, pre = _parse(version)
    return {"major": major, "minor": minor, "patch": patch,
            "prerelease": pre, "valid": bool(re.match(r'^\d+\.\d+\.\d+', str(version).lstrip("v")))}


def semver_compare(a: str, b: str) -> int:
    """Compare two versions. Returns -1 (a<b), 0 (equal), 1 (a>b).
    Example: semver_compare("1.2.3", "1.3.0") → -1"""
    pa, pb = _parse(a), _parse(b)
    # Compare major.minor.patch.
    for i in range(3):
        if pa[i] < pb[i]: return -1
        if pa[i] > pb[i]: return 1
    # Prerelease: version with prerelease < version without.
    if pa[3] and not pb[3]: return -1
    if not pa[3] and pb[3]: return 1
    if pa[3] < pb[3]: return -1
    if pa[3] > pb[3]: return 1
    return 0


def semver_satisfies(version: str, constraint: str) -> bool:
    """Check if a version satisfies a constraint.
    Supports: ^1.2.3 (compatible), ~1.2.3 (patch-level), >=, <=, >, <, =.

    Examples:
      semver_satisfies("1.5.0", "^1.2.3") → True (same major, >= minor)
      semver_satisfies("2.0.0", "^1.2.3") → False (different major)
      semver_satisfies("1.2.5", "~1.2.3") → True (same major.minor, >= patch)
    """
    constraint = constraint.strip()
    v = _parse(version)

    if constraint.startswith("^"):
        # Caret: compatible with version (same major, >= specified).
        c = _parse(constraint[1:])
        if v[0] != c[0]: return False  # Major must match.
        return semver_compare(version, constraint[1:]) >= 0

    if constraint.startswith("~"):
        # Tilde: patch-level changes (same major.minor, >= specified).
        c = _parse(constraint[1:])
        if v[0] != c[0] or v[1] != c[1]: return False
        return v[2] >= c[2]

    if constraint.startswith(">="):
        return semver_compare(version, constraint[2:].strip()) >= 0
    if constraint.startswith("<="):
        return semver_compare(version, constraint[2:].strip()) <= 0
    if constraint.startswith(">") and not constraint.startswith(">="):
        return semver_compare(version, constraint[1:].strip()) > 0
    if constraint.startswith("<") and not constraint.startswith("<="):
        return semver_compare(version, constraint[1:].strip()) < 0
    if constraint.startswith("=") or constraint.startswith("=="):
        c = constraint.lstrip("=").strip()
        return semver_compare(version, c) == 0

    # Plain version = exact match.
    return semver_compare(version, constraint) == 0


def semver_bump(version: str, part: str = "patch") -> str:
    """Bump a version. part = "major", "minor", or "patch".
    Example: semver_bump("1.2.3", "minor") → "1.3.0" """
    major, minor, patch, _ = _parse(version)
    if part == "major":
        return f"{major + 1}.0.0"
    elif part == "minor":
        return f"{major}.{minor + 1}.0"
    else:
        return f"{major}.{minor}.{patch + 1}"


def semver_sort(versions: list, reverse: bool = False) -> list:
    """Sort a list of version strings.
    Example: semver_sort(["1.0.0", "2.1.0", "1.5.3"]) → ["1.0.0", "1.5.3", "2.1.0"]"""
    return sorted(versions, key=lambda v: _parse(v), reverse=bool(reverse))


def is_breaking_change(old: str, new: str) -> bool:
    """Check if upgrading from old to new is a breaking change (major bump).
    Example: is_breaking_change("1.5.0", "2.0.0") → True"""
    return _parse(old)[0] != _parse(new)[0]


SEMVER_FUNCTIONS = {
    "semver_parse": semver_parse,
    "semver_compare": semver_compare,
    "semver_satisfies": semver_satisfies,
    "semver_bump": semver_bump,
    "semver_sort": semver_sort,
    "is_breaking_change": is_breaking_change,
}
