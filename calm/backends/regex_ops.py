"""
CALM regex backend — verified pattern matching.

The model writes "this regex matches emails" — the engine tests it
against actual strings and reports exact matches.

Functions: test, find_all, match_groups, replace, split, explain.
"""

from __future__ import annotations

import re
from typing import List, Optional


def regex_test(pattern: str, text: str) -> bool:
    """Test if a pattern matches anywhere in text."""
    try:
        return bool(re.search(pattern, text))
    except re.error as e:
        return f"regex error: {e}"


def regex_find_all(pattern: str, text: str) -> list:
    """Find all matches of a pattern in text."""
    try:
        return re.findall(pattern, text)
    except re.error as e:
        return [f"regex error: {e}"]


def regex_match_groups(pattern: str, text: str) -> Optional[dict]:
    """Match pattern and return named/numbered groups."""
    try:
        m = re.search(pattern, text)
        if not m:
            return None
        result = {"full": m.group(0), "groups": list(m.groups())}
        if m.groupdict():
            result["named"] = m.groupdict()
        return result
    except re.error as e:
        return {"error": str(e)}


def regex_replace(pattern: str, replacement: str, text: str) -> str:
    """Replace all matches of pattern in text."""
    try:
        return re.sub(pattern, replacement, text)
    except re.error as e:
        return f"regex error: {e}"


def regex_split(pattern: str, text: str) -> list:
    """Split text by pattern."""
    try:
        return re.split(pattern, text)
    except re.error as e:
        return [f"regex error: {e}"]


def regex_validate(pattern: str) -> dict:
    """Check if a regex pattern is valid. Returns {valid, error, groups}."""
    try:
        compiled = re.compile(pattern)
        return {
            "valid": True,
            "groups": compiled.groups,
            "groupindex": dict(compiled.groupindex) if compiled.groupindex else {},
        }
    except re.error as e:
        return {"valid": False, "error": str(e)}


def regex_count(pattern: str, text: str) -> int:
    """Count matches of pattern in text."""
    try:
        return len(re.findall(pattern, text))
    except re.error:
        return 0


REGEX_FUNCTIONS = {
    "regex_test": regex_test,
    "regex_find_all": regex_find_all,
    "regex_match_groups": regex_match_groups,
    "regex_replace": regex_replace,
    "regex_split": regex_split,
    "regex_validate": regex_validate,
    "regex_count": regex_count,
}

REGEX_NL_PATTERNS = [
    (r'(?:does|test|check)\s+["\'](.+?)["\']\s+match\s+(?:the\s+)?(?:regex|pattern)\s+["\'](.+?)["\']', 'regex_test("{0}", "{1}")'),
    (r'(?:find all|extract)\s+(?:matches\s+)?(?:of\s+)?(?:regex|pattern)\s+["\'](.+?)["\']\s+in\s+["\'](.+?)["\']', 'regex_find_all("{1}", "{0}")'),
    (r'(?:is)\s+["\'](.+?)["\']\s+(?:a\s+)?valid\s+regex', 'regex_validate("{0}")'),
    (r'(?:count)\s+(?:matches\s+)?(?:of\s+)?(?:regex|pattern)\s+["\'](.+?)["\']\s+in\s+["\'](.+?)["\']', 'regex_count("{1}", "{0}")'),
]
