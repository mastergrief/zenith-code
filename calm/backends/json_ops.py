"""
CALM JSON/data format backend — verified structure operations.

The model writes "this JSON has 3 keys" — the engine parses and counts.

Functions: validate, path extraction, diff, schema check, transform.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional, Union


def json_validate(text: str) -> dict:
    """Validate JSON string. Returns {valid, error, type, keys/length}."""
    try:
        data = json.loads(text)
        result = {"valid": True, "type": type(data).__name__}
        if isinstance(data, dict):
            result["keys"] = list(data.keys())
            result["key_count"] = len(data)
        elif isinstance(data, list):
            result["length"] = len(data)
        return result
    except json.JSONDecodeError as e:
        return {"valid": False, "error": str(e), "line": e.lineno, "col": e.colno}


def json_path(data: Any, path: str) -> Any:
    """Extract a value by dot-notation path. Supports array indices.
    Example: json_path(data, "users.0.name") → "Alice" """
    current = data
    for key in path.split("."):
        if isinstance(current, dict):
            if key in current:
                current = current[key]
            else:
                return None
        elif isinstance(current, list):
            try:
                current = current[int(key)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return current


def json_keys(data: Any, depth: int = 1) -> list:
    """List all keys at given depth. depth=1 for top-level."""
    if depth <= 0 or not isinstance(data, dict):
        return []
    if depth == 1:
        return list(data.keys())
    result = []
    for k, v in data.items():
        sub = json_keys(v, depth - 1)
        result.extend(f"{k}.{s}" for s in sub)
    return result


def json_flatten(data: Any, prefix: str = "") -> dict:
    """Flatten nested JSON to dot-notation keys.
    {"a": {"b": 1}} → {"a.b": 1}"""
    result = {}
    if isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{prefix}.{k}" if prefix else k
            result.update(json_flatten(v, new_key))
    elif isinstance(data, list):
        for i, v in enumerate(data):
            new_key = f"{prefix}.{i}" if prefix else str(i)
            result.update(json_flatten(v, new_key))
    else:
        result[prefix] = data
    return result


def json_diff(a: Any, b: Any, path: str = "") -> list:
    """Diff two JSON structures. Returns list of {path, type, old, new}."""
    diffs = []
    if type(a) != type(b):
        diffs.append({"path": path or "/", "type": "type_change",
                      "old": type(a).__name__, "new": type(b).__name__})
        return diffs
    if isinstance(a, dict):
        all_keys = set(a.keys()) | set(b.keys())
        for k in sorted(all_keys):
            p = f"{path}.{k}" if path else k
            if k not in a:
                diffs.append({"path": p, "type": "added", "new": b[k]})
            elif k not in b:
                diffs.append({"path": p, "type": "removed", "old": a[k]})
            else:
                diffs.extend(json_diff(a[k], b[k], p))
    elif isinstance(a, list):
        for i in range(max(len(a), len(b))):
            p = f"{path}.{i}" if path else str(i)
            if i >= len(a):
                diffs.append({"path": p, "type": "added", "new": b[i]})
            elif i >= len(b):
                diffs.append({"path": p, "type": "removed", "old": a[i]})
            else:
                diffs.extend(json_diff(a[i], b[i], p))
    elif a != b:
        diffs.append({"path": path or "/", "type": "changed", "old": a, "new": b})
    return diffs


def json_format(text: str, indent: int = 2) -> str:
    """Pretty-print a JSON string."""
    try:
        return json.dumps(json.loads(text), indent=indent, ensure_ascii=False)
    except json.JSONDecodeError as e:
        return f"invalid JSON: {e}"


def json_minify(text: str) -> str:
    """Minify a JSON string."""
    try:
        return json.dumps(json.loads(text), separators=(",", ":"))
    except json.JSONDecodeError as e:
        return f"invalid JSON: {e}"


JSON_FUNCTIONS = {
    "json_validate": json_validate,
    "json_path": json_path,
    "json_keys": json_keys,
    "json_flatten": json_flatten,
    "json_diff": json_diff,
    "json_format": json_format,
    "json_minify": json_minify,
}

JSON_NL_PATTERNS = [
    (r'(?:validate|is valid|check)\s+(?:this\s+)?(?:JSON|json)', None),
    (r'(?:format|pretty.?print|indent)\s+(?:this\s+)?(?:JSON|json)', None),
    (r'(?:minify|compact|compress)\s+(?:this\s+)?(?:JSON|json)', None),
    (r'(?:diff|compare)\s+(?:these\s+)?(?:two\s+)?(?:JSON|json)', None),
]
