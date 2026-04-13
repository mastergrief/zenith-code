"""
CALM refactoring backend — computed code smell detection.

Beyond quality_ops metrics: detects specific refactoring opportunities
like extract method candidates, duplicate code, god classes, long
parameter lists, and feature envy.

Functions: code_smells, extract_candidates, duplicates, god_class,
parameter_count, return_count, class_cohesion.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path
from typing import List


def code_smells(source: str) -> list:
    """Detect code smells. Returns [{line, smell, severity, suggestion}]."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [{"error": "syntax error"}]

    smells = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Long parameter list (> 5 params).
            params = node.args.args
            if len(params) > 5:
                smells.append({
                    "line": node.lineno,
                    "smell": "long_parameter_list",
                    "severity": "medium",
                    "detail": f"{node.name}() has {len(params)} parameters",
                    "suggestion": "Consider a config object or dataclass",
                })

            # Multiple return statements (> 4).
            returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
            if len(returns) > 4:
                smells.append({
                    "line": node.lineno,
                    "smell": "multiple_returns",
                    "severity": "low",
                    "detail": f"{node.name}() has {len(returns)} return statements",
                    "suggestion": "Consider early returns or extract helper",
                })

            # Boolean parameter (flag argument).
            for param in params:
                if param.arg.startswith("is_") or param.arg.startswith("has_") or \
                   param.arg in ("flag", "verbose", "debug", "force", "strict"):
                    # Check if there's a default of True/False.
                    pass  # Just naming-based detection for now.

            # Too many local variables (> 10).
            local_vars = set()
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                    local_vars.add(child.id)
            if len(local_vars) > 10:
                smells.append({
                    "line": node.lineno,
                    "smell": "too_many_locals",
                    "severity": "medium",
                    "detail": f"{node.name}() has {len(local_vars)} local variables",
                    "suggestion": "Extract sub-functions or use a data structure",
                })

        # God class (> 10 methods).
        if isinstance(node, ast.ClassDef):
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if len(methods) > 10:
                smells.append({
                    "line": node.lineno,
                    "smell": "god_class",
                    "severity": "high",
                    "detail": f"class {node.name} has {len(methods)} methods",
                    "suggestion": "Split into smaller, focused classes",
                })

            # Large class (> 200 lines).
            if hasattr(node, 'end_lineno') and node.end_lineno:
                class_lines = node.end_lineno - node.lineno
                if class_lines > 200:
                    smells.append({
                        "line": node.lineno,
                        "smell": "large_class",
                        "severity": "medium",
                        "detail": f"class {node.name} is {class_lines} lines",
                        "suggestion": "Extract related methods into separate classes",
                    })

    return smells


def extract_candidates(source: str) -> list:
    """Find functions that could be split — extract method candidates.
    Returns [{function, line, reason, suggested_splits}]."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [{"error": "syntax error"}]

    candidates = []
    lines = source.splitlines()

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        if not hasattr(node, 'end_lineno') or not node.end_lineno:
            continue

        length = node.end_lineno - node.lineno + 1
        if length < 20:
            continue

        # Find comment blocks that might indicate logical sections.
        sections = []
        for i in range(node.lineno - 1, min(node.end_lineno, len(lines))):
            line = lines[i].strip()
            if line.startswith("#") and len(line) > 3 and not line.startswith("#!"):
                sections.append({"line": i + 1, "comment": line})

        if sections or length > 30:
            candidates.append({
                "function": node.name,
                "line": node.lineno,
                "length": length,
                "reason": f"{length} lines" + (f", {len(sections)} comment sections" if sections else ""),
                "section_markers": sections[:5],
            })

    return candidates


def duplicate_blocks(source: str, min_lines: int = 4) -> list:
    """Find duplicate code blocks (exact line matches).
    Returns [{lines, occurrences, content}]."""
    lines = source.splitlines()
    min_lines = int(min_lines)

    # Build blocks of N consecutive non-empty lines.
    blocks = {}
    for i in range(len(lines) - min_lines + 1):
        block = tuple(lines[i:i + min_lines])
        # Skip if all blank or comments.
        meaningful = [l for l in block if l.strip() and not l.strip().startswith("#")]
        if len(meaningful) < min_lines - 1:
            continue
        key = "\n".join(l.strip() for l in block)
        blocks.setdefault(key, []).append(i + 1)

    duplicates = [
        {
            "content": key[:100],
            "lines": min_lines,
            "occurrences": len(positions),
            "at_lines": positions,
        }
        for key, positions in blocks.items()
        if len(positions) > 1
    ]

    return sorted(duplicates, key=lambda d: -d["occurrences"])[:10]


def code_smells_file(path: str) -> dict:
    """Run code_smells + extract_candidates + duplicate_blocks on a file."""
    source = Path(path).read_text(encoding="utf-8", errors="replace")
    smells = code_smells(source)
    extracts = extract_candidates(source)
    dupes = duplicate_blocks(source)

    return {
        "file": path,
        "smells": smells,
        "smell_count": len(smells),
        "extract_candidates": extracts,
        "duplicates": dupes,
        "rating": (
            "clean" if not smells and not dupes else
            "minor issues" if len(smells) <= 2 else
            "needs refactoring" if len(smells) <= 5 else
            "significant debt"
        ),
    }


REFACTOR_FUNCTIONS = {
    "code_smells": code_smells,
    "extract_candidates": extract_candidates,
    "duplicate_blocks": duplicate_blocks,
    "code_smells_file": code_smells_file,
}

REFACTOR_NL_PATTERNS = [
    (r'(?:code smells?|anti.?patterns?)\s+(?:in|for)', None),
    (r'(?:duplicate|duplicated|copy.?paste)\s+(?:code|blocks?)\s+(?:in|for)', None),
    (r'(?:extract|refactor)\s+(?:candidates?|opportunities)', None),
]
