"""
CALM code quality backend — turns "is this code clean?" into numbers.

The model says "this function is too complex" — the engine computes
cyclomatic complexity = 14, max nesting = 6, and the model's vague
judgment becomes a verified fact.

Functions: complexity, naming, structure, duplication, coverage metrics.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List, Optional


def cyclomatic_complexity(source: str) -> list:
    """Compute cyclomatic complexity per function.
    CC = 1 + branches (if/elif/for/while/except/and/or/with/assert).
    Returns [{name, line, complexity, rating}]."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [{"error": "syntax error"}]

    results = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            cc = 1
            for child in ast.walk(node):
                if isinstance(child, (ast.If, ast.IfExp)):
                    cc += 1
                elif isinstance(child, (ast.For, ast.AsyncFor)):
                    cc += 1
                elif isinstance(child, (ast.While,)):
                    cc += 1
                elif isinstance(child, ast.ExceptHandler):
                    cc += 1
                elif isinstance(child, (ast.With, ast.AsyncWith)):
                    cc += 1
                elif isinstance(child, ast.Assert):
                    cc += 1
                elif isinstance(child, ast.BoolOp):
                    # Each `and`/`or` adds a branch
                    cc += len(child.values) - 1

            rating = (
                "simple" if cc <= 5 else
                "moderate" if cc <= 10 else
                "complex" if cc <= 20 else
                "very complex"
            )
            results.append({
                "name": node.name,
                "line": node.lineno,
                "complexity": cc,
                "rating": rating,
            })

    return results


def max_nesting_depth(source: str) -> list:
    """Compute maximum nesting depth per function.
    Returns [{name, line, depth, rating}]."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [{"error": "syntax error"}]

    _NESTING = (
        ast.If, ast.For, ast.While, ast.With, ast.Try,
        ast.AsyncFor, ast.AsyncWith,
    )

    def _depth(node, current=0):
        """Recursively find max depth."""
        if isinstance(node, _NESTING):
            current += 1
        mx = current
        for child in ast.iter_child_nodes(node):
            mx = max(mx, _depth(child, current))
        return mx

    results = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            d = _depth(node)
            rating = (
                "flat" if d <= 2 else
                "acceptable" if d <= 4 else
                "deep" if d <= 6 else
                "too deep"
            )
            results.append({
                "name": node.name, "line": node.lineno,
                "depth": d, "rating": rating,
            })
    return results


def function_length(source: str) -> list:
    """Compute lines per function.
    Returns [{name, line, length, rating}]."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [{"error": "syntax error"}]

    lines = source.splitlines()
    results = []
    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

    for i, node in enumerate(funcs):
        # Find end line — next function start or end of file.
        if hasattr(node, 'end_lineno') and node.end_lineno:
            length = node.end_lineno - node.lineno + 1
        else:
            # Fallback: count until next function or dedent.
            length = 1
            for j in range(node.lineno, len(lines)):
                length = j - node.lineno + 1

        rating = (
            "concise" if length <= 15 else
            "acceptable" if length <= 30 else
            "long" if length <= 60 else
            "too long"
        )
        results.append({
            "name": node.name, "line": node.lineno,
            "length": length, "rating": rating,
        })
    return results


def naming_check(source: str) -> list:
    """Check naming conventions.
    Returns [{name, line, kind, issue}] for naming violations."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [{"error": "syntax error"}]

    issues = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            name = node.name
            if not name.startswith('_') and not re.match(r'^[a-z_][a-z0-9_]*$', name):
                issues.append({
                    "name": name, "line": node.lineno,
                    "kind": "function", "issue": "not snake_case",
                })
            if len(name) < 3 and not name.startswith('_'):
                issues.append({
                    "name": name, "line": node.lineno,
                    "kind": "function", "issue": "too short (< 3 chars)",
                })
        elif isinstance(node, ast.ClassDef):
            name = node.name
            if not re.match(r'^[A-Z][a-zA-Z0-9]*$', name):
                issues.append({
                    "name": name, "line": node.lineno,
                    "kind": "class", "issue": "not CamelCase",
                })
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            name = node.id
            # Single-letter vars outside comprehensions are suspicious.
            if len(name) == 1 and name not in ('_', 'i', 'j', 'k', 'n', 'x', 'y', 'f'):
                issues.append({
                    "name": name, "line": node.lineno,
                    "kind": "variable", "issue": "single-letter name",
                })
    return issues


def dead_code(source: str) -> list:
    """Detect potential dead code: unreachable after return, pass in non-empty body.
    Returns [{line, issue}]."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [{"error": "syntax error"}]

    issues = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            for i, stmt in enumerate(body[:-1]):
                if isinstance(stmt, ast.Return):
                    issues.append({
                        "line": body[i + 1].lineno,
                        "issue": f"unreachable code after return on line {stmt.lineno}",
                    })
            # pass in non-trivial body
            if len(body) > 1:
                for stmt in body:
                    if isinstance(stmt, ast.Pass):
                        issues.append({
                            "line": stmt.lineno,
                            "issue": "unnecessary pass in non-empty function",
                        })
    return issues


def code_quality(source: str) -> dict:
    """Full quality report — runs all checks, produces a summary score.
    Score 0-100, higher is better."""
    cc = cyclomatic_complexity(source)
    nesting = max_nesting_depth(source)
    lengths = function_length(source)
    naming = naming_check(source)
    dead = dead_code(source)

    # Score: start at 100, deduct for issues.
    score = 100
    for f in cc:
        if f.get("complexity", 0) > 10: score -= 5
        elif f.get("complexity", 0) > 20: score -= 15
    for f in nesting:
        if f.get("depth", 0) > 4: score -= 5
        elif f.get("depth", 0) > 6: score -= 10
    for f in lengths:
        if f.get("length", 0) > 30: score -= 3
        elif f.get("length", 0) > 60: score -= 8
    score -= len(naming) * 2
    score -= len(dead) * 5
    score = max(0, min(100, score))

    rating = (
        "excellent" if score >= 90 else
        "good" if score >= 75 else
        "needs work" if score >= 50 else
        "poor"
    )

    return {
        "score": score,
        "rating": rating,
        "functions": len(cc),
        "complex_functions": sum(1 for f in cc if f.get("complexity", 0) > 10),
        "deep_functions": sum(1 for f in nesting if f.get("depth", 0) > 4),
        "long_functions": sum(1 for f in lengths if f.get("length", 0) > 30),
        "naming_issues": len(naming),
        "dead_code": len(dead),
        "details": {
            "complexity": cc,
            "nesting": nesting,
            "lengths": lengths,
            "naming": naming,
            "dead_code": dead,
        },
    }


def code_quality_file(path: str) -> dict:
    """Run code_quality on a file."""
    source = Path(path).read_text(encoding="utf-8", errors="replace")
    result = code_quality(source)
    result["file"] = path
    result["lines"] = len(source.splitlines())
    return result


QUALITY_FUNCTIONS = {
    "cyclomatic_complexity": cyclomatic_complexity,
    "max_nesting_depth": max_nesting_depth,
    "function_length": function_length,
    "naming_check": naming_check,
    "dead_code": dead_code,
    "code_quality": code_quality,
    "code_quality_file": code_quality_file,
}

QUALITY_NL_PATTERNS = [
    (r'(?:cyclomatic|code)\s+complexity\s+(?:of|for|in)', None),
    (r'(?:nesting|indentation)\s+depth\s+(?:of|for|in)', None),
    (r'(?:function|method)\s+(?:length|size|lines)\s+(?:of|for|in)', None),
    (r'(?:naming|variable names?)\s+(?:check|quality|convention)', None),
    (r'(?:dead code|unused)\s+(?:in|for)', None),
]
