"""
CALM documentation backend — computed docstring analysis.

Models claim "this is well documented" without checking. This backend
measures documentation coverage, quality, and completeness.

Functions: docstring_coverage, undocumented, docstring_quality,
parameter_docs, module_doc_check.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List


def docstring_coverage(source: str) -> dict:
    """Measure docstring coverage for functions and classes.
    Returns {total, documented, undocumented, coverage_pct, rating}."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {"error": "syntax error"}

    total = 0
    documented = 0
    undocumented_list = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_") and node.name != "__init__":
                    continue

            total += 1
            docstring = ast.get_docstring(node)
            if docstring:
                documented += 1
            else:
                undocumented_list.append({
                    "name": node.name,
                    "line": node.lineno,
                    "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                })

    coverage = round(documented / total * 100, 1) if total > 0 else 100

    return {
        "total": total,
        "documented": documented,
        "undocumented": total - documented,
        "coverage_pct": coverage,
        "rating": (
            "fully documented" if coverage == 100 else
            "well documented" if coverage >= 80 else
            "partially documented" if coverage >= 50 else
            "poorly documented"
        ),
        "undocumented_items": undocumented_list[:10],
    }


def docstring_coverage_file(path: str) -> dict:
    """Run docstring_coverage on a file."""
    source = Path(path).read_text(encoding="utf-8", errors="replace")
    result = docstring_coverage(source)
    result["file"] = path
    return result


def docstring_quality(source: str) -> list:
    """Assess quality of existing docstrings.
    Checks: length, parameter documentation, return docs, examples.
    Returns [{name, line, issues}]."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [{"error": "syntax error"}]

    results = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        docstring = ast.get_docstring(node)
        if not docstring:
            continue

        issues = []
        params = [a.arg for a in node.args.args if a.arg != "self"]

        # Check length.
        if len(docstring) < 10:
            issues.append("too short (< 10 chars)")

        # Check if params are documented.
        if params:
            undoc_params = [
                p for p in params
                if p not in docstring and f":{p}" not in docstring
                and f"{p} " not in docstring
            ]
            if undoc_params:
                issues.append(f"undocumented params: {', '.join(undoc_params)}")

        # Check return documentation.
        if node.returns and "return" not in docstring.lower() and "→" not in docstring:
            issues.append("has return annotation but no return docs")

        if issues:
            results.append({
                "name": node.name,
                "line": node.lineno,
                "issues": issues,
            })

    return results


def module_doc_check(path: str) -> dict:
    """Check module-level documentation.
    Returns {has_module_docstring, docstring_length, has_usage, has_examples}."""
    try:
        source = Path(path).read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except (SyntaxError, FileNotFoundError):
        return {"error": f"cannot parse {path}"}

    docstring = ast.get_docstring(tree)

    return {
        "file": path,
        "has_module_docstring": docstring is not None,
        "docstring_length": len(docstring) if docstring else 0,
        "has_usage": bool(docstring and ("usage" in docstring.lower() or "example" in docstring.lower())),
        "has_examples": bool(docstring and (">>>" in docstring or "```" in docstring)),
    }


DOC_FUNCTIONS = {
    "docstring_coverage": docstring_coverage,
    "docstring_coverage_file": docstring_coverage_file,
    "docstring_quality": docstring_quality,
    "module_doc_check": module_doc_check,
}

DOC_NL_PATTERNS = [
    (r'(?:docstring|documentation)\s+coverage\s+(?:of|for|in)', None),
    (r'(?:are|is)\s+(?:all\s+)?(?:functions?|classes?)\s+documented', None),
    (r'(?:quality|check)\s+(?:of\s+)?docstrings?\s+(?:in|for)', None),
]
