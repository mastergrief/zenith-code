"""
CALM AST backend — verified Python AST analysis and transforms.

Actual code transforms, not suggestions. Parse, inspect, and rewrite
Python source deterministically via the ast module.

Functions: ast_parse, ast_functions, ast_classes, ast_imports,
ast_complexity, ast_rename, ast_extract_function.
"""

from __future__ import annotations

import ast
import re
import textwrap
from typing import Dict, List, Optional


def ast_parse(source: str) -> dict:
    """Parse Python source and return AST summary.
    Returns {valid, node_count, functions, classes, imports, errors}."""
    source = str(source)
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return {"valid": False, "errors": [f"SyntaxError: {e.msg} (line {e.lineno})"],
                "node_count": 0, "functions": [], "classes": [], "imports": []}

    funcs = []
    classes = []
    imports = []
    node_count = 0

    for node in ast.walk(tree):
        node_count += 1
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]
            funcs.append({
                "name": node.name,
                "line": node.lineno,
                "args": args,
                "decorators": [_decorator_name(d) for d in node.decorator_list],
                "is_async": isinstance(node, ast.AsyncFunctionDef),
            })
        elif isinstance(node, ast.ClassDef):
            bases = [_name_of(b) for b in node.bases]
            methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            classes.append({
                "name": node.name,
                "line": node.lineno,
                "bases": bases,
                "methods": methods,
            })
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({"module": alias.name, "alias": alias.asname, "line": node.lineno})
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.append({
                    "module": f"{node.module}.{alias.name}" if node.module else alias.name,
                    "alias": alias.asname, "line": node.lineno,
                })

    return {
        "valid": True, "errors": [],
        "node_count": node_count,
        "functions": funcs,
        "classes": classes,
        "imports": imports,
    }


def ast_functions(source: str) -> list:
    """List all functions in Python source with signatures.
    Returns [{name, args, line, decorators, is_async}, ...]."""
    parsed = ast_parse(source)
    return parsed.get("functions", [])


def ast_classes(source: str) -> list:
    """List all classes in Python source.
    Returns [{name, bases, methods, line}, ...]."""
    parsed = ast_parse(source)
    return parsed.get("classes", [])


def ast_imports(source: str) -> list:
    """List all imports in Python source.
    Returns [{module, alias, line}, ...]."""
    parsed = ast_parse(source)
    return parsed.get("imports", [])


def ast_complexity(source: str) -> dict:
    """Measure code complexity metrics from AST.
    Returns {total_nodes, functions, classes, max_depth, branches, loops}."""
    source = str(source)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {"error": "SyntaxError", "total_nodes": 0}

    branches = 0
    loops = 0
    max_depth = 0
    total = 0

    def _walk(node, depth=0):
        nonlocal branches, loops, max_depth, total
        total += 1
        max_depth = max(max_depth, depth)
        if isinstance(node, (ast.If, ast.IfExp)):
            branches += 1
        elif isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
            loops += 1
        elif isinstance(node, ast.BoolOp):
            branches += len(node.values) - 1
        elif isinstance(node, (ast.ExceptHandler,)):
            branches += 1
        for child in ast.iter_child_nodes(node):
            _walk(child, depth + 1)

    _walk(tree)

    funcs = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    cls = sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef))

    return {
        "total_nodes": total,
        "functions": funcs,
        "classes": cls,
        "max_depth": max_depth,
        "branches": branches,
        "loops": loops,
    }


def ast_rename(source: str, old_name: str, new_name: str) -> str:
    """Rename a symbol (function, variable, class) throughout Python source.
    Uses AST-aware renaming — only renames actual identifiers, not strings.
    Example: ast_rename("def foo(): return foo()", "foo", "bar")
    → "def bar(): return bar()" """
    source = str(source)
    old_name, new_name = str(old_name), str(new_name)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Fall back to regex if source doesn't parse.
        return re.sub(rf'\b{re.escape(old_name)}\b', new_name, source)

    # Collect positions to rename (line, col_offset, end_col_offset).
    positions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == old_name:
            positions.append((node.lineno, node.col_offset, node.col_offset + len(old_name)))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == old_name:
            positions.append((node.lineno, node.col_offset + 4, node.col_offset + 4 + len(old_name)))
        elif isinstance(node, ast.ClassDef) and node.name == old_name:
            positions.append((node.lineno, node.col_offset + 6, node.col_offset + 6 + len(old_name)))
        elif isinstance(node, ast.arg) and node.arg == old_name:
            positions.append((node.lineno, node.col_offset, node.col_offset + len(old_name)))
        elif isinstance(node, ast.alias) and node.name == old_name:
            if hasattr(node, "lineno"):
                positions.append((node.lineno, node.col_offset, node.col_offset + len(old_name)))

    if not positions:
        return source

    # Apply replacements in reverse order (bottom-right to top-left).
    lines = source.splitlines(keepends=True)
    for lineno, col_start, col_end in sorted(positions, reverse=True):
        idx = lineno - 1
        if idx < len(lines):
            line = lines[idx]
            lines[idx] = line[:col_start] + new_name + line[col_end:]

    return "".join(lines)


def ast_extract_function(source: str, name: str) -> str:
    """Extract a single function's source code by name.
    Example: ast_extract_function(code, "my_func") → "def my_func(x):\\n    return x + 1" """
    source = str(source)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""

    lines = source.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, "end_lineno") and node.end_lineno else start + 1
            return "\n".join(lines[start:end])
    return ""


def _decorator_name(node: ast.expr) -> str:
    """Get decorator name from AST node."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return f"{_name_of(node.value)}.{node.attr}"
    elif isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return "?"


def _name_of(node: ast.expr) -> str:
    """Get name from AST expression node."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return f"{_name_of(node.value)}.{node.attr}"
    elif isinstance(node, ast.Constant):
        return str(node.value)
    return "?"


AST_FUNCTIONS = {
    "ast_parse": ast_parse,
    "ast_functions": ast_functions,
    "ast_classes": ast_classes,
    "ast_imports": ast_imports,
    "ast_complexity": ast_complexity,
    "ast_rename": ast_rename,
    "ast_extract_function": ast_extract_function,
}
