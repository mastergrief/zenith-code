"""
CALM performance backend — computed complexity analysis.

Models say "this is O(n)" without measuring. This backend counts
nested loops, estimates complexity, times execution, and measures
memory. Every performance claim becomes verifiable.

Functions: complexity, loop_depth, time_function, memory_estimate,
benchmark_compare, bottleneck_find.
"""

from __future__ import annotations

import ast
import sys
import time
import re
from pathlib import Path
from typing import List, Optional


def estimate_complexity(source: str) -> list:
    """Estimate Big-O complexity per function by counting nested loops.
    Returns [{name, line, loop_depth, estimated_complexity, rating}].

    loop_depth 0 = O(1), 1 = O(n), 2 = O(n²), 3 = O(n³), etc.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [{"error": "syntax error"}]

    _LOOP_TYPES = (ast.For, ast.AsyncFor, ast.While)

    def _max_loop_depth(node, current=0):
        mx = current
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _LOOP_TYPES):
                mx = max(mx, _max_loop_depth(child, current + 1))
            else:
                mx = max(mx, _max_loop_depth(child, current))
        return mx

    results = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            depth = _max_loop_depth(node)

            # Check for recursive calls.
            is_recursive = False
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                    if child.func.id == node.name:
                        is_recursive = True

            # Check for sorted/sort calls (adds O(n log n)).
            has_sort = False
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name) and child.func.id == "sorted":
                        has_sort = True
                    elif isinstance(child.func, ast.Attribute) and child.func.attr == "sort":
                        has_sort = True

            if is_recursive:
                complexity = "O(2^n) or O(n!) — recursive, depends on branching"
                rating = "exponential"
            elif has_sort and depth >= 1:
                complexity = f"O(n^{depth} * n log n)"
                rating = "expensive"
            elif has_sort:
                complexity = "O(n log n)"
                rating = "good"
            elif depth == 0:
                complexity = "O(1)"
                rating = "constant"
            elif depth == 1:
                complexity = "O(n)"
                rating = "linear"
            elif depth == 2:
                complexity = "O(n²)"
                rating = "quadratic"
            elif depth == 3:
                complexity = "O(n³)"
                rating = "cubic"
            else:
                complexity = f"O(n^{depth})"
                rating = "polynomial"

            results.append({
                "name": node.name,
                "line": node.lineno,
                "loop_depth": depth,
                "has_sort": has_sort,
                "is_recursive": is_recursive,
                "estimated_complexity": complexity,
                "rating": rating,
            })

    return results


def estimate_complexity_file(path: str) -> list:
    """Run estimate_complexity on a file."""
    source = Path(path).read_text(encoding="utf-8", errors="replace")
    results = estimate_complexity(source)
    for r in results:
        r["file"] = path
    return results


def loop_depth(source: str) -> dict:
    """Summary: max and average loop nesting across all functions.
    Returns {max_depth, avg_depth, deepest_function, functions}.
    """
    estimates = estimate_complexity(source)
    estimates = [e for e in estimates if "error" not in e]
    if not estimates:
        return {"max_depth": 0, "avg_depth": 0, "functions": 0}

    depths = [e["loop_depth"] for e in estimates]
    max_d = max(depths)
    deepest = [e["name"] for e in estimates if e["loop_depth"] == max_d]

    return {
        "max_depth": max_d,
        "avg_depth": round(sum(depths) / len(depths), 1),
        "deepest_function": deepest[0] if deepest else None,
        "functions": len(estimates),
    }


def time_expression(expr: str, n: int = 100) -> dict:
    """Time a Python expression N times.
    Returns {avg_ms, min_ms, max_ms, total_ms}.

    Example: time_expression("sorted(range(1000))", 100)
    """
    from calm.expression import safe_eval
    times = []
    for _ in range(int(n)):
        t0 = time.perf_counter()
        safe_eval(expr)
        elapsed = (time.perf_counter() - t0) * 1000  # ms
        times.append(elapsed)

    return {
        "expression": expr,
        "iterations": int(n),
        "avg_ms": round(sum(times) / len(times), 4),
        "min_ms": round(min(times), 4),
        "max_ms": round(max(times), 4),
        "total_ms": round(sum(times), 2),
    }


def memory_estimate(source: str) -> dict:
    """Estimate memory usage patterns in code.
    Detects: list comprehensions over large ranges, string concatenation
    in loops, unbounded growth patterns.
    Returns {warnings: [{line, issue, severity}]}.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {"error": "syntax error"}

    warnings = []

    for node in ast.walk(tree):
        # Large range in list comprehension.
        if isinstance(node, ast.ListComp):
            for gen in node.generators:
                if isinstance(gen.iter, ast.Call) and isinstance(gen.iter.func, ast.Name):
                    if gen.iter.func.id == "range" and gen.iter.args:
                        arg = gen.iter.args[-1]
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                            if arg.value > 10000:
                                warnings.append({
                                    "line": node.lineno,
                                    "issue": f"list comprehension over range({arg.value}) — consider generator",
                                    "severity": "medium",
                                })

        # String concatenation in loop.
        if isinstance(node, (ast.For, ast.While)):
            for child in ast.walk(node):
                if isinstance(child, ast.AugAssign) and isinstance(child.op, ast.Add):
                    if isinstance(child.target, ast.Name):
                        warnings.append({
                            "line": child.lineno,
                            "issue": f"string/list concatenation in loop — consider join() or list.append()",
                            "severity": "low",
                        })

        # Unbounded list.append in while loop.
        if isinstance(node, ast.While):
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                    if child.func.attr == "append":
                        warnings.append({
                            "line": child.lineno,
                            "issue": "append in while loop — may grow unbounded",
                            "severity": "medium",
                        })

    return {"warnings": warnings, "count": len(warnings)}


def perf_summary(path: str) -> dict:
    """Full performance summary for a file.
    Combines complexity + memory analysis."""
    source = Path(path).read_text(encoding="utf-8", errors="replace")
    complexity = estimate_complexity(source)
    memory = memory_estimate(source)
    ld = loop_depth(source)

    expensive = [c for c in complexity if c.get("rating") in ("quadratic", "cubic", "polynomial", "exponential", "expensive")]

    return {
        "file": path,
        "functions": len(complexity),
        "max_loop_depth": ld["max_depth"],
        "expensive_functions": len(expensive),
        "memory_warnings": memory.get("count", 0),
        "hotspots": [{"name": c["name"], "complexity": c["estimated_complexity"]} for c in expensive],
        "rating": (
            "efficient" if not expensive and memory.get("count", 0) == 0 else
            "has concerns" if len(expensive) <= 1 else
            "needs optimization"
        ),
    }


PERF_FUNCTIONS = {
    "estimate_complexity": estimate_complexity,
    "estimate_complexity_file": estimate_complexity_file,
    "loop_depth": loop_depth,
    "time_expression": time_expression,
    "memory_estimate": memory_estimate,
    "perf_summary": perf_summary,
}

PERF_NL_PATTERNS = [
    (r'(?:what is|estimate|analyze)\s+(?:the\s+)?(?:time\s+)?complexity\s+(?:of|for)', None),
    (r'(?:how much|estimate)\s+memory\s+(?:does|for|used)', None),
    (r'(?:how deep|nesting|loop)\s+depth\s+(?:of|in|for)', None),
]
