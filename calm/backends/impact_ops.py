"""
CALM impact analysis backend — computed blast radius.

"If I change this function, what breaks?" is a graph problem,
not a judgment call. This backend traces call graphs, import chains,
and test coverage to answer it deterministically.

Functions: call_graph, dependents, blast_radius, test_coverage_map,
import_chain, dead_functions, coupling_score.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set


def _parse_file(path: str) -> Optional[ast.Module]:
    """Parse a Python file, return AST or None."""
    try:
        return ast.parse(Path(path).read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, FileNotFoundError):
        return None


def call_graph(path: str) -> dict:
    """Build a call graph for a Python file.
    Returns {function_name: [functions_it_calls]}.

    Example: call_graph("app.py")
    → {"main": ["parse_args", "run_server"], "run_server": ["handle_request"]}
    """
    tree = _parse_file(path)
    if not tree:
        return {"error": f"cannot parse {path}"}

    # Collect all function definitions.
    functions = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            calls = []
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name):
                        calls.append(child.func.id)
                    elif isinstance(child.func, ast.Attribute):
                        calls.append(child.func.attr)
            functions[node.name] = sorted(set(calls))

    return functions


def dependents(path: str, function_name: str) -> dict:
    """Find all functions in a file that call the given function.
    Returns {callers: [names], count: N}.

    "Who depends on me?" — the reverse call graph.
    """
    graph = call_graph(path)
    if "error" in graph:
        return graph

    callers = [
        name for name, calls in graph.items()
        if function_name in calls and name != function_name
    ]
    return {"function": function_name, "callers": callers, "count": len(callers)}


def blast_radius(path: str, function_name: str) -> dict:
    """Compute the full blast radius of changing a function.
    Traces transitive dependents — if A calls B calls target,
    changing target affects both B and A.

    Returns {direct: [...], transitive: [...], total: N}.
    """
    graph = call_graph(path)
    if "error" in graph:
        return graph

    # Build reverse graph.
    reverse = {}
    for caller, callees in graph.items():
        for callee in callees:
            reverse.setdefault(callee, []).append(caller)

    # BFS from function_name through reverse graph.
    direct = reverse.get(function_name, [])
    visited = set(direct)
    queue = list(direct)
    transitive = []

    while queue:
        current = queue.pop(0)
        for upstream in reverse.get(current, []):
            if upstream not in visited and upstream != function_name:
                visited.add(upstream)
                transitive.append(upstream)
                queue.append(upstream)

    return {
        "function": function_name,
        "direct": sorted(direct),
        "transitive": sorted(transitive),
        "total": len(direct) + len(transitive),
        "risk": (
            "low" if len(direct) + len(transitive) <= 2 else
            "medium" if len(direct) + len(transitive) <= 5 else
            "high"
        ),
    }


def import_chain(directory: str, module_name: str) -> dict:
    """Find all files that import a given module.
    Returns {importers: [{file, line}], count: N}.

    "Who imports me?" — the dependency chain.
    """
    directory = str(directory)
    importers = []

    for root, dirs, files in os.walk(directory):
        # Skip hidden dirs and __pycache__.
        dirs[:] = [d for d in dirs if not d.startswith(('.', '__'))]
        for f in files:
            if not f.endswith('.py'):
                continue
            filepath = os.path.join(root, f)
            try:
                tree = ast.parse(Path(filepath).read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if module_name in alias.name:
                            importers.append({"file": filepath, "line": node.lineno})
                elif isinstance(node, ast.ImportFrom):
                    if node.module and module_name in node.module:
                        importers.append({"file": filepath, "line": node.lineno})

    return {"module": module_name, "importers": importers, "count": len(importers)}


def dead_functions(path: str) -> list:
    """Find functions that are defined but never called within the file.
    Returns [{name, line}] — candidates for removal.

    Note: only checks within-file calls. Cross-file analysis needs import_chain.
    """
    graph = call_graph(path)
    if "error" in graph:
        return [graph]

    all_defined = set(graph.keys())
    all_called = set()
    for calls in graph.values():
        all_called.update(calls)

    # Functions defined but never called (except likely entry points).
    _ENTRY_POINTS = {'main', '__init__', 'setUp', 'tearDown', 'setup', 'teardown'}
    uncalled = [
        name for name in all_defined
        if name not in all_called
        and name not in _ENTRY_POINTS
        and not name.startswith('test_')
        and not name.startswith('_')
    ]

    tree = _parse_file(path)
    results = []
    if tree:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in uncalled:
                    results.append({"name": node.name, "line": node.lineno})

    return results


def coupling_score(path: str) -> dict:
    """Measure coupling within a file — how interconnected are the functions?
    Returns {score: 0-100, avg_calls, max_calls, isolated, tightly_coupled}.

    Low coupling (< 30) = functions are independent.
    High coupling (> 70) = everything calls everything.
    """
    graph = call_graph(path)
    if "error" in graph:
        return graph

    if not graph:
        return {"score": 0, "functions": 0}

    # Count internal calls (calls to functions defined in the same file).
    internal_names = set(graph.keys())
    call_counts = []
    for name, calls in graph.items():
        internal = [c for c in calls if c in internal_names and c != name]
        call_counts.append(len(internal))

    avg = sum(call_counts) / len(call_counts) if call_counts else 0
    max_calls = max(call_counts) if call_counts else 0
    isolated = sum(1 for c in call_counts if c == 0)

    # Score: 0-100 based on average internal coupling.
    n = len(graph)
    max_possible = n - 1  # each function could call all others
    score = int(avg / max_possible * 100) if max_possible > 0 else 0
    score = min(100, score)

    return {
        "score": score,
        "rating": (
            "low coupling" if score < 30 else
            "moderate coupling" if score < 60 else
            "high coupling"
        ),
        "functions": n,
        "avg_internal_calls": round(avg, 1),
        "max_internal_calls": max_calls,
        "isolated_functions": isolated,
        "tightly_coupled": sum(1 for c in call_counts if c >= 3),
    }


def change_risk(path: str, function_name: str) -> dict:
    """Comprehensive risk assessment for changing a function.
    Combines blast_radius + coupling + dead_code analysis.

    Returns {risk_level, blast_radius, coupling, is_dead, recommendation}.
    """
    br = blast_radius(path, function_name)
    cp = coupling_score(path)
    dead = dead_functions(path)
    is_dead = any(d.get("name") == function_name for d in dead)

    if isinstance(br, dict) and "error" in br:
        return br

    risk = br.get("risk", "unknown")
    recommendation = []

    if is_dead:
        recommendation.append("function appears unused — consider removing instead of modifying")
        risk = "low"
    if br.get("total", 0) > 5:
        recommendation.append(f"high blast radius ({br['total']} dependents) — add tests before changing")
    if cp.get("score", 0) > 60:
        recommendation.append("file has high coupling — changes may cascade unpredictably")
    if not recommendation:
        recommendation.append("low risk — safe to modify")

    return {
        "function": function_name,
        "risk_level": risk,
        "blast_radius": br.get("total", 0),
        "direct_callers": br.get("direct", []),
        "coupling_score": cp.get("score", 0),
        "is_unused": is_dead,
        "recommendations": recommendation,
    }


IMPACT_FUNCTIONS = {
    "call_graph": call_graph,
    "dependents": dependents,
    "blast_radius": blast_radius,
    "import_chain": import_chain,
    "dead_functions": dead_functions,
    "coupling_score": coupling_score,
    "change_risk": change_risk,
}
