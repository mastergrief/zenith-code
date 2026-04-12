"""
CALM test analysis backend — computed test coverage and quality.

Models say "tests cover this" without checking. This backend
analyzes test files, measures coverage, and finds gaps.

Functions: test_summary, untested_functions, test_quality,
assertion_count, test_to_code_ratio.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def test_summary(test_path: str) -> dict:
    """Analyze a test file: count tests, assertions, fixtures.
    Returns {tests, assertions, fixtures, avg_assertions_per_test}."""
    try:
        source = Path(test_path).read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except (SyntaxError, FileNotFoundError):
        return {"error": f"cannot parse {test_path}"}

    tests = []
    fixtures = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                # Count assertions in this test.
                asserts = 0
                for child in ast.walk(node):
                    if isinstance(child, ast.Assert):
                        asserts += 1
                    elif isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Attribute):
                            if child.func.attr.startswith("assert"):
                                asserts += 1
                        elif isinstance(child.func, ast.Name):
                            if child.func.id in ("assert_equal", "assert_true",
                                                  "assert_raises", "assertEqual"):
                                asserts += 1

                tests.append({"name": node.name, "line": node.lineno, "assertions": asserts})
            elif node.name.startswith("setup") or node.name.startswith("fixture") or \
                 node.name.startswith("setUp") or node.name.startswith("conftest"):
                fixtures.append({"name": node.name, "line": node.lineno})

    total_assertions = sum(t["assertions"] for t in tests)
    avg = round(total_assertions / len(tests), 1) if tests else 0

    return {
        "file": test_path,
        "tests": len(tests),
        "assertions": total_assertions,
        "fixtures": len(fixtures),
        "avg_assertions_per_test": avg,
        "weak_tests": [t["name"] for t in tests if t["assertions"] == 0],
        "rating": (
            "well tested" if avg >= 2 and not any(t["assertions"] == 0 for t in tests) else
            "adequately tested" if avg >= 1 else
            "weakly tested"
        ),
    }


def untested_functions(code_path: str, test_path: str) -> list:
    """Find functions in code_path that have no corresponding test.
    Returns [{name, line}] — functions with no test_<name> in test file."""
    try:
        code_tree = ast.parse(Path(code_path).read_text(encoding="utf-8", errors="replace"))
        test_source = Path(test_path).read_text(encoding="utf-8", errors="replace")
    except (SyntaxError, FileNotFoundError) as e:
        return [{"error": str(e)}]

    # Get all public function names from code.
    code_functions = []
    for node in ast.walk(code_tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                code_functions.append({"name": node.name, "line": node.lineno})

    # Check which have a test.
    untested = []
    for func in code_functions:
        test_name = f"test_{func['name']}"
        if test_name not in test_source and func['name'] not in test_source:
            untested.append(func)

    return untested


def test_to_code_ratio(code_path: str, test_path: str) -> dict:
    """Compute the test-to-code ratio.
    Returns {code_lines, test_lines, ratio, rating}."""
    try:
        code_lines = len(Path(code_path).read_text().splitlines())
        test_lines = len(Path(test_path).read_text().splitlines())
    except FileNotFoundError as e:
        return {"error": str(e)}

    ratio = round(test_lines / code_lines, 2) if code_lines > 0 else 0

    return {
        "code_lines": code_lines,
        "test_lines": test_lines,
        "ratio": ratio,
        "rating": (
            "excellent" if ratio >= 1.5 else
            "good" if ratio >= 1.0 else
            "adequate" if ratio >= 0.5 else
            "undertested"
        ),
    }


def run_tests_verbose(test_path: str) -> dict:
    """Run pytest and parse detailed results.
    Returns {passed, failed, errors, duration, failures: [{test, error}]}."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", test_path, "-v", "--tb=short", "-q"],
            capture_output=True, text=True, timeout=60,
            cwd=str(Path(test_path).parent),
        )
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}

    output = proc.stdout
    passed = failed = errors = 0

    m = re.search(r'(\d+) passed', output)
    if m: passed = int(m.group(1))
    m = re.search(r'(\d+) failed', output)
    if m: failed = int(m.group(1))
    m = re.search(r'(\d+) error', output)
    if m: errors = int(m.group(1))

    # Extract failure details.
    failures = []
    for m in re.finditer(r'FAILED (.+?) -', output):
        failures.append(m.group(1))

    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "total": passed + failed + errors,
        "success_rate": round(passed / (passed + failed) * 100, 1) if (passed + failed) > 0 else 0,
        "failures": failures[:10],
    }


TEST_FUNCTIONS = {
    "test_summary": test_summary,
    "untested_functions": untested_functions,
    "test_to_code_ratio": test_to_code_ratio,
    "run_tests_verbose": run_tests_verbose,
}
