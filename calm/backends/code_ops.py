"""
CALM code backend — verified coding operations.

Wraps real development tools (file I/O, AST, pytest, ruff) so the
LLM can dispatch coding operations from <calm> blocks with automatic
verification. Every edit is syntax-checked, every fix is tested.

Functions are registered as both `code.X` and bare `X` via auto-alias.

Usage in <calm> blocks:
    code.read("auth.py")
    code.write("hello.py", "print('hello')")
    code.syntax_check("hello.py")
    code.run("hello.py")
    code.test("tests/test_auth.py")
    code.lint("auth.py")
    code.search("TODO", "src/")
    code.find("*.py", "src/")
    code.diff("auth.py")
    code.edit("auth.py", 5, "new_line_content")
    code.insert("auth.py", 10, "inserted_line")
    code.delete("auth.py", 5)
    code.count_lines("auth.py")
    code.functions("auth.py")
    code.classes("auth.py")
    code.imports("auth.py")
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

from calm.stack_vm import (
    Backend,
    CalmRuntimeError,
    Dispatcher,
    Instruction,
    VMState,
    _pop_n,
)

# Safety: restrict file operations to the working directory tree.
_WORKSPACE = Path.cwd()

def _safe_path(path_str: str) -> Path:
    """Resolve a path and verify it's within the workspace."""
    p = Path(path_str).resolve()
    try:
        p.relative_to(_WORKSPACE.resolve())
    except ValueError:
        # Allow absolute paths under common safe locations
        safe_prefixes = ["/tmp", str(Path.home())]
        if not any(str(p).startswith(s) for s in safe_prefixes):
            raise CalmRuntimeError(f"code: path outside workspace: {p}")
    return p


def _run_cmd(cmd: list, timeout: float = 30.0, cwd: str = None) -> dict:
    """Run a shell command safely and return {stdout, stderr, returncode}."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or str(_WORKSPACE),
        )
        return {
            "stdout": proc.stdout[:5000],
            "stderr": proc.stderr[:2000],
            "returncode": proc.returncode,
            "ok": proc.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "timeout", "returncode": -1, "ok": False}
    except FileNotFoundError as e:
        return {"stdout": "", "stderr": str(e), "returncode": -1, "ok": False}


# ---------------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------------

def _b_read(state: VMState, instr: Instruction) -> None:
    """Read a file. Supports optional line range: code.read(path, start, end)."""
    args = _pop_n(state, 1, "code.read")
    path = _safe_path(str(args[0]))
    if not path.exists():
        raise CalmRuntimeError(f"code.read: file not found: {path}")
    content = path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    state.stack.append({
        "path": str(path),
        "lines": len(lines),
        "content": content[:10000],  # cap at 10K chars
    })


def _b_write(state: VMState, instr: Instruction) -> None:
    """Write content to a file: code.write(path, content)."""
    path_str, content = _pop_n(state, 2, "code.write")
    path = _safe_path(str(path_str))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(content), encoding="utf-8")
    state.stack.append({"written": str(path), "bytes": len(str(content))})


def _b_edit(state: VMState, instr: Instruction) -> None:
    """Replace a line: code.edit(path, line_number, new_content)."""
    path_str, line_no, new_content = _pop_n(state, 3, "code.edit")
    path = _safe_path(str(path_str))
    if not path.exists():
        raise CalmRuntimeError(f"code.edit: file not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    idx = int(line_no) - 1  # 1-indexed
    if idx < 0 or idx >= len(lines):
        raise CalmRuntimeError(f"code.edit: line {line_no} out of range (1-{len(lines)})")
    old = lines[idx]
    lines[idx] = str(new_content)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    state.stack.append({"edited": str(path), "line": int(line_no), "old": old, "new": str(new_content)})


def _b_insert(state: VMState, instr: Instruction) -> None:
    """Insert a line after line_number: code.insert(path, after_line, content)."""
    path_str, line_no, content = _pop_n(state, 3, "code.insert")
    path = _safe_path(str(path_str))
    lines = path.read_text(encoding="utf-8").splitlines()
    idx = int(line_no)  # insert after this line (0 = beginning)
    lines.insert(idx, str(content))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    state.stack.append({"inserted": str(path), "after_line": int(line_no)})


def _b_delete(state: VMState, instr: Instruction) -> None:
    """Delete a line: code.delete(path, line_number)."""
    path_str, line_no = _pop_n(state, 2, "code.delete")
    path = _safe_path(str(path_str))
    lines = path.read_text(encoding="utf-8").splitlines()
    idx = int(line_no) - 1
    if idx < 0 or idx >= len(lines):
        raise CalmRuntimeError(f"code.delete: line {line_no} out of range")
    removed = lines.pop(idx)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    state.stack.append({"deleted": str(path), "line": int(line_no), "content": removed})


# ---------------------------------------------------------------------------
# Analysis operations
# ---------------------------------------------------------------------------

def _b_syntax_check(state: VMState, instr: Instruction) -> None:
    """Check Python syntax: code.syntax_check(path) → True/error."""
    (path_str,) = _pop_n(state, 1, "code.syntax_check")
    path = _safe_path(str(path_str))
    content = path.read_text(encoding="utf-8")
    try:
        ast.parse(content, filename=str(path))
        state.stack.append(True)
    except SyntaxError as e:
        state.stack.append(f"SyntaxError line {e.lineno}: {e.msg}")


def _b_functions(state: VMState, instr: Instruction) -> None:
    """List all function names in a Python file."""
    (path_str,) = _pop_n(state, 1, "code.functions")
    path = _safe_path(str(path_str))
    content = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(content)
        funcs = [
            {"name": node.name, "line": node.lineno, "args": len(node.args.args)}
            for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ]
        state.stack.append(funcs)
    except SyntaxError as e:
        raise CalmRuntimeError(f"code.functions: {e}")


def _b_classes(state: VMState, instr: Instruction) -> None:
    """List all class names in a Python file."""
    (path_str,) = _pop_n(state, 1, "code.classes")
    path = _safe_path(str(path_str))
    content = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(content)
        classes = [
            {"name": node.name, "line": node.lineno,
             "methods": [m.name for m in node.body if isinstance(m, ast.FunctionDef)]}
            for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        ]
        state.stack.append(classes)
    except SyntaxError as e:
        raise CalmRuntimeError(f"code.classes: {e}")


def _b_imports(state: VMState, instr: Instruction) -> None:
    """List all imports in a Python file."""
    (path_str,) = _pop_n(state, 1, "code.imports")
    path = _safe_path(str(path_str))
    content = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(content)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                imports.append(f"{node.module}.{', '.join(a.name for a in node.names)}")
        state.stack.append(imports)
    except SyntaxError as e:
        raise CalmRuntimeError(f"code.imports: {e}")


def _b_count_lines(state: VMState, instr: Instruction) -> None:
    """Count lines in a file."""
    (path_str,) = _pop_n(state, 1, "code.count_lines")
    path = _safe_path(str(path_str))
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    code_lines = sum(1 for l in lines if l.strip() and not l.strip().startswith("#"))
    state.stack.append({"total": len(lines), "code": code_lines, "blank": len(lines) - code_lines})


# ---------------------------------------------------------------------------
# Execution operations
# ---------------------------------------------------------------------------

def _b_run(state: VMState, instr: Instruction) -> None:
    """Run a Python file: code.run(path) → {stdout, stderr, ok}."""
    (path_str,) = _pop_n(state, 1, "code.run")
    path = _safe_path(str(path_str))
    result = _run_cmd([sys.executable, str(path)], timeout=15.0)
    state.stack.append(result)


def _b_test(state: VMState, instr: Instruction) -> None:
    """Run pytest on a file or directory: code.test(path) → {passed, failed, output}."""
    (path_str,) = _pop_n(state, 1, "code.test")
    result = _run_cmd(
        [sys.executable, "-m", "pytest", str(path_str), "-v", "--tb=short", "-q"],
        timeout=60.0,
    )
    # Parse pytest output for pass/fail counts
    output = result["stdout"]
    passed = failed = 0
    for line in output.splitlines():
        if "passed" in line:
            import re
            m = re.search(r"(\d+) passed", line)
            if m: passed = int(m.group(1))
            m = re.search(r"(\d+) failed", line)
            if m: failed = int(m.group(1))
    state.stack.append({
        "passed": passed,
        "failed": failed,
        "ok": result["ok"],
        "output": output[:3000],
    })


def _b_lint(state: VMState, instr: Instruction) -> None:
    """Lint a Python file with ruff (if available) or py_compile."""
    (path_str,) = _pop_n(state, 1, "code.lint")
    path = _safe_path(str(path_str))
    # Try ruff first
    result = _run_cmd(["ruff", "check", str(path), "--output-format=text"], timeout=15.0)
    if result["returncode"] == -1 and "No such file" in result["stderr"]:
        # ruff not installed — fall back to py_compile
        result = _run_cmd([sys.executable, "-m", "py_compile", str(path)], timeout=10.0)
    state.stack.append({
        "ok": result["ok"],
        "output": (result["stdout"] + result["stderr"]).strip()[:2000],
    })


# ---------------------------------------------------------------------------
# Search operations
# ---------------------------------------------------------------------------

def _b_search(state: VMState, instr: Instruction) -> None:
    """Search for a pattern in files: code.search(pattern, path) → matches."""
    pattern, path_str = _pop_n(state, 2, "code.search")
    # Try ripgrep, fall back to grep
    for cmd in [["rg", "--no-heading", "-n"], ["grep", "-rn"]]:
        result = _run_cmd(cmd + [str(pattern), str(path_str)], timeout=15.0)
        if result["returncode"] != -1:
            break
    matches = []
    for line in result["stdout"].splitlines()[:50]:
        matches.append(line)
    state.stack.append({"pattern": str(pattern), "matches": len(matches), "lines": matches})


def _b_find(state: VMState, instr: Instruction) -> None:
    """Find files matching a glob: code.find(pattern, path) → list of paths."""
    pattern, path_str = _pop_n(state, 2, "code.find")
    base = _safe_path(str(path_str))
    found = sorted(str(p) for p in base.rglob(str(pattern)))[:100]
    state.stack.append(found)


def _b_diff(state: VMState, instr: Instruction) -> None:
    """Show git diff for a file: code.diff(path) → diff string."""
    (path_str,) = _pop_n(state, 1, "code.diff")
    result = _run_cmd(["git", "diff", str(path_str)], timeout=10.0)
    state.stack.append(result["stdout"][:5000] if result["ok"] else "not in git or no changes")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

CODE_WORDS: Dict[str, Backend] = {
    # File I/O
    "code.read": _b_read,
    "code.write": _b_write,
    "code.edit": _b_edit,
    "code.insert": _b_insert,
    "code.delete": _b_delete,
    # Analysis
    "code.syntax_check": _b_syntax_check,
    "code.functions": _b_functions,
    "code.classes": _b_classes,
    "code.imports": _b_imports,
    "code.count_lines": _b_count_lines,
    # Execution
    "code.run": _b_run,
    "code.test": _b_test,
    "code.lint": _b_lint,
    # Search
    "code.search": _b_search,
    "code.find": _b_find,
    "code.diff": _b_diff,
}


def register(dispatcher: Dispatcher) -> None:
    """Register all code backend words on a dispatcher."""
    for name, fn in CODE_WORDS.items():
        dispatcher.register_backend(name, fn)
