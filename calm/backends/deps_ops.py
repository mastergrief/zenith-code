"""
CALM dependency backend — verified package/import analysis.

Models say "pip install X" without checking conflicts. This backend
inspects installed packages, parses requirements, and detects issues.

Functions: installed_packages, check_import, requirements_parse,
version_check, dependency_tree, conflict_detect.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import List, Optional


def installed_version(package: str) -> Optional[str]:
    """Get installed version of a package. None if not installed."""
    try:
        mod = importlib.import_module(package.replace("-", "_"))
        return getattr(mod, "__version__", getattr(mod, "VERSION", "installed"))
    except ImportError:
        # Try importlib.metadata (Python 3.8+).
        try:
            from importlib.metadata import version
            return version(package)
        except Exception:
            return None


def is_installed(package: str) -> bool:
    """Check if a package is installed."""
    return installed_version(package) is not None


def requirements_parse(path: str) -> list:
    """Parse a requirements.txt file.
    Returns [{name, version_spec, line}]."""
    import re
    results = []
    try:
        for i, line in enumerate(Path(path).read_text().splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            m = re.match(r'^([a-zA-Z0-9_\-\.]+)\s*([><=!~]+.+)?$', line)
            if m:
                results.append({
                    "name": m.group(1),
                    "version_spec": m.group(2) or "any",
                    "line": i,
                })
    except FileNotFoundError:
        return [{"error": f"file not found: {path}"}]
    return results


def imports_used(path: str) -> list:
    """Extract all imports from a Python file.
    Returns [{module, is_stdlib, is_installed}]."""
    import ast

    _STDLIB = {
        "abc", "argparse", "ast", "asyncio", "base64", "bisect",
        "calendar", "collections", "concurrent", "configparser",
        "contextlib", "copy", "csv", "dataclasses", "datetime",
        "decimal", "difflib", "email", "enum", "errno", "fnmatch",
        "fractions", "functools", "glob", "gzip", "hashlib", "heapq",
        "html", "http", "importlib", "inspect", "io", "ipaddress",
        "itertools", "json", "logging", "math", "mimetypes",
        "multiprocessing", "operator", "os", "pathlib", "pickle",
        "platform", "pprint", "queue", "random", "re", "shlex",
        "shutil", "signal", "socket", "sqlite3", "statistics",
        "string", "struct", "subprocess", "sys", "tempfile",
        "textwrap", "threading", "time", "timeit", "traceback",
        "typing", "unittest", "urllib", "uuid", "warnings",
        "weakref", "xml", "zipfile", "zlib",
    }

    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, FileNotFoundError):
        return [{"error": f"cannot parse {path}"}]

    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])

    results = []
    for mod in sorted(modules):
        results.append({
            "module": mod,
            "is_stdlib": mod in _STDLIB,
            "is_installed": is_installed(mod),
        })
    return results


def missing_imports(path: str) -> list:
    """Find imports that aren't installed.
    Returns [{module}] — packages you need to pip install."""
    imports = imports_used(path)
    return [
        imp for imp in imports
        if not imp.get("is_stdlib") and not imp.get("is_installed")
        and "error" not in imp
    ]


def python_path() -> list:
    """Current Python path entries."""
    return sys.path


DEPS_FUNCTIONS = {
    "installed_version": installed_version,
    "is_installed": is_installed,
    "requirements_parse": requirements_parse,
    "imports_used": imports_used,
    "missing_imports": missing_imports,
    "python_path": python_path,
}

DEPS_NL_PATTERNS = [
    (r'(?:is)\s+(\w+)\s+(?:installed|available)', 'is_installed("{0}")'),
    (r'(?:what version|version of)\s+(\w[\w-]+)', 'installed_version("{0}")'),
    (r'(?:missing|uninstalled)\s+imports?\s+(?:in|for)', None),
]
