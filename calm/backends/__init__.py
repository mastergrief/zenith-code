"""
CALM backend registry — auto-discovers and registers all backends.

Two naming conventions:
  *_ops.py — compute backends (functions that DO something)
  *_kb.py  — knowledge backends (functions that LOOK UP something)

Each backend exports a *_FUNCTIONS dict. This module scans for both
at import time and builds a unified registry. Adding a backend is:
write the file, done. Zero other files to edit.

Usage:
    from calm.backends import BACKEND_FUNCTIONS, BACKEND_MODULES
    # BACKEND_FUNCTIONS: {func_name: callable, ...} — all 411+ functions
    # BACKEND_MODULES: {module_path: category, ...} — for system prompt
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Dict, Callable, Tuple

# All registered functions from all backends.
BACKEND_FUNCTIONS: Dict[str, Callable] = {}

# Module → category mapping for the system prompt builder.
# category is derived from module name: foo_ops → foo
BACKEND_MODULES: Dict[str, str] = {}

# Which functions came from which category.
FUNCTION_CATEGORIES: Dict[str, str] = {}

# NL patterns collected from backends for auto-precompute.
# Each entry: (compiled_regex, template_string)
# template_string uses {0}, {1}, etc. for regex groups, e.g. 'circle_area({0})'
NL_PATTERNS: list = []


def _discover_backends():
    """Scan calm/backends/ for *_ops.py modules and register their functions."""
    package_dir = Path(__file__).parent

    for _importer, modname, _ispkg in pkgutil.iter_modules([str(package_dir)]):
        if not (modname.endswith("_ops") or modname.endswith("_kb")):
            continue

        full_name = f"calm.backends.{modname}"
        # Derive category from module name: date_ops → date, country_kb → country
        category = modname.replace("_ops", "").replace("_kb", "")

        try:
            mod = importlib.import_module(full_name)
        except ImportError:
            continue

        # Find the *_FUNCTIONS dict and optional *_NL_PATTERNS in the module.
        for attr_name in dir(mod):
            if attr_name.endswith("_FUNCTIONS") and isinstance(getattr(mod, attr_name), dict):
                funcs = getattr(mod, attr_name)
                BACKEND_FUNCTIONS.update(funcs)
                BACKEND_MODULES[full_name] = category
                for func_name in funcs:
                    FUNCTION_CATEGORIES[func_name] = category
                break  # One *_FUNCTIONS dict per module.

        # Collect NL patterns if present.
        for attr_name in dir(mod):
            if attr_name.endswith("_NL_PATTERNS") and isinstance(getattr(mod, attr_name), list):
                import re as _re
                for pattern, template in getattr(mod, attr_name):
                    NL_PATTERNS.append((_re.compile(pattern, _re.IGNORECASE), template))
                break


_discover_backends()
