"""
CALM Python language backend — verified Python-specific operations.

Models hallucinate Python builtins, stdlib APIs, and version features.
This backend verifies claims about the language itself.

Functions: builtin_exists, module_exists, has_method, python_version,
type_check, exception_hierarchy, magic_methods, stdlib_functions.
"""

from __future__ import annotations

import builtins
import importlib
import inspect
import sys
from typing import List, Optional


def builtin_exists(name: str) -> bool:
    """Check if a Python builtin exists.
    Models sometimes hallucinate builtins like 'flatten' or 'chunk'."""
    return hasattr(builtins, name)


def module_exists(name: str) -> bool:
    """Check if a Python module/package is importable."""
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def has_method(type_name: str, method: str) -> bool:
    """Check if a type has a method.
    Example: has_method("str", "removeprefix") → True (3.9+)
    Models often hallucinate methods like str.contains() or list.flatten()."""
    type_map = {
        "str": str, "list": list, "dict": dict, "set": set,
        "tuple": tuple, "int": int, "float": float, "bytes": bytes,
        "bytearray": bytearray, "frozenset": frozenset,
    }
    t = type_map.get(type_name)
    if t is None:
        return False
    return hasattr(t, method)


def list_methods(type_name: str) -> list:
    """List all public methods of a type.
    Example: list_methods("str") → ["capitalize", "casefold", ...]"""
    type_map = {
        "str": str, "list": list, "dict": dict, "set": set,
        "tuple": tuple, "int": int, "float": float, "bytes": bytes,
    }
    t = type_map.get(type_name)
    if t is None:
        return []
    return sorted(m for m in dir(t) if not m.startswith('_'))


def python_version() -> dict:
    """Current Python version info."""
    return {
        "version": sys.version.split()[0],
        "major": sys.version_info.major,
        "minor": sys.version_info.minor,
        "micro": sys.version_info.micro,
    }


def type_of(value: str) -> str:
    """Determine the type of a Python literal.
    Example: type_of("42") → "int", type_of("[1,2]") → "list" """
    import ast
    try:
        node = ast.literal_eval(value)
        return type(node).__name__
    except (ValueError, SyntaxError):
        return "unknown"


def exception_hierarchy(exc_name: str) -> list:
    """Get the MRO (method resolution order) for an exception.
    Example: exception_hierarchy("ValueError") → ["ValueError", "Exception", "BaseException"]
    Models sometimes get exception inheritance wrong."""
    exc_map = {
        "ValueError": ValueError, "TypeError": TypeError,
        "KeyError": KeyError, "IndexError": IndexError,
        "AttributeError": AttributeError, "NameError": NameError,
        "ImportError": ImportError, "FileNotFoundError": FileNotFoundError,
        "OSError": OSError, "IOError": IOError,
        "RuntimeError": RuntimeError, "StopIteration": StopIteration,
        "ZeroDivisionError": ZeroDivisionError,
        "OverflowError": OverflowError, "MemoryError": MemoryError,
        "RecursionError": RecursionError, "NotImplementedError": NotImplementedError,
        "PermissionError": PermissionError, "TimeoutError": TimeoutError,
        "ConnectionError": ConnectionError, "UnicodeError": UnicodeError,
    }
    exc = exc_map.get(exc_name)
    if exc is None:
        return [f"unknown: {exc_name}"]
    return [c.__name__ for c in exc.__mro__]


def magic_methods(category: str = "all") -> list:
    """List Python magic/dunder methods by category.
    Categories: comparison, arithmetic, container, string, context, iterator, all."""
    _METHODS = {
        "comparison": ["__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__"],
        "arithmetic": ["__add__", "__sub__", "__mul__", "__truediv__", "__floordiv__",
                       "__mod__", "__pow__", "__neg__", "__abs__"],
        "container": ["__len__", "__getitem__", "__setitem__", "__delitem__",
                      "__contains__", "__iter__", "__reversed__"],
        "string": ["__str__", "__repr__", "__format__", "__bytes__"],
        "context": ["__enter__", "__exit__"],
        "iterator": ["__iter__", "__next__"],
        "callable": ["__call__"],
        "hashing": ["__hash__", "__eq__"],
        "attribute": ["__getattr__", "__setattr__", "__delattr__", "__getattribute__"],
    }
    if category == "all":
        all_methods = []
        for methods in _METHODS.values():
            all_methods.extend(methods)
        return sorted(set(all_methods))
    return _METHODS.get(category, [])


def stdlib_search(keyword: str) -> list:
    """Search stdlib module names containing a keyword.
    Example: stdlib_search("json") → ["json", "json.decoder", "json.encoder"]"""
    # Common stdlib modules (not exhaustive but covers most).
    _STDLIB = [
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
    ]
    keyword = keyword.lower()
    return [m for m in _STDLIB if keyword in m]


PYTHON_FUNCTIONS = {
    "builtin_exists": builtin_exists,
    "module_exists": module_exists,
    "has_method": has_method,
    "list_methods": list_methods,
    "python_version": python_version,
    "type_of": type_of,
    "exception_hierarchy": exception_hierarchy,
    "magic_methods": magic_methods,
    "stdlib_search": stdlib_search,
}

PYTHON_NL_PATTERNS = [
    (r'(?:is|does)\s+(\w+)\s+(?:a\s+)?(?:Python\s+)?(?:builtin|built-in)', 'builtin_exists("{0}")'),
    (r'(?:is|does)\s+(\w+)\s+(?:a\s+)?(?:Python\s+)?(?:module|package|library)', 'module_exists("{0}")'),
    (r'(?:what|list)\s+(?:methods?|attributes?)\s+(?:does|of|on)\s+(\w+)', 'list_methods("{0}")'),
    (r'(?:what is|what\'s)\s+the\s+(?:exception|error)\s+hierarchy\s+(?:of|for)\s+(\w+)', 'exception_hierarchy("{0}")'),
]
