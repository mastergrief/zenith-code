"""R25: Scrape Python stdlib for (docstring, function-signature) pairs.

Stdlib provides ~3000-5000 functions with clean signatures + first-line
docstrings. Perfect-quality data: signatures are compiler-verified,
docstrings are canonical-form problem statements ("Return the X of
sequence S"), and arg names are consistent with the broader Python
ecosystem that Claude-authored corpora come from.

Expected yield: 1500-3000 pairs after vocab/length filters. Directly
addresses rare-class tail: stdlib has extensive `file`/`path`/`url`/
`pattern`/`bytes` argument usage that current corpora under-sample.

Usage:
    PYTHONPATH=. python3 scripts/build_stdlib_corpus.py \\
        --out agents/distill/data/stdlib_signatures.jsonl

Output is messages-schema JSONL compatible with CodeExampleDB.

Extraction rules:
  - module.submodule walk via pkgutil (stdlib only, no pip packages)
  - for each function/method: inspect.signature() + __doc__
  - first-line docstring as the problem (if >= 20 chars)
  - skip: dunder methods, C-extension functions without signature,
    private names (leading `_`), names over 40 chars
  - skip modules that require extra deps or raise on import
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import pkgutil
import sys
import warnings
from pathlib import Path


# Stdlib modules to scan. Explicit list avoids accidentally hitting
# deprecated / platform-specific modules.
_STDLIB_MODULES = [
    # Data / collections
    "collections", "collections.abc", "itertools", "functools", "operator",
    "heapq", "bisect", "array", "queue", "copy", "contextlib", "reprlib",
    "types",
    # Numbers / math
    "math", "cmath", "decimal", "fractions", "random", "statistics",
    "numbers",
    # Text
    "string", "re", "textwrap", "unicodedata", "difflib", "shlex",
    "stringprep",
    # Binary data
    "struct", "codecs", "base64", "binascii", "hashlib", "hmac", "secrets",
    # Dates / times
    "datetime", "calendar", "time", "zoneinfo",
    # Data format
    "json", "csv", "configparser", "tomllib", "plistlib",
    "xml.etree.ElementTree", "xml.dom.minidom", "html", "html.parser",
    "html.entities",
    # File / path
    "pathlib", "os.path", "shutil", "tempfile", "glob", "fnmatch",
    "linecache", "io", "fileinput",
    # Networking / internet
    "urllib.parse", "urllib.request", "urllib.response", "urllib.error",
    "urllib.robotparser", "socket", "ipaddress", "email.utils",
    "email.parser", "http.client", "http.cookies",
    # Compression / archiving
    "gzip", "zlib", "bz2", "lzma", "tarfile", "zipfile", "shutil",
    # System
    "sys", "platform", "argparse", "logging", "getopt", "os",
    # Typing / classes
    "typing", "dataclasses", "enum", "abc", "weakref",
    # Concurrency
    "concurrent.futures", "asyncio",
    # Debug / profiling
    "traceback", "pdb", "inspect",
]


# Popular pip packages — attempt import, skip if not installed. Each
# adds 100-1000 signatures on typical ML/web-dev Python environments.
_PIP_MODULES = [
    # HTTP / networking
    "requests", "httpx", "urllib3", "aiohttp",
    # Web frameworks
    "flask", "fastapi", "django",
    # Data science
    "numpy", "pandas", "scipy",
    # ML
    "sklearn", "torch", "torchvision", "transformers",
    # DB / ORM
    "sqlalchemy", "pymongo", "redis",
    # Validation / parsing
    "pydantic", "marshmallow", "yaml", "lxml", "bs4", "beautifulsoup4",
    # Async / concurrent
    "trio", "anyio",
    # CLI / config
    "click", "typer", "rich",
    # Testing
    "pytest", "hypothesis",
    # Utils
    "toml", "pillow", "PIL",
]


_ALLOWED = set(
    "0123456789+-*/()=.,:; "
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "_><"
)


def _clean_doc(doc, max_len: int = 180) -> str | None:
    """First-line docstring, normalized to vocab + length bounds."""
    if not doc or not isinstance(doc, str):
        return None
    first_line = doc.strip().split("\n")[0].strip().rstrip(".")
    if len(first_line) < 20:
        return None
    # Collapse whitespace; drop vocab-foreign chars
    s = " ".join(first_line.split())
    s = "".join(c if c in _ALLOWED else " " for c in s)
    s = " ".join(s.split())
    if 20 <= len(s) <= max_len:
        return s
    return None


def _clean_sig(sig: inspect.Signature, placeholder: str = "FN") -> str | None:
    """Render signature as `def FN(<args>):` with normalized args."""
    parts = []
    for name, param in sig.parameters.items():
        if param.kind == param.VAR_POSITIONAL:
            parts.append(f"*{name}")
        elif param.kind == param.VAR_KEYWORD:
            parts.append(f"**{name}")
        elif param.default is not param.empty:
            # Strip default value (canonical form)
            parts.append(name)
        else:
            parts.append(name)
    skeleton = f"def {placeholder}({', '.join(parts)}):"
    if len(skeleton) > 80:
        return None
    if not all(c in _ALLOWED for c in skeleton):
        return None
    return skeleton


def _iter_callables(module_name: str):
    """Yield (name, callable, doc) from a module, catching import errors."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = importlib.import_module(module_name)
    except BaseException as e:  # catch pyo3 PanicException + SystemExit etc
        print(f"  [skip] {module_name}: {type(e).__name__}: {str(e)[:80]}")
        return

    try:
        attrs = dir(mod)
    except BaseException as e:
        print(f"  [skip-dir] {module_name}: {e}")
        return

    for attr_name in attrs:
        if attr_name.startswith("_"):
            continue
        if len(attr_name) > 40:
            continue
        try:
            obj = getattr(mod, attr_name)
        except Exception:
            continue
        if not callable(obj):
            continue
        # Prefer first-line of obj.__doc__ as the "problem"
        doc = getattr(obj, "__doc__", None)
        if not doc:
            continue
        yield attr_name, obj, doc


def _discover_installed_packages() -> list[str]:
    """R28: auto-scan all top-level installed packages (stdlib + pip).
    Returns list of importable module names in the current Python env.
    Skips deprecated / known-problematic packages.

    ⚠ WARNING — IMPORT-TIME SIDE EFFECTS
    -----------------------------------
    `pkgutil.iter_modules()` with `PYTHONPATH=.` (the normal harness
    invocation) enumerates BOTH stdlib/pip AND local repo modules. A
    subsequent `importlib.import_module(name)` RUNS module-level code.

    Observed 2026-04-22: `scripts/r53_phase2_bench.py` ran its full
    ablation at import time — loaded Gemma (2.06 GB) into GPU, ran
    perf probes. Similar risk exists with ANY pip package whose top
    level performs eager work (`transformers` loads configs, some CUDA
    libs call `cuda.init`, telemetry packages send pings).

    Mitigations in place:
    1. The repo-exclusion `skip` set below filters our own top-level
       packages (calm, agents, scripts, src, tests, etc.) — these are
       the highest-impact side effects (GPU, disk, model loads).
    2. `--auto-scan` is OPT-IN. Default pipeline uses the hand-curated
       `_STDLIB_MODULES + _PIP_MODULES` lists which are known safe.
    3. `_iter_callables` catches BaseException on import to survive
       pyo3 panics.

    Before enabling `--auto-scan`:
      - Verify no NEW repo-local package names need adding to `skip`
      - Run first with `--verbose` and eyeball the module list
      - Monitor GPU VRAM during scan (`watch -n1 nvidia-smi`)
    """
    skip = {
        "pip", "pkg_resources", "setuptools", "wheel", "easy_install",
        "distutils", "ensurepip", "tkinter", "turtle", "turtledemo",
        "idlelib", "antigravity", "this", "__pycache__",
        "_distutils_hack", "_virtualenv", "_yaml",
        # R28 — exclude our OWN repo to avoid self-referential training data
        "calm", "agents", "scripts", "src", "tests", "rust", "bin",
        "models", "unsloth_compiled_cache", "RESEARCH",
    }
    found: set = set()
    for mod in pkgutil.iter_modules():
        name = mod.name
        if name.startswith("_") or name in skip:
            continue
        if len(name) < 2 or len(name) > 40:
            continue
        found.add(name)
    # Also include every stdlib top-level
    for name in getattr(sys, "stdlib_module_names", ()):
        if name.startswith("_") or name in skip:
            continue
        found.add(name)
    return sorted(found)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="agents/distill/data/stdlib_signatures.jsonl")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--auto-scan", action="store_true",
                    help="R28: auto-discover all installed packages.")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_skels: dict = {}  # skeleton → example
    seen_prob_hashes: set = set()

    n_modules = 0
    n_functions = 0
    n_kept = 0
    n_rejected_doc = 0
    n_rejected_sig = 0
    n_duplicate_prob = 0

    if args.auto_scan:
        auto_mods = _discover_installed_packages()
        all_modules = list(dict.fromkeys(list(_STDLIB_MODULES) +
                                          list(_PIP_MODULES) +
                                          auto_mods))
        print(f"auto-scan: {len(auto_mods)} packages discovered, "
              f"{len(all_modules)} total modules to scan")
    else:
        all_modules = list(_STDLIB_MODULES) + list(_PIP_MODULES)
    for mod_name in all_modules:
        source_tag = "stdlib" if mod_name in _STDLIB_MODULES else "pip"
        print(f"scanning {mod_name} ({source_tag}) ...")
        mod_kept = 0
        for attr_name, obj, doc in _iter_callables(mod_name):
            n_functions += 1
            try:
                sig = inspect.signature(obj)
            except BaseException:  # pyo3/Rust methods raise panics
                n_rejected_sig += 1
                continue
            skel = _clean_sig(sig)
            if skel is None:
                n_rejected_sig += 1
                continue
            prob = _clean_doc(doc)
            if prob is None:
                n_rejected_doc += 1
                continue
            # Dedup on problem text hash
            prob_hash = hash(prob)
            if prob_hash in seen_prob_hashes:
                n_duplicate_prob += 1
                continue
            seen_prob_hashes.add(prob_hash)
            # Keep — one record per function
            seen_skels[f"{mod_name}.{attr_name}"] = {
                "problem": prob,
                "solution": f"{skel}\n    pass",  # skeleton + stub body
                "fn_name": attr_name,
                "module": mod_name,
                "category": "stdlib",
            }
            mod_kept += 1
            n_kept += 1
            if args.verbose and mod_kept <= 3:
                print(f"  + {attr_name}({list(sig.parameters)})")
        if mod_kept:
            print(f"  kept {mod_kept}")
        n_modules += 1

    # Write as messages-schema JSONL for CodeExampleDB compatibility
    with out_path.open("w") as f:
        for key, rec in sorted(seen_skels.items()):
            messages = [
                {"role": "user", "content": rec["problem"]},
                {"role": "assistant", "content": rec["solution"]},
            ]
            f.write(json.dumps({
                "messages": messages,
                "fn_name": rec["fn_name"],
                "module": rec["module"],
                "category": rec["category"],
            }) + "\n")

    print(f"\n{'='*50}")
    print(f"Stats:")
    print(f"  modules scanned:      {n_modules}")
    print(f"  callables found:      {n_functions}")
    print(f"  kept (written):       {n_kept}")
    print(f"  rejected (bad sig):   {n_rejected_sig}")
    print(f"  rejected (bad doc):   {n_rejected_doc}")
    print(f"  rejected (duplicate): {n_duplicate_prob}")
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
