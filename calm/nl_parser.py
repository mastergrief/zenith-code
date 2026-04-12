"""
CALM v0.1 natural-language instruction parser.

LLMs naturally emit compute instructions in varied forms:
  - "multiply 17 by 23"
  - "is_prime(391)"
  - "sqrt of 1764"
  - "gcd(391, 782)"
  - "17 + 23"
  - "check if 391 is prime"

This module translates these into stack_vm instructions so the
interceptor can execute them. It's a fuzzy front-end to the strict
parser — catches common patterns, falls through to the standard
parser for anything it doesn't recognize.

Usage:
    from calm.nl_parser import normalize_calm_line
    stack_code = normalize_calm_line("multiply 17 by 23")
    # -> "push 17\npush 23\nmul"
"""

from __future__ import annotations

import re
from typing import Optional


# Patterns: (regex, replacement template)
# Templates use {0}, {1}, {2} for captured groups.
_PATTERNS = [
    # Function call syntax: func(a, b) or func(a)
    (r'^(sqrt|is_prime|is_prime|isprime|neg|abs|floor|ceil|log)\s*\(\s*([^,)]+)\s*\)$',
     lambda m: f"push {m.group(2).strip()}\n{_resolve(m.group(1))}"),
    (r'^(add|sub|mul|div|mod|gcd|pow)\s*\(\s*([^,)]+)\s*,\s*([^,)]+)\s*\)$',
     lambda m: f"push {m.group(2).strip()}\npush {m.group(3).strip()}\n{_resolve(m.group(1))}"),

    # "X op Y" infix: "17 + 23", "391 / 17"
    (r'^(-?[\d.]+)\s*\+\s*(-?[\d.]+)$',
     lambda m: f"push {m.group(1)}\npush {m.group(2)}\nadd"),
    (r'^(-?[\d.]+)\s*-\s*(-?[\d.]+)$',
     lambda m: f"push {m.group(1)}\npush {m.group(2)}\nsub"),
    (r'^(-?[\d.]+)\s*\*\s*(-?[\d.]+)$',
     lambda m: f"push {m.group(1)}\npush {m.group(2)}\nmul"),
    (r'^(-?[\d.]+)\s*/\s*(-?[\d.]+)$',
     lambda m: f"push {m.group(1)}\npush {m.group(2)}\ndiv"),
    (r'^(-?[\d.]+)\s*%\s*(-?[\d.]+)$',
     lambda m: f"push {m.group(1)}\npush {m.group(2)}\nmod"),

    # "multiply X by Y", "add X and Y", "subtract Y from X"
    (r'^multiply\s+(-?[\d.]+)\s+by\s+(-?[\d.]+)$',
     lambda m: f"push {m.group(1)}\npush {m.group(2)}\nmul"),
    (r'^add\s+(-?[\d.]+)\s+and\s+(-?[\d.]+)$',
     lambda m: f"push {m.group(1)}\npush {m.group(2)}\nadd"),
    (r'^subtract\s+(-?[\d.]+)\s+from\s+(-?[\d.]+)$',
     lambda m: f"push {m.group(2)}\npush {m.group(1)}\nsub"),
    (r'^divide\s+(-?[\d.]+)\s+by\s+(-?[\d.]+)$',
     lambda m: f"push {m.group(1)}\npush {m.group(2)}\ndiv"),

    # "X to the power of Y", "X raised to Y"
    (r'^(-?[\d.]+)\s+(?:to the power of|raised to)\s+(-?[\d.]+)$',
     lambda m: f"push {m.group(1)}\npush {m.group(2)}\nmath.pow"),

    # "square root of X", "sqrt of X"
    (r'^(?:square root|sqrt)\s+of\s+(-?[\d.]+)$',
     lambda m: f"push {m.group(1)}\nmath.sqrt"),

    # "gcd of X and Y"
    (r'^gcd\s+of\s+(-?[\d.]+)\s+and\s+(-?[\d.]+)$',
     lambda m: f"push {m.group(1)}\npush {m.group(2)}\nmath.gcd"),

    # "is X prime", "check if X is prime"
    (r'^(?:check\s+)?(?:if\s+)?(-?[\d]+)\s+is\s+prime$',
     lambda m: f"push {m.group(1)}\nmath.is_prime"),
    (r'^is\s+(-?[\d]+)\s+prime\??$',
     lambda m: f"push {m.group(1)}\nmath.is_prime"),

    # "factorize X", "factor X", "factors of X"
    (r'^(?:factorize|factor|factors\s+of)\s+(-?[\d]+)$',
     lambda m: f"push {m.group(1)}\nmath.factorize"),

    # Bare "op X" for unary: "sqrt 1764", "is_prime 391"
    (r'^(sqrt|is_prime|isprime|neg|abs|floor|ceil|log)\s+(-?[\d.]+)$',
     lambda m: f"push {m.group(2)}\n{_resolve(m.group(1))}"),

    # Bare "op X Y" for binary: "gcd 391 782", "pow 2 10"
    (r'^(gcd|pow|add|sub|mul|div|mod)\s+(-?[\d.]+)\s+(-?[\d.]+)$',
     lambda m: f"push {m.group(2)}\npush {m.group(3)}\n{_resolve(m.group(1))}"),
]

# Word resolution: common names → CALM words
_WORD_MAP = {
    "sqrt": "math.sqrt",
    "is_prime": "math.is_prime",
    "isprime": "math.is_prime",
    "gcd": "math.gcd",
    "pow": "math.pow",
    "floor": "math.floor",
    "ceil": "math.ceil",
    "log": "math.log",
    "factorize": "math.factorize",
    "factor": "math.factorize",
    "add": "add",
    "sub": "sub",
    "mul": "mul",
    "div": "div",
    "mod": "mod",
    "neg": "neg",
    "abs": "abs",
}


def _resolve(name: str) -> str:
    """Resolve a word name to its CALM form."""
    return _WORD_MAP.get(name.lower().strip(), name)


def normalize_calm_line(line: str) -> Optional[str]:
    """
    Try to convert a natural-language instruction into stack_vm code.
    Returns the normalized code string, or None if no pattern matches
    (in which case the standard parser should handle it).
    """
    line = line.strip()
    if not line:
        return None

    for pattern, template in _PATTERNS:
        m = re.match(pattern, line, re.IGNORECASE)
        if m:
            return template(m)

    return None


def normalize_calm_block(block: str) -> str:
    """
    Normalize an entire CALM block. Each line is either:
    - Already valid stack code (returned as-is)
    - Natural language (translated via normalize_calm_line)
    - A claim suffix (preserved as-is)
    """
    lines = []
    for raw_line in block.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("\\") or stripped.startswith("//"):
            lines.append(raw_line)
            continue

        # Strip claim suffix for NL parsing, then re-attach.
        claim = ""
        for suffix_re in (r'\s*->\s*<pending>$', r'\s*->\s*\[[^\]]*\]$'):
            m = re.search(suffix_re, stripped)
            if m:
                claim = stripped[m.start():]
                stripped = stripped[:m.start()].strip()
                break

        normalized = normalize_calm_line(stripped)
        if normalized:
            # Multi-line normalized code — attach claim to last line.
            norm_lines = normalized.splitlines()
            norm_lines[-1] += claim
            lines.extend(norm_lines)
        else:
            lines.append(raw_line)

    return "\n".join(lines)
