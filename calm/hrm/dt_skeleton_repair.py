"""Post-decode AST-style repair for DT skeleton outputs.

DT emits `def FN(<args>):` — a narrow output grammar. Round 1/2
diagnostic showed ~15-25% of missed outputs are one-char-off from
valid (`d FN(n):`, `def FN(x,:`, `def m, n):`, `def FN(sel:`,
`def):`). These are trivially repairable by a handful of regex
rewrites, no LLM in the loop.

Pattern follows `calm/llm_computer/facades/ast_repair.py` — walker
rewrites driven by deterministic rules on raw output. Inference-only,
zero training cost, composes with any DT checkpoint.

Usage:
    from calm.hrm.dt_skeleton_repair import repair_skeleton
    out = "d FN(n):"          # DT's malformed output
    repaired = repair_skeleton(out)  # "def FN(n):"

Rules (applied in order, first-match-wins):
    R1  'd<something>' → 'def<something>' (missing 'ef')
    R2  'def <ident>,:' → 'def FN(<ident>):' (missing FN wrapper)
    R3  'def FN(x,:' or 'def FN(x:' → 'def FN(x):' (unclosed paren)
    R4  'def FN(x,,:' → 'def FN(x):' (trailing-comma cleanup)

Output invariant: if repair_skeleton returns X != input, then X is a
valid `def FN(<args>):` skeleton. If input is already valid or can't
be repaired, input is returned unchanged.
"""
from __future__ import annotations

import re


# Validity: matches DT's target grammar exactly.
_VALID_RE = re.compile(r"^def FN\(([^)]*)\)\s*:$")

# Rewrites applied in order. Each is (pattern, replacement_callable_or_str).
# When callable, it gets the match object and returns the new string.
_DEF_MISSING_RE = re.compile(r"^d([^e].*)$")
_FN_MISSING_RE = re.compile(r"^def ([a-zA-Z_][a-zA-Z_0-9]*,.*)$")
# "def FN(xyz,:" or "def FN(xyz:"  — missing closing paren
_UNCLOSED_PAREN_RE = re.compile(r"^(def FN\([^)]*?)([,]?\s*):\s*$")
_NO_PAREN_OPEN_RE = re.compile(r"^def FN([^(].*):$")


def _is_valid(s: str) -> bool:
    return bool(_VALID_RE.match(s.strip()))


def _valid_args(args: str) -> bool:
    """Args between parens must be a comma-sep list of identifiers
    (possibly with spaces). Empty OK."""
    s = args.strip()
    if not s:
        return True
    for tok in [t.strip() for t in s.split(",")]:
        if not tok:
            return False
        # Allow identifiers + *args / **kwargs / self
        if not re.match(r"^\*{0,2}[a-zA-Z_][a-zA-Z_0-9]*$", tok):
            return False
    return True


def repair_skeleton(output: str) -> str:
    """Return a validity-preserving repaired skeleton, or original
    string unchanged if already valid or unrepairable."""
    s = output.strip()
    if _is_valid(s):
        return s

    # R1: "d FN(n):" → "def FN(n):" (missing "ef")
    m = _DEF_MISSING_RE.match(s)
    if m and m.group(1).startswith(" FN"):
        candidate = "def" + m.group(1)
        if _is_valid(candidate):
            return candidate

    # R2: "def m, n):" → "def FN(m, n):" (missing FN( wrapper)
    #     detect by: starts "def <ident>," and has ") at end" or ":" at end
    m = re.match(r"^def ([a-zA-Z_][a-zA-Z_0-9]*(?:,[^)]*)?)\)?:$", s)
    if m and not s.startswith("def FN"):
        args_raw = m.group(1)
        # Trim trailing ',' or ', '
        args = args_raw.rstrip(", ")
        candidate = f"def FN({args}):"
        if _is_valid(candidate) and _valid_args(args):
            return candidate

    # R3: "def FN(xyz,:" or "def FN(xyz:" or "def FN(sel:" → add ")"
    m = _UNCLOSED_PAREN_RE.match(s)
    if m:
        args_raw = m.group(1)[len("def FN("):]
        args = args_raw.rstrip(", ")
        candidate = f"def FN({args}):"
        if _is_valid(candidate) and _valid_args(args):
            return candidate

    # R4: "def FN(x,,:" etc — drop any trailing punctuation before building
    m = re.match(r"^def FN\((.*)$", s)
    if m:
        rest = m.group(1)
        # Strip trailing :, ), ,, spaces
        rest = rest.rstrip(":) ,")
        args = rest.rstrip(", ")
        candidate = f"def FN({args}):"
        if _is_valid(candidate) and _valid_args(args):
            return candidate

    # Unrepairable.
    return output
