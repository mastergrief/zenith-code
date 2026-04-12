"""
String compute backend — CPU-native string operations.

Words like str.len, str.upper, str.lower, str.contains, str.split,
str.concat. Useful for problems where the model needs to manipulate
text precisely (regex, formatting, validation).
"""

from __future__ import annotations

import re as _re
from typing import Dict

from calm.stack_vm import (
    Backend,
    CalmRuntimeError,
    Dispatcher,
    VMState,
    Instruction,
    _pop_n,
)


def _b_len(state: VMState, instr: Instruction) -> None:
    (s,) = _pop_n(state, 1, "str.len")
    if not isinstance(s, str):
        raise CalmRuntimeError(f"str.len: need string, got {type(s).__name__}")
    state.stack.append(len(s))


def _b_upper(state: VMState, instr: Instruction) -> None:
    (s,) = _pop_n(state, 1, "str.upper")
    if not isinstance(s, str):
        raise CalmRuntimeError(f"str.upper: need string")
    state.stack.append(s.upper())


def _b_lower(state: VMState, instr: Instruction) -> None:
    (s,) = _pop_n(state, 1, "str.lower")
    if not isinstance(s, str):
        raise CalmRuntimeError(f"str.lower: need string")
    state.stack.append(s.lower())


def _b_contains(state: VMState, instr: Instruction) -> None:
    haystack, needle = _pop_n(state, 2, "str.contains")
    if not (isinstance(haystack, str) and isinstance(needle, str)):
        raise CalmRuntimeError(f"str.contains: need two strings")
    state.stack.append(needle in haystack)


def _b_concat(state: VMState, instr: Instruction) -> None:
    a, b = _pop_n(state, 2, "str.concat")
    if not (isinstance(a, str) and isinstance(b, str)):
        raise CalmRuntimeError(f"str.concat: need two strings")
    state.stack.append(a + b)


def _b_regex_match(state: VMState, instr: Instruction) -> None:
    """Pop (string, pattern), push bool for whether pattern matches."""
    text, pattern = _pop_n(state, 2, "str.regex_match")
    if not (isinstance(text, str) and isinstance(pattern, str)):
        raise CalmRuntimeError(f"str.regex_match: need two strings")
    try:
        state.stack.append(bool(_re.search(pattern, text)))
    except _re.error as e:
        raise CalmRuntimeError(f"str.regex_match: bad pattern: {e}")


def _b_replace(state: VMState, instr: Instruction) -> None:
    """Pop (string, old, new), push result."""
    s, old, new = _pop_n(state, 3, "str.replace")
    if not all(isinstance(x, str) for x in (s, old, new)):
        raise CalmRuntimeError(f"str.replace: need three strings")
    state.stack.append(s.replace(old, new))


STRING_WORDS: Dict[str, Backend] = {
    "str.len": _b_len,
    "str.upper": _b_upper,
    "str.lower": _b_lower,
    "str.contains": _b_contains,
    "str.concat": _b_concat,
    "str.regex_match": _b_regex_match,
    "str.replace": _b_replace,
}


def register(dispatcher: Dispatcher) -> None:
    """Register all string backend words on a dispatcher."""
    for name, fn in STRING_WORDS.items():
        dispatcher.register_backend(name, fn)
