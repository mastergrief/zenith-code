"""
Wasm compute backend — arithmetic via WebAssembly.

Loads calm_math.wat (compiled to wasm at import time via wasmtime),
registers words like wasm.add, wasm.mul, wasm.sqrt, wasm.gcd that
execute at native speed through the wasm VM.

The point: these operations are 10^6-10^9x faster than running them
through a transformer forward pass. The LLM just says "wasm.mul",
the CPU does the work.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from calm.stack_vm import (
    Backend,
    CalmRuntimeError,
    Dispatcher,
    Instruction,
    VMState,
    _is_number,
    _pop_n,
)

_WAT_PATH = Path(__file__).parent / "calm_math.wat"

# Lazy-initialized wasm instance.
_instance = None
_store = None


def _ensure_wasm():
    """Compile and instantiate the wasm module (lazy, once)."""
    global _instance, _store
    if _instance is not None:
        return

    try:
        import wasmtime
    except ImportError:
        raise RuntimeError(
            "wasmtime not installed — run: pip install wasmtime"
        )

    _store = wasmtime.Store()
    module = wasmtime.Module.from_file(_store.engine, str(_WAT_PATH))
    _instance = wasmtime.Instance(_store, module, [])


def _call(func_name: str, *args):
    """Call a wasm export and return the result."""
    _ensure_wasm()
    fn = _instance.exports(_store)[func_name]
    try:
        return fn(_store, *args)
    except Exception as e:
        # wasmtime.Trap on div-by-zero, overflow, etc.
        msg = str(e)
        if "divide by zero" in msg:
            raise CalmRuntimeError(f"{func_name}: division by zero")
        raise CalmRuntimeError(f"{func_name}: wasm trap: {msg}")


# ---------------------------------------------------------------------------
# Integer wasm backends (pop Python int, call i64 export, push int back)
# ---------------------------------------------------------------------------

def _make_i_binop(wasm_name: str, calm_name: str) -> Backend:
    def _fn(state: VMState, instr: Instruction) -> None:
        a, b = _pop_n(state, 2, calm_name)
        if not (isinstance(a, int) and isinstance(b, int)
                and not isinstance(a, bool) and not isinstance(b, bool)):
            raise CalmRuntimeError(f"{calm_name}: need two ints")
        try:
            result = _call(wasm_name, a, b)
        except Exception as e:
            raise CalmRuntimeError(f"{calm_name}: wasm error: {e}")
        state.stack.append(int(result))
    return _fn


def _make_i_unop(wasm_name: str, calm_name: str) -> Backend:
    def _fn(state: VMState, instr: Instruction) -> None:
        (a,) = _pop_n(state, 1, calm_name)
        if not (isinstance(a, int) and not isinstance(a, bool)):
            raise CalmRuntimeError(f"{calm_name}: need int")
        result = _call(wasm_name, a)
        state.stack.append(int(result))
    return _fn


# ---------------------------------------------------------------------------
# Float wasm backends (pop Python number, call f64 export, push float back)
# ---------------------------------------------------------------------------

def _make_f_binop(wasm_name: str, calm_name: str) -> Backend:
    def _fn(state: VMState, instr: Instruction) -> None:
        a, b = _pop_n(state, 2, calm_name)
        if not (_is_number(a) and _is_number(b)):
            raise CalmRuntimeError(f"{calm_name}: need numeric operands")
        result = _call(wasm_name, float(a), float(b))
        state.stack.append(float(result))
    return _fn


def _make_f_unop(wasm_name: str, calm_name: str) -> Backend:
    def _fn(state: VMState, instr: Instruction) -> None:
        (a,) = _pop_n(state, 1, calm_name)
        if not _is_number(a):
            raise CalmRuntimeError(f"{calm_name}: need numeric")
        result = _call(wasm_name, float(a))
        state.stack.append(float(result))
    return _fn


# ---------------------------------------------------------------------------
# Mixed: auto-dispatch int or float based on operand types
# ---------------------------------------------------------------------------

def _make_auto_binop(i_name: str, f_name: str, calm_name: str) -> Backend:
    """Integer path if both args are int, float path otherwise."""
    def _fn(state: VMState, instr: Instruction) -> None:
        a, b = _pop_n(state, 2, calm_name)
        if not (_is_number(a) and _is_number(b)):
            raise CalmRuntimeError(f"{calm_name}: need numeric operands")
        if isinstance(a, int) and isinstance(b, int):
            result = _call(i_name, a, b)
            state.stack.append(int(result))
        else:
            result = _call(f_name, float(a), float(b))
            state.stack.append(float(result))
    return _fn


# ---------------------------------------------------------------------------
# Word registry
# ---------------------------------------------------------------------------

WASM_WORDS: Dict[str, Backend] = {
    # Auto-dispatch (int when possible, float otherwise)
    "wasm.add": _make_auto_binop("i_add", "f_add", "wasm.add"),
    "wasm.sub": _make_auto_binop("i_sub", "f_sub", "wasm.sub"),
    "wasm.mul": _make_auto_binop("i_mul", "f_mul", "wasm.mul"),
    "wasm.div": _make_auto_binop("i_div", "f_div", "wasm.div"),
    "wasm.mod": _make_i_binop("i_mod", "wasm.mod"),
    "wasm.neg": _make_i_unop("i_neg", "wasm.neg"),
    "wasm.abs": _make_i_unop("i_abs", "wasm.abs"),
    "wasm.pow": _make_i_binop("i_pow", "wasm.pow"),
    "wasm.gcd": _make_i_binop("i_gcd", "wasm.gcd"),
    # Float-only
    "wasm.sqrt": _make_f_unop("f_sqrt", "wasm.sqrt"),
    "wasm.floor": _make_f_unop("f_floor", "wasm.floor"),
    "wasm.ceil": _make_f_unop("f_ceil", "wasm.ceil"),
    "wasm.fabs": _make_f_unop("f_abs", "wasm.fabs"),
    # Explicit integer path
    "wasm.iadd": _make_i_binop("i_add", "wasm.iadd"),
    "wasm.imul": _make_i_binop("i_mul", "wasm.imul"),
    "wasm.idiv": _make_i_binop("i_div", "wasm.idiv"),
    # Explicit float path
    "wasm.fadd": _make_f_binop("f_add", "wasm.fadd"),
    "wasm.fmul": _make_f_binop("f_mul", "wasm.fmul"),
    "wasm.fdiv": _make_f_binop("f_div", "wasm.fdiv"),
}


def register(dispatcher: Dispatcher) -> None:
    """Register all wasm backend words on a dispatcher."""
    _ensure_wasm()  # fail fast at registration time
    for name, fn in WASM_WORDS.items():
        dispatcher.register_backend(name, fn)
