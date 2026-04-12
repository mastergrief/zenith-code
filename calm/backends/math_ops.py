"""
Math compute backend — CPU-native math operations.

Registers words like math.sqrt, math.pow, math.is_prime, math.gcd,
math.factorize. These run at native CPU speed, not model-predicted.
The model just says `call math.sqrt`, the backend pops from the stack,
computes the real answer, pushes it back.
"""

from __future__ import annotations

import math as _math
from typing import Dict

from calm.stack_vm import (
    Backend,
    CalmRuntimeError,
    Dispatcher,
    VMState,
    Instruction,
    _is_number,
    _pop_n,
)


def _b_sqrt(state: VMState, instr: Instruction) -> None:
    (a,) = _pop_n(state, 1, "math.sqrt")
    if not _is_number(a):
        raise CalmRuntimeError(f"math.sqrt: need numeric, got {type(a).__name__}")
    if a < 0:
        raise CalmRuntimeError(f"math.sqrt: negative input {a}")
    state.stack.append(_math.sqrt(a))


def _b_pow(state: VMState, instr: Instruction) -> None:
    base, exp = _pop_n(state, 2, "math.pow")
    if not (_is_number(base) and _is_number(exp)):
        raise CalmRuntimeError(f"math.pow: need numeric operands")
    state.stack.append(base ** exp)


def _b_is_prime(state: VMState, instr: Instruction) -> None:
    (n,) = _pop_n(state, 1, "math.is_prime")
    if not isinstance(n, int) or isinstance(n, bool):
        raise CalmRuntimeError(f"math.is_prime: need int, got {type(n).__name__}")
    if n < 2:
        state.stack.append(False)
        return
    if n < 4:
        state.stack.append(True)
        return
    if n % 2 == 0 or n % 3 == 0:
        state.stack.append(False)
        return
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            state.stack.append(False)
            return
        i += 6
    state.stack.append(True)


def _b_gcd(state: VMState, instr: Instruction) -> None:
    a, b = _pop_n(state, 2, "math.gcd")
    if not (isinstance(a, int) and isinstance(b, int)
            and not isinstance(a, bool) and not isinstance(b, bool)):
        raise CalmRuntimeError(f"math.gcd: need int operands")
    state.stack.append(_math.gcd(a, b))


def _b_factorize(state: VMState, instr: Instruction) -> None:
    """Push prime factors as individual stack values (smallest first)."""
    (n,) = _pop_n(state, 1, "math.factorize")
    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        raise CalmRuntimeError(f"math.factorize: need int >= 2, got {n}")
    factors = []
    d = 2
    tmp = n
    while d * d <= tmp:
        while tmp % d == 0:
            factors.append(d)
            tmp //= d
        d += 1
    if tmp > 1:
        factors.append(tmp)
    # Push factor count first, then each factor — caller knows how many to pop.
    state.stack.append(len(factors))
    for f in factors:
        state.stack.append(f)


def _b_floor(state: VMState, instr: Instruction) -> None:
    (a,) = _pop_n(state, 1, "math.floor")
    if not _is_number(a):
        raise CalmRuntimeError(f"math.floor: need numeric")
    state.stack.append(int(_math.floor(a)))


def _b_ceil(state: VMState, instr: Instruction) -> None:
    (a,) = _pop_n(state, 1, "math.ceil")
    if not _is_number(a):
        raise CalmRuntimeError(f"math.ceil: need numeric")
    state.stack.append(int(_math.ceil(a)))


def _b_log(state: VMState, instr: Instruction) -> None:
    (a,) = _pop_n(state, 1, "math.log")
    if not _is_number(a) or a <= 0:
        raise CalmRuntimeError(f"math.log: need positive numeric, got {a}")
    state.stack.append(_math.log(a))


def _b_pi(state: VMState, instr: Instruction) -> None:
    state.stack.append(_math.pi)


MATH_WORDS: Dict[str, Backend] = {
    "math.sqrt": _b_sqrt,
    "math.pow": _b_pow,
    "math.is_prime": _b_is_prime,
    "math.gcd": _b_gcd,
    "math.factorize": _b_factorize,
    "math.floor": _b_floor,
    "math.ceil": _b_ceil,
    "math.log": _b_log,
    "math.pi": _b_pi,
}


def register(dispatcher: Dispatcher) -> None:
    """Register all math backend words on a dispatcher."""
    for name, fn in MATH_WORDS.items():
        dispatcher.register_backend(name, fn)
