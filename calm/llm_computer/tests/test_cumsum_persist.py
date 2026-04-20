"""R18 port: CumSum + PersistLinear compute nodes + interpreter.

Mirrors transformer-vm's CumSumDimension + PersistDimension. Upstream
has these as evaluator-only (no transformer-weight compilation); this
port mirrors the semantic via the project's interpreter path.
"""

from __future__ import annotations

from calm.llm_computer.gate_graph import (
    BinOp, Const, CumSum, GateGraph, PersistLinear, Result,
)
from calm.llm_computer.interpret import interpret


# ---------------------------------------------------------------------
# CumSum
# ---------------------------------------------------------------------


def test_cumsum_single_step_equals_source():
    """First eval of a CumSum = source value (accumulator starts at 0)."""
    g = GateGraph()
    x = g.add(Const("x", value=7))
    cs = g.add(CumSum("cs", source=x))
    g.add(Result("r", source=cs))
    # Interpreter is single-step (one pass over nodes) — running total = x
    assert interpret(g) == 7.0


def test_cumsum_accumulates_across_reinterpret():
    """Multiple interpret() calls on the same graph accumulate, because
    CumSum._accum persists as node state."""
    g = GateGraph()
    x = g.add(Const("x", value=3))
    cs = g.add(CumSum("cs", source=x))
    g.add(Result("r", source=cs))
    assert interpret(g) == 3.0   # accum 0 → 3
    assert interpret(g) == 6.0   # accum 3 → 6
    assert interpret(g) == 9.0   # accum 6 → 9


def test_cumsum_mixed_with_binop():
    """CumSum can feed into a BinOp downstream."""
    g = GateGraph()
    x = g.add(Const("x", value=5))
    cs = g.add(CumSum("cs", source=x))
    two = g.add(Const("two", value=2))
    doubled = g.add(BinOp("doubled", op="mul", left=cs, right=two))
    g.add(Result("r", source=doubled))
    # First interpret: accum 0→5, then doubled = 10
    assert interpret(g) == 10.0


def test_cumsum_sign_source_const():
    """Source constant can be negative."""
    g = GateGraph()
    x = g.add(Const("x", value=-4))
    cs = g.add(CumSum("cs", source=x))
    g.add(Result("r", source=cs))
    assert interpret(g) == -4.0


# ---------------------------------------------------------------------
# PersistLinear
# ---------------------------------------------------------------------


def test_persist_linear_identity():
    """Single source coef=1.0 → persist value = source."""
    g = GateGraph()
    x = g.add(Const("x", value=10))
    p = g.add(PersistLinear("p", coefs=[(x, 1.0)]))
    g.add(Result("r", source=p))
    assert interpret(g) == 10


def test_persist_linear_combination():
    """Standard 3-source linear combo: 3x - 2y + z."""
    g = GateGraph()
    x = g.add(Const("x", value=2))
    y = g.add(Const("y", value=5))
    z = g.add(Const("z", value=7))
    p = g.add(PersistLinear("p", coefs=[(x, 3.0), (y, -2.0), (z, 1.0)]))
    g.add(Result("r", source=p))
    # 3*2 - 2*5 + 7 = 6 - 10 + 7 = 3
    assert interpret(g) == 3


def test_persist_empty_coefs_is_zero():
    """PersistLinear with no coefs — value is 0 (sum over empty set)."""
    g = GateGraph()
    p = g.add(PersistLinear("p", coefs=[]))
    g.add(Result("r", source=p))
    assert interpret(g) == 0


def test_persist_and_cumsum_composed():
    """CumSum feeds into PersistLinear, proves the two primitives
    compose cleanly through the interpreter."""
    g = GateGraph()
    a = g.add(Const("a", value=4))
    b = g.add(Const("b", value=3))
    cs = g.add(CumSum("cs", source=a))
    # PersistLinear: cs - b
    p = g.add(PersistLinear("p", coefs=[(cs, 1.0), (b, -1.0)]))
    g.add(Result("r", source=p))
    # cs accum: 0→4, then p = 4 - 3 = 1
    assert interpret(g) == 1.0
