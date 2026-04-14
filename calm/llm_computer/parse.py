"""Parse HRM scratchpad traces into GateGraph.

The HRM emits a scratchpad trace like:
    "factorial(5) + 3 * 4 = <call>factorial(5)<end_call>120 + 3 * 4 = 120 + 12 = 132"

We only look at the **first segment** (before the first `=`), which is
the problem expression. The reductions that follow may contain value
errors from the HRM — we don't trust those. Instead, we parse the
problem expression into a `GateGraph` of `Const` / `BinOp` / `Delegate`
nodes and hand it to the interpreter, which recomputes every value
analytically.

Strategy:
  - HRM provides the **structure** (operator precedence, parens, which
    sub-problems to delegate via `<call>` markers).
  - Interpreter provides the **values** (integer arithmetic + backend
    calls resolved via `safe_eval`).

Delegation markers in the problem segment tell us which function calls
HRM wants routed to a verified backend. In practice, HRM's scratchpad
format puts `<call>fn(args)<end_call>` only in the REDUCTION segments,
not the original expression. So the parser finds function calls by
syntax (matching the `fn(args)` pattern) and emits `Delegate` nodes for
the known CALM backend functions. Unknown function names fall back to
raw `ast.parse` which would fail noisily — caller should catch.

Design: use Python's `ast` module as the front-end parser. The
expression syntax we support is a strict subset of Python: integer
literals, `+ - * /`, parentheses, and named function calls.
"""

from __future__ import annotations

import ast
import re
from typing import Dict

from calm.llm_computer.gate_graph import (
    BinOp, Const, Delegate, GateGraph, Node, Result,
)


# Function names known to resolve via the CALM backend registry — same
# list the HRM trace generator uses (data.py:_DELEGATED_FUNCS).
_DELEGATED_FUNCS = {
    "gcd", "factorial", "fibonacci", "is_prime",
    "euler_totient", "digital_root",
}


class ParseError(ValueError):
    pass


def parse_expression(expr: str) -> GateGraph:
    """Parse an expression string into a GateGraph.

    The graph contains exactly one `Result` node (`name='answer'`)
    whose `source` points to the root of the parsed expression. All
    intermediate nodes are chained through `BinOp` and `Delegate`
    dependencies so the interpreter can walk them in topo order.
    """
    expr = expr.strip()
    if not expr:
        raise ParseError("empty expression")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ParseError(f"cannot parse {expr!r}: {e}") from e

    graph = GateGraph()
    counter = {"n": 0}

    def fresh_name(prefix: str) -> str:
        counter["n"] += 1
        return f"{prefix}{counter['n']}"

    def visit(node: ast.AST) -> Node:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ParseError(f"non-numeric constant: {node.value!r}")
            value = int(node.value) if node.value == int(node.value) else node.value
            return graph.add(Const(name=fresh_name("c"), value=value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = visit(node.operand)
            if isinstance(inner, Const):
                # Fold the negation into a literal.
                inner.value = -inner.value
                return inner
            # General case: 0 - inner.
            zero = graph.add(Const(name=fresh_name("c"), value=0))
            return graph.add(BinOp(name=fresh_name("neg"), op="sub",
                                    left=zero, right=inner))
        if isinstance(node, ast.BinOp):
            left = visit(node.left)
            right = visit(node.right)
            op_map = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul",
                      ast.Div: "div", ast.FloorDiv: "div"}
            op_name = op_map.get(type(node.op))
            if op_name is None:
                raise ParseError(f"unsupported operator: {ast.dump(node.op)}")
            return graph.add(BinOp(name=fresh_name(op_name), op=op_name,
                                    left=left, right=right))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ParseError(f"complex call target unsupported: {ast.dump(node.func)}")
            fn_name = node.func.id
            if fn_name not in _DELEGATED_FUNCS:
                raise ParseError(f"unknown function: {fn_name}")
            args = [visit(a) for a in node.args]
            return graph.add(Delegate(name=fresh_name(fn_name), fn_name=fn_name,
                                       args=args))
        raise ParseError(f"unsupported AST node: {ast.dump(node)}")

    root = visit(tree)
    graph.add(Result(name="answer", source=root))
    return graph


def extract_problem_from_trace(trace: str) -> str:
    """Return the segment of the trace before the first `=`, which is
    the problem HRM is trying to solve.

    The HRM emits a trace like "expr = step1 = step2 = final". Everything
    after the first `=` is HRM's (possibly buggy) value computation.
    We only trust the problem shape; values get recomputed.

    Also strips any `<call>fn(args)<end_call>result` artifacts that
    leaked into the first segment (should be rare — the trace generator
    only emits these in reduction segments, but be defensive).
    """
    if "=" in trace:
        problem = trace.split("=", 1)[0]
    else:
        problem = trace
    # Strip <call>...<end_call>VALUE sequences — keep only the original expr.
    problem = re.sub(r"<call>([^<]+)<end_call>[^=;]*", r"\1", problem)
    return problem.strip()
