"""Interpret a GateGraph — evaluate each node in topo order.

This is the "analytical correctness" half of the HRM-+-LLM-Computer
split. The interpreter walks the compute subset of the GateGraph
(`Const`, `BinOp`, `Delegate`, `Result`) and produces the final answer
by evaluating every node exactly. No approximation, no memorization.

`Delegate` nodes route through `safe_eval` from our existing CALM
engine — so the entire registry of 1000+ verified backend functions is
available behind `Delegate(fn_name, args)`.

Future direction: swap this Python interpreter for a compiled
Small2DTransformer that runs the gate graph as transformer weights
(doc 03 §8). Same IR, different execution substrate.
"""

from __future__ import annotations

from typing import Any, Dict

from calm.expression import ExpressionError, safe_eval
from calm.llm_computer.gate_graph import (
    BinOp, Const, Delegate, GateGraph, Node, Result,
)


class InterpreterError(RuntimeError):
    pass


def interpret(graph: GateGraph) -> Any:
    """Walk the compute nodes of `graph` and return the Result value.

    Nodes are evaluated in insertion order, which is a valid topological
    order because `parse.py` builds children before parents. If the
    graph has no `Result`, the value of the last compute node is
    returned.
    """
    values: Dict[str, Any] = {}

    def val_of(node: Node) -> Any:
        if node.name not in values:
            raise InterpreterError(f"node {node.name!r} referenced before eval")
        return values[node.name]

    last_value: Any = None
    for node in graph.nodes:
        if isinstance(node, Const):
            values[node.name] = node.value
        elif isinstance(node, BinOp):
            if node.left is None or node.right is None:
                raise InterpreterError(f"BinOp {node.name!r} missing operand")
            a = val_of(node.left)
            b = val_of(node.right)
            if node.op == "add":
                values[node.name] = a + b
            elif node.op == "sub":
                values[node.name] = a - b
            elif node.op == "mul":
                values[node.name] = a * b
            elif node.op == "div":
                # Integer division for whole-number results, else float.
                if isinstance(a, int) and isinstance(b, int) and a % b == 0:
                    values[node.name] = a // b
                else:
                    values[node.name] = a / b
            else:
                raise InterpreterError(f"unknown BinOp {node.op!r}")
        elif isinstance(node, Delegate):
            args_str = ", ".join(str(val_of(a)) for a in node.args)
            try:
                values[node.name] = safe_eval(f"{node.fn_name}({args_str})")
            except ExpressionError as e:
                raise InterpreterError(f"delegate {node.fn_name}({args_str}): {e}") from e
        elif isinstance(node, Result):
            if node.source is None:
                raise InterpreterError(f"Result {node.name!r} has no source")
            values[node.name] = val_of(node.source)
        else:
            # Unknown or hardware node — skip for compute interpretation.
            continue
        last_value = values[node.name]

    results = graph.results()
    if results:
        return values[results[-1].name]
    return last_value
