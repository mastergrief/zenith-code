"""Multi-step reasoning compute backend.

Deterministic functions for chained arithmetic, comparison, conditionals,
sequence costing, syllogisms, extrema, percentages, and ratios. Each
function handles one reasoning pattern that the PT extracts from NL.

CALM verifies every intermediate step — the model reasons, these
functions compute, the engine cross-checks.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Union

Number = Union[int, float]


def chained_eval(a: Number, op1: str, b: Number, op2: str, c: Number) -> Number:
    """Evaluate a chained expression: a op1 b op2 c (left-to-right)."""
    ops = {"+": lambda x, y: x + y, "-": lambda x, y: x - y,
           "*": lambda x, y: x * y, "/": lambda x, y: x / y if y != 0 else float("inf")}
    if op1 not in ops or op2 not in ops:
        raise ValueError(f"Unknown operator: {op1} or {op2}")
    intermediate = ops[op1](a, b)
    return ops[op2](intermediate, c)


def compare(a: Number, op: str, b: Number) -> bool:
    """Compare two values: a op b → bool."""
    cmp = {">": lambda x, y: x > y, "<": lambda x, y: x < y,
           ">=": lambda x, y: x >= y, "<=": lambda x, y: x <= y,
           "==": lambda x, y: x == y, "!=": lambda x, y: x != y}
    if op not in cmp:
        raise ValueError(f"Unknown comparator: {op}")
    return cmp[op](a, b)


def conditional_eval(condition: bool, val_true: Number, val_false: Number) -> Number:
    """If condition then val_true else val_false."""
    return val_true if condition else val_false


def sequence_cost(*args: Number) -> Number:
    """Compute total cost from (quantity, price) pairs.

    Args alternate: q1, p1, q2, p2, ... → q1*p1 + q2*p2 + ...
    """
    if len(args) % 2 != 0:
        raise ValueError("Need even number of args: q1, p1, q2, p2, ...")
    total = 0
    for i in range(0, len(args), 2):
        total += args[i] * args[i + 1]
    return total


def syllogism_check(a_rel_b: bool, b_rel_c: bool) -> bool:
    """Transitive syllogism: if A→B and B→C then A→C."""
    return a_rel_b and b_rel_c


def multi_max(*args: Number) -> Number:
    """Maximum of N values."""
    if not args:
        raise ValueError("Need at least one value")
    return max(args)


def multi_min(*args: Number) -> Number:
    """Minimum of N values."""
    if not args:
        raise ValueError("Need at least one value")
    return min(args)


def percentage(x: Number, y: Number) -> Number:
    """x percent of y → x/100 * y."""
    return x / 100 * y


def ratio_simplify(x: int, y: int) -> str:
    """Simplify ratio x:y → simplified string."""
    if y == 0:
        return f"{x}:0"
    g = math.gcd(abs(x), abs(y))
    return f"{x // g}:{y // g}"


def ratio_decimal(x: Number, y: Number) -> float:
    """Ratio as decimal: x / y."""
    if y == 0:
        return float("inf")
    return x / y


def step_by_step(expression: str) -> Number:
    """Evaluate a multi-step expression with standard precedence.

    Uses Python's ast-safe evaluation via the CALM expression engine.
    """
    from calm.expression import safe_eval
    return safe_eval(expression)


REASONING_FUNCTIONS = {
    "chained_eval": chained_eval,
    "compare": compare,
    "conditional_eval": conditional_eval,
    "sequence_cost": sequence_cost,
    "syllogism_check": syllogism_check,
    "multi_max": multi_max,
    "multi_min": multi_min,
    "percentage": percentage,
    "ratio_simplify": ratio_simplify,
    "ratio_decimal": ratio_decimal,
    "step_by_step": step_by_step,
}

REASONING_NL_PATTERNS = [
    # Chained arithmetic
    (r"(?:if|when)\s+.*(?:and then|then)\s+.*(?:how (?:much|many)|what|total)", "step_by_step"),
    (r"(?:first|start).*(?:then|next|after).*(?:how (?:much|many)|total|result)", "step_by_step"),
    # Comparison
    (r"(?:is|are)\s+\d+\s+(?:greater|larger|bigger|more|less|smaller|fewer)\s+than\s+\d+", "compare"),
    (r"(?:which is|who has)\s+(?:more|less|bigger|smaller)", "compare"),
    # Percentage
    (r"\d+\s*(?:percent|%)\s+of\s+\d+", "percentage"),
    (r"what is\s+\d+\s*%\s+of\s+\d+", "percentage"),
    # Sequence cost
    (r"\d+\s+(?:items?|things?|units?)\s+(?:at|for|costing)\s+\d+\s+(?:each|per|dollars)", "sequence_cost"),
    (r"(?:buy|purchase|order)\s+\d+.*(?:at|for)\s+\d+.*(?:and|plus)\s+\d+.*(?:at|for)\s+\d+", "sequence_cost"),
    # Ratio
    (r"ratio\s+of\s+\d+\s+(?:to|and)\s+\d+", "ratio_simplify"),
    (r"\d+\s+out\s+of\s+\d+", "ratio_decimal"),
    # Max/min
    (r"(?:largest|biggest|maximum|greatest|highest)\s+(?:of|among|between)", "multi_max"),
    (r"(?:smallest|minimum|least|lowest)\s+(?:of|among|between)", "multi_min"),
    # Conditional
    (r"if\s+.*(?:then|,)\s+.*(?:otherwise|else)", "conditional_eval"),
    # Syllogism
    (r"if\s+.*(?:and|,)\s+.*(?:therefore|then|so)\s+.*(?:is|must)", "syllogism_check"),
]
