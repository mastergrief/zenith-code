"""Parse + interpret round-trip tests.

Each case: parse a math expression into a GateGraph, interpret it,
verify the result matches `safe_eval`. The interpreter and safe_eval
are independent code paths (interpreter walks `BinOp`/`Delegate` nodes,
safe_eval evaluates a Python AST via the backend function registry) —
agreement between them is the correctness signal.
"""

from __future__ import annotations

from calm.expression import safe_eval
from calm.llm_computer.interpret import interpret
from calm.llm_computer.parse import extract_problem_from_trace, parse_expression


CASES = [
    "17 * 23",
    "347 * 289",
    "25 * 88",
    "71 + 29",
    "17 - 25",
    "14 * 87",
    "(30 + 24) * (12 + 8)",
    "3 * (4 + 5)",
    "factorial(5)",
    "factorial(5) + 3 * 4",
    "gcd(48, 180)",
    "gcd(48, 180) + 100",
    "fibonacci(12)",
    "fibonacci(7) * 2",
    "is_prime(17)",
    "is_prime(105)",
    "1 - 38 * 28 + 38",
    "46 * 47 - 2",
    "factorial(7) - gcd(48, 180)",
]


def test_parse_interpret_matches_safe_eval():
    for expr in CASES:
        graph = parse_expression(expr)
        got = interpret(graph)
        expected = safe_eval(expr)
        # Bool/int equivalence for is_prime etc.
        assert got == expected, f"{expr!r}: interpret={got} safe_eval={expected}"


TRACES = [
    # Clean scratchpad — first segment is the expression.
    ("17 * 23 = 391", "17 * 23"),
    # With place-value decomp — first segment still the expression.
    ("25 * 88 = (25*80 + 25*8) = (2000 + 25*8) = (2000 + 200) = 2200", "25 * 88"),
    # Function-only trace with <call> markers — should unwrap.
    ("<call>factorial(5)<end_call>120 = 120", "factorial(5)"),
    # Mixed: function + arithmetic.
    ("factorial(5) + 3 * 4 = <call>factorial(5)<end_call>120 + 3 * 4 = 120 + 12 = 132",
     "factorial(5) + 3 * 4"),
]


def test_extract_problem_strips_call_markers():
    for trace, expected in TRACES:
        got = extract_problem_from_trace(trace)
        assert got == expected, f"trace {trace!r}: got {got!r}, expected {expected!r}"


def test_end_to_end_trace_to_answer():
    """Full pipeline: trace string → problem → graph → interpret → answer.
    Verify output matches direct safe_eval on the problem."""
    for trace, _ in TRACES:
        problem = extract_problem_from_trace(trace)
        graph = parse_expression(problem)
        got = interpret(graph)
        expected = safe_eval(problem)
        assert got == expected, f"trace {trace!r}: got {got} expected {expected}"


if __name__ == "__main__":
    test_parse_interpret_matches_safe_eval()
    print("[ok] parse_interpret_matches_safe_eval")
    test_extract_problem_strips_call_markers()
    print("[ok] extract_problem_strips_call_markers")
    test_end_to_end_trace_to_answer()
    print("[ok] end_to_end_trace_to_answer")
