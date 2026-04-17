"""Round 46.1: raw-path parser + executor for MultiStepReasoningFacade.

No Gemma required — this tests the parse → evaluate pipeline against
Python eval on hand-crafted + randomly generated chains.

Gate: ≥ 95/100 parse success, 100/100 evaluation correctness on the
subset that parses. Parse failures on pathological NL are acceptable;
wrong values on a parsed expression are a bug.
"""

from __future__ import annotations

import random
import re

import pytest

from calm.llm_computer.facades.multi_step import MultiStepReasoningFacade


# --- Hand-crafted prompts, expected (expression, value) ---
# Python eval gives the ground truth.
HAND_PROMPTS = [
    # Algebraic, multi-step
    ("What is 17 * 23 + 45?",                 "17 * 23 + 45", 17*23+45),
    ("Compute 100 / 4 - 7",                   "100 / 4 - 7",  100//4-7),
    ("3 + 4 * 5",                             "3 + 4 * 5",    3+4*5),
    ("(3 + 4) * 5",                           "(3 + 4) * 5",  (3+4)*5),
    ("What is 8*7 - 13?",                     "8*7 - 13",     8*7-13),
    # NL operators
    ("What is 17 times 23 plus 45?",          "17 * 23 + 45", 17*23+45),
    ("Compute 100 divided by 4 minus 7",      "100 / 4 - 7",  100//4-7),
    ("7 plus 8 times 9",                      "7 + 8 * 9",    7+8*9),
    ("17 multiplied by 23 plus 5",            "17 * 23 + 5",  17*23+5),
    # Three-step
    ("What is 2 + 3 * 4 - 5?",                "2 + 3 * 4 - 5", 2+3*4-5),
    ("Compute 10 - 2 * 3 + 7",                "10 - 2 * 3 + 7", 10-2*3+7),
    # Single step (should still parse — smallest multi-step)
    ("What is 47 * 19?",                      "47 * 19",      47*19),
    ("Compute 123 + 456",                     "123 + 456",    123+456),
    # Edge: decimal, negative result
    ("What is 5 - 17?",                       "5 - 17",       5-17),
    ("Compute 20 * 3 - 100",                  "20 * 3 - 100", 20*3-100),
]


def test_hand_parse_and_evaluate():
    facade = MultiStepReasoningFacade()
    failures = []
    for prompt, exp_expr, exp_val in HAND_PROMPTS:
        parsed = facade.parse(prompt)
        if parsed is None:
            failures.append((prompt, "PARSE_FAIL", None, exp_val))
            continue
        value = facade.evaluate(parsed)
        if value != exp_val:
            failures.append((prompt, parsed, value, exp_val))
    if failures:
        msg = "\n".join(
            f"  {p!r} -> expr={e!r} val={v!r}  (expected {ev})"
            for p, e, v, ev in failures)
        pytest.fail(f"{len(failures)}/{len(HAND_PROMPTS)} failed:\n{msg}")


def _gen_random_chain(rng, n_ops: int) -> tuple[str, int]:
    """Generate a random algebraic chain: returns (expression, value).

    Integer ops only: +, -, *. Operands in [1, 99]. // when a is
    cleanly divisible (otherwise skipped to avoid Python int-div rules
    diverging from safe_eval truediv)."""
    ops = ["+", "-", "*"]
    a = rng.randint(1, 99)
    parts = [str(a)]
    for _ in range(n_ops):
        op = rng.choice(ops)
        b = rng.randint(1, 99)
        parts.append(op)
        parts.append(str(b))
    expr = " ".join(parts)
    # Safe_eval on "/" returns float; we're testing int-only here.
    val = eval(expr)  # trusted input: we built it
    return expr, val


def test_random_algebraic_100():
    facade = MultiStepReasoningFacade()
    rng = random.Random(42)
    n_parsed = 0
    n_correct = 0
    failures = []
    for _ in range(100):
        n_ops = rng.choice([1, 2, 3, 4])
        expr, val = _gen_random_chain(rng, n_ops)
        prompt = f"What is {expr}?"
        parsed = facade.parse(prompt)
        if parsed is None:
            failures.append((prompt, "PARSE_FAIL", None, val))
            continue
        n_parsed += 1
        result = facade.evaluate(parsed)
        if result == val:
            n_correct += 1
        else:
            failures.append((prompt, parsed, result, val))

    print(f"\n  parsed {n_parsed}/100, correct {n_correct}/100")
    assert n_parsed >= 95, (
        f"parser coverage below gate ({n_parsed}/100):\n"
        + "\n".join(f"    {p!r} (expected {v})"
                    for p, _, _, v in failures[:5]))
    assert n_correct == n_parsed, (
        f"executor returned wrong values on {n_parsed - n_correct} "
        f"parsed prompts:\n"
        + "\n".join(f"    {p!r} expr={e!r} got {r!r} expected {v}"
                    for p, e, r, v in failures[:5]))


def _gen_nl_random_chain(rng, n_ops: int) -> tuple[str, str, int]:
    """Random chain using NL-word operators: returns (prompt, expr, value)."""
    nl_ops = [("plus", "+"), ("minus", "-"), ("times", "*")]
    a = rng.randint(1, 99)
    prompt_parts = [str(a)]
    expr_parts = [str(a)]
    for _ in range(n_ops):
        nl, sym = rng.choice(nl_ops)
        b = rng.randint(1, 99)
        prompt_parts.append(nl)
        prompt_parts.append(str(b))
        expr_parts.append(sym)
        expr_parts.append(str(b))
    prompt = "Compute " + " ".join(prompt_parts)
    expr = " ".join(expr_parts)
    val = eval(expr)
    return prompt, expr, val


def test_random_nl_50():
    facade = MultiStepReasoningFacade()
    rng = random.Random(123)
    n_parsed = 0
    n_correct = 0
    failures = []
    for _ in range(50):
        n_ops = rng.choice([1, 2, 3])
        prompt, expr, val = _gen_nl_random_chain(rng, n_ops)
        parsed = facade.parse(prompt)
        if parsed is None:
            failures.append((prompt, "PARSE_FAIL", None, val))
            continue
        n_parsed += 1
        result = facade.evaluate(parsed)
        if result == val:
            n_correct += 1
        else:
            failures.append((prompt, parsed, result, val))

    print(f"\n  NL parsed {n_parsed}/50, correct {n_correct}/50")
    assert n_parsed >= 45, (
        f"NL parser coverage below gate ({n_parsed}/50):\n"
        + "\n".join(f"    {p!r} (expected {v})"
                    for p, _, _, v in failures[:5]))
    assert n_correct == n_parsed, (
        f"NL executor wrong on {n_parsed - n_correct} prompts:\n"
        + "\n".join(f"    {p!r} expr={e!r} got {r!r} expected {v}"
                    for p, e, r, v in failures[:5]))
