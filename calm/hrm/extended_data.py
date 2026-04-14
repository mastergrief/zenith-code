"""Extended training data — 6 formats beyond the original 4 domains.

These are the exact formats the h=32 and h=64 multi-task HRMs failed on
in `scripts/eval_hrm_ood.py`. Adding them to training tests the
distribution hypothesis: does format diversity teach format-invariance?

The 6 new training generators mirror the OOD test's format categories:
  - code_var:    `x = 17; y = 23; result = x op y`
  - prefix_op:   `add 8 and 14`, `subtract 12 from 50`, `multiply 7 and 9`
  - distractor:  narrative with irrelevant details, one real computation
  - units:       `3 meters plus 7 meters equals`
  - let_bound:   `if a=5 and b=12, what is a+b`
  - eq_complete: `50 + 30 = ?`

For Experiment 2's NEW held-out OOD test, we use format VARIATIONS (not
duplicates) of these — e.g., train on `if a=X and b=Y, what is a+b` but
test on `let a be X, let b be Y, find a+b`. That disentangles "learned
the exact template" from "learned format-invariance."
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from calm.expression import ExpressionError, safe_eval


@dataclass
class ExtFormatProblem:
    """Uniform shape: input is the surface text, expression is the target."""
    source: str       # format category, for telemetry
    input: str        # encoder input
    expression: str   # decoder target (math expression)


# ----- Generator helpers ------------------------------------------------

def _eval_safe(expr: str):
    try:
        ans = safe_eval(expr)
        if isinstance(ans, float) and ans == int(ans):
            ans = int(ans)
        return ans
    except (ExpressionError, OverflowError):
        return None


_OPS = ["+", "-", "*"]
_OP_NAMES = {"+": "plus", "-": "minus", "*": "times"}
_ADD_VERBS = ["add"]
_SUB_VERBS = ["subtract"]
_MUL_VERBS = ["multiply"]


# --- 1. code_var: `x = A; y = B; result = x op y` ---------------------

def _gen_code_var(rng: random.Random) -> ExtFormatProblem:
    a = rng.randint(1, 999)
    b = rng.randint(1, 999)
    op = rng.choice(_OPS)
    # Pick varied variable names.
    names = rng.choice([("x", "y"), ("a", "b"), ("p", "q"), ("m", "n")])
    result_name = rng.choice(["result", "sum", "diff", "prod", "r"])
    inp = f"{names[0]} = {a}; {names[1]} = {b}; {result_name} = {names[0]} {op} {names[1]}"
    return ExtFormatProblem("code_var", inp, f"{a} {op} {b}")


# --- 2. prefix_op: `add A and B`, `subtract A from B`, `multiply A and B` ---

def _gen_prefix_op(rng: random.Random) -> ExtFormatProblem:
    a = rng.randint(1, 999)
    b = rng.randint(1, 999)
    kind = rng.choice(["add", "sub", "mul"])
    if kind == "add":
        inp = f"add {a} and {b}"
        expr = f"{a} + {b}"
    elif kind == "sub":
        # "subtract A from B" means B - A
        inp = f"subtract {a} from {b}"
        expr = f"{b} - {a}"
    else:
        inp = f"multiply {a} and {b}"
        expr = f"{a} * {b}"
    return ExtFormatProblem("prefix_op", inp, expr)


# --- 3. distractor: narrative with filler + one real computation ------

_ACTORS = ["tom", "lisa", "alice", "bob", "jane", "tim", "sam", "eve"]
_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_PLACES = ["the market", "the store", "the bakery", "the library", "the shop"]
_ITEMS = ["apples", "cookies", "books", "marbles", "coins"]

_DISTRACTOR_TEMPLATES: List[Callable[[Dict, random.Random], Tuple[str, str]]] = [
    lambda v, r: (
        f"on {r.choice(_DAYS)} {r.choice(_ACTORS)} went to {r.choice(_PLACES)}. "
        f"{r.choice(_ACTORS)} bought {v['a']} {r.choice(_ITEMS)} and then {v['b']} more. how many {r.choice(_ITEMS)}",
        f"{v['a']} + {v['b']}",
    ),
    lambda v, r: (
        f"{r.choice(_ACTORS)} drove for {v['a']} hours at {v['b']} mph. what distance did {r.choice(_ACTORS)} cover",
        f"{v['a']} * {v['b']}",
    ),
    lambda v, r: (
        f"a library has {v['a']} books. yesterday {v['b']} were borrowed. how many remain",
        f"{v['a']} - {v['b']}",
    ),
    lambda v, r: (
        f"at {v['a']} past noon {r.choice(_ACTORS)} left the house and drove {v['b']} miles. how many miles did {r.choice(_ACTORS)} drive",
        f"0 + {v['b']}",
    ),
]


def _gen_distractor(rng: random.Random) -> ExtFormatProblem:
    tmpl = rng.choice(_DISTRACTOR_TEMPLATES)
    v = {"a": rng.randint(1, 99), "b": rng.randint(1, 99)}
    inp, expr = tmpl(v, rng)
    return ExtFormatProblem("distractor", inp, expr)


# --- 4. units: `A meters plus B meters equals` ------------------------

_UNITS = ["meters", "kilograms", "liters", "grams", "seconds", "feet", "dollars"]


def _gen_units(rng: random.Random) -> ExtFormatProblem:
    a = rng.randint(1, 999)
    b = rng.randint(1, 999)
    op = rng.choice(_OPS)
    unit = rng.choice(_UNITS)
    op_word = _OP_NAMES[op]
    if op == "*":
        # "A units times B" — doesn't require "units" on the second operand
        inp = f"{a} {unit} {op_word} {b} equals"
    else:
        inp = f"{a} {unit} {op_word} {b} {unit} equals"
    return ExtFormatProblem("units", inp, f"{a} {op} {b}")


# --- 5. let_bound: `if a=A and b=B, what is a op b` -------------------

_LET_TEMPLATES: List[Callable[[int, int, str, random.Random], Tuple[str, str]]] = [
    lambda a, b, op, r: (f"if a={a} and b={b}, what is a{op}b", f"{a} {op} {b}"),
    lambda a, b, op, r: (f"if a = {a} and b = {b}, what is a {op} b", f"{a} {op} {b}"),
    lambda a, b, op, r: (
        f"if x equals {a} and y equals {b}, what is x {op} y",
        f"{a} {op} {b}",
    ),
    lambda a, b, op, r: (f"given p={a} and q={b}, compute p {op} q", f"{a} {op} {b}"),
]


def _gen_let_bound(rng: random.Random) -> ExtFormatProblem:
    a = rng.randint(1, 999)
    b = rng.randint(1, 999)
    op = rng.choice(_OPS)
    tmpl = rng.choice(_LET_TEMPLATES)
    inp, expr = tmpl(a, b, op, rng)
    return ExtFormatProblem("let_bound", inp, expr)


# --- 6. eq_complete: `A op B = ?` -------------------------------------

def _gen_eq_complete(rng: random.Random) -> ExtFormatProblem:
    a = rng.randint(1, 999)
    b = rng.randint(1, 999)
    op = rng.choice(_OPS)
    inp = f"{a} {op} {b} = ?"
    return ExtFormatProblem("eq_complete", inp, f"{a} {op} {b}")


# --- Top-level generator ----------------------------------------------

_GENERATORS = {
    "code_var": _gen_code_var,
    "prefix_op": _gen_prefix_op,
    "distractor": _gen_distractor,
    "units": _gen_units,
    "let_bound": _gen_let_bound,
    "eq_complete": _gen_eq_complete,
}


class ExtendedFormatGenerator:
    """Generates balanced samples across the 6 new formats."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def generate(self, n: int = 1200) -> List[ExtFormatProblem]:
        """n/6 per format, shuffled. Enforces encoder length bound (≤ 128)."""
        per = n // 6
        all_problems: List[ExtFormatProblem] = []
        for fmt, gen in _GENERATORS.items():
            made = 0
            attempts = 0
            while made < per and attempts < per * 10:
                attempts += 1
                p = gen(self._rng)
                if _eval_safe(p.expression) is None:
                    continue
                if len(p.input) + 2 > 128:
                    continue
                if len(p.expression) + 2 + 1 > 28:
                    continue
                all_problems.append(p)
                made += 1
        self._rng.shuffle(all_problems)
        return all_problems
