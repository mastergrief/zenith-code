"""10 MORE training formats beyond the first extended set.

Goal: push the distribution-probe scaling curve by adding another 10
formats. Each is deliberately different from the existing 10:

  11. fn-call       — `f(17, 23, op=mul)`, `compute(50, 30, +)`
  12. phrasal-verb  — `take 8 and 14 and sum them`
  13. past-narr     — `she had 100. she gave 25. how much left`
  14. alt-let       — `let a be 5, let b be 12, find a+b`
  15. eq-var        — `what is the result of 7 * 9`
  16. three-operand — `A + B + C`, `A * B - C`
  17. possessive-of — `the sum of 8 and 14`, `the product of 7 and 9`
  18. verb-by       — `multiply 7 by 9`, `divide 48 by 12` (not / div)
  19. question-first — `what's 17 times 23`
  20. when-then     — `when a is 5 and b is 12, a+b`

For the held-out test (20 formats → held-out), we'll design formats
that aren't close to ANY of the 20 trained formats — testing whether
the scaling curve keeps climbing or plateaus.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Tuple

from calm.expression import ExpressionError, safe_eval


@dataclass
class Ext2Problem:
    source: str
    input: str
    expression: str


_OPS = ["+", "-", "*"]
_OP_NAMES = {"+": "plus", "-": "minus", "*": "times"}


def _eval_safe(expr: str):
    try:
        ans = safe_eval(expr)
        if isinstance(ans, float) and ans == int(ans):
            ans = int(ans)
        return ans
    except (ExpressionError, OverflowError):
        return None


# 11. fn-call: `f(a, b, op=NAME)` or `compute(a, b, OP)` or `apply(a, b, VERB)`

def _gen_fn_call(rng: random.Random) -> Ext2Problem:
    a = rng.randint(1, 999)
    b = rng.randint(1, 999)
    op = rng.choice(_OPS)
    style = rng.choice(["op=", "infix", "verb"])
    if style == "op=":
        op_name = {"+": "add", "-": "sub", "*": "mul"}[op]
        inp = f"f({a}, {b}, op={op_name})"
    elif style == "infix":
        inp = f"compute({a}, {b}, {op})"
    else:
        verb = {"+": "add", "-": "subtract", "*": "multiply"}[op]
        inp = f"apply({a}, {b}, {verb})"
    return Ext2Problem("fn_call", inp, f"{a} {op} {b}")


# 12. phrasal-verb: `take A and B and sum them`, `take A and B and multiply`

def _gen_phrasal(rng: random.Random) -> Ext2Problem:
    a = rng.randint(1, 999)
    b = rng.randint(1, 999)
    op = rng.choice(_OPS)
    verb = {"+": "sum them", "-": "find the difference", "*": "multiply"}[op]
    inp = f"take {a} and {b} and {verb}"
    return Ext2Problem("phrasal", inp, f"{a} {op} {b}")


# 13. past-narr (simpler than 'distractor' — no actors, just pure narrative)

def _gen_past_narr(rng: random.Random) -> Ext2Problem:
    a = rng.randint(1, 99)
    b = rng.randint(1, min(a, 99))  # ensure non-negative for subtraction stories
    op = rng.choice(["+", "-"])
    if op == "+":
        templates = [
            (f"there were {a} items. {b} more arrived. how many items total", f"{a} + {b}"),
            (f"she had {a} coins. she earned {b} more. how many coins does she have", f"{a} + {b}"),
        ]
    else:
        templates = [
            (f"she had {a} dollars. she gave away {b}. how much is left", f"{a} - {b}"),
            (f"there were {a} birds. {b} flew away. how many remain", f"{a} - {b}"),
        ]
    inp, expr = rng.choice(templates)
    return Ext2Problem("past_narr2", inp, expr)


# 14. alt-let: `let a be A, let b be B, find a op b`

def _gen_alt_let(rng: random.Random) -> Ext2Problem:
    a = rng.randint(1, 999)
    b = rng.randint(1, 999)
    op = rng.choice(_OPS)
    templates = [
        (f"let a be {a}, let b be {b}, find a{op}b", f"{a} {op} {b}"),
        (f"set x to {a}, set y to {b}, compute x{op}y", f"{a} {op} {b}"),
        (f"define m={a}, n={b}, evaluate m{op}n", f"{a} {op} {b}"),
    ]
    inp, expr = rng.choice(templates)
    return Ext2Problem("alt_let", inp, expr)


# 15. eq-var: `what is the result of A op B`, `compute A op B`, `evaluate A op B`

def _gen_eq_var(rng: random.Random) -> Ext2Problem:
    a = rng.randint(1, 999)
    b = rng.randint(1, 999)
    op = rng.choice(_OPS)
    verb = rng.choice(["what is the result of", "compute", "evaluate"])
    inp = f"{verb} {a} {op} {b}"
    return Ext2Problem("eq_var", inp, f"{a} {op} {b}")


# 16. three-operand: `A + B + C` or `A * B - C`

def _gen_three_operand(rng: random.Random) -> Ext2Problem:
    a = rng.randint(1, 99)
    b = rng.randint(1, 99)
    c = rng.randint(1, 99)
    op1 = rng.choice(_OPS)
    op2 = rng.choice(_OPS)
    inp = f"{a} {op1} {b} {op2} {c}"
    return Ext2Problem("three_op", inp, f"{a} {op1} {b} {op2} {c}")


# 17. possessive-of: `the sum of A and B`, `the product of A and B`

def _gen_possessive(rng: random.Random) -> Ext2Problem:
    a = rng.randint(1, 999)
    b = rng.randint(1, 999)
    op = rng.choice(_OPS)
    noun = {"+": "sum", "-": "difference", "*": "product"}[op]
    inp = f"the {noun} of {a} and {b}"
    return Ext2Problem("possessive", inp, f"{a} {op} {b}")


# 18. verb-by: `multiply A by B`, `subtract A from B`

def _gen_verb_by(rng: random.Random) -> Ext2Problem:
    a = rng.randint(1, 999)
    b = rng.randint(1, 999)
    op = rng.choice(["+", "-", "*"])
    if op == "+":
        inp = f"add {a} to {b}"
        expr = f"{b} + {a}"
    elif op == "-":
        inp = f"subtract {a} from {b}"
        expr = f"{b} - {a}"
    else:
        inp = f"multiply {a} by {b}"
        expr = f"{a} * {b}"
    return Ext2Problem("verb_by", inp, expr)


# 19. question-first: `what's A times B`, `how much is A plus B`

def _gen_question_first(rng: random.Random) -> Ext2Problem:
    a = rng.randint(1, 999)
    b = rng.randint(1, 999)
    op = rng.choice(_OPS)
    op_word = _OP_NAMES[op]
    frames = [
        f"what's {a} {op_word} {b}",
        f"how much is {a} {op_word} {b}",
        f"tell me {a} {op_word} {b}",
    ]
    inp = rng.choice(frames)
    return Ext2Problem("question_first", inp, f"{a} {op} {b}")


# 20. when-then: `when a is A and b is B, a op b`

def _gen_when_then(rng: random.Random) -> Ext2Problem:
    a = rng.randint(1, 999)
    b = rng.randint(1, 999)
    op = rng.choice(_OPS)
    frames = [
        f"when a is {a} and b is {b}, a{op}b",
        f"given that x = {a} and y = {b}, x {op} y",
        f"for a={a} and b={b}, compute a {op} b",
    ]
    inp = rng.choice(frames)
    return Ext2Problem("when_then", inp, f"{a} {op} {b}")


_GENERATORS = {
    "fn_call": _gen_fn_call,
    "phrasal": _gen_phrasal,
    "past_narr2": _gen_past_narr,
    "alt_let": _gen_alt_let,
    "eq_var": _gen_eq_var,
    "three_op": _gen_three_operand,
    "possessive": _gen_possessive,
    "verb_by": _gen_verb_by,
    "question_first": _gen_question_first,
    "when_then": _gen_when_then,
}


class Extended2FormatGenerator:
    """Generates balanced samples across the 10 NEW formats."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def generate(self, n: int = 10000) -> List[Ext2Problem]:
        per = n // 10
        out: List[Ext2Problem] = []
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
                out.append(p)
                made += 1
        self._rng.shuffle(out)
        return out
