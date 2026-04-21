"""Multi-step variable-chain reasoning data.

Round 6b task: chain assignments with variable references. Tests whether
the model can resolve nested variable references and emit the flattened
expression.

Format:
  chain_1:  'a = 2 + 3 ; a'         → '2 + 3'
  chain_2:  'a = 2 + 3 ; b = a * 4 ; b'   → '( 2 + 3 ) * 4'
  chain_3:  'a = 2 + 3 ; b = a * 4 ; c = b - 1 ; c'
            → '( ( 2 + 3 ) * 4 ) - 1'

This is STRUCTURE ONLY (substrate computes the values via safe_eval);
the model's task is to expand each variable reference to its
definition, producing a flattened expression. Tests working-memory /
associative-recall of variable bindings across positions.

Plain PT vs PT+DeltaNet comparison: the hypothesis is that at
chain_length ≥ 3, PT+Delta's recurrent state helps track bindings
better than plain PT's softmax lookback.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

from calm.hrm.data import _CHAR_TO_ID


@dataclass
class ChainProblem:
    """NL chain question with flattened expression target + computed answer."""
    question: str
    expression: str
    answer: int
    chain_length: int


_VAR_NAMES = "abcde"
_OPS = ["+", "-", "*"]


def _eval_expr(expr: str) -> int:
    """Evaluate a flattened expression via safe Python eval on integers."""
    # Restrict to digits, ops, parens, spaces; then eval.
    allowed = set("0123456789+-* ()")
    if not all(c in allowed for c in expr):
        raise ValueError(f"unexpected char in {expr!r}")
    return eval(expr)


def _gen_one(n_steps: int, rng: random.Random, max_op: int = 9) -> ChainProblem:
    """Generate one chain problem with exactly n_steps assignments."""
    if n_steps < 1:
        raise ValueError("n_steps >= 1")
    if n_steps > len(_VAR_NAMES):
        raise ValueError(f"n_steps > {len(_VAR_NAMES)}")

    vars_used = list(_VAR_NAMES[:n_steps])
    assignments: List[Tuple[str, str]] = []
    for i in range(n_steps):
        if i == 0:
            A = rng.randint(1, max_op)
            B = rng.randint(1, max_op)
            op = rng.choice(_OPS)
            expr = f"{A} {op} {B}"
        else:
            prev_var = vars_used[i - 1]
            B = rng.randint(1, max_op)
            op = rng.choice(_OPS)
            if rng.random() < 0.5:
                expr = f"{prev_var} {op} {B}"
            else:
                expr = f"{B} {op} {prev_var}"
        assignments.append((vars_used[i], expr))

    final_var = vars_used[-1]

    # NL question: semicolon-separated assignments + final reference.
    q_parts = [f"{v} = {e}" for v, e in assignments]
    q_parts.append(final_var)
    question = " ; ".join(q_parts)

    # Flattened expression by recursive substitution.
    assignment_map = dict(assignments)

    def expand(tok: str) -> str:
        if tok not in assignment_map:
            return tok
        parts = assignment_map[tok].split()
        out = []
        for p in parts:
            if p in assignment_map:
                out.append(f"( {expand(p)} )")
            else:
                out.append(p)
        return " ".join(out)

    flat = expand(final_var)
    # Parenthesize top-level when chain length > 1 (makes eval unambiguous).
    if n_steps > 1:
        # Outer doesn't NEED parens, but for consistency with nested form,
        # leave as-is. safe_eval handles it.
        pass
    answer = _eval_expr(flat)
    return ChainProblem(question=question, expression=flat,
                        answer=answer, chain_length=n_steps)


class ChainDataGenerator:
    """Generates ChainProblems at mixed chain lengths.

    Args:
      lengths: which chain lengths to generate (e.g. [1, 2, 3]).
      seed:    RNG seed.
      max_op:  max literal operand value (default 9 — single digit).
    """
    def __init__(self, lengths: List[int] = (1, 2, 3), seed: int = 42,
                 max_op: int = 9):
        self.lengths = list(lengths)
        self.rng = random.Random(seed)
        self.max_op = max_op

    def generate(self, n: int) -> List[ChainProblem]:
        """Generate n problems, uniformly distributed across chain lengths."""
        probs: List[ChainProblem] = []
        for i in range(n):
            L = self.lengths[i % len(self.lengths)]
            probs.append(_gen_one(L, self.rng, self.max_op))
        self.rng.shuffle(probs)
        # Filter: ensure every char is in the tokenizer vocab.
        vocab_chars = set(_CHAR_TO_ID.keys())
        return [p for p in probs
                if all(c in vocab_chars for c in p.question)
                and all(c in vocab_chars for c in p.expression)]


def filter_by_length(probs: List[ChainProblem], n_steps: int) -> List[ChainProblem]:
    """Slice problems matching a specific chain length."""
    return [p for p in probs if p.chain_length == n_steps]


if __name__ == "__main__":
    gen = ChainDataGenerator(lengths=[1, 2, 3], seed=0)
    probs = gen.generate(15)
    for p in probs:
        print(f"[L{p.chain_length}] Q: {p.question}")
        print(f"        E: {p.expression}  = {p.answer}")
