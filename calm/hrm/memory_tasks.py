"""Working-memory stress-test tasks: MQAR, reassignment, scratchpad.

Three task generators designed to stress PT vs PT+DeltaNet where the
R5-R9 arc predicts DeltaNet's advantage should manifest.

Unlike calm/hrm/chain_data.py (L=1-5 sequential assignments that plain
PT handles trivially), these tasks push the memory-capacity axis:

  MQAR       — N associative recall pairs, query one by key
  Reassign   — variable reassignment with interleaved operations
  Scratchpad — multi-step with intermediate-value emission

Same token vocabulary as calm/hrm/data.py (_CHAR_TO_ID, VOCAB_SIZE=82).
Compatible with SeqDataset (each problem has .question, .expression,
.answer attributes).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

from calm.hrm.data import _CHAR_TO_ID


@dataclass
class MemProblem:
    """Unified problem format for all three task types."""
    question: str
    expression: str
    answer: str
    task_kind: str         # "mqar" | "reassign" | "scratchpad"
    difficulty: int        # task-specific difficulty parameter (N, chain length, etc.)


def _valid_chars(s: str) -> bool:
    """All chars in s must be in the tokenizer vocab."""
    return all(c in _CHAR_TO_ID for c in s)


# -----------------------------------------------------------------------------
# TASK 1: MQAR (Multi-Query Associative Recall)
#
# Format:  k1 v1 k2 v2 ... kN vN ; k_q
# Target:  v_q  (the value paired with k_q, copied from prefix)
#
# This is the canonical DeltaNet paper test. Plain PT should handle it
# via content-based softmax copy pointer. PT+Delta should handle it via
# state-based lookup. At high N, the question is which scales.
# -----------------------------------------------------------------------------

def _gen_mqar(n_pairs: int, rng: random.Random,
              key_pool: List[str], val_pool: List[str]) -> MemProblem:
    """One MQAR problem: n_pairs KV bindings, query one."""
    keys = rng.sample(key_pool, n_pairs)
    values = [rng.choice(val_pool) for _ in range(n_pairs)]
    q_idx = rng.randrange(n_pairs)
    query_key = keys[q_idx]
    target_val = values[q_idx]

    parts = []
    for k, v in zip(keys, values):
        parts.append(f"{k} {v}")
    question = " ".join(parts) + f" ; {query_key}"
    # Expression is just the target value (copied from prefix).
    return MemProblem(
        question=question, expression=target_val, answer=target_val,
        task_kind="mqar", difficulty=n_pairs,
    )


# Small letter keys (a-z), digit values (0-9) — fits vocab cleanly.
_MQAR_KEY_POOL = list("abcdefghijklmnopqrstuvwxyz")
_MQAR_VAL_POOL = [str(d) for d in range(10)]


def gen_mqar_batch(n_pairs: int, count: int, seed: int) -> List[MemProblem]:
    """Generate `count` MQAR problems at fixed n_pairs."""
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        out.append(_gen_mqar(n_pairs, rng, _MQAR_KEY_POOL, _MQAR_VAL_POOL))
    return [p for p in out if _valid_chars(p.question) and _valid_chars(p.expression)]


# -----------------------------------------------------------------------------
# TASK 2: Reassignment — latest-value semantics
#
# Format:  "x = A ; ... more ops ... ; x = B ; x"
# Target:  B (the LATEST assignment, not A)
#
# Multiple assignments to the same variable. Model must produce the
# most recent binding, not an earlier one. Softmax attention CAN do this
# (attend to later occurrence); DeltaNet's update rule INHERENTLY handles
# it (old binding subtracted when re-written).
# -----------------------------------------------------------------------------

_VARS = list("abcde")


def _gen_reassign(n_reassigns: int, rng: random.Random,
                  max_op: int = 9) -> MemProblem:
    """Variable reassigned n_reassigns times; query picks one reassigned var."""
    target_var = rng.choice(_VARS)
    # Track each var's current value (for computing ground truth).
    values = {}
    parts = []
    for i in range(n_reassigns):
        # Each step either reassigns target_var or assigns/reassigns another var.
        if i == n_reassigns - 1 or rng.random() < 0.4:
            # Reassign target_var.
            val = rng.randint(1, max_op)
            parts.append(f"{target_var} = {val}")
            values[target_var] = val
        else:
            other = rng.choice([v for v in _VARS if v != target_var])
            val = rng.randint(1, max_op)
            parts.append(f"{other} = {val}")
            values[other] = val

    parts.append(target_var)
    question = " ; ".join(parts)
    # Target = latest value of target_var.
    target_val = values[target_var]
    expression = str(target_val)

    return MemProblem(
        question=question, expression=expression, answer=str(target_val),
        task_kind="reassign", difficulty=n_reassigns,
    )


def gen_reassign_batch(n_reassigns: int, count: int, seed: int) -> List[MemProblem]:
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        p = _gen_reassign(n_reassigns, rng)
        if _valid_chars(p.question) and _valid_chars(p.expression):
            out.append(p)
    return out


# -----------------------------------------------------------------------------
# TASK 3: Scratchpad arithmetic with intermediate-value references
#
# Format: "compute ( A op B ) op C"
# Target: stepwise:  "A op B = R1 ; R1 op C = R2"
#
# The stepwise emission forces the model to produce intermediate values
# and reference them in the next step. Plain PT must re-derive R1 from
# input context; PT+Delta can store R1 in state and retrieve it.
# -----------------------------------------------------------------------------

_OPS = ["+", "-", "*"]


def _gen_scratchpad(depth: int, rng: random.Random, max_op: int = 9) -> MemProblem:
    """Generate nested expression, target is step-by-step evaluation."""
    # Build a left-nested expression: ((a op b) op c) op d ...
    operands = [rng.randint(1, max_op) for _ in range(depth + 1)]
    ops = [rng.choice(_OPS) for _ in range(depth)]

    # Question: the nested expression
    q_parts = [str(operands[0])]
    for i in range(depth):
        q_parts = ["("] + q_parts + [ops[i], str(operands[i + 1]), ")"]
    question = " ".join(q_parts)

    # Step-by-step evaluation
    steps = []
    running = operands[0]
    for i in range(depth):
        before = running
        op = ops[i]
        right = operands[i + 1]
        if op == "+":
            running = before + right
        elif op == "-":
            running = before - right
        elif op == "*":
            running = before * right
        # Emit this step: "before op right = running"
        steps.append(f"{before} {op} {right} = {running}")
    expression = " ; ".join(steps)

    return MemProblem(
        question=question, expression=expression, answer=str(running),
        task_kind="scratchpad", difficulty=depth,
    )


def gen_scratchpad_batch(depth: int, count: int, seed: int) -> List[MemProblem]:
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        p = _gen_scratchpad(depth, rng)
        if _valid_chars(p.question) and _valid_chars(p.expression):
            out.append(p)
    return out


if __name__ == "__main__":
    print("=== MQAR samples ===")
    for p in gen_mqar_batch(5, 3, seed=0)[:3]:
        print(f"  Q: {p.question}")
        print(f"  E: {p.expression}\n")
    for p in gen_mqar_batch(10, 3, seed=1)[:2]:
        print(f"  [N=10] Q: {p.question}")
        print(f"         E: {p.expression}\n")

    print("\n=== Reassign samples ===")
    for p in gen_reassign_batch(5, 3, seed=0)[:3]:
        print(f"  Q: {p.question}")
        print(f"  E: {p.expression}  (target var latest value)\n")
    for p in gen_reassign_batch(15, 2, seed=1)[:2]:
        print(f"  [K=15] Q: {p.question}")
        print(f"          E: {p.expression}\n")

    print("\n=== Scratchpad samples ===")
    for p in gen_scratchpad_batch(2, 3, seed=0)[:3]:
        print(f"  Q: {p.question}")
        print(f"  E: {p.expression}\n")
    for p in gen_scratchpad_batch(4, 2, seed=1)[:2]:
        print(f"  [D=4] Q: {p.question}")
        print(f"         E: {p.expression}\n")
