"""2-digit adder correctness.

Exhaustive: all 10,000 (a, b) pairs with a, b in [0, 99] produce the
correct sum via greedy decoding at output position 1. Scaling proof —
the same IR + compiler that handles 4 tiny primitives + 1-digit adder
handles this 10K-case program with no compiler change.
"""

from __future__ import annotations

import itertools

import torch

from calm.llm_computer.programs.adder import MAX_OPERAND, build_adder


def test_adder_exhaustive():
    model = build_adder()
    pairs = list(itertools.product(range(MAX_OPERAND + 1), repeat=2))
    inputs = torch.tensor(pairs, dtype=torch.long)
    with torch.no_grad():
        preds = model(inputs)[:, 1, :].argmax(dim=-1).tolist()
    wrong = [(a, b, a + b, p) for (a, b), p in zip(pairs, preds) if p != a + b]
    assert not wrong, f"adder mismatches: {wrong[:5]}... ({len(wrong)} total)"


def test_adder_param_count():
    """Lock the param count so regressions shift are visible."""
    model = build_adder()
    assert model.param_count() == 486_012


if __name__ == "__main__":
    test_adder_exhaustive()
    print("[ok] adder exhaustive 10,000/10,000")
    test_adder_param_count()
    print("[ok] adder param count locked at 486,012")
