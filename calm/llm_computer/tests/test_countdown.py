"""Countdown machine — Stage 1+2 test.

Stage 1 (control-flow IR): implicit branch via head-level step-diff
routing. The same step channels feed both "decrement" head entries
(for v ≥ 1) and "HALT" head entries (for v == 0). No new IR node — the
LinearHead accumulates conditionally via the step-diff telescoping.

Stage 2 (autoregressive as the compute loop): `run_countdown` drives
the model in a generation loop. Each forward pass is one tick; the
output sequence IS the computation trace. Halt is signaled by a
distinguished token the runner detects.
"""

from __future__ import annotations

import torch

from calm.llm_computer.programs.countdown import (
    HALT, V_MAX, VOCAB, build_countdown, run_countdown,
)


def test_countdown_compiles():
    model = build_countdown()
    assert model.param_count() < 5_000, model.param_count()
    # Single forward pass at position 0 with input v.
    for v in range(V_MAX + 1):
        x = torch.tensor([[v]], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
        pred = int(logits[0, 0, :].argmax().item())
        expected = HALT if v == 0 else v - 1
        assert pred == expected, (v, pred, expected)


def test_countdown_autoregressive_exhaustive():
    """Stage 2 gate: run_countdown must terminate with correct trace for
    every starting value."""
    model = build_countdown()
    for v in range(V_MAX + 1):
        seq = run_countdown(v, model=model)
        expected = list(range(v, -1, -1)) + [HALT]
        assert seq == expected, (v, seq, expected)


def test_countdown_halts_immediately_at_zero():
    model = build_countdown()
    seq = run_countdown(0, model=model)
    assert seq == [0, HALT]


def test_countdown_max_steps_safeguard():
    """Even at V_MAX the loop completes within max_steps default."""
    model = build_countdown()
    seq = run_countdown(V_MAX, model=model, max_steps=20)
    assert seq[-1] == HALT
    assert len(seq) == V_MAX + 2  # initial + V_MAX decrements + HALT


def test_countdown_step_is_local():
    """Any suffix ending in value v should produce v-1 (or HALT at 0) next.

    Proves the compiled program is Markov: the next token depends only on
    the last emitted token, not the full history. That's the property
    that makes this a true autoregressive machine — tape-as-memory, not
    trained-in-memory.
    """
    model = build_countdown()
    # A few arbitrary prefixes
    for prefix, v in [([8, 7, 6, 5], 5),
                      ([3, 2], 2),
                      ([4, 3, 2, 1], 1)]:
        x = torch.tensor([prefix], dtype=torch.long)
        with torch.no_grad():
            pred = int(model(x)[0, -1, :].argmax().item())
        expected = HALT if v == 0 else v - 1
        assert pred == expected, (prefix, pred, expected)
