"""Tiny-ISA tests — Stage 3 of UTM path.

Covers the 2-opcode (INC, DEC) accumulator machine compiled on the
Small2DTransformer substrate and driven autoregressively.
"""

from __future__ import annotations

import torch

from calm.llm_computer.programs.isa import (
    DEC, HALT, INC, V_MAX, VOCAB,
    build_isa, run_isa, simulate_expected,
)


def test_isa_compiles_small():
    model = build_isa()
    assert model.param_count() < 20_000, model.param_count()


def test_isa_inc_exhaustive():
    model = build_isa()
    for v in range(V_MAX + 1):
        seq = run_isa(INC, v, model=model)
        assert seq == simulate_expected(INC, v), (v, seq)


def test_isa_dec_exhaustive():
    model = build_isa()
    for v in range(V_MAX + 1):
        seq = run_isa(DEC, v, model=model)
        assert seq == simulate_expected(DEC, v), (v, seq)


def test_isa_halts_at_boundary():
    model = build_isa()
    seq_inc = run_isa(INC, V_MAX, model=model)
    assert seq_inc[-1] == HALT
    seq_dec = run_isa(DEC, 0, model=model)
    assert seq_dec[-1] == HALT


def test_isa_single_step_local():
    """Single forward pass on a partial sequence must predict correctly."""
    model = build_isa()
    # [INC, 3, 4, 5]  — last token is 5, next should be 6 under INC
    x = torch.tensor([[INC, 3, 4, 5]], dtype=torch.long)
    with torch.no_grad():
        pred = int(model(x)[0, -1, :].argmax().item())
    assert pred == 6, pred

    # [DEC, 4, 3, 2]  — next under DEC should be 1
    x = torch.tensor([[DEC, 4, 3, 2]], dtype=torch.long)
    with torch.no_grad():
        pred = int(model(x)[0, -1, :].argmax().item())
    assert pred == 1, pred


def test_isa_halts_gracefully_at_max_steps():
    """If for some reason HALT is never emitted within max_steps, the
    runner returns a truncated sequence rather than spinning forever."""
    model = build_isa()
    # DEC starting from V_MAX takes V_MAX+1 steps + HALT + start opcode
    seq = run_isa(DEC, V_MAX, max_steps=5, model=model)
    # max_steps=5 means we generate at most 5 new tokens after input
    # [DEC, V_MAX] = 2 tokens + 5 = 7 tokens max
    assert len(seq) <= 7
