"""Exhaustive test for the compiled opcode dispatcher."""

from __future__ import annotations

import itertools
import math

import torch

from calm.llm_computer.programs.dispatched import (
    FACT_MAX_N, FACT_SLOT_BASE,
    GCD_BASE,
    PRIME_MAX_N, PRIME_MIN_N, PRIME_SLOT_FALSE, PRIME_SLOT_TRUE,
    build_dispatched, decode_output, run_program,
)
from calm.llm_computer.programs.is_prime import _is_prime


def test_dispatched_gcd_exhaustive():
    model = build_dispatched()
    pairs = list(itertools.product(range(GCD_BASE), repeat=2))
    x = torch.tensor([(a, b, 0) for a, b in pairs], dtype=torch.long)
    with torch.no_grad():
        preds = model(x)[:, 2, :].argmax(dim=-1).tolist()
    expected = [math.gcd(a, b) for a, b in pairs]
    failures = [(a, b, p, e) for (a, b), p, e in zip(pairs, preds, expected) if p != e]
    assert not failures, f"{len(failures)} gcd failures, first: {failures[0]}"


def test_dispatched_factorial_exhaustive():
    model = build_dispatched()
    x = torch.tensor([(n, 0, 1) for n in range(FACT_MAX_N + 1)], dtype=torch.long)
    with torch.no_grad():
        preds = model(x)[:, 2, :].argmax(dim=-1).tolist()
    for n, slot in enumerate(preds):
        assert slot == FACT_SLOT_BASE + n, (n, slot)
        assert decode_output(1, slot) == math.factorial(n)


def test_dispatched_is_prime_exhaustive():
    model = build_dispatched()
    x = torch.tensor([(n, 0, 2) for n in range(PRIME_MIN_N, PRIME_MAX_N + 1)],
                     dtype=torch.long)
    with torch.no_grad():
        preds = model(x)[:, 2, :].argmax(dim=-1).tolist()
    for n, slot in zip(range(PRIME_MIN_N, PRIME_MAX_N + 1), preds):
        expected_slot = PRIME_SLOT_TRUE if _is_prime(n) else PRIME_SLOT_FALSE
        assert slot == expected_slot, (n, slot, expected_slot)


def test_dispatched_mixed_batch():
    """One batch, all three opcodes interleaved — no cross-contamination."""
    model = build_dispatched()
    cases = [
        (0, 12, 15, math.gcd(12, 15)),
        (1, 5, 0, math.factorial(5)),
        (2, 7, 0, True),
        (0, 7, 13, math.gcd(7, 13)),
        (1, 0, 0, math.factorial(0)),
        (2, 9, 0, False),
        (0, 15, 15, 15),
        (1, 8, 0, math.factorial(8)),
        (2, 2, 0, True),
    ]
    for opcode, a, b, expected in cases:
        got = run_program(model, opcode, a, b)
        assert got == expected, (opcode, a, b, got, expected)
