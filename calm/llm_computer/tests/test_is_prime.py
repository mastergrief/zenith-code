"""Exhaustive test for the compiled is_prime program."""

from __future__ import annotations

import torch

from calm.llm_computer.programs.is_prime import (
    MAX_N, MIN_N, _is_prime, build_is_prime,
)


def test_is_prime_exhaustive():
    model = build_is_prime()
    inputs = torch.tensor([[n] for n in range(MIN_N, MAX_N + 1)], dtype=torch.long)
    with torch.no_grad():
        logits = model(inputs)
        preds = logits[:, 0, :].argmax(dim=-1).tolist()
    expected = [1 if _is_prime(n) else 0 for n in range(MIN_N, MAX_N + 1)]
    failures = [(n, p, e) for n, p, e in zip(range(MIN_N, MAX_N + 1), preds, expected) if p != e]
    assert not failures, f"failures: {failures}"


def test_is_prime_fits_param_budget():
    model = build_is_prime()
    assert model.param_count() < 2_000_000, model.param_count()
