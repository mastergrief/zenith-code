"""Exhaustive test for the compiled factorial program."""

from __future__ import annotations

import math

import torch

from calm.llm_computer.programs.factorial import MAX_N, build_factorial


def test_factorial_exhaustive():
    model = build_factorial()
    inputs = torch.tensor([[n] for n in range(MAX_N + 1)], dtype=torch.long)
    with torch.no_grad():
        logits = model(inputs)
        preds = logits[:, 0, :].argmax(dim=-1).tolist()
    expected = [math.factorial(n) for n in range(MAX_N + 1)]
    failures = [(n, p, e) for n, p, e in zip(range(MAX_N + 1), preds, expected) if p != e]
    assert not failures, f"failures: {failures}"


def test_factorial_fits_param_budget():
    model = build_factorial()
    assert model.param_count() < 2_000_000, model.param_count()
