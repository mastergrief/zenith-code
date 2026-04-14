"""Exhaustive test for the compiled gcd program."""

from __future__ import annotations

import itertools
import math

import torch

from calm.llm_computer.programs.gcd import BASE, build_gcd


def test_gcd_exhaustive():
    model = build_gcd()
    pairs = list(itertools.product(range(BASE), repeat=2))
    inputs = torch.tensor(pairs, dtype=torch.long)
    with torch.no_grad():
        logits = model(inputs)
        preds = logits[:, 1, :].argmax(dim=-1).tolist()
    expected = [math.gcd(a, b) for a, b in pairs]
    failures = [(a, b, p, e) for (a, b), p, e in zip(pairs, preds, expected) if p != e]
    assert not failures, f"{len(failures)} failures, first: {failures[0]}"


def test_gcd_fits_param_budget():
    model = build_gcd()
    # Under the 2M-per-program ceiling set by the plan.
    assert model.param_count() < 2_000_000, model.param_count()
