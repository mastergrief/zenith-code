"""Tests for card grammars as generation constraints."""

from __future__ import annotations

import pytest
import torch

from calm.llm_computer.card_grammar import (
    AdderOutputGrammar, BooleanGrammar, CardGrammar, GrammarComposition,
    GrammarContext, VocabRangeGrammar, apply_grammar,
)


def test_adder_grammar_applies_at_position_1():
    g = AdderOutputGrammar()
    ctx0 = GrammarContext(position=0, prompt=(1, 2))
    ctx1 = GrammarContext(position=1, prompt=(1, 2))
    assert not g.applies(ctx0)
    assert g.applies(ctx1)
    assert g.valid_tokens(ctx1) == frozenset(range(7))


def test_adder_grammar_silent_at_other_positions():
    g = AdderOutputGrammar()
    for pos in [0, 2, 3, 4]:
        ctx = GrammarContext(position=pos, prompt=(0, 0))
        assert g.valid_tokens(ctx) == frozenset()


def test_apply_grammar_hard_mode_zeros_invalid():
    g = AdderOutputGrammar(max_sum=6, output_position=1)
    logits = torch.tensor([0.5, 1.0, 0.3, 0.8, 0.2, 0.1, 0.4, 2.0, 0.7])  # 9 tokens
    ctx = GrammarContext(position=1, prompt=(0, 0))
    out = apply_grammar(logits, g, ctx, mode="hard")
    # Token 7 and 8 should be -inf (adder max is 6)
    assert out[7] == float("-inf")
    assert out[8] == float("-inf")
    # Tokens 0..6 unchanged
    for t in range(7):
        assert out[t] == logits[t]


def test_apply_grammar_soft_mode_penalizes():
    g = AdderOutputGrammar()
    logits = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    ctx = GrammarContext(position=1, prompt=(0, 0))
    out = apply_grammar(logits, g, ctx, mode="soft", soft_penalty=4.0)
    assert out[0] == 0.5  # valid
    assert out[7] == 0.5 - 4.0  # invalid
    assert out[8] == 0.5 - 4.0


def test_apply_grammar_silent_when_not_applicable():
    g = AdderOutputGrammar()
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    ctx = GrammarContext(position=0, prompt=(0, 0))  # wrong position
    out = apply_grammar(logits, g, ctx)
    assert torch.equal(out, logits)


def test_vocab_range_grammar():
    g = VocabRangeGrammar(lo=8, hi=12, output_position=1)
    ctx = GrammarContext(position=1, prompt=(0, 0))
    assert g.applies(ctx)
    assert g.valid_tokens(ctx) == frozenset({8, 9, 10, 11})


def test_boolean_grammar():
    g = BooleanGrammar(output_position=2)
    ctx = GrammarContext(position=2, prompt=(0, 0))
    assert g.applies(ctx)
    assert g.valid_tokens(ctx) == frozenset({0, 1})
    ctx_other = GrammarContext(position=0, prompt=(0, 0))
    assert not g.applies(ctx_other)


def test_composition_intersects_applicable_grammars():
    """At position 1: adder says [0..6], boolean says {0, 1}. Intersection = {0, 1}."""
    adder = AdderOutputGrammar(max_sum=6, output_position=1)
    boolean = BooleanGrammar(output_position=1)
    composed = GrammarComposition([adder, boolean])
    ctx = GrammarContext(position=1, prompt=(0, 0))
    assert composed.applies(ctx)
    assert composed.valid_tokens(ctx) == frozenset({0, 1})


def test_composition_ignores_non_applicable():
    adder = AdderOutputGrammar(output_position=1)
    bool_at_2 = BooleanGrammar(output_position=2)
    composed = GrammarComposition([adder, bool_at_2])
    ctx = GrammarContext(position=1, prompt=(0, 0))
    assert composed.applies(ctx)
    # Only adder applies at position 1 → constraint = adder's
    assert composed.valid_tokens(ctx) == frozenset(range(7))


def test_composition_empty_when_nothing_applies():
    g = GrammarComposition([AdderOutputGrammar(output_position=1)])
    ctx = GrammarContext(position=0, prompt=(0, 0))
    assert not g.applies(ctx)
    assert g.valid_tokens(ctx) == frozenset()


def test_apply_grammar_batched():
    g = AdderOutputGrammar()
    logits = torch.zeros(3, 9)  # batch=3
    ctx = GrammarContext(position=1, prompt=(0, 0))
    out = apply_grammar(logits, g, ctx, mode="hard")
    assert out.shape == (3, 9)
    assert (out[:, 7] == float("-inf")).all()
    assert (out[:, 8] == float("-inf")).all()


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode must be"):
        apply_grammar(
            torch.zeros(9), AdderOutputGrammar(),
            GrammarContext(position=1, prompt=(0, 0)),
            mode="what",
        )


if __name__ == "__main__":
    test_adder_grammar_applies_at_position_1()
    print("[ok] adder grammar applies at position 1")
    test_adder_grammar_silent_at_other_positions()
    print("[ok] adder silent at non-output positions")
    test_apply_grammar_hard_mode_zeros_invalid()
    print("[ok] hard mode sends invalid to -inf")
    test_apply_grammar_soft_mode_penalizes()
    print("[ok] soft mode subtracts penalty")
    test_apply_grammar_silent_when_not_applicable()
    print("[ok] silent when grammar doesn't apply")
    test_vocab_range_grammar()
    print("[ok] vocab range grammar")
    test_boolean_grammar()
    print("[ok] boolean grammar")
    test_composition_intersects_applicable_grammars()
    print("[ok] composition intersects valid sets")
    test_composition_ignores_non_applicable()
    print("[ok] composition ignores silent grammars")
    test_composition_empty_when_nothing_applies()
    print("[ok] composition empty when all silent")
    test_apply_grammar_batched()
    print("[ok] apply_grammar batched")
    test_invalid_mode_raises()
    print("[ok] invalid mode raises")
