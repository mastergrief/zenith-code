"""Compiled cards as generation grammars — hard constraints during decoding.

Currently: compiled cards produce correct outputs but don't STOP the
model from emitting tokens that violate their invariants. Example: the
compiled adder knows `a+b ∈ [0, 6]` for (a, b) ∈ [0,3]². But the
trained LM head rows 0-6 could still emit token 7 as "the sum" with
positive probability.

This module wraps a compiled card's output range into a logit mask
that can be applied at generation time. Invalid tokens get -inf
logits before softmax; they can't be sampled.

Two modes:
  - Hard grammar: logits outside the valid range set to -inf
  - Soft grammar: logits outside the valid range get a fixed penalty
    (callable can tune gentleness during training, harden at inference)

MVP ships:
  - `CardGrammar` abstract: defines `valid_tokens(context)` returning
    a set of allowed token IDs given the input context.
  - `AdderOutputGrammar`: adder always emits sum ∈ [0, 6]; returns
    {0..6} as valid if we're decoding at position 1 (adder's output).
  - `apply_grammar(logits, grammar, context)` zeros invalid logit mass.

This is a COMPOSITIONAL SAFETY primitive. A multi-card unified model
can combine grammars: "the adder-output-at-position-1 must be in
[0, 6] AND the is-prime-at-position-2 must be 0 or 1".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Optional

import torch


@dataclass
class GrammarContext:
    """Metadata a grammar can consult to decide which tokens are valid.

    Attributes:
        position: which sequence position we're generating (0-indexed).
        prompt: the input tokens seen so far.
        card_outputs: dict of card_name → set of observed/valid output
            tokens, populated by upstream cards.
    """
    position: int
    prompt: tuple[int, ...]
    card_outputs: dict[str, frozenset[int]] = None  # type: ignore

    def __post_init__(self):
        if self.card_outputs is None:
            self.card_outputs = {}


class CardGrammar(ABC):
    """Abstract: given a context, return the set of valid token IDs."""

    @abstractmethod
    def valid_tokens(self, context: GrammarContext) -> frozenset[int]:
        """Return allowed tokens for this context. Empty set means no
        constraint from this grammar (caller should ignore)."""
        ...

    @abstractmethod
    def applies(self, context: GrammarContext) -> bool:
        """True if this grammar has an opinion about the context."""
        ...


class AdderOutputGrammar(CardGrammar):
    """Adder emits a+b ∈ [0, max_sum] at a specific output position.

    For the compiled adder_tiny where (a, b) ∈ [0, 3]², max_sum=6 and
    output_position=1. Only tokens {0, 1, 2, 3, 4, 5, 6} are valid
    predictions at position 1.

    If the model emits at position ≠ 1, this grammar is silent (returns
    `applies=False`) so it doesn't constrain other outputs.
    """

    def __init__(self, max_sum: int = 6, output_position: int = 1):
        self.max_sum = max_sum
        self.output_position = output_position
        self._valid = frozenset(range(max_sum + 1))

    def valid_tokens(self, context: GrammarContext) -> frozenset[int]:
        if not self.applies(context):
            return frozenset()
        return self._valid

    def applies(self, context: GrammarContext) -> bool:
        return context.position == self.output_position


class VocabRangeGrammar(CardGrammar):
    """Generic: emits only tokens in [lo, hi) at a given position."""

    def __init__(self, lo: int, hi: int, output_position: int):
        self.lo = lo
        self.hi = hi
        self.output_position = output_position
        self._valid = frozenset(range(lo, hi))

    def valid_tokens(self, context: GrammarContext) -> frozenset[int]:
        if not self.applies(context):
            return frozenset()
        return self._valid

    def applies(self, context: GrammarContext) -> bool:
        return context.position == self.output_position


class BooleanGrammar(CardGrammar):
    """Emits 0 or 1 at a given output position — for boolean card
    outputs (is_prime, is_even, is_positive, etc.)."""

    def __init__(self, output_position: int, false_token: int = 0,
                 true_token: int = 1):
        self.output_position = output_position
        self._valid = frozenset({false_token, true_token})

    def valid_tokens(self, context: GrammarContext) -> frozenset[int]:
        if not self.applies(context):
            return frozenset()
        return self._valid

    def applies(self, context: GrammarContext) -> bool:
        return context.position == self.output_position


class GrammarComposition(CardGrammar):
    """Intersect multiple grammars: a token is valid iff EVERY grammar
    that applies to the context accepts it.

    This lets us combine card constraints: at position 1, adder says
    "∈ [0,6]" AND is_prime says "∈ {0, 1}". Intersection gives {0, 1}
    (because {0, 1} is the subset of [0,6] that both accept).
    """

    def __init__(self, grammars: Iterable[CardGrammar]):
        self._grammars = list(grammars)

    def valid_tokens(self, context: GrammarContext) -> frozenset[int]:
        applicable = [g for g in self._grammars if g.applies(context)]
        if not applicable:
            return frozenset()
        valid = applicable[0].valid_tokens(context)
        for g in applicable[1:]:
            valid = valid & g.valid_tokens(context)
        return valid

    def applies(self, context: GrammarContext) -> bool:
        return any(g.applies(context) for g in self._grammars)


def apply_grammar(
    logits: torch.Tensor,
    grammar: CardGrammar,
    context: GrammarContext,
    mode: str = "hard",
    soft_penalty: float = 4.0,
) -> torch.Tensor:
    """Apply grammar constraints to logits.

    Args:
        logits: (vocab_size,) or (B, vocab_size) tensor.
        grammar: CardGrammar with opinion about the context.
        context: the generation context.
        mode: "hard" (set invalid to -inf) or "soft" (subtract penalty).
        soft_penalty: only used if mode="soft". Subtracted from invalid
            logits so they become less likely but not impossible.

    Returns:
        New tensor with grammar applied. If grammar.applies=False, logits
        are returned unchanged.
    """
    if not grammar.applies(context):
        return logits
    valid = grammar.valid_tokens(context)
    if not valid:
        return logits  # no constraint
    vocab_size = logits.size(-1)
    # Build mask: True where token is valid
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=logits.device)
    for t in valid:
        if 0 <= t < vocab_size:
            mask[t] = True
    # Invert for penalty
    invalid = ~mask
    result = logits.clone()
    if mode == "hard":
        result = result.masked_fill(invalid, float("-inf"))
    elif mode == "soft":
        result = result - soft_penalty * invalid.to(result.dtype)
    else:
        raise ValueError(f"mode must be 'hard' or 'soft', got {mode!r}")
    return result
