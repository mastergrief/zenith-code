"""Tied output head / token embedding — Gemma-style weight sharing.

In Gemma (and most modern LMs) the final `lm_head` is a transposed view of
the token embedding matrix: logits[i] = residual[i] @ tok.weight.T.
This cuts one of the two largest weights (d_model × vocab) in half and
is the convention baked into Gemma GGUF files (they store only
`token_embd.weight` and reuse it at the head).

Our `Small2DTransformer` stores `tok` and `head` as independent Linear
modules. To host Gemma faithfully we need their weights to agree on the
Gemma vocab/channel rectangle. Compiled cards that occupy disjoint
rectangles stay independent — their `head` entries describe logit
coefficients per slot, which is NOT the same as the token's embedding
vector, so we must NOT tie there.

Utilities:
  - `tie_head_to_tok(substrate, tok_range, ch_range)` — copy the
    specified rectangle from `tok.weight` into `head.weight`.
  - `verify_tied(substrate, tok_range, ch_range)` — assert the bytes
    match after tying (for test assertions).
  - `tied_logits(substrate, residual, tok_range)` — reference
    implementation of tied forward for a residual tensor; used in
    tests to compare against `substrate.head(residual)`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from calm.llm_computer.model import Small2DTransformer


Range = Tuple[int, int]


def _resolve(r: Optional[Range], full: int) -> Range:
    if r is None:
        return (0, full)
    lo, hi = r
    assert 0 <= lo < hi <= full, f"invalid range {r} for size {full}"
    return lo, hi


def tie_head_to_tok(
    substrate: Small2DTransformer,
    tok_range: Optional[Range] = None,
    ch_range: Optional[Range] = None,
) -> None:
    """Copy `substrate.tok.weight[tok_range, ch_range]` into
    `substrate.head.weight[tok_range, ch_range]`.

    After this call, logits for vocab slots in `tok_range` computed over
    channels in `ch_range` equal the Gemma-style tied-embedding form
    `residual[ch_range] @ tok.weight[tok_range, ch_range].T`.

    Rectangles outside the tied region are left unchanged — compiled
    card head entries continue to operate on their own slot/channel
    ranges without disturbance.
    """
    tok_lo, tok_hi = _resolve(tok_range, substrate.config.vocab_size)
    ch_lo, ch_hi = _resolve(ch_range, substrate.config.d_model)
    with torch.no_grad():
        substrate.head.weight[tok_lo:tok_hi, ch_lo:ch_hi] = \
            substrate.tok.weight[tok_lo:tok_hi, ch_lo:ch_hi]


def verify_tied(
    substrate: Small2DTransformer,
    tok_range: Optional[Range] = None,
    ch_range: Optional[Range] = None,
) -> bool:
    """True iff `head.weight` agrees bit-for-bit with `tok.weight` on the
    specified rectangle."""
    tok_lo, tok_hi = _resolve(tok_range, substrate.config.vocab_size)
    ch_lo, ch_hi = _resolve(ch_range, substrate.config.d_model)
    return torch.equal(
        substrate.head.weight[tok_lo:tok_hi, ch_lo:ch_hi],
        substrate.tok.weight[tok_lo:tok_hi, ch_lo:ch_hi],
    )


def tied_logits(
    substrate: Small2DTransformer,
    residual: torch.Tensor,
    tok_range: Optional[Range] = None,
) -> torch.Tensor:
    """Reference tied-head logit computation: `residual @ tok.weight.T`
    restricted to `tok_range`. Returns shape (..., tok_hi - tok_lo).

    Used in tests to compare against `substrate.head(residual)[..., tok_range]`
    — they must agree after `tie_head_to_tok()` has been called.
    """
    tok_lo, tok_hi = _resolve(tok_range, substrate.config.vocab_size)
    w = substrate.tok.weight[tok_lo:tok_hi]
    return residual @ w.T
