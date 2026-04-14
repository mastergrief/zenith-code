"""Hand-written `increment_counter` weights.

Program: output `p` at every position `p`, regardless of input token.
Exercises the position embedding — the first program with a non-zero pos
table. Attention + FFN stay zero (they'd be needed for the full cumsum-
via-uniform-keys construction, which requires softmax mode; this
simpler version routes p directly through the position embedding).

Residual layout (d_model = 2 * vocab_size):
  - dims 0..V-1: token embedding (identity one-hot, carried through
    untouched — unused by this program).
  - dims V..2V-1: position embedding. pos[p] is one-hot at dim V+p.

At position p, residual upper half is `e_p` (in the upper-half basis).
Head reads only the upper half: head[j, V+j] = 1 → logits[j] = 1 iff j==p.
Argmax = p.

This is the cleanest expression of "the model's own position is its
answer." Real cumsum programs that aggregate values across positions
(e.g., running counts, parity checks) need softmax-mode uniform-key
attention and a ReGLU multiplier — add in a later program once the
compiler exposes those primitives.
"""

from __future__ import annotations

import torch

from calm.llm_computer.model import Small2DTransformer, Small2DConfig


def build_increment_counter(vocab_size: int = 8) -> Small2DTransformer:
    V = vocab_size
    cfg = Small2DConfig(
        vocab_size=V,
        d_model=2 * V,
        n_heads=V,
        n_layers=1,
        d_ffn=2 * V,
        max_len=V,           # program only defined for p in 0..V-1
        use_hard_max=True,
    )
    assert cfg.d_head == 2, f"expected d_head=2, got {cfg.d_head}"

    model = Small2DTransformer(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # Token embedding: identity on lower half. This program ignores
        # the input, but we keep tok non-zero so downstream programs can
        # combine position + tok cleanly if they reuse this config.
        for k in range(V):
            model.tok.weight[k, k] = 1.0

        # Position embedding: pos[p] has 1 at dim V+p.
        for p in range(V):
            model.pos.weight[p, V + p] = 1.0

        # Attention + FFN weights stay zero.

        # Head reads upper half: head[j, V+j] = 1.
        for j in range(V):
            model.head.weight[j, V + j] = 1.0

    return model


def run_increment_counter(model: Small2DTransformer, length: int) -> list[int]:
    """Feed an arbitrary-token prompt of `length` tokens; return argmax per position."""
    with torch.no_grad():
        # Arbitrary input; result should be independent of token values.
        idx = torch.zeros(1, length, dtype=torch.long)
        logits = model(idx)
        return logits[0].argmax(dim=-1).tolist()


if __name__ == "__main__":
    V = 8
    model = build_increment_counter(vocab_size=V)
    print(f"[increment_counter] built Small2DTransformer, {model.param_count():,} params")
    for length in (1, 3, 5, 8):
        got = run_increment_counter(model, length)
        expected = list(range(length))
        status = "ok" if got == expected else "FAIL"
        print(f"  [{status}] length={length}, output={got} (expected {expected})")

    # Robustness: different input tokens should NOT change output.
    with torch.no_grad():
        logits_a = model(torch.tensor([[3, 7, 1, 4]], dtype=torch.long))
        logits_b = model(torch.tensor([[0, 0, 0, 0]], dtype=torch.long))
    a = logits_a[0].argmax(-1).tolist()
    b = logits_b[0].argmax(-1).tolist()
    print(f"  [{'ok' if a == b == [0, 1, 2, 3] else 'FAIL'}] "
          f"input-invariance: tokens [3,7,1,4]→{a}, [0,0,0,0]→{b}")
