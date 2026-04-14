"""Hand-written `add_one` weights for Small2DTransformer.

Smallest possible compile-to-weights demo:
  - Input token k → output token (k+1) mod vocab.
  - Attention and FFN are zeroed; residual just passes tok+pos embeddings.
  - Token embedding is the identity matrix (tok[k] = e_k).
  - Positional embedding is zero (positions don't matter for this program).
  - Linear head implements the cyclic shift: head[j, k] = 1 iff j = (k+1) mod vocab.

This is not a useful program — it's the "hello world" of compile-to-weights,
there to prove that manually-constructed weights in a standard
Small2DTransformer architecture execute a deterministic function under
greedy decoding.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from calm.llm_computer.model import Small2DTransformer, Small2DConfig


def build_add_one(vocab_size: int = 8) -> Small2DTransformer:
    """Return a Small2DTransformer whose weights implement `add_one`.

    Requires `d_model == vocab_size` so the token embedding can be the
    identity matrix (one-hot per token). Hard-max attention is the
    analytically-clean choice here but the result is the same under
    softmax since attention output is zeroed.
    """
    cfg = Small2DConfig(
        vocab_size=vocab_size,
        d_model=vocab_size,   # identity tok embedding
        n_heads=vocab_size // 2 if vocab_size >= 2 else 1,  # d_head = 2
        n_layers=2,
        d_ffn=vocab_size,
        max_len=32,
        use_hard_max=True,
    )
    assert cfg.d_head == 2, f"expected d_head=2, got {cfg.d_head}"

    model = Small2DTransformer(cfg)

    with torch.no_grad():
        # Zero every parameter; then set only what the program needs.
        for p in model.parameters():
            p.zero_()

        # Token embedding: tok[k] = e_k (k-th standard basis vector in d_model).
        model.tok.weight.copy_(torch.eye(vocab_size))
        # Position embedding stays zero.
        # W_qkv, W_out, ff_in, ff_out all zero → attention and FFN contribute 0.
        # Linear head: head.weight[j, k] = 1 iff j == (k+1) mod vocab_size.
        shift = torch.zeros(vocab_size, vocab_size)
        for k in range(vocab_size):
            shift[(k + 1) % vocab_size, k] = 1.0
        model.head.weight.copy_(shift)

    return model


def run_add_one(model: Small2DTransformer, input_token: int) -> int:
    """Greedy-decode: feed input_token, return argmax of output logits."""
    with torch.no_grad():
        x = torch.tensor([[input_token]], dtype=torch.long)
        logits = model(x)  # (1, 1, vocab)
        return int(logits[0, 0].argmax().item())


if __name__ == "__main__":
    model = build_add_one(vocab_size=8)
    print(f"[add_one] built Small2DTransformer, {model.param_count():,} params")
    for k in range(8):
        got = run_add_one(model, k)
        expected = (k + 1) % 8
        status = "ok" if got == expected else "FAIL"
        print(f"  [{status}] input={k}, output={got} (expected {expected})")
