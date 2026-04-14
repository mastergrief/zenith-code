"""Hand-written `copy_past` weights for Small2DTransformer.

Program semantics: at every output position, emit the token that was at
input position 0. Exercises the attention primitive — the first program
that actually USES a non-zero attention head.

How the weights work (the "trick"):
  - Residual layout: d_model = 2 * vocab_size. Low half (dims 0..V-1)
    carries the current-position tok embedding; upper half (dims V..2V-1)
    is reserved as a "retrieved from position 0" slot.
  - W_q and W_k both zero across all heads → attention scores are 0 for
    all valid (causal) past positions. `torch.argmax` with first-tie
    semantics therefore picks past-position 0 at every query position.
  - Heads V/2..V-1 (the "upper" heads) have their v projection set to
    copy x's dims 0..V-1 into the head's 2-dim slot. After reshape, the
    attention output at every position reconstructs tok(input[0]) in the
    upper half (dims V..2V-1).
  - W_out routes that upper half into residual dims V..2V-1 (identity on
    those dims, zero elsewhere).
  - Head reads only the upper half: head.weight[j, V+j] = 1 picks out
    position j's dim from the retrieved embedding.

Result: greedy decoding produces argmax = input[0] at every position.
"""

from __future__ import annotations

import torch

from calm.llm_computer.model import Small2DTransformer, Small2DConfig


def build_copy_past(vocab_size: int = 8) -> Small2DTransformer:
    V = vocab_size
    cfg = Small2DConfig(
        vocab_size=V,
        d_model=2 * V,          # low half: tok, upper half: retrieved-from-pos-0
        n_heads=V,              # d_head = 2 with 2V-dim residual → V heads
        n_layers=1,             # one layer suffices
        d_ffn=2 * V,
        max_len=32,
        use_hard_max=True,
    )
    assert cfg.d_head == 2, f"expected d_head=2, got {cfg.d_head}"

    model = Small2DTransformer(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # Token embedding: tok[k][j] = 1 iff j == k, for j in 0..V-1.
        # Upper half (dims V..2V-1) stays zero. This keeps the "retrieved"
        # slot empty until attention writes into it.
        for k in range(V):
            model.tok.weight[k, k] = 1.0
        # Position embedding is zero.

        # W_qkv[0]: 3*d_model = 6V outputs from d_model = 2V inputs.
        # Output-dim layout (one chunk per {q,k,v} × head × d_head):
        #   q: output dims 0..(2V-1)        ← q for heads 0..V-1
        #   k: output dims 2V..(4V-1)       ← k for heads 0..V-1
        #   v: output dims 4V..(6V-1)       ← v for heads 0..V-1
        # Inside each chunk: head h occupies `[chunk_start + 2h, +1]`.
        #
        # q and k stay zero (scores all zero → causal argmax picks pos 0).
        # For v: heads V/2..V-1 copy x dims 0..V-1 into the upper v-slots.
        #
        # For i in 0..V-1, the (i/2)-th upper head's d_head dim i%2 should
        # read x dim i (low half of residual = tok embedding). That head
        # sits at head index h = V/2 + i//2, so v-slot = 4V + 2h + (i%2) =
        # 4V + V + i = 5V + i.
        for i in range(V):
            model.W_qkv[0].weight[5 * V + i, i] = 1.0

        # Reshape back-fact: head h's attention output, after the
        # transpose(1,2).reshape, lands in residual dims 2h..2h+1. So upper
        # heads V/2..V-1 write to residual dims V..2V-1 — exactly the slot
        # we reserved. W_out then needs to route attn[V..2V-1] into
        # residual[V..2V-1]; upper-half identity suffices.
        for d in range(V, 2 * V):
            model.W_out[0].weight[d, d] = 1.0

        # Head: only consult the upper half. head.weight[j, V+j] = 1 picks
        # the dim that tok(input[0]) wrote at dim V+j.
        for j in range(V):
            model.head.weight[j, V + j] = 1.0

    return model


def run_copy_past(model: Small2DTransformer, input_tokens: list[int]) -> list[int]:
    """Greedy-decode: feed all input tokens, return argmax at every position."""
    with torch.no_grad():
        x = torch.tensor([input_tokens], dtype=torch.long)
        logits = model(x)  # (1, S, V)
        return logits[0].argmax(dim=-1).tolist()


if __name__ == "__main__":
    model = build_copy_past(vocab_size=8)
    print(f"[copy_past] built Small2DTransformer, {model.param_count():,} params")
    tests = [
        [3, 7, 2, 5, 1],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [7, 0],
        [5],
    ]
    for inp in tests:
        got = run_copy_past(model, inp)
        expected = [inp[0]] * len(inp)
        status = "ok" if got == expected else "FAIL"
        print(f"  [{status}] input={inp}, output={got} (expected {expected})")
