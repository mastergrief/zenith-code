"""`retrieve_by_index` — parabolic-key LookUpExact validation.

Program: given a sequence of value tokens followed by a query-index
token, return the token whose value was stored at the queried position.

Example (V=4, N=4 stored values):
  input [2, 0, 3, 1, 0] → output at last position = 2  (= token at index 0)
  input [2, 0, 3, 1, 2] → output at last position = 3  (= token at index 2)
  input [2, 0, 3, 1, 3] → output at last position = 1  (= token at index 3)

This is the first program to use `LookUpExact`. The retrieval is
exact (not argmax-ties-to-pos-0 like `LookUp`) — the head's hard-max
over parabolic keys selects exactly `j = query_idx`.

Residual channel layout (d_model = 2V + 4):
  ch 0..V-1: tok one-hot (at every position, marks that position's
             input token)
  ch V:      tok scalar (at the query position this = query_idx)
  ch V+1:    bias (constant 1, used as q[1])
  ch V+2:    2p (parabolic-key k[0])
  ch V+3:    -p² (parabolic-key k[1])
  ch V+4..2V+3: retrieved one-hot (LookUpExact writes V channels here)
"""

from __future__ import annotations

from calm.llm_computer.compile import compile_program
from calm.llm_computer.gate_graph import (
    GateGraph, LinearHead, LookUpExact, PosEmbed, TokenEmbed,
)
from calm.llm_computer.model import Small2DTransformer


def build_retrieve_by_index(vocab_size: int = 4,
                              max_len: int = 8) -> Small2DTransformer:
    V = vocab_size
    graph = GateGraph(vocab_size=V)

    # ch 0..V-1: tok one-hot (so v_projection has something to pull).
    graph.add(TokenEmbed(
        name="tok_onehot",
        entries=[(k, k, 1.0) for k in range(V)],
    ))
    # ch V: tok scalar — at the query position this = query_idx.
    graph.add(TokenEmbed(
        name="tok_scalar",
        entries=[(k, V, float(k)) for k in range(V)],
    ))
    # ch V+1: bias.
    # ch V+2: 2p.
    # ch V+3: -p².
    graph.add(PosEmbed(
        name="pos_parabolic",
        entries=(
            [(p, V + 1, 1.0) for p in range(max_len)]
            + [(p, V + 2, 2.0 * p) for p in range(max_len)]
            + [(p, V + 3, -float(p * p)) for p in range(max_len)]
        ),
    ))
    # LookUpExact: V heads, one per value channel.
    graph.add(LookUpExact(
        name="retrieve",
        layer=0,
        pos_key0_channel=V + 2,
        pos_key1_channel=V + 3,
        query_key_channel=V,
        bias_channel=V + 1,
        value_source_channels=list(range(V)),
        out_channels=[V + 4 + k for k in range(V)],
    ))
    # Head: read retrieved one-hot.
    graph.add(LinearHead(
        name="head",
        entries=[(k, V + 4 + k, 1.0) for k in range(V)],
    ))

    d_model = 2 * V + 4
    return compile_program(
        graph,
        d_model=d_model,
        n_heads=d_model // 2,
        n_layers=1,
        d_ffn=d_model,
        max_len=max_len,
        vocab_size=V,
    )


if __name__ == "__main__":
    import torch
    V = 4
    N = 4  # 4 values stored at positions 0..3, query at position 4
    model = build_retrieve_by_index(vocab_size=V, max_len=N + 1)
    print(f"[retrieve_by_index] built Small2DTransformer, "
          f"{model.param_count():,} params")
    all_ok = True
    # Test all (values, query_idx) combos.
    test_cases = [
        ([2, 0, 3, 1, 0], 2),
        ([2, 0, 3, 1, 1], 0),
        ([2, 0, 3, 1, 2], 3),
        ([2, 0, 3, 1, 3], 1),
        ([3, 3, 3, 3, 0], 3),
        ([0, 1, 2, 3, 2], 2),
    ]
    for inp, expected in test_cases:
        x = torch.tensor([inp], dtype=torch.long)
        with torch.no_grad():
            got = int(model(x)[0, N].argmax().item())
        ok = got == expected
        all_ok = all_ok and ok
        print(f"  [{'ok' if ok else 'FAIL'}] {inp} (query idx={inp[-1]}) "
              f"→ {got} (expected {expected})")
    print(f"\noverall: {'PASS' if all_ok else 'FAIL'}")
