"""`retrieve_threshold` — LookUpExact feeding ReGLU in one layer.

Program: inputs [v0, v1, v2, v3, query_idx]; output at position 4 is
`1` if `v_{query_idx} >= 2` else `0`. Composition test — layer 0's
attention does the parabolic-key retrieval, layer 0's FFN thresholds
the retrieved scalar, all in a single layer (the transformer's attn-
before-FFN ordering means the FFN sees the residual after attention's
write).

Residual channel layout (d_model = V + 6):
  ch 0..V-1: tok one-hot (unused here — we use the scalar)
  ch V:      tok scalar (= stored value at each past position,
                         = query_idx at query position — same channel
                         serves both because tok embed is per-token)
  ch V+1:    bias (constant 1)
  ch V+2:    2p  (parabolic key 0)
  ch V+3:    -p² (parabolic key 1)
  ch V+4:    retrieved scalar (LookUpExact writes; ReGLU reads)
  ch V+5:    step-function output (ReGLU writes)
"""

from __future__ import annotations

from calm.llm_computer.compile import compile_program
from calm.llm_computer.gate_graph import (
    GateGraph, LinearHead, LookUpExact, PosEmbed, ReGLU, TokenEmbed,
)
from calm.llm_computer.model import Small2DTransformer


def build_retrieve_threshold(vocab_size: int = 4, threshold: int = 2,
                               max_len: int = 8) -> Small2DTransformer:
    V = vocab_size
    T = threshold
    d_model = V + 6
    graph = GateGraph(vocab_size=V)

    graph.add(TokenEmbed(
        name="tok_onehot",
        entries=[(k, k, 1.0) for k in range(V)],
    ))
    graph.add(TokenEmbed(
        name="tok_scalar",
        entries=[(k, V, float(k)) for k in range(V)],
    ))
    graph.add(PosEmbed(
        name="pos_parabolic_bias",
        entries=(
            [(p, V + 1, 1.0) for p in range(max_len)]
            + [(p, V + 2, 2.0 * p) for p in range(max_len)]
            + [(p, V + 3, -float(p * p)) for p in range(max_len)]
        ),
    ))
    # Retrieve the scalar at position query_idx → ch V+4.
    graph.add(LookUpExact(
        name="retrieve_scalar",
        layer=0,
        pos_key0_channel=V + 2,
        pos_key1_channel=V + 3,
        query_key_channel=V,
        bias_channel=V + 1,
        value_source_channels=[V],
        out_channels=[V + 4],
    ))
    # Step function 1[v >= T] via 2 ReGLU neurons. Reads ch V+4
    # (written by LookUpExact, same layer attn happens before FFN).
    graph.add(ReGLU(
        name="step_hi",
        layer=0,
        gate=[(V + 4, 1.0), (V + 1, -(T - 1))],
        val=[(V + 1, 1.0)],
        output_channel=V + 5,
        output_coef=1.0,
    ))
    graph.add(ReGLU(
        name="step_lo",
        layer=0,
        gate=[(V + 4, 1.0), (V + 1, -T)],
        val=[(V + 1, 1.0)],
        output_channel=V + 5,
        output_coef=-1.0,
    ))
    # Head: logits[1] = step output. All others zero → argmax = 0 when
    # step = 0, argmax = 1 when step = 1.
    graph.add(LinearHead(
        name="read_step",
        entries=[(1, V + 5, 1.0)],
    ))

    return compile_program(
        graph,
        d_model=d_model,
        n_heads=d_model // 2,
        n_layers=1,
        d_ffn=2,
        max_len=max_len,
        vocab_size=V,
    )


if __name__ == "__main__":
    import torch
    V, T = 4, 2
    N = 4
    model = build_retrieve_threshold(vocab_size=V, threshold=T, max_len=N + 1)
    print(f"[retrieve_threshold] built Small2DTransformer, "
          f"{model.param_count():,} params (V={V}, T={T})")
    all_ok = True
    cases = [
        ([3, 0, 2, 1, 0], 1),  # v_0 = 3 >= 2 → 1
        ([3, 0, 2, 1, 1], 0),  # v_1 = 0 < 2 → 0
        ([3, 0, 2, 1, 2], 1),  # v_2 = 2 >= 2 → 1
        ([3, 0, 2, 1, 3], 0),  # v_3 = 1 < 2 → 0
        ([1, 1, 1, 1, 0], 0),  # all 1 < 2 → 0
        ([2, 2, 2, 2, 2], 1),  # all 2 >= 2 → 1
        ([0, 1, 2, 3, 3], 1),  # v_3 = 3 >= 2 → 1
    ]
    for inp, expected in cases:
        x = torch.tensor([inp], dtype=torch.long)
        with torch.no_grad():
            got = int(model(x)[0, N].argmax().item())
        ok = got == expected
        all_ok = all_ok and ok
        q = inp[-1]
        print(f"  [{'ok' if ok else 'FAIL'}] values={inp[:N]} q={q} "
              f"v_q={inp[q]} → {got} (expected {expected})")
    print(f"\noverall: {'PASS' if all_ok else 'FAIL'}")
