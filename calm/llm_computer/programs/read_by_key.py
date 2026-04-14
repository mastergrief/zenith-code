"""`read_by_key` — semantic-key retrieval (KV store primitive).

Program: `N` positions each store a distinct key (permutation of
`0..N-1`); the query at the last position names a key. Output is
the position index where that key was stored.

Exercises two new compiler capabilities:
  1. LookUpExact with coefficients (`pos_key0_coef=2.0`) — scales a
     scalar key channel into `2·key` for the parabolic construction,
     instead of needing a precomputed `2p` table per position.
  2. ReGLU squaring — `-k² = -k · ReLU(k)` for non-negative integer
     `k`, lifting semantic keys into the `(2k, -k²)` parabolic form.

Layering: 2 layers.
  layer 0 attn: unused (zero heads). layer 0 FFN: ReGLU squares `k`.
  layer 1 attn: LookUpExact with semantic keys. layer 1 FFN: step
                functions decode the retrieved scalar into a one-hot
                over possible positions.

Residual channels (d_model = 10):
  ch 0: k_ch         — tok scalar (stored key / query key)
  ch 1: bias_ch      — constant 1 (PosEmbed)
  ch 2: pos_scalar   — p (PosEmbed; this is the "value" for each key)
  ch 3: neg_k2       — written by layer 0 FFN (computes -k²)
  ch 4: retrieved_p  — written by layer 1 attn (the matched position)
  ch 5..8: step_S for S in [0, 3] — one-hot decoding (layer 1 FFN)
  ch 9: unused (padding for d_head=2)

Head: logits[k] = step_{k-1} - step_k (step_{-1} = 1, step_N = 0 via
wiring), giving a one-hot over `[0, N-1]` when retrieved_p is an
integer in that range.

Head wiring notation: step_{-1} is absent, so logits[0] = -step_0 + 1
wouldn't be possible with a pure linear head. Instead we use the
standard step-diff: logits[k] = step_k - step_{k+1} — which gives a
one-hot indicating "retrieved_p == k". For correct argmax over k in
[0, N-1], step_k for k=0..N-1 is sufficient.
"""

from __future__ import annotations

from calm.llm_computer.compile import compile_program
from calm.llm_computer.gate_graph import (
    GateGraph, LinearHead, LookUpExact, PosEmbed, ReGLU, TokenEmbed,
)
from calm.llm_computer.model import Small2DTransformer


def build_read_by_key(vocab_size: int = 4, max_len: int = 8) -> Small2DTransformer:
    V = vocab_size   # also = number of stored positions + 1 (query)
    N = V            # stored positions 0..V-1
    d_model = 10
    graph = GateGraph(vocab_size=V)

    # ch 0: stored key scalar at each position (or query key at query pos).
    graph.add(TokenEmbed(
        name="key_scalar",
        entries=[(k, 0, float(k)) for k in range(V)],
    ))
    # ch 1: bias. ch 2: position scalar.
    graph.add(PosEmbed(
        name="bias_and_pos",
        entries=(
            [(p, 1, 1.0) for p in range(max_len)]
            + [(p, 2, float(p)) for p in range(max_len)]
        ),
    ))
    # Layer 0 FFN: compute -k² via ReGLU.
    # out += 1 * (-k) * ReLU(k) = -k * k for k >= 0 = -k²
    graph.add(ReGLU(
        name="square_neg",
        layer=0,
        gate=[(0, 1.0)],       # gate = k (from ch 0)
        val=[(0, -1.0)],       # val = -k
        output_channel=3,      # write to neg_k2 channel
        output_coef=1.0,
    ))
    # Layer 1 attn: semantic-key LookUpExact.
    graph.add(LookUpExact(
        name="retrieve_by_key",
        layer=1,
        pos_key0_channel=0, pos_key0_coef=2.0,  # k[0] = 2 * residual[k_ch]
        pos_key1_channel=3, pos_key1_coef=1.0,  # k[1] = residual[neg_k2]
        query_key_channel=0, query_key_coef=1.0,
        bias_channel=1, bias_coef=1.0,
        value_source_channels=[2],  # fetch the position scalar
        out_channels=[4],           # write retrieved position into ch 4
    ))
    # Layer 1 FFN: step functions 1[retrieved_p >= S] for S in [0, N-1].
    # 2 ReGLU neurons per S → 2N total.
    for S in range(N):
        # +ReLU(retrieved - (S-1)) → step_hi
        graph.add(ReGLU(
            name=f"step_{S}_hi",
            layer=1,
            gate=[(4, 1.0), (1, -(S - 1))],
            val=[(1, 1.0)],
            output_channel=5 + S,
            output_coef=1.0,
        ))
        # -ReLU(retrieved - S) → step_lo
        graph.add(ReGLU(
            name=f"step_{S}_lo",
            layer=1,
            gate=[(4, 1.0), (1, -S)],
            val=[(1, 1.0)],
            output_channel=5 + S,
            output_coef=-1.0,
        ))
    # Head: logits[k] = step_k - step_{k+1}. For k=N-1, step_N=0 (not
    # wired), so just logits[N-1] = step_{N-1}.
    head_entries = []
    for k in range(N):
        head_entries.append((k, 5 + k, 1.0))
        if k + 1 < N:
            head_entries.append((k, 5 + k + 1, -1.0))
    graph.add(LinearHead(name="decode", entries=head_entries))

    return compile_program(
        graph,
        d_model=d_model,
        n_heads=d_model // 2,
        n_layers=2,
        d_ffn=2 * N,  # max across layers (layer 1 needs 2N; layer 0 needs 1)
        max_len=max_len,
        vocab_size=V,
    )


if __name__ == "__main__":
    import itertools
    import torch

    V = 4
    model = build_read_by_key(vocab_size=V, max_len=V + 1)
    print(f"[read_by_key] built Small2DTransformer, "
          f"{model.param_count():,} params (V={V})")

    all_ok = True
    # Test every permutation of 4 keys, every query in [0, V).
    for perm in itertools.permutations(range(V)):
        for query_key in range(V):
            inp = list(perm) + [query_key]
            expected_pos = perm.index(query_key)
            x = torch.tensor([inp], dtype=torch.long)
            with torch.no_grad():
                got = int(model(x)[0, V].argmax().item())
            ok = got == expected_pos
            all_ok = all_ok and ok
            if not ok:
                print(f"  [FAIL] keys={perm} query={query_key} "
                      f"→ {got} (expected pos {expected_pos})")

    n_cases = len(list(itertools.permutations(range(V)))) * V
    print(f"\noverall: {'PASS' if all_ok else 'FAIL'} "
          f"({n_cases} (perm, query) cases)")
