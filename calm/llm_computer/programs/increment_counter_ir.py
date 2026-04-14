"""`increment_counter` as a gate graph.

Compare to `increment_counter.py` (hand-wired). Exercises `TokenEmbed`
(unused lower-half identity — kept for shape parity with hand-wired),
`PosEmbed` (write per-position one-hot into upper half), and
`LinearHead` (read upper half).

Program semantics: output `p` at every output position `p`, regardless
of input token.

Residual channel layout (d_model = 2V):
  ch 0..V-1: tok embedding (identity; unused by this program but kept
             so it reuses the same config as copy_past)
  ch V..2V-1: position one-hot (pos[p, V+p] = 1)
"""

from __future__ import annotations

from calm.llm_computer.compile import compile_program
from calm.llm_computer.gate_graph import (
    GateGraph, LinearHead, PosEmbed, TokenEmbed,
)
from calm.llm_computer.model import Small2DTransformer


def build_increment_counter_ir(vocab_size: int = 8) -> Small2DTransformer:
    V = vocab_size
    graph = GateGraph(vocab_size=V)

    # tok: identity on lower half (unused, but matches hand-wired layout).
    graph.add(TokenEmbed(
        name="tok_lower_identity",
        entries=[(k, k, 1.0) for k in range(V)],
    ))
    # pos: pos[p, V + p] = 1 → one-hot in upper half at position p.
    graph.add(PosEmbed(
        name="pos_upper_onehot",
        entries=[(p, V + p, 1.0) for p in range(V)],
    ))
    # head: read upper half — head[j, V + j] = 1.
    graph.add(LinearHead(
        name="head_upper",
        entries=[(j, V + j, 1.0) for j in range(V)],
    ))

    return compile_program(
        graph,
        d_model=2 * V,
        n_heads=V,
        n_layers=1,
        d_ffn=2 * V,
        max_len=V,  # program only defined for p in 0..V-1
        vocab_size=V,
    )


if __name__ == "__main__":
    import torch
    V = 8
    model = build_increment_counter_ir(vocab_size=V)
    print(f"[increment_counter_ir] built Small2DTransformer, {model.param_count():,} params")
    all_ok = True
    for length in (1, 3, 5, 8):
        x = torch.zeros(1, length, dtype=torch.long)
        with torch.no_grad():
            got = model(x)[0].argmax(dim=-1).tolist()
        expected = list(range(length))
        ok = got == expected
        all_ok = all_ok and ok
        print(f"  [{'ok' if ok else 'FAIL'}] length={length} → {got} (expected {expected})")
    # Input invariance.
    with torch.no_grad():
        a = model(torch.tensor([[3, 7, 1, 4]]))[0].argmax(-1).tolist()
        b = model(torch.tensor([[0, 0, 0, 0]]))[0].argmax(-1).tolist()
    inv_ok = a == b == [0, 1, 2, 3]
    all_ok = all_ok and inv_ok
    print(f"  [{'ok' if inv_ok else 'FAIL'}] input-invariance: [3,7,1,4]→{a}, [0,0,0,0]→{b}")
    print(f"overall: {'PASS' if all_ok else 'FAIL'}")
