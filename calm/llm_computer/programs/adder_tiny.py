"""1-digit adder — the first compositional program in the IR.

Program: input `[a, b]` with `a, b ∈ [0, 3]`; output at position 1 is
the token representing `a + b` (value in `[0, 6]`). Exercises both
`LookUp` (copy `a` from position 0) AND `ReGLU` (step functions that
decompose the sum into a one-hot) in one graph.

Construction (RESEARCH/03 §4c + §9 inspired):

  Residual channel layout (d_model = 10):
    ch 0: b scalar (from TokenEmbed, pos 1 writes its own token)
    ch 1: bias 1 (from PosEmbed, every position)
    ch 2: a scalar (from LookUp — copies pos 0's ch 0 at every query)
    ch 3..9: step functions step_S = 1[a + b >= S] for S ∈ {0..6}

  Step function S via two ReGLU neurons per S:
    neuron 2S:   gate = a + b - (S - 1), val = +1  →  +ReLU(a+b-S+1)
    neuron 2S+1: gate = a + b - S,       val = +1  →  -ReLU(a+b-S)
    sum: step_S = ReLU(a+b-S+1) - ReLU(a+b-S) = 1[a+b >= S]

  Final head: logits[k] = step_k - step_{k+1} (= indicator a+b == k),
  treating step_7 as 0. argmax picks k = a + b.

Tested for a, b ∈ [0, 3] only. Extending to 3-bit operands needs more
step functions (≥ 15 for sum ∈ [0, 14]); same template.
"""

from __future__ import annotations

from calm.llm_computer.compile import compile_program
from calm.llm_computer.gate_graph import (
    GateGraph, LinearHead, LookUp, PosEmbed, ReGLU, TokenEmbed,
)
from calm.llm_computer.model import Small2DTransformer


def build_adder_tiny(vocab_size: int = 8, max_len: int = 4) -> Small2DTransformer:
    V = vocab_size
    MAX_SUM = 6  # since a, b in [0, 3]
    graph = GateGraph(vocab_size=V)

    # ch 0: own token scalar.
    graph.add(TokenEmbed(
        name="own_scalar",
        entries=[(k, 0, float(k)) for k in range(V)],
    ))
    # ch 1: bias 1.
    graph.add(PosEmbed(
        name="bias",
        entries=[(p, 1, 1.0) for p in range(max_len)],
    ))
    # LookUp: copy pos 0's ch 0 into ch 2 at every query position.
    # After this, at query pos i, residual[2] = tok[input[0]][0] = a.
    graph.add(LookUp(
        name="copy_a",
        layer=0,
        v_source_channels=[0],
        out_channels=[2],
    ))

    # FFN: 7 step functions, 14 ReGLU neurons.
    for S in range(MAX_SUM + 1):
        # neuron 2S: +ReLU(a + b - S + 1)
        graph.add(ReGLU(
            name=f"step_{S}_hi",
            layer=0,
            gate=[(0, 1.0), (2, 1.0), (1, -(S - 1))],  # b + a - (S-1)
            val=[(1, 1.0)],
            output_channel=3 + S,
            output_coef=1.0,
        ))
        # neuron 2S+1: -ReLU(a + b - S)
        graph.add(ReGLU(
            name=f"step_{S}_lo",
            layer=0,
            gate=[(0, 1.0), (2, 1.0), (1, -S)],  # b + a - S
            val=[(1, 1.0)],
            output_channel=3 + S,
            output_coef=-1.0,
        ))

    # Head: logits[k] = step_k - step_{k+1}. step_7 = 0 by convention.
    head_entries = []
    for k in range(MAX_SUM + 1):
        head_entries.append((k, 3 + k, 1.0))
        if k + 1 <= MAX_SUM:
            head_entries.append((k, 3 + k + 1, -1.0))
    graph.add(LinearHead(name="onehot_via_steps", entries=head_entries))

    return compile_program(
        graph,
        d_model=10,
        n_heads=5,
        n_layers=1,
        d_ffn=14,
        max_len=max_len,
        vocab_size=V,
    )


if __name__ == "__main__":
    import torch
    model = build_adder_tiny(vocab_size=8, max_len=4)
    print(f"[adder_tiny] built Small2DTransformer, {model.param_count():,} params")
    all_ok = True
    for a in range(4):
        for b in range(4):
            x = torch.tensor([[a, b]], dtype=torch.long)
            with torch.no_grad():
                got = int(model(x)[0, 1].argmax().item())  # position 1's output
            expected = a + b
            ok = got == expected
            all_ok = all_ok and ok
            status = "ok" if ok else "FAIL"
            print(f"  [{status}] {a} + {b} = {got} (expected {expected})")
    print(f"\noverall: {'PASS' if all_ok else 'FAIL'}")
