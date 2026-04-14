"""`threshold` as a gate graph.

Compare to `threshold.py` (hand-wired). Exercises `TokenEmbed`,
`PosEmbed`, two `ReGLU` neurons, and `LinearHead`. This is the first
IR program that uses the FFN compile path — its correctness is the
strongest signal that `_apply_reglu` in `compile.py` wires ff_in and
ff_out correctly.

Program semantics: output `1` if input_token ≥ T, else `0`. Implements
`1[z ≥ 0] = ReLU(z + 1) − ReLU(z)` via two ReGLU neurons, per the step-
function primitive in RESEARCH/03 §4c.

Residual channel layout (d_model = 4):
  ch 0: input scalar (TokenEmbed: tok[k, 0] = k)
  ch 1: bias (PosEmbed: pos[p, 1] = 1 at every position)
  ch 2: step-function output (ReGLU neurons accumulate here)
  ch 3: unused
"""

from __future__ import annotations

from calm.llm_computer.compile import compile_program
from calm.llm_computer.gate_graph import (
    GateGraph, LinearHead, PosEmbed, ReGLU, TokenEmbed,
)
from calm.llm_computer.model import Small2DTransformer


def build_threshold_ir(vocab_size: int = 8, threshold_value: int = 4,
                        max_len: int = 16) -> Small2DTransformer:
    V = vocab_size
    T = threshold_value
    graph = GateGraph(vocab_size=V)

    # ch 0: input scalar.
    graph.add(TokenEmbed(
        name="input_scalar",
        entries=[(k, 0, float(k)) for k in range(V)],
    ))
    # ch 1: constant 1 bias.
    graph.add(PosEmbed(
        name="bias_channel",
        entries=[(p, 1, 1.0) for p in range(max_len)],
    ))
    # ReGLU #0: gate = input − (T − 1), val = +1  →  +ReLU(input − (T − 1))
    graph.add(ReGLU(
        name="step_hi",
        layer=0,
        gate=[(0, 1.0), (1, -(T - 1))],
        val=[(1, 1.0)],
        output_channel=2,
        output_coef=1.0,
    ))
    # ReGLU #1: gate = input − T, val = −1  →  −ReLU(input − T)
    graph.add(ReGLU(
        name="step_lo",
        layer=0,
        gate=[(0, 1.0), (1, -T)],
        val=[(1, -1.0)],
        output_channel=2,
        output_coef=1.0,
    ))
    # head: token 1 logits = residual[2]. All other tokens zero; first-tie
    # argmax picks 0 when step == 0, 1 when step == 1.
    graph.add(LinearHead(
        name="read_step",
        entries=[(1, 2, 1.0)],
    ))

    return compile_program(
        graph,
        d_model=4,
        n_heads=2,
        n_layers=1,
        d_ffn=2,
        max_len=max_len,
        vocab_size=V,
    )


if __name__ == "__main__":
    import torch
    V, T = 8, 4
    model = build_threshold_ir(vocab_size=V, threshold_value=T)
    print(f"[threshold_ir] built Small2DTransformer, {model.param_count():,} params")
    all_ok = True
    for k in range(V):
        x = torch.tensor([[k]], dtype=torch.long)
        with torch.no_grad():
            got = int(model(x)[0, 0].argmax().item())
        expected = 1 if k >= T else 0
        ok = got == expected
        all_ok = all_ok and ok
        print(f"  [{'ok' if ok else 'FAIL'}] input={k} → {got} (expected {expected})")
    print(f"overall: {'PASS' if all_ok else 'FAIL'}")
