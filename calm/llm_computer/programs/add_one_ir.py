"""`add_one` as a gate graph.

Compare to `add_one.py`: the hand-wired version pokes tensors directly;
this one declares the program as an IR and has `compile_program` emit
the weights. Should produce bit-identical weights because the IR is
1:1 with the tensor-poking pattern (identity tok embedding + cyclic-
shift linear head, no attention or FFN).
"""

from __future__ import annotations

from calm.llm_computer.compile import compile_program
from calm.llm_computer.gate_graph import (
    GateGraph, LinearHead, TokenEmbed,
)
from calm.llm_computer.model import Small2DTransformer


def build_add_one_ir(vocab_size: int = 8) -> Small2DTransformer:
    V = vocab_size
    graph = GateGraph(vocab_size=V)

    # tok.weight[k, k] = 1 → identity.
    graph.add(TokenEmbed(
        name="tok_id",
        entries=[(k, k, 1.0) for k in range(V)],
    ))
    # head.weight[(k+1) % V, k] = 1 → cyclic shift.
    graph.add(LinearHead(
        name="head_shift",
        entries=[((k + 1) % V, k, 1.0) for k in range(V)],
    ))

    return compile_program(
        graph,
        d_model=V,
        n_heads=V // 2,
        n_layers=2,  # match add_one.py
        d_ffn=V,
        max_len=32,
        vocab_size=V,
    )


if __name__ == "__main__":
    import torch
    model = build_add_one_ir(vocab_size=8)
    print(f"[add_one_ir] built Small2DTransformer, {model.param_count():,} params")
    all_ok = True
    for k in range(8):
        x = torch.tensor([[k]], dtype=torch.long)
        with torch.no_grad():
            got = int(model(x)[0, 0].argmax().item())
        expected = (k + 1) % 8
        ok = got == expected
        all_ok = all_ok and ok
        print(f"  [{'ok' if ok else 'FAIL'}] input={k} → {got} (expected {expected})")
    print(f"overall: {'PASS' if all_ok else 'FAIL'}")
