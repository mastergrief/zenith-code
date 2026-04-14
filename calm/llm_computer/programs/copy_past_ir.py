"""`copy_past` as a gate graph.

Compare to `copy_past.py` (hand-wired). Exercises the `LookUp` node —
the first IR program that uses attention.

Program semantics: at every output position, emit tok[input[0]] (the
token that was at input position 0). LookUp's zero-q/k first-tie
semantics deterministically picks past position 0 at every query
position, so the V projection drops tok(input[0]) into the residual.

Residual channel layout (d_model = 2V):
  ch 0..V-1: tok embedding (identity, written by TokenEmbed)
  ch V..2V-1: attention output (written by LookUp, read by LinearHead)

NOTE on bit-match: the hand-wired version packs 2 channels per head
(upper heads V/2..V-1 each read 2 input dims); this IR uses 1 channel
per head (heads 0..V-1 each read 1 input dim). Both produce the same
forward-pass output but weight tensors differ. Bit-match is therefore
FALSE for copy_past; behavioral match passes.
"""

from __future__ import annotations

from calm.llm_computer.compile import compile_program
from calm.llm_computer.gate_graph import (
    GateGraph, LinearHead, LookUp, TokenEmbed,
)
from calm.llm_computer.model import Small2DTransformer


def build_copy_past_ir(vocab_size: int = 8) -> Small2DTransformer:
    V = vocab_size
    graph = GateGraph(vocab_size=V)

    # tok identity in lower half.
    graph.add(TokenEmbed(
        name="tok_lower_identity",
        entries=[(k, k, 1.0) for k in range(V)],
    ))
    # LookUp: copy residual channels 0..V-1 from position 0 into channels V..2V-1.
    graph.add(LookUp(
        name="copy_from_pos_0",
        layer=0,
        v_source_channels=list(range(V)),
        out_channels=[V + k for k in range(V)],
    ))
    # head reads upper half.
    graph.add(LinearHead(
        name="read_upper",
        entries=[(j, V + j, 1.0) for j in range(V)],
    ))

    return compile_program(
        graph,
        d_model=2 * V,
        n_heads=V,
        n_layers=1,
        d_ffn=2 * V,
        max_len=32,
        vocab_size=V,
    )


if __name__ == "__main__":
    import torch
    V = 8
    model = build_copy_past_ir(vocab_size=V)
    print(f"[copy_past_ir] built Small2DTransformer, {model.param_count():,} params")
    all_ok = True
    for inp in ([3, 7, 2, 5, 1], [0, 1, 2, 3, 4, 5, 6, 7], [7, 0], [5]):
        x = torch.tensor([inp], dtype=torch.long)
        with torch.no_grad():
            got = model(x)[0].argmax(dim=-1).tolist()
        expected = [inp[0]] * len(inp)
        ok = got == expected
        all_ok = all_ok and ok
        print(f"  [{'ok' if ok else 'FAIL'}] input={inp} → {got} (expected {expected})")
    print(f"overall: {'PASS' if all_ok else 'FAIL'}")
