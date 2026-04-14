"""gcd — compile-time Euclidean GCD via gate-graph IR.

Program: pos 0 = a, pos 1 = b, both tokens in [0, MAX_OP]. Output at
pos 1: argmax logits[gcd(a, b)].

Construction (step-diff LUT over binary key):
  - TokenEmbed ch 0 = token scalar (a at pos 0, b at pos 1)
  - PosEmbed ch 1 = bias 1
  - LookUp copies pos 0's ch 0 into ch 2 at pos 1  (so at pos 1, ch 2 = a)
  - Layer 0 FFN: two ReGLUs compute key = (MAX_OP+1)*a + b into ch 3
  - Layer 1 FFN: 2*(MAX_KEY+1) step ReGLUs decode key into per-pair
    indicators step_k = 1[key >= k] in channels [4, 4 + MAX_KEY]
  - LinearHead: for each (A, B) pair, logits[gcd(A, B)] gets
      +step_{BASE*A+B} - step_{BASE*A+B+1}
    so the unique matching pair fires exactly +1 at logits[gcd(a, b)].

Not a learned or recursive Euclidean — this is a compiled lookup table,
the same pattern as `adder.py` extended to a 2D key. Purpose: validate
the existing IR compiles arbitrary binary-input backends with no new
primitives. Auto-scheduler places ReGLUs across layers.
"""

from __future__ import annotations

import math

from calm.llm_computer.compile import compile_program
from calm.llm_computer.gate_graph import (
    GateGraph, LinearHead, LookUp, PosEmbed, ReGLU, TokenEmbed,
)
from calm.llm_computer.model import Small2DTransformer
from calm.llm_computer.schedule import auto_schedule


MAX_OP = 15
BASE = MAX_OP + 1          # 16
MAX_KEY = BASE * BASE - 1  # 255
VOCAB = BASE               # 16 — input tokens AND output tokens fit


def build_gcd(max_len: int = 4) -> Small2DTransformer:
    V = VOCAB
    graph = GateGraph(vocab_size=V)

    graph.add(TokenEmbed(
        name="own_scalar",
        entries=[(k, 0, float(k)) for k in range(V)],
    ))
    graph.add(PosEmbed(
        name="bias",
        entries=[(p, 1, 1.0) for p in range(max_len)],
    ))
    graph.add(LookUp(
        name="copy_a",
        v_source_channels=[0],
        out_channels=[2],
    ))
    # key = BASE * a + b. Two ReGLUs both writing to ch 3.
    graph.add(ReGLU(
        name="scale_a",
        gate=[(1, 1.0)],
        val=[(2, 1.0)],
        output_channel=3,
        output_coef=float(BASE),
    ))
    graph.add(ReGLU(
        name="add_b",
        gate=[(1, 1.0)],
        val=[(0, 1.0)],
        output_channel=3,
        output_coef=1.0,
    ))
    # Step decode: step_k = 1[key >= k], two ReGLUs per k.
    for k in range(MAX_KEY + 1):
        graph.add(ReGLU(
            name=f"step_{k}_hi",
            gate=[(3, 1.0), (1, -(k - 1))],
            val=[(1, 1.0)],
            output_channel=4 + k,
            output_coef=1.0,
        ))
        graph.add(ReGLU(
            name=f"step_{k}_lo",
            gate=[(3, 1.0), (1, -k)],
            val=[(1, 1.0)],
            output_channel=4 + k,
            output_coef=-1.0,
        ))
    # Head: logits[gcd(A, B)] = sum over (A, B) of step_{k} - step_{k+1}
    # where k = BASE * A + B. Only the matching pair (A, B) = (a, b)
    # contributes +1; all others cancel to 0.
    head_entries = []
    for A in range(BASE):
        for B in range(BASE):
            k = BASE * A + B
            g = math.gcd(A, B)
            head_entries.append((g, 4 + k, 1.0))
            if k + 1 <= MAX_KEY:
                head_entries.append((g, 4 + k + 1, -1.0))
    graph.add(LinearHead(name="gcd_head", entries=head_entries))

    n_layers = auto_schedule(graph)

    d_model = 4 + (MAX_KEY + 1)       # 260
    n_heads = d_model // 2            # 130
    d_ffn = 2 * (MAX_KEY + 1)         # 512
    return compile_program(
        graph,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ffn=d_ffn,
        max_len=max_len,
        vocab_size=V,
    )


if __name__ == "__main__":
    import itertools
    import time
    import torch

    t0 = time.time()
    model = build_gcd()
    t_build = time.time() - t0
    print(f"[gcd] built in {t_build:.1f}s, {model.param_count():,} params")

    pairs = list(itertools.product(range(BASE), repeat=2))
    inputs = torch.tensor(pairs, dtype=torch.long)  # (256, 2)
    t0 = time.time()
    with torch.no_grad():
        logits = model(inputs)
        preds = logits[:, 1, :].argmax(dim=-1).tolist()
    t_run = time.time() - t0
    expected = [math.gcd(a, b) for a, b in pairs]
    correct = sum(1 for p, e in zip(preds, expected) if p == e)
    print(f"[gcd] ran in {t_run:.2f}s ({t_run * 1e6 / len(pairs):.1f}us/case)")
    print(f"[gcd] {correct}/{len(pairs)} = {correct / len(pairs):.1%} correct")

    print("\nDemo (6x6 table, a x b):")
    cols = (0, 3, 6, 9, 12, 15)
    header = "    " + " ".join(f"b={b:>2}" for b in cols)
    print(header)
    for a in cols:
        row = [f"a={a:>2}"]
        for b in cols:
            x = torch.tensor([[a, b]], dtype=torch.long)
            with torch.no_grad():
                got = int(model(x)[0, 1].argmax().item())
            ok = "✓" if got == math.gcd(a, b) else "✗"
            row.append(f"{got:>2}{ok}")
        print("  " + " ".join(f"{c}" for c in row))
