"""factorial — compile-time factorial lookup via gate-graph IR.

Program: pos 0 = n ∈ [0, MAX_N]. Output at pos 0: argmax logits[n!].

Construction: single-input step-diff decode.
  - TokenEmbed ch 0 = token scalar
  - PosEmbed ch 1 = bias 1
  - Layer 0 FFN: 2·(MAX_N+1) step ReGLUs decode n into indicators
    step_k = 1[n >= k] in ch [2, 2 + MAX_N]
  - LinearHead: logits[factorial(k)] += step_k - step_{k+1} for each k

No attention needed — all attention weights zero (no-op). The head
accumulates +1 at logits[n!] because for input n, step_n - step_{n+1} = 1
and all other step-diffs are 0.

Vocab must cover both inputs ([0, MAX_N]) and outputs ([1, MAX_N!]). For
MAX_N=8 that's vocab = 9! + 1 = 362881 — large but only a few slots
carry nonzero weights.
"""

from __future__ import annotations

import math

from calm.llm_computer.compile import compile_program
from calm.llm_computer.gate_graph import (
    GateGraph, LinearHead, PosEmbed, ReGLU, TokenEmbed,
)
from calm.llm_computer.model import Small2DTransformer
from calm.llm_computer.schedule import auto_schedule


MAX_N = 8
VOCAB = math.factorial(MAX_N) + 1   # 40321


def build_factorial(max_len: int = 2) -> Small2DTransformer:
    V = VOCAB
    graph = GateGraph(vocab_size=V)

    # Token embedding: only input slots [0, MAX_N] carry the identity.
    graph.add(TokenEmbed(
        name="own_scalar",
        entries=[(k, 0, float(k)) for k in range(MAX_N + 1)],
    ))
    graph.add(PosEmbed(
        name="bias",
        entries=[(p, 1, 1.0) for p in range(max_len)],
    ))
    # Step decode: step_k = 1[n >= k] for k in [0, MAX_N].
    for k in range(MAX_N + 1):
        graph.add(ReGLU(
            name=f"step_{k}_hi",
            gate=[(0, 1.0), (1, -(k - 1))],
            val=[(1, 1.0)],
            output_channel=2 + k,
            output_coef=1.0,
        ))
        graph.add(ReGLU(
            name=f"step_{k}_lo",
            gate=[(0, 1.0), (1, -k)],
            val=[(1, 1.0)],
            output_channel=2 + k,
            output_coef=-1.0,
        ))
    head_entries = []
    for k in range(MAX_N + 1):
        fk = math.factorial(k)
        head_entries.append((fk, 2 + k, 1.0))
        if k + 1 <= MAX_N:
            head_entries.append((fk, 2 + k + 1, -1.0))
    graph.add(LinearHead(name="factorial_head", entries=head_entries))

    n_layers = auto_schedule(graph)

    d_model = 2 + (MAX_N + 1)
    if d_model % 2 != 0:
        d_model += 1
    n_heads = d_model // 2
    d_ffn = 2 * (MAX_N + 1)
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
    import time
    import torch

    t0 = time.time()
    model = build_factorial()
    t_build = time.time() - t0
    print(f"[factorial] built in {t_build:.1f}s, {model.param_count():,} params")

    inputs = torch.tensor([[n] for n in range(MAX_N + 1)], dtype=torch.long)
    t0 = time.time()
    with torch.no_grad():
        logits = model(inputs)
        preds = logits[:, 0, :].argmax(dim=-1).tolist()
    t_run = time.time() - t0
    expected = [math.factorial(n) for n in range(MAX_N + 1)]
    correct = sum(1 for p, e in zip(preds, expected) if p == e)
    print(f"[factorial] ran in {t_run:.3f}s")
    print(f"[factorial] {correct}/{len(expected)} correct")
    for n, p, e in zip(range(MAX_N + 1), preds, expected):
        ok = "✓" if p == e else "✗"
        print(f"  {n}! = {p}  (expected {e}) {ok}")
