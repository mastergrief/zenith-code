"""2-operand compiled multiplier — `a × b`, operand range [0, 99],
product range [0, 999]. MAX_PRODUCT capped at 999 to keep d_model
tractable on 8 GB VRAM; covers the observed Gemma failures (17×23=391,
47×19=893) but not 99×99=9801.

Construction:

  Layer 0 attention:  LookUp copies a from position 0 into ch_a at
                      position 1 (same pattern as adder).
  Layer 0 FFN:        One ReGLU computes `a · ReLU(b) = a·b` (docstring
                      of ReGLU cites this exact trick for integer
                      multiplication). Writes product into ch_prod.
  Layer 1 FFN:        2 × (MAX_PRODUCT + 1) ReGLU step functions over
                      ch_prod. ch_step[P] fires = 1 for prod ≥ P.
  Head:               logit[k] = ch_step[k] − ch_step[k+1] (indicator
                      of prod == k, same as adder).

Size: d_model = 1004, n_heads = 502, d_ffn = 2000, vocab = 1000,
n_layers = 2. ~88 MB FP32. Fits comfortably alongside Gemma on 8 GB.
"""

from __future__ import annotations

from calm.llm_computer.compile import compile_program
from calm.llm_computer.gate_graph import (
    GateGraph, LinearHead, LookUp, PosEmbed, ReGLU, TokenEmbed,
)
from calm.llm_computer.model import Small2DTransformer


MAX_OPERAND = 99
MAX_PRODUCT = 999     # capped; products >= 1000 produce undefined output
VOCAB = MAX_PRODUCT + 1  # 1000 — covers both operand and product tokens


def build_multiplier(max_len: int = 4) -> Small2DTransformer:
    V = VOCAB
    CH_OWN = 0      # token scalar at current position (b at position 1)
    CH_BIAS = 1     # constant 1 from PosEmbed
    CH_A = 2        # copy of pos-0's CH_OWN (= a)
    CH_PROD = 3     # a * b (computed in layer 0 FFN)
    CH_STEP = 4     # start of step-indicator channels

    graph = GateGraph(vocab_size=V)

    # ch 0: per-position token scalar (a at pos 0, b at pos 1).
    graph.add(TokenEmbed(
        name="own_scalar",
        entries=[(k, CH_OWN, float(k)) for k in range(V)],
    ))
    # ch 1: bias = 1 everywhere.
    graph.add(PosEmbed(
        name="bias",
        entries=[(p, CH_BIAS, 1.0) for p in range(max_len)],
    ))
    # Layer 0 attention: copy pos-0's CH_OWN into CH_A at every query.
    graph.add(LookUp(
        name="copy_a",
        layer=0,
        v_source_channels=[CH_OWN],
        out_channels=[CH_A],
    ))

    # Layer 0 FFN: one ReGLU, output = a · ReLU(b) = a·b (for b ≥ 0).
    graph.add(ReGLU(
        name="product",
        layer=0,
        gate=[(CH_OWN, 1.0)],   # gate = b
        val=[(CH_A, 1.0)],       # val = a
        output_channel=CH_PROD,
        output_coef=1.0,
    ))

    # Layer 1 FFN: step-function indicators over CH_PROD.
    # For each threshold P: ch_step[P] = 1 if prod ≥ P else 0.
    for P in range(MAX_PRODUCT + 1):
        graph.add(ReGLU(
            name=f"step_{P}_hi",
            layer=1,
            gate=[(CH_PROD, 1.0), (CH_BIAS, -(P - 1))],  # prod - (P-1)
            val=[(CH_BIAS, 1.0)],
            output_channel=CH_STEP + P,
            output_coef=1.0,
        ))
        graph.add(ReGLU(
            name=f"step_{P}_lo",
            layer=1,
            gate=[(CH_PROD, 1.0), (CH_BIAS, -P)],        # prod - P
            val=[(CH_BIAS, 1.0)],
            output_channel=CH_STEP + P,
            output_coef=-1.0,
        ))

    # Head: logit[k] = ch_step[k] - ch_step[k+1] = indicator(prod == k).
    head_entries = []
    for k in range(MAX_PRODUCT + 1):
        head_entries.append((k, CH_STEP + k, 1.0))
        if k + 1 <= MAX_PRODUCT:
            head_entries.append((k, CH_STEP + k + 1, -1.0))
    graph.add(LinearHead(name="onehot_via_steps", entries=head_entries))

    d_model = CH_STEP + (MAX_PRODUCT + 1)  # 4 + 1000 = 1004
    n_heads = d_model // 2                  # 502
    d_ffn = 2 * (MAX_PRODUCT + 1)           # 2000

    return compile_program(
        graph,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=2,
        d_ffn=d_ffn,
        max_len=max_len,
        vocab_size=V,
    )


if __name__ == "__main__":
    import itertools
    import time
    import torch

    print("[multiplier] building model...")
    t0 = time.time()
    model = build_multiplier()
    t_build = time.time() - t0
    print(f"[multiplier] built in {t_build:.1f}s, "
          f"{model.param_count():,} params")

    # Exhaustive over in-range products (< 1000).
    pairs = [(a, b) for a in range(MAX_OPERAND + 1)
             for b in range(MAX_OPERAND + 1)
             if a * b <= MAX_PRODUCT]
    print(f"[multiplier] running {len(pairs)} in-range cases...")
    t0 = time.time()
    inputs = torch.tensor(pairs, dtype=torch.long)
    with torch.no_grad():
        logits = model(inputs)
        preds = logits[:, 1, :].argmax(dim=-1).tolist()
    t_run = time.time() - t0
    expected = [a * b for a, b in pairs]
    correct = sum(1 for p, e in zip(preds, expected) if p == e)
    print(f"[multiplier] ran in {t_run:.2f}s "
          f"({t_run * 1e6 / max(len(pairs), 1):.1f}us/case)")
    print(f"[multiplier] {correct}/{len(pairs)} = "
          f"{correct / max(len(pairs), 1):.1%} correct")

    # Show a few specific cases including Gemma's failures.
    print("\nSpot checks (including Gemma's documented failures):")
    for a, b in [(17, 23), (47, 19), (2, 3), (5, 5), (10, 10),
                  (99, 10), (31, 31), (0, 99), (1, 99)]:
        if a * b > MAX_PRODUCT:
            status = f"(out of range, prod={a*b})"
        else:
            x = torch.tensor([[a, b]], dtype=torch.long)
            with torch.no_grad():
                got = int(model(x)[0, 1].argmax().item())
            status = f"got {got} {'✓' if got == a*b else '✗ (exp ' + str(a*b) + ')'}"
        print(f"  {a:2} × {b:2} = {a*b:<5}  {status}")
