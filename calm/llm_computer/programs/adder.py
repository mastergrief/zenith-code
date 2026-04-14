"""2-digit adder — scales `adder_tiny` to `a, b ∈ [0, 99]`.

Same construction as `adder_tiny.py`: LookUp copies `a` from position 0,
ReGLU step-function pair per possible sum, LinearHead decodes step-
differences into a one-hot over the sum value. Only the dimensions
grow: `MAX_SUM = 198` needs 199 step functions (398 ReGLU neurons),
`d_model` = 202 residual channels, `vocab` = 200 tokens.

This is the proof of compositionality — the same IR that compiles 4
tiny primitives and a toy 1-digit adder also compiles the 10,000-case
2-digit adder with no compiler change. The only thing that grows is
the declaration.
"""

from __future__ import annotations

from calm.llm_computer.compile import compile_program
from calm.llm_computer.gate_graph import (
    GateGraph, LinearHead, LookUp, PosEmbed, ReGLU, TokenEmbed,
)
from calm.llm_computer.model import Small2DTransformer


MAX_OPERAND = 99
MAX_SUM = MAX_OPERAND * 2  # 198
VOCAB = MAX_SUM + 2        # 200, spare slot 199 unused


def build_adder(max_len: int = 4) -> Small2DTransformer:
    V = VOCAB
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
    graph.add(LookUp(
        name="copy_a",
        layer=0,
        v_source_channels=[0],
        out_channels=[2],
    ))

    # FFN: MAX_SUM + 1 step functions, 2 × (MAX_SUM + 1) ReGLU neurons.
    for S in range(MAX_SUM + 1):
        graph.add(ReGLU(
            name=f"step_{S}_hi",
            layer=0,
            gate=[(0, 1.0), (2, 1.0), (1, -(S - 1))],
            val=[(1, 1.0)],
            output_channel=3 + S,
            output_coef=1.0,
        ))
        graph.add(ReGLU(
            name=f"step_{S}_lo",
            layer=0,
            gate=[(0, 1.0), (2, 1.0), (1, -S)],
            val=[(1, 1.0)],
            output_channel=3 + S,
            output_coef=-1.0,
        ))

    # Head: logits[k] = step_k - step_{k+1} (step_{MAX_SUM+1} = 0).
    head_entries = []
    for k in range(MAX_SUM + 1):
        head_entries.append((k, 3 + k, 1.0))
        if k + 1 <= MAX_SUM:
            head_entries.append((k, 3 + k + 1, -1.0))
    graph.add(LinearHead(name="onehot_via_steps", entries=head_entries))

    d_model = 3 + (MAX_SUM + 1)      # 202
    n_heads = d_model // 2           # 101
    d_ffn = 2 * (MAX_SUM + 1)        # 398

    return compile_program(
        graph,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=1,
        d_ffn=d_ffn,
        max_len=max_len,
        vocab_size=V,
    )


if __name__ == "__main__":
    import itertools
    import time
    import torch

    print("[adder] building model...")
    t0 = time.time()
    model = build_adder()
    t_build = time.time() - t0
    print(f"[adder] built in {t_build:.1f}s, {model.param_count():,} params")

    print(f"[adder] running all {(MAX_OPERAND + 1) ** 2} cases...")
    t0 = time.time()
    # Batch all 10K cases.
    pairs = list(itertools.product(range(MAX_OPERAND + 1), repeat=2))
    inputs = torch.tensor(pairs, dtype=torch.long)  # (10000, 2)
    with torch.no_grad():
        logits = model(inputs)  # (10000, 2, V)
        preds = logits[:, 1, :].argmax(dim=-1).tolist()
    t_run = time.time() - t0
    expected = [a + b for a, b in pairs]
    correct = sum(1 for p, e in zip(preds, expected) if p == e)
    print(f"[adder] ran in {t_run:.2f}s ({t_run * 1e6 / len(pairs):.1f}us/case)")
    print(f"[adder] {correct}/{len(pairs)} = {correct / len(pairs):.1%} correct")

    print("\nDemo (5x5 table, a x b):")
    print("     b=0  b=25  b=50  b=75  b=99")
    for a in (0, 25, 50, 75, 99):
        row = [f"a={a:2}"]
        for b in (0, 25, 50, 75, 99):
            x = torch.tensor([[a, b]], dtype=torch.long)
            with torch.no_grad():
                got = int(model(x)[0, 1].argmax().item())
            ok = "✓" if got == a + b else "✗"
            row.append(f"{got}{ok}")
        print("  " + " ".join(f"{c:>6}" for c in row))
