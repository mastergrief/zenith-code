"""Option 2 fusion MVP — prove compiled + learned can coexist in one substrate.

Step 1 — empty layer 0 + compiled adder at layer 1. If adder still hits
  16/16 exhaustive, the compiled program survives being wrapped in extra
  transformer layers. This is the minimal proof of fusion mechanics.

Step 2 — replace layer 0 with random trained weights. Confirm adder
  behavior degrades (if layer 0 writes to channels adder reads) and
  recovers when we constrain layer 0 to write only to unused channels.

Step 3 — stack two truly distinct programs: a learned 1-layer NL
  encoder (producing digit tokens at positions 0, 1) + compiled adder
  at layer 1 reading from those positions. Single forward pass,
  NL → answer.
"""

from __future__ import annotations

import torch

from calm.llm_computer.compile import compile_program
from calm.llm_computer.gate_graph import (
    GateGraph, LinearHead, LookUp, PosEmbed, ReGLU, TokenEmbed,
)


def _exhaustive_adder(model) -> int:
    correct = 0
    for a in range(4):
        for b in range(4):
            x = torch.tensor([[a, b]], dtype=torch.long)
            with torch.no_grad():
                got = int(model(x)[0, 1].argmax().item())
            correct += int(got == a + b)
    return correct


def build_adder_at_layer(target_layer: int, n_layers: int):
    """Build adder_tiny with all hardware nodes pinned to `target_layer`.

    This replicates adder_tiny's construction but puts LookUp + ReGLU
    in a configurable layer, so we can verify correctness when the
    compiled program lives at layer 1 of a 2-layer substrate (with
    layer 0 empty/learned/trained).
    """
    V = 8
    MAX_SUM = 6
    g = GateGraph(vocab_size=V)
    g.add(TokenEmbed(name="own_scalar",
                     entries=[(k, 0, float(k)) for k in range(V)]))
    g.add(PosEmbed(name="bias", entries=[(p, 1, 1.0) for p in range(4)]))
    g.add(LookUp(name="copy_a", layer=target_layer,
                  v_source_channels=[0], out_channels=[2]))
    for S in range(MAX_SUM + 1):
        g.add(ReGLU(name=f"step_{S}_hi", layer=target_layer,
                     gate=[(0, 1.0), (2, 1.0), (1, -(S - 1))],
                     val=[(1, 1.0)],
                     output_channel=3 + S, output_coef=1.0))
        g.add(ReGLU(name=f"step_{S}_lo", layer=target_layer,
                     gate=[(0, 1.0), (2, 1.0), (1, -S)],
                     val=[(1, 1.0)],
                     output_channel=3 + S, output_coef=-1.0))
    head = []
    for k in range(MAX_SUM + 1):
        head.append((k, 3 + k, 1.0))
        if k + 1 <= MAX_SUM:
            head.append((k, 3 + k + 1, -1.0))
    g.add(LinearHead(name="onehot", entries=head))

    return compile_program(g, d_model=10, n_heads=5, n_layers=n_layers,
                           d_ffn=14, max_len=4, vocab_size=V)


def step1_empty_layer0_plus_adder_layer1():
    """Layer 0 all zeros + compiled adder at layer 1. Expect 16/16."""
    print("\n=== step 1: empty layer 0 + compiled adder at layer 1 ===")
    model = build_adder_at_layer(target_layer=1, n_layers=2)
    acc = _exhaustive_adder(model)
    print(f"  {acc}/16 ({'PASS' if acc == 16 else 'FAIL'})")
    return acc


def step2_random_noise_layer0():
    """Layer 0 filled with small random noise. Expect degradation."""
    print("\n=== step 2: random-noise layer 0 + compiled adder at layer 1 ===")
    for sigma in (1e-6, 1e-4, 1e-2):
        torch.manual_seed(0)
        model = build_adder_at_layer(target_layer=1, n_layers=2)
        with torch.no_grad():
            for p in (model.W_qkv[0].weight, model.W_out[0].weight,
                      model.ff_in[0].weight, model.ff_out[0].weight):
                p.add_(torch.randn_like(p) * sigma)
        acc = _exhaustive_adder(model)
        print(f"  sigma={sigma:.0e}: {acc}/16")


def step3_trained_layer0_real_fusion():
    """Layer 0 trained on a real aux task (identity-copy), Layer 1 adder.

    The aux task: make layer 0 learn to preserve (a, b) at positions 0, 1.
    If we only supervise on the final answer, gradients flow through both
    layers. Layer 1's weights are initialized from the compiled adder.
    Question: can gradient descent on layer 0 IMPROVE behavior without
    breaking the compiled layer 1?
    """
    import torch.nn.functional as F
    print("\n=== step 3: trained layer 0 + compiled adder at layer 1 ===")

    torch.manual_seed(0)
    model = build_adder_at_layer(target_layer=1, n_layers=2)
    # Initialize layer 0 with small random weights (so it has something to learn from).
    sigma = 1e-4
    with torch.no_grad():
        for p in (model.W_qkv[0].weight, model.W_out[0].weight,
                  model.ff_in[0].weight, model.ff_out[0].weight):
            p.add_(torch.randn_like(p) * sigma)
    pre = _exhaustive_adder(model)

    # Freeze layer 1 (compiled adder). Only train layer 0 parameters + embeddings.
    layer1_params = []
    for name, p in model.named_parameters():
        if ".1." in name or name.startswith("head.") or name.startswith("tok.") or name.startswith("pos."):
            p.requires_grad = False
            layer1_params.append(name)
        else:
            p.requires_grad = True
    print(f"  frozen: {len(layer1_params)} tensors (layer 1 + embeddings + head)")

    # Train layer 0 to recover correctness.
    model.config.use_hard_max = False
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=5e-4,
    )
    all_cases = [(a, b) for a in range(4) for b in range(4)]
    for epoch in range(30):
        total = 0.0
        for a, b in all_cases:
            x = torch.tensor([[a, b]], dtype=torch.long)
            target = torch.tensor([a + b], dtype=torch.long)
            logits = model(x)[:, 1, :]
            loss = F.cross_entropy(logits, target)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if epoch % 10 == 9:
            model.config.use_hard_max = True
            post = _exhaustive_adder(model)
            print(f"  epoch {epoch+1}: loss={total/16:.4f}, hard-max acc={post}/16")
            model.config.use_hard_max = False

    model.config.use_hard_max = True
    post = _exhaustive_adder(model)
    print(f"  pre-train layer 0: {pre}/16")
    print(f"  post-train layer 0: {post}/16")


if __name__ == "__main__":
    step1_empty_layer0_plus_adder_layer1()
    step2_random_noise_layer0()
    step3_trained_layer0_real_fusion()
