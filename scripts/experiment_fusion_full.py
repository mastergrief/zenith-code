"""Full fusion demo — learned layer 0 + compiled adder at layer 1.

Task:  input `[A, B]` tokens at positions 0, 1.  A, B ∈ {0, 1, 2, 3}.
Target: A + B as a token at position 1.

Architecture (single Small2DTransformer, 2 layers):
  Layer 0 (learned, random init, trainable): self-attention that learns
    to copy own-token scalar (ch 0) into channel 5 at each position.
    Uses softmax so gradients flow. Purely local — compatible with
    causal mask.
  Layer 1 (compiled, frozen): modified adder that reads A from position
    0 (via LookUp into channel 2) and B from channel 5 at position 1
    (written by layer 0). Step-decodes A+B.

Residual channels (d_model = 16):
  0   — own_scalar (TokenEmbed)
  1   — bias 1 (PosEmbed)
  2   — copy_a (LookUp write at layer 1)
  3-9 — step_S channels (layer 1 FFN)
  5   — "extracted B" slot; written by layer 0
  10-15 — unused

Before training: layer 0 all-zero, so ch 5 at pos 1 = 0, adder decodes
A + 0 = A. Only 4/16 correct (cases where B=0 → answer=A and A=A, so
actually 16/16 *by coincidence* — wait no, the step decode gives A as
output, and the true answer is A+B; when B=0 answer is A, so yes 4/16
correct only when B=0).

After training: if layer 0 learns to copy ch 0 → ch 5, ch 5 at pos 1
becomes B, adder decodes A + B → correct. Target: ≥14/16.

This is a MEANINGFUL fusion test: the compiled adder alone cannot
solve the task because it reads ch 5 (which only layer 0 populates).
The learned layer 0 is required for end-to-end correctness.
"""

from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F

from calm.llm_computer.compile import compile_program
from calm.llm_computer.gate_graph import (
    GateGraph, LinearHead, LookUp, PosEmbed, ReGLU, TokenEmbed,
)


VOCAB = 8
D_MODEL = 16
N_HEADS = 8
D_FFN = 16
N_LAYERS = 2
MAX_LEN = 2
MAX_SUM = 6


def build_fused_skeleton():
    """Layer 1 contains the full adder; layer 0 starts empty (to be trained)."""
    V = VOCAB
    g = GateGraph(vocab_size=V)
    # Channel 5 is the "extracted B" slot — layer 0 writes it, layer 1 reads it.
    g.add(TokenEmbed(name="own_scalar",
                     entries=[(k, 0, float(k)) for k in range(V)]))
    g.add(PosEmbed(name="bias", entries=[(p, 1, 1.0) for p in range(MAX_LEN)]))

    # Layer 1: LookUp copies A from pos 0's ch 0 to ch 2 at all query positions.
    g.add(LookUp(name="copy_a", layer=1,
                  v_source_channels=[0], out_channels=[2]))
    # Layer 1: step_S = 1[a + b_extracted >= S] where a = ch 2, b = ch 5.
    for S in range(MAX_SUM + 1):
        g.add(ReGLU(name=f"step_{S}_hi", layer=1,
                     gate=[(5, 1.0), (2, 1.0), (1, -(S - 1))],  # b_from_ch5 + a_from_ch2 - (S-1)
                     val=[(1, 1.0)],
                     output_channel=3 + S, output_coef=1.0))
        g.add(ReGLU(name=f"step_{S}_lo", layer=1,
                     gate=[(5, 1.0), (2, 1.0), (1, -S)],
                     val=[(1, 1.0)],
                     output_channel=3 + S, output_coef=-1.0))
    head = []
    for k in range(MAX_SUM + 1):
        head.append((k, 3 + k, 1.0))
        if k + 1 <= MAX_SUM:
            head.append((k, 3 + k + 1, -1.0))
    g.add(LinearHead(name="onehot", entries=head))

    return compile_program(g, d_model=D_MODEL, n_heads=N_HEADS,
                           n_layers=N_LAYERS, d_ffn=D_FFN,
                           max_len=MAX_LEN, vocab_size=V)


def _all_cases():
    return [(a, b) for a in range(4) for b in range(4)]


def _exhaustive(model, hard_max: bool) -> int:
    orig = model.config.use_hard_max
    model.config.use_hard_max = hard_max
    correct = 0
    total = 0
    with torch.no_grad():
        for a, b in _all_cases():
            inp = torch.tensor([[a, b]], dtype=torch.long)
            pred = int(model(inp)[0, 1].argmax().item())
            total += 1
            correct += int(pred == a + b)
    model.config.use_hard_max = orig
    return correct, total


def main():
    torch.manual_seed(0)
    model = build_fused_skeleton()

    # Before training: layer 0 all zero → position 1's ch 5 = 0 → adder sees
    # A + 0 = A. Compute expected accuracy.
    pre_hard_correct, total = _exhaustive(model, hard_max=True)
    pre_soft_correct, _ = _exhaustive(model, hard_max=False)
    print(f"[fusion] pre-train: hard-max {pre_hard_correct}/{total}, "
          f"softmax {pre_soft_correct}/{total}")

    # Initialize layer 0 with small random weights for gradient symmetry breaking.
    sigma = 1e-2
    with torch.no_grad():
        for p in (model.W_qkv[0].weight, model.W_out[0].weight,
                  model.ff_in[0].weight, model.ff_out[0].weight):
            p.add_(torch.randn_like(p) * sigma)

    # Freeze everything except layer 0's attention + FFN params.
    frozen = []
    trainable = []
    for name, p in model.named_parameters():
        if ".0." in name and ("W_qkv" in name or "W_out" in name
                               or "ff_in" in name or "ff_out" in name):
            p.requires_grad = True
            trainable.append(name)
        else:
            p.requires_grad = False
            frozen.append(name)
    print(f"[fusion] trainable: {trainable}")
    print(f"[fusion] frozen   : {len(frozen)} tensors")

    # Train with softmax attention (hard-max argmax is non-differentiable).
    model.config.use_hard_max = False
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=5e-3,
    )
    cases = _all_cases()
    for epoch in range(300):
        total_loss = 0.0
        for a, b in cases:
            inp = torch.tensor([[a, b]], dtype=torch.long)
            target = torch.tensor([a + b], dtype=torch.long)
            logits = model(inp)[:, 1, :]
            loss = F.cross_entropy(logits, target)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 50 == 0:
            hc, _ = _exhaustive(model, hard_max=True)
            sc, _ = _exhaustive(model, hard_max=False)
            print(f"  epoch {epoch+1:3d}: loss={total_loss/len(cases):.4f}, "
                  f"hard-max {hc}/{total}, softmax {sc}/{total}")

    hc, _ = _exhaustive(model, hard_max=True)
    sc, _ = _exhaustive(model, hard_max=False)
    print(f"\n[fusion] FINAL: hard-max {hc}/{total}, softmax {sc}/{total}")
    if hc >= int(0.9 * total):
        print(f"[fusion] ✓ PASS — fused learned+compiled model generalizes "
              f"to distractor-token task")
    else:
        print(f"[fusion] ⚠ partial — {hc}/{total}, diagnose layer 0 behavior")

    # Spot check a few cases for visibility.
    print("\n[fusion] sample predictions (hard-max):")
    model.config.use_hard_max = True
    for a, b in [(0, 0), (1, 2), (3, 3), (2, 1)]:
        inp = torch.tensor([[a, b]], dtype=torch.long)
        with torch.no_grad():
            pred = int(model(inp)[0, 1].argmax().item())
        mark = "✓" if pred == a + b else "✗"
        print(f"  {a} + {b} → {pred} (expected {a+b}) {mark}")


if __name__ == "__main__":
    main()
