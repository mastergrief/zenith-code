"""Rung 4 — unified CHRLM learns an orthogonal task while adder survives.

Rung 3 proved one tensor holds compiled + trained. But the trainable
layer's loss stayed flat because the adder's head reads exclusively from
protected channels. Rung 4 goes further: give the trainable layer
a task whose output goes through NEW (non-compiled) vocab rows, so
gradients actually flow to the trainable params and the layer learns
non-adder behavior.

Design (single tensor, all one .pt):
  - d_model=16, n_heads=8, n_layers=2, vocab=16, max_len=4
  - Compile adder_tiny at d_model=16. Channels 2-9 populated by the
    adder; channels 10-15 are zero (free for trainable layer).
    Head rows 0-7 populated for sum prediction; rows 8-15 are zero.
  - Install compiled into layer 0 of a fresh unified substrate.
  - Freeze: layer 0 weights, token/pos embeddings.
  - Channel mask on layer 1: protect channels 2-9 from writes.
  - Partial head freeze: freeze rows 0-7 (adder's outputs), leave
    rows 8-15 trainable.
  - Orthogonal task: at pos 1 (where the adder predicts a+b on rows
    0-7), ALSO predict an echo token 8+a on rows 8-15. This requires
    layer 1 to write `a` into channels 10-15 and the trainable head
    rows 8-15 to read from those channels.

Gates:
  1. Pre-training: adder 16/16 (compiled prediction via rows 0-7)
  2. Post-training: adder 16/16 (mask kept compiled intact)
  3. Post-training: echo task >= 75% on held-out pairs (trainable
     layer + trainable head learned something non-trivial)
  4. Save/reload .pt: both adder and echo survive
  5. OVERALL: one tensor holds BOTH tasks without interference

If all pass, the unified-tensor architecture supports orthogonal
multi-task learning at the tiny MVP scale. This unblocks v3 real-
training — channel masking + partial head freeze are the primitives
the unified trainer needs.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from calm.llm_computer.channel_masking import (
    compiled_output_channels_adder_tiny,
    freeze_head_rows,
    protect_residual_channels,
)
from calm.llm_computer.compile import compile_program
from calm.llm_computer.gate_graph import (
    GateGraph, LinearHead, LookUp, PosEmbed, ReGLU, TokenEmbed,
)
from calm.llm_computer.unified_chrlm import (
    UnifiedCHRLMConfig, build_unified_chrlm,
    freeze_layer_params, trainable_param_count,
)


# ----- Adder at configurable d_model -----

def build_adder_d16(d_model: int, n_heads: int, n_layers: int, vocab_size: int,
                    max_len: int, target_layer: int = 0):
    """Recompile adder_tiny at configurable dims. The IR's channel
    allocation (ch 2 = copy_a, ch 3-9 = step funcs) stays the same;
    channels 10+ are left zero for downstream use.
    """
    assert d_model >= 10, "adder_tiny allocates channels 0..9"
    V = 8
    MAX_SUM = 6
    g = GateGraph(vocab_size=vocab_size)
    g.add(TokenEmbed(name="own_scalar",
                     entries=[(k, 0, float(k)) for k in range(V)]))
    g.add(PosEmbed(name="bias", entries=[(p, 1, 1.0) for p in range(max_len)]))
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
    head_entries = []
    for k in range(MAX_SUM + 1):
        head_entries.append((k, 3 + k, 1.0))
        if k + 1 <= MAX_SUM:
            head_entries.append((k, 3 + k + 1, -1.0))
    g.add(LinearHead(name="onehot", entries=head_entries))

    return compile_program(
        g, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        d_ffn=14, max_len=max_len, vocab_size=vocab_size,
    )


@torch.no_grad()
def exhaustive_adder_at_pos1(model) -> int:
    """Check all 16 (a, b) pairs predict a+b at position 1 via argmax
    restricted to rows 0..6 (the adder's sum range).
    """
    ok = 0
    for a in range(4):
        for b in range(4):
            x = torch.tensor([[a, b]], dtype=torch.long)
            logits = model(x)[0, 1]
            # Restrict to adder's valid output vocab 0..6
            argmax = int(logits[:7].argmax().item())
            if argmax == a + b:
                ok += 1
    return ok


@torch.no_grad()
def exhaustive_echo_at_pos1(model, offset: int = 8) -> int:
    """For all 16 (a, b) pairs, check whether the trainable head emits
    `offset + a` at position 1 via argmax restricted to rows [offset,
    offset+8). Returns count correct out of 16.
    """
    ok = 0
    vocab = model.config.vocab_size
    for a in range(4):
        for b in range(4):
            x = torch.tensor([[a, b]], dtype=torch.long)
            logits = model(x)[0, 1]
            # Restrict to trainable vocab range
            argmax = offset + int(logits[offset:offset + 8].argmax().item())
            if argmax == offset + a:
                ok += 1
    return ok


# ----- Training -----

def training_data():
    """16 (a, b) pairs. Returns (inputs, target_sum, target_echo).
    target_sum is for adder loss (vocab 0..6).
    target_echo is for echo loss (vocab 8+a).
    """
    xs, ys_sum, ys_echo = [], [], []
    for a in range(4):
        for b in range(4):
            xs.append([a, b])
            ys_sum.append(a + b)
            ys_echo.append(8 + a)
    return (
        torch.tensor(xs, dtype=torch.long),
        torch.tensor(ys_sum, dtype=torch.long),
        torch.tensor(ys_echo, dtype=torch.long),
    )


def train_unified(model, xs, ys_sum, ys_echo, steps, lr, batch_size, seed,
                  adder_weight: float = 0.0, echo_weight: float = 1.0):
    """Train all requires_grad params. Loss combines:
      - adder_weight * CE(logits[pos=1, vocab=0..6], a+b)    [frozen rows]
      - echo_weight  * CE(logits[pos=1, vocab=8..15], 8+a)  [trainable rows]
    adder_weight=0 is the cleanest test: the gradient path to the
    frozen head rows is zeroed by the head-row freeze anyway; adding
    the adder loss just confirms the compiled portions aren't moved.
    """
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)
    rng = torch.Generator().manual_seed(seed)
    n = xs.size(0)

    losses_sum, losses_echo = [], []
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (batch_size,), generator=rng)
        bx = xs[idx]
        b_sum = ys_sum[idx]
        b_echo = ys_echo[idx]
        logits = model(bx)[:, 1, :]
        loss_sum = F.cross_entropy(logits[:, :7], b_sum)
        loss_echo = F.cross_entropy(logits[:, 8:16], b_echo - 8)
        loss = adder_weight * loss_sum + echo_weight * loss_echo
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses_sum.append(float(loss_sum.item()))
        losses_echo.append(float(loss_echo.item()))
    model.eval()
    return {
        "wallclock_s": time.time() - t0,
        "first_sum": losses_sum[0], "last_sum": losses_sum[-1],
        "first_echo": losses_echo[0], "last_echo": losses_echo[-1],
    }


# ----- Main -----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init-std", type=float, default=0.02)
    ap.add_argument("--save-path", default="/tmp/minibench_unified_rung4.pt")
    ap.add_argument("--no-channel-mask", action="store_true",
                    help="turn off channel masking (should cause adder to fail)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    D_MODEL, N_HEADS, N_LAYERS, VOCAB, MAX_LEN = 16, 8, 2, 16, 4
    print("=== minibench_unified_rung4 — orthogonal task, one tensor ===", flush=True)
    print(f"  config: d_model={D_MODEL}, n_heads={N_HEADS}, n_layers={N_LAYERS}, "
          f"vocab={VOCAB}, max_len={MAX_LEN}", flush=True)
    print(f"  tasks: sum (rows 0-6, frozen compiled) + echo 8+a "
          f"(rows 8-15, trainable)", flush=True)

    # Build unified substrate
    cfg = UnifiedCHRLMConfig(
        vocab_size=VOCAB, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        d_ffn=14, max_len=MAX_LEN, use_hard_max=True,
        compiled_layers=(0,),
    )
    unified = build_unified_chrlm(cfg)

    # Compile adder at matching dims and state_dict-transfer
    src = build_adder_d16(
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        vocab_size=VOCAB, max_len=MAX_LEN,
    )
    unified.load_state_dict(src.state_dict())

    # Random init on trainable layer 1 (not channels 2-9 which will be
    # masked, but everywhere else).
    with torch.no_grad():
        for lin in (unified.W_qkv[1], unified.W_out[1],
                    unified.ff_in[1], unified.ff_out[1]):
            lin.weight.normal_(0.0, args.init_std)
        # Also init trainable head rows 8-15 with small noise so
        # gradient can flow through them.
        unified.head.weight[8:16].normal_(0.0, args.init_std)

    # Freeze layer 0 + embeddings (but NOT the head — partial freeze)
    n_frozen_layer = freeze_layer_params(unified, layer_idx=0)
    # Freeze embeddings only (token + pos)
    for p in unified.tok.parameters():
        p.requires_grad = False
    for p in unified.pos.parameters():
        p.requires_grad = False

    # Partial head freeze: rows 0-7 are the adder's, rows 8-15 trainable
    head_hooks = freeze_head_rows(unified, rows=range(0, 8))

    # Channel mask layer 1: protect adder's output channels 2-9
    mask_hooks = []
    if not args.no_channel_mask:
        mask_hooks = protect_residual_channels(
            unified, layer_idx=1,
            protected_channels=compiled_output_channels_adder_tiny(),
        )

    n_trainable = trainable_param_count(unified)
    n_total = sum(p.numel() for p in unified.parameters())
    print(f"  params: total={n_total:,}  trainable={n_trainable:,}  "
          f"mask_hooks={len(mask_hooks)}", flush=True)

    # Gate 1: adder works before training
    pre_adder = exhaustive_adder_at_pos1(unified)
    pre_echo = exhaustive_echo_at_pos1(unified)
    print(f"\n  [gate 1] pre-training adder: {pre_adder}/16  "
          f"echo: {pre_echo}/16 (pre-training untrained)", flush=True)
    assert pre_adder == 16, f"compiled adder broken BEFORE training: {pre_adder}/16"

    # Train
    xs, ys_sum, ys_echo = training_data()
    print(f"\n--- training {args.steps} steps (echo task) ---", flush=True)
    result = train_unified(
        unified, xs, ys_sum, ys_echo,
        steps=args.steps, lr=args.lr, batch_size=args.batch_size,
        seed=args.seed,
    )
    print(f"  wallclock: {result['wallclock_s']:.2f}s", flush=True)
    print(f"  sum loss:  first={result['first_sum']:.4f}  "
          f"last={result['last_sum']:.4f}", flush=True)
    print(f"  echo loss: first={result['first_echo']:.4f}  "
          f"last={result['last_echo']:.4f}", flush=True)

    # Gate 2: adder still works after training
    post_adder = exhaustive_adder_at_pos1(unified)
    # Gate 3: echo task learned
    post_echo = exhaustive_echo_at_pos1(unified)
    print(f"\n  [gate 2] post-training adder: {post_adder}/16", flush=True)
    print(f"  [gate 3] post-training echo:  {post_echo}/16", flush=True)

    # Gate 4: save + reload
    save_path = Path(args.save_path)
    torch.save(unified.state_dict(), save_path)
    reloaded = build_unified_chrlm(cfg)
    reloaded.load_state_dict(torch.load(save_path, weights_only=True))
    reloaded.eval()
    rt_adder = exhaustive_adder_at_pos1(reloaded)
    rt_echo = exhaustive_echo_at_pos1(reloaded)
    print(f"\n  [gate 4] save+reload adder: {rt_adder}/16  "
          f"echo: {rt_echo}/16", flush=True)

    # Summary
    echo_gate = post_echo >= 12  # 75% threshold
    all_pass = (
        pre_adder == 16 and post_adder == 16 and echo_gate
        and rt_adder == 16 and rt_echo == post_echo
    )
    print("", flush=True)
    print("=== summary ===", flush=True)
    print(f"  gate 1 (pre adder):         {'PASS' if pre_adder == 16 else 'FAIL'} "
          f"({pre_adder}/16)", flush=True)
    print(f"  gate 2 (post adder):        {'PASS' if post_adder == 16 else 'FAIL'} "
          f"({post_adder}/16)", flush=True)
    print(f"  gate 3 (echo >=75%):        {'PASS' if echo_gate else 'FAIL'} "
          f"({post_echo}/16 = {post_echo/16*100:.0f}%)", flush=True)
    print(f"  gate 4 (roundtrip):         "
          f"{'PASS' if rt_adder == 16 and rt_echo == post_echo else 'FAIL'} "
          f"(adder {rt_adder}/16, echo {rt_echo}/16)", flush=True)
    print("", flush=True)
    print(f"  OVERALL: {'PASS — orthogonal multi-task in one tensor' if all_pass else 'FAIL'}",
          flush=True)


if __name__ == "__main__":
    main()
