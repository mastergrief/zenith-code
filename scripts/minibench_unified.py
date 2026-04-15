"""Minibench — unified CHRLM as ONE tensor, compiled + trained fused.

Rung 3 of the unbundling plan. Rung 1 (substrate primitives solo) tests
D3/D5 in isolation. This script tests the OTHER half: can a single
Small2DTransformer hold a frozen compiled program AND a trainable layer,
in one .pt file, without the training corrupting the compiled behavior?

MVP config (smallest that could possibly work):
  - d_model=10, n_heads=5 (adder_tiny's native dims — hard constraint)
  - vocab=8, max_len=4 (adder's native)
  - n_layers=2: layer 0 = compiled adder (frozen)
                layer 1 = trainable (starts zero)
  - Freeze token/pos embeddings and LM head too — they're compiled-shared

Training task: the SAME adder task the compiled layer already solves.
Input [a, b] for a,b ∈ {0,1,2,3}; target = adder output at pos 1 = a+b.
Gradient signal says "keep doing what you're doing" so trainable layer 1
should converge to near-zero updates. If adder still passes 16/16 after
training, the unified-tensor architecture survives.

Gates (all must pass for rung 3 PASS):
  1. Pre-training: exhaustive adder 16/16
  2. Post-training: exhaustive adder 16/16
  3. Save + reload .pt: exhaustive adder 16/16
  4. Trainable-param count > 0 (we're actually training something)
  5. At least one training step ran (wallclock > 0)

If this passes, we know the unified tensor holds the adder through
gradient updates. Next: swap in an orthogonal task (NL→structure) and
prove the trainable layer learns something without breaking the adder.
That's rung 4.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from calm.llm_computer.channel_masking import (
    compiled_output_channels_adder_tiny, protect_residual_channels,
)
from calm.llm_computer.unified_chrlm import (
    UnifiedCHRLMConfig, build_unified_chrlm,
    freeze_layer_params, freeze_embeddings_and_head,
    trainable_param_count,
)
from scripts.experiment_fast_weights_fusion import (
    build_adder_tiny_small2d, exhaustive_adder,
)


def install_compiled_at_layer_0(unified) -> None:
    """Mirror the pattern in test_unified_chrlm.py: build a source
    Small2DTransformer with the adder at our target layer, then
    state_dict-transfer into the unified empty substrate.
    """
    src = build_adder_tiny_small2d(target_layer=0, n_layers=unified.config.n_layers)
    unified.load_state_dict(src.state_dict())


def init_trainable_layers(unified, std: float, compiled_layers: tuple[int, ...]) -> None:
    """Kaiming-small random init on non-compiled layers so gradients can
    flow once freezing is applied. Without this, zero-init + hard_max
    attention traps the trainable layer at norm=0 (gradients are zero
    because activations are zero).
    """
    with torch.no_grad():
        for layer_idx in range(unified.config.n_layers):
            if layer_idx in compiled_layers:
                continue
            for lin in (unified.W_qkv[layer_idx], unified.W_out[layer_idx],
                        unified.ff_in[layer_idx], unified.ff_out[layer_idx]):
                lin.weight.normal_(mean=0.0, std=std)


def adder_training_data():
    """All 16 (a, b) pairs for a, b ∈ {0..3}. Returns (inputs, targets)
    where inputs is shape (16, 2) and targets is shape (16,) giving
    the expected sum at position 1.
    """
    xs, ys = [], []
    for a in range(4):
        for b in range(4):
            xs.append([a, b])
            ys.append(a + b)
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def train_unified(model, xs, ys, steps: int, lr: float, batch_size: int,
                  seed: int) -> dict:
    """Train whatever parameters are still trainable on the adder task.
    Loss: CE at position 1's logits vs target sum.
    """
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable, "no trainable parameters — nothing to test"

    opt = torch.optim.AdamW(trainable, lr=lr)
    rng = torch.Generator().manual_seed(seed)
    n = xs.size(0)

    losses = []
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (batch_size,), generator=rng)
        bx = xs[idx]
        by = ys[idx]
        logits = model(bx)           # (B, 2, V)
        pos1_logits = logits[:, 1, :]  # predict at position 1
        loss = F.cross_entropy(pos1_logits, by)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    model.eval()
    return {
        "wallclock_s": time.time() - t0,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "mean_loss": sum(losses) / len(losses),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-path", default="/tmp/minibench_unified.pt")
    ap.add_argument("--init-std", type=float, default=0.02,
                    help="std of Gaussian init on trainable layer weights "
                         "(0.0 = leave at zero init, 0.02 ≈ GPT-style)")
    ap.add_argument("--channel-mask", action="store_true",
                    help="protect adder's output channels from trainable "
                         "layer writes via gradient hooks")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    print("=== minibench_unified — one tensor, compiled + trained ===", flush=True)
    print(f"  config: d_model=10, n_heads=5, n_layers=2, vocab=8, max_len=4",
          flush=True)

    # Build unified substrate (2 layers, d_head=2)
    cfg = UnifiedCHRLMConfig(
        vocab_size=8, d_model=10, n_heads=5, n_layers=2, d_ffn=14,
        max_len=4, use_hard_max=True,
        compiled_layers=(0,),
    )
    unified = build_unified_chrlm(cfg)

    # Install compiled adder at layer 0
    install_compiled_at_layer_0(unified)

    # Random init on trainable layers BEFORE freezing compiled portions.
    # Zero-init + hard_max attention traps gradients at 0.
    if args.init_std > 0:
        init_trainable_layers(unified, std=args.init_std, compiled_layers=(0,))

    # Freeze: layer 0 + embeddings + head. Layer 1 stays trainable.
    n_frozen_layer = freeze_layer_params(unified, layer_idx=0)
    n_frozen_shared = freeze_embeddings_and_head(unified)

    # Channel masking: protect adder's output channels from trainable
    # layer 1 residual writes via gradient hooks.
    hook_handles = []
    if args.channel_mask:
        protected = compiled_output_channels_adder_tiny()
        hook_handles = protect_residual_channels(
            unified, layer_idx=1, protected_channels=protected,
        )
        print(f"  channel mask: protected channels {protected} on layer 1 "
              f"({len(hook_handles)} hooks)", flush=True)

    n_trainable = trainable_param_count(unified)
    n_total = sum(p.numel() for p in unified.parameters())
    print(f"  params: total={n_total:,}  frozen={n_frozen_layer + n_frozen_shared:,}  "
          f"trainable={n_trainable:,}", flush=True)

    # Gate 1: pre-training adder 16/16
    pre = exhaustive_adder(unified)
    print(f"\n  [gate 1] pre-training adder: {pre}/16", flush=True)
    assert pre == 16, f"compiled adder broken BEFORE training: {pre}/16"

    # Training
    xs, ys = adder_training_data()
    print(f"\n--- training {args.steps} steps on adder task ---", flush=True)
    result = train_unified(
        unified, xs, ys,
        steps=args.steps, lr=args.lr,
        batch_size=args.batch_size, seed=args.seed,
    )
    print(f"  wallclock: {result['wallclock_s']:.2f}s", flush=True)
    print(f"  loss:  first={result['first_loss']:.4f}  "
          f"last={result['last_loss']:.4f}  mean={result['mean_loss']:.4f}",
          flush=True)

    # Gate 2: post-training adder 16/16
    post = exhaustive_adder(unified)
    print(f"\n  [gate 2] post-training adder: {post}/16", flush=True)

    # Gate 3: save + reload .pt
    save_path = Path(args.save_path)
    torch.save(unified.state_dict(), save_path)
    print(f"\n  saved unified CHRLM: {save_path}  "
          f"({save_path.stat().st_size} bytes)", flush=True)

    reloaded = build_unified_chrlm(cfg)
    reloaded.load_state_dict(torch.load(save_path, weights_only=True))
    reloaded.eval()
    roundtrip = exhaustive_adder(reloaded)
    print(f"\n  [gate 3] save+reload adder: {roundtrip}/16", flush=True)

    # Inspect how much layer 1 moved
    with torch.no_grad():
        l1_norm = sum(p.norm().item() for p in (
            unified.W_qkv[1].weight, unified.W_out[1].weight,
            unified.ff_in[1].weight, unified.ff_out[1].weight,
        ))
        # Compare to frozen layer 0 magnitude
        l0_norm = sum(p.norm().item() for p in (
            unified.W_qkv[0].weight, unified.W_out[0].weight,
            unified.ff_in[0].weight, unified.ff_out[0].weight,
        ))
    print(f"\n  weight norms: layer0 (frozen compiled)={l0_norm:.3f}  "
          f"layer1 (trained)={l1_norm:.3f}", flush=True)

    # Summary
    all_pass = (pre == 16) and (post == 16) and (roundtrip == 16) and (n_trainable > 0)
    print("", flush=True)
    print("=== summary ===", flush=True)
    print(f"  gate 1 (pre):       {'PASS' if pre == 16 else 'FAIL'} "
          f"({pre}/16)", flush=True)
    print(f"  gate 2 (post):      {'PASS' if post == 16 else 'FAIL'} "
          f"({post}/16)", flush=True)
    print(f"  gate 3 (roundtrip): {'PASS' if roundtrip == 16 else 'FAIL'} "
          f"({roundtrip}/16)", flush=True)
    print(f"  trainable params:   {n_trainable}", flush=True)
    print(f"  wallclock:          {result['wallclock_s']:.2f}s", flush=True)
    print("", flush=True)
    print(f"  OVERALL: {'PASS — unified tensor architecture holds' if all_pass else 'FAIL'}",
          flush=True)


if __name__ == "__main__":
    main()
