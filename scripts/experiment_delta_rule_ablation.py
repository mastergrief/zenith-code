"""Round 5c — DeltaNet ablation: isolate what regresses vs Round 1 at n=5.

Round 5a (hybrid softmax+DeltaNet) and 5b (pure DeltaNet) both hit
~20% at n=5, massively below Round 1 plain fast-weights (65.1%).
Softmax parallel path was NOT the culprit — pure DeltaNet had same
trajectory. Something in the mechanism itself over-constrains the
optimization at d_head=2 substrate scale.

Ablating the paper's two feature-map choices (SiLU + L2-norm) at n=5
to find which piece carries the regression.

Variants:
  paper      — SiLU + L2-norm (both on; Round 5a/b canonical)
  no-silu    — L2-norm only
  no-l2      — SiLU only
  raw        — neither (just the delta-rule Householder with raw Q/K)

n=5 only, 100 epochs, same test harness.
"""

from __future__ import annotations

import random, sys, time
import torch
import torch.nn.functional as F

from calm.llm_computer.delta_rule import DeltaNetConfig, DeltaNetSmall2DTransformer
from scripts.experiment_fast_weights import (
    BATCH_SIZE, D_FFN, D_MODEL, EPOCHS, LR, MAX_LEN, N_HEADS, N_LAYERS,
    STEPS_PER_EPOCH, TOKEN_POOL, VOCAB_SIZE, build_batch, eval_recall,
)

VARIANTS = {
    "paper":   dict(use_silu_feat=True,  use_l2_norm=True),
    "no-silu": dict(use_silu_feat=False, use_l2_norm=True),
    "no-l2":   dict(use_silu_feat=True,  use_l2_norm=False),
    "raw":     dict(use_silu_feat=False, use_l2_norm=False),
}


def make(flags):
    cfg = DeltaNetConfig(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        d_ffn=D_FFN, max_len=MAX_LEN, use_hard_max=False,
        use_softmax_attn=False, **flags,
    )
    return DeltaNetSmall2DTransformer(cfg)


def train(m, n_pairs, label):
    rng = random.Random(42 + n_pairs)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    m.config.use_hard_max = False
    m.train()
    t0 = time.time()
    for epoch in range(EPOCHS):
        loss_s, correct, total = 0.0, 0, 0
        for _ in range(STEPS_PER_EPOCH):
            x, tgt = build_batch(n_pairs, TOKEN_POOL, BATCH_SIZE, rng)
            logits = m(x)[:, -1, :]
            loss = F.cross_entropy(logits, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_s += loss.item() * x.size(0); correct += (logits.argmax(-1) == tgt).sum().item(); total += x.size(0)
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"    [{label}] ep{epoch+1:3d} loss={loss_s/total:.4f} train={correct/total:.1%} t={time.time()-t0:.1f}s", flush=True)
    m.eval()


def main():
    print("=== Round 5c: DeltaNet Ablation @ n=5 ===")
    print("  target: find which of {SiLU, L2-norm} carries the regression")
    print(f"  baseline: Round 1 plain fast-weights n=5 = 65.1%")
    print(f"  Round 5a hybrid DeltaNet n=5 = 21.8%  (38pp regression)")
    print(f"  Round 5b pure DeltaNet n=5 ≈ 20% at ep40 (killed early)")
    sys.stdout.flush()

    results = {}
    for name, flags in VARIANTS.items():
        print(f"\n[variant={name}] flags={flags}", flush=True)
        torch.manual_seed(0)
        m = make(flags)
        train(m, 5, name)
        c, t = eval_recall(m, 5, TOKEN_POOL)
        results[name] = (c, t)
        print(f"    eval n=5: {c}/{t} = {c/t:.1%}", flush=True)

    print("\n" + "=" * 60)
    print("Round 5c Results — DeltaNet ablation at n=5")
    print("=" * 60)
    print("\n  variant      silu  l2     n=5 recall")
    print("  -------      ----  ----   ----------")
    for name, flags in VARIANTS.items():
        c, t = results[name]
        s = "Y" if flags["use_silu_feat"] else "N"
        l = "Y" if flags["use_l2_norm"] else "N"
        print(f"  {name:<10}   {s:<4}  {l:<4}   {c}/{t} = {c/t:.1%}")
    print(f"\n  Round 1 baseline at n=5: 65.1%")

    best = max(results, key=lambda v: results[v][0])
    best_rate = results[best][0] / results[best][1]
    print(f"\n  Best variant: {best} = {best_rate:.1%}")
    if best_rate >= 0.60:
        print("  DECISION: PASS — ablation recovers Round-1-class performance.")
    elif best_rate >= 0.40:
        print("  DECISION: PARTIAL — mechanism partially recovers; more tuning needed.")
    else:
        print("  DECISION: FAIL — neither ablation reaches Round 1. Root cause")
        print("            is elsewhere (learned β_t? update magnitude? training?)")
    print("=" * 60)


if __name__ == "__main__":
    main()
