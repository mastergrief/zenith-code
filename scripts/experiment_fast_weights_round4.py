"""Round 4 — delta rule + hypernet-gated writes for fast weights.

Round 3 proved capacity scaling (d_model 64 → 128) does NOT close the n=10
interference ceiling: +1.8pp lift vs +32.8pp at n=5. Diagnosis: off-target
cross-terms in `W_fast @ q = sum_i v_i (k_i · q)` dominate when 10 random
keys crowd the same matrix.

Round 4 switches mechanism:

  (A) Delta rule (Schlag 2021, full form) — subtract the current binding
      for k before writing a new (k, v). Mathematically: new W_fast
      contribution is outer(v - W_fast@k, k), which replaces the existing
      v-estimate for this k. Reduces interference from re-writes.

  (B) Write gate (hypernet) — tiny per-layer MLP emits a per-position
      sigmoid gate that scales the update term. Lets the model learn to
      NOT write at SEP / query tokens. Half the writes at n=10 were noise
      from non-KV positions; silencing them strips that noise.

Four-way ablation at d_model=64, n_pairs ∈ {5, 10}:
  plain              — Round 1 baseline (reference: 65.1% / 12.2%)
  delta-rule only    — (A) alone
  gate only          — (B) alone
  delta + gate       — both stacked

Decision on n=10:
  PASS if delta+gate >= 50%
  PARTIAL if 30-50%  → scope Round 5 (normalized φ(k) nonlinearity)
  FAIL if <30%       → interference is structural even with both fixes;
                       consider SRWM or multi-head FW

Runtime: ~15-25 min on CPU (8 training runs).
"""

from __future__ import annotations

import random
import time

import torch
import torch.nn.functional as F

from calm.llm_computer.fast_weights import (
    FastWeightConfig, FastWeightSmall2DTransformer,
)
from scripts.experiment_fast_weights import (
    BATCH_SIZE, EPOCHS, ETA_WRITE, LAMBDA_DECAY, LR, MAX_LEN, STEPS_PER_EPOCH,
    TOKEN_POOL, VOCAB_SIZE, build_batch, eval_recall,
)


VARIANTS = {
    "plain":      dict(use_delta_rule=False, use_write_gate=False),
    "delta":      dict(use_delta_rule=True,  use_write_gate=False),
    "gate":       dict(use_delta_rule=False, use_write_gate=True),
    "delta+gate": dict(use_delta_rule=True,  use_write_gate=True),
}


def make_model(variant_flags: dict) -> FastWeightSmall2DTransformer:
    cfg = FastWeightConfig(
        vocab_size=VOCAB_SIZE,
        d_model=64, n_heads=32, n_layers=2, d_ffn=128, max_len=MAX_LEN,
        use_hard_max=False,
        lambda_decay=LAMBDA_DECAY, eta_write=ETA_WRITE, use_fast_weights=True,
        gate_hidden=16,
        **variant_flags,
    )
    return FastWeightSmall2DTransformer(cfg)


def train_model(model, n_pairs: int, label: str):
    rng = random.Random(42 + n_pairs)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    orig_hard_max = model.config.use_hard_max
    model.config.use_hard_max = False
    model.train()
    t0 = time.time()
    for epoch in range(EPOCHS):
        total_loss, total_correct, total = 0.0, 0, 0
        for _ in range(STEPS_PER_EPOCH):
            x, target = build_batch(n_pairs, TOKEN_POOL, BATCH_SIZE, rng)
            logits = model(x)[:, -1, :]
            loss = F.cross_entropy(logits, target)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(-1) == target).sum().item()
            total += x.size(0)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(
                f"    [{label}] epoch {epoch+1:3d}/{EPOCHS}: "
                f"loss={total_loss/total:.4f}  train={total_correct/total:.2%}  "
                f"{elapsed:5.1f}s",
                flush=True,
            )
    model.config.use_hard_max = orig_hard_max
    model.eval()


def run_variant(variant: str, flags: dict, n_pairs_list: list[int]):
    results = {}
    for n_pairs in n_pairs_list:
        label = f"{variant} n={n_pairs}"
        print(f"\n  training {label}...", flush=True)
        torch.manual_seed(0)
        model = make_model(flags)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    params={n_params}", flush=True)
        train_model(model, n_pairs, label)
        correct, total = eval_recall(model, n_pairs, TOKEN_POOL)
        results[n_pairs] = (correct, total)
        print(f"    eval: {correct}/{total} = {correct/total:.1%}", flush=True)
    return results


def main():
    print("=== Round 4: Delta Rule + Write Gate Ablation ===")
    print(f"  vocab={VOCAB_SIZE}  d_model=64  n_heads=32  d_head=2  n_layers=2")
    print(f"  epochs={EPOCHS}  batch={BATCH_SIZE}  lr={LR}")
    print(f"  fast weights: lambda={LAMBDA_DECAY}  eta={ETA_WRITE}")
    print(f"  target: close n=10 ceiling (Round 1 baseline = 12.2%)")

    n_pairs_list = [5, 10]
    all_results = {}

    for variant, flags in VARIANTS.items():
        print(f"\n[variant = {variant}]")
        all_results[variant] = run_variant(variant, flags, n_pairs_list)

    print("\n" + "=" * 68)
    print("Round 4 Results — held-out recall by variant")
    print("=" * 68)
    header = "  n_pairs  " + "  ".join(f"{v:<12}" for v in VARIANTS)
    print(f"\n{header}")
    print("  -------  " + "  ".join("-" * 12 for _ in VARIANTS))
    for n in n_pairs_list:
        row = f"  {n:>7}  "
        for v in VARIANTS:
            c, t = all_results[v][n]
            row += f"{c/t:>11.1%}  "
        print(row)

    print("\n  Round 1 baseline at same config: n=5: 65.1%   n=10: 12.2%")

    # Decision on n=10
    best_name = max(VARIANTS, key=lambda v: all_results[v][10][0])
    best_c, best_t = all_results[best_name][10]
    best_rate = best_c / best_t

    print("\n" + "=" * 68)
    print(f"Best variant at n=10: {best_name}  = {best_rate:.1%}")
    print(f"Lift vs Round 1 baseline: {(best_rate - 0.122) * 100:+.1f} pp")
    if best_rate >= 0.50:
        print("DECISION: PASS — delta rule / gate closes the n=10 ceiling.")
        print("  Next: try n=20 to see where the new ceiling lands.")
    elif best_rate >= 0.30:
        print("DECISION: PARTIAL — mechanisms help substantially but don't saturate.")
        print("  Round 5 candidate: positive-normalized keys (softmax or ELU+1 on k_t)")
        print("  before the outer product, or combine with d_model scaling.")
    else:
        print("DECISION: FAIL — interference is structural even with delta+gate.")
        print("  Round 5 must switch to qualitatively different memory: SRWM-style")
        print("  learned update rules, or multi-head fast weights (per-head W_fast).")
    print("=" * 68)


if __name__ == "__main__":
    main()
