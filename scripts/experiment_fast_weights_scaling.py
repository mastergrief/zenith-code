"""Round 3 — d_model capacity scaling on 10-pair associative recall.

Round 1 hit a ceiling at n=10: fast-weights 12.2%, vanilla 8.6%. The
hypothesis is that W_fast capacity (d_model² entries) is the bottleneck
— with 10 simultaneous bindings at λ=0.95 in a 64×64 = 4096-entry matrix,
destructive interference dominates.

Round 3 tests that hypothesis by scaling d_model on the exact same task:

  d_model=64   (baseline)  → W_fast = 4096 entries
  d_model=128              → W_fast = 16384 entries (4×)

If d_model=128 crosses ~40% on n=10, capacity was the lever. Ship + plan a
Round 4 at 256 to see the scaling law. If still <20%, interference is
structural and needs a different storage mechanism (normalized Schlag form,
per-head fast weights, or reversible writes).

We also re-measure n=5 at each size to check that higher d_model lifts the
whole capacity curve, not just the tail case.

Runtime: ~15-25 min on CPU (d_model=128 forward is ~4× d_model=64).
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
    SEP_TOKEN, TOKEN_POOL, VOCAB_SIZE, MAX_LEN,
    BATCH_SIZE, EPOCHS, LR, N_EVAL, STEPS_PER_EPOCH,
    LAMBDA_DECAY, ETA_WRITE,
    build_batch, eval_recall,
)


def make_fast(d_model: int) -> FastWeightSmall2DTransformer:
    """Build FastWeightSmall2DTransformer with specified d_model.
    Keeps d_head=2 (substrate invariant) by scaling n_heads proportionally.
    """
    assert d_model % 2 == 0
    cfg = FastWeightConfig(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        n_heads=d_model // 2,  # d_head = 2
        n_layers=2,
        d_ffn=2 * d_model,     # d_ffn = 2 * d_model (same ratio as Round 1)
        max_len=MAX_LEN,
        use_hard_max=False,
        lambda_decay=LAMBDA_DECAY,
        eta_write=ETA_WRITE,
        use_fast_weights=True,
    )
    return FastWeightSmall2DTransformer(cfg)


def train_model(model, n_pairs: int, label: str):
    """Identical to scripts.experiment_fast_weights.train_model but
    re-implemented here because d_model is no longer shared."""
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
                f"loss={total_loss/total:.4f}  train_acc={total_correct/total:.2%}  "
                f"{elapsed:5.1f}s",
                flush=True,
            )
    model.config.use_hard_max = orig_hard_max
    model.eval()


def run(d_model: int, n_pairs_list: list[int]) -> dict:
    results = {}
    for n_pairs in n_pairs_list:
        label = f"d={d_model} n={n_pairs}"
        print(f"\n  training {label}...", flush=True)
        torch.manual_seed(0)
        model = make_fast(d_model)
        n_params = sum(p.numel() for p in model.parameters())
        w_fast_entries = d_model * d_model
        print(f"    params={n_params}  W_fast={w_fast_entries} entries", flush=True)
        train_model(model, n_pairs, label)
        correct, total = eval_recall(model, n_pairs, TOKEN_POOL)
        results[n_pairs] = (correct, total)
        print(f"    eval: {correct}/{total} = {correct/total:.1%}", flush=True)
    return results


def main():
    print("=== Round 3: Fast-Weight d_model Capacity Scaling ===")
    print(f"  vocab={VOCAB_SIZE}  epochs={EPOCHS}  batch={BATCH_SIZE}  lr={LR}")
    print(f"  fast weights: lambda={LAMBDA_DECAY}  eta={ETA_WRITE}")
    print(f"  d_head=2 (substrate invariant); n_heads scales with d_model")

    d_model_list = [64, 128]
    n_pairs_list = [5, 10]

    all_results = {}
    for d_model in d_model_list:
        print(f"\n[d_model = {d_model}]  W_fast = {d_model*d_model} entries")
        all_results[d_model] = run(d_model, n_pairs_list)

    print("\n" + "=" * 60)
    print("Round 3 Results — held-out recall by d_model")
    print("=" * 60)
    header = "  n_pairs  " + "  ".join(f"d_model={d:<6}" for d in d_model_list)
    print(f"\n{header}")
    print("  -------  " + "  ".join("-" * 14 for _ in d_model_list))
    for n in n_pairs_list:
        row = f"  {n:>7}  "
        for d in d_model_list:
            c, t = all_results[d][n]
            row += f"{c:>4}/{t} = {c/t:5.1%}  "
        print(row)

    # Reference points from Round 1 (d_model=64):
    #   n=5:  65.1%   n=10: 12.2%
    print("\n  Round 1 reference at d_model=64:")
    print("    n=5: 65.1%   n=10: 12.2%")

    # Decision for the n=10 ceiling
    c_64_10, t_64_10 = all_results[64][10]
    c_128_10, t_128_10 = all_results[128][10]
    r_64_10 = c_64_10 / t_64_10
    r_128_10 = c_128_10 / t_128_10
    delta_10 = (r_128_10 - r_64_10) * 100

    print("\n" + "=" * 60)
    print(f"n=10 scaling: d_model 64 → 128  gives  "
          f"{r_64_10:.1%} → {r_128_10:.1%}  ({delta_10:+.1f} pp)")
    if r_128_10 >= 0.40:
        print("DECISION: PASS — capacity scaling closes the n=10 ceiling.")
        print("  Interference was the bottleneck; d_model² capacity matters.")
        print("  Round 4 candidate: d_model=256 to measure scaling law slope.")
    elif r_128_10 >= 0.20:
        print("DECISION: PARTIAL — scaling helps but doesn't saturate.")
        print("  Capacity matters, but interference structure also matters.")
        print("  Round 4: try per-head fast weights OR normalized Schlag form.")
    else:
        print("DECISION: FAIL — scaling alone does not close the ceiling.")
        print("  Interference is structural, not capacity-bound.")
        print("  Round 4 MUST switch mechanism (normalized Schlag, per-head FW, "
              "or hypernet-modulated η/λ).")
    print("=" * 60)


if __name__ == "__main__":
    main()
