"""Round 5 — DeltaNet vs fast-weights Round-4 null on associative recall.

HYPOTHESIS
  DeltaNet (L2-norm + SiLU + Householder form + learned β_t + no λ decay)
  breaks the Round-4 n=10 ceiling. Round 4's delta+gate nulled at 10.5-12.2%
  because:
    (a) λ=0.95 decayed ALL stored bindings, not just overwritten ones
    (b) No L2-norm, so β·k k^T wasn't a clean projection
    (c) Fixed η instead of learned β_t

MEASUREMENT
  Same test harness as experiment_fast_weights.py (build_sequence + eval_recall),
  same architecture sizes (d_model=64, n_heads=32, d_head=2, n_layers=2),
  same training budget (100 epochs × 32 steps/epoch × 32 batch = 102,400 seqs),
  same held-out protocol (fresh RNG, same 256-token pool).

  n_pairs ∈ {3, 5, 10}. Compare against Round 1 baseline (fast-weights plain)
  and Round 4 best (delta+gate).

DECISION
  PASS if DeltaNet n=10 >= 80%   (breaks the n=10 ceiling)
  PARTIAL if 50-80%              (meaningful lift but not saturated)
  FAIL if <50%                   (mechanism does not transfer)

Runtime: ~5-10 min on CPU.
"""

from __future__ import annotations

import random
import sys
import time

import torch
import torch.nn.functional as F

from calm.llm_computer.delta_rule import (
    DeltaNetConfig, DeltaNetSmall2DTransformer,
)
from scripts.experiment_fast_weights import (
    BATCH_SIZE, D_FFN, D_MODEL, EPOCHS, LR, MAX_LEN, N_HEADS, N_LAYERS,
    STEPS_PER_EPOCH, TOKEN_POOL, VOCAB_SIZE,
    build_batch, eval_recall,
)


def make_delta():
    cfg = DeltaNetConfig(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ffn=D_FFN,
        max_len=MAX_LEN, use_hard_max=False,
        use_delta_net=True,
    )
    return DeltaNetSmall2DTransformer(cfg)


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


def run_at_n(n_pairs: int):
    print(f"\n  training delta-net n={n_pairs}...", flush=True)
    torch.manual_seed(0)
    model = make_delta()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    params={n_params}", flush=True)
    train_model(model, n_pairs, f"delta n={n_pairs}")
    correct, total = eval_recall(model, n_pairs, TOKEN_POOL)
    print(f"    eval: {correct}/{total} = {correct/total:.1%}", flush=True)
    return correct, total


def main():
    print("=== Round 5: DeltaNet on Associative Recall ===")
    print(f"  vocab={VOCAB_SIZE}  d_model={D_MODEL}  n_heads={N_HEADS}  "
          f"d_head={D_MODEL//N_HEADS}  n_layers={N_LAYERS}")
    print(f"  epochs={EPOCHS}  batch={BATCH_SIZE}  lr={LR}")
    print(f"  mechanism: L2-norm K/Q + SiLU + Householder (no λ decay) + learned β_t")
    sys.stdout.flush()

    n_pairs_list = [3, 5, 10]
    results = {}
    for n in n_pairs_list:
        results[n] = run_at_n(n)

    print("\n" + "=" * 68)
    print("Round 5 Results — DeltaNet held-out recall")
    print("=" * 68)
    print("\n  n_pairs  DeltaNet       R1 plain   R4 delta+gate")
    print("  -------  ----------     --------   --------------")
    r1_baseline = {3: 0.991, 5: 0.651, 10: 0.122}   # Round 1 plain fast-weights
    r4_best     = {3: None,  5: None,  10: 0.122}   # Round 4 delta+gate n=10
    for n in n_pairs_list:
        c, t = results[n]
        r = c / t
        r1 = r1_baseline.get(n)
        r4 = r4_best.get(n)
        r1s = f"{r1:.1%}" if r1 is not None else "   —"
        r4s = f"{r4:.1%}" if r4 is not None else "     —"
        print(f"  {n:>7}  {c:>4}/{t} = {r:5.1%}    {r1s:>8}   {r4s:>14}")

    rate_n10 = results[10][0] / results[10][1]
    print("\n" + "=" * 68)
    print(f"DeltaNet at n=10: {rate_n10:.1%}")
    print(f"Lift vs R1 baseline (12.2%):  {(rate_n10 - 0.122)*100:+.1f} pp")
    print(f"Lift vs R4 best (~12.2%):      {(rate_n10 - 0.122)*100:+.1f} pp")
    if rate_n10 >= 0.80:
        print("DECISION: PASS — DeltaNet breaks the n=10 ceiling.")
        print("  Next: sweep n ∈ {20, 40, 64} to map the new ceiling.")
    elif rate_n10 >= 0.50:
        print("DECISION: PARTIAL — meaningful lift, not yet saturated.")
        print("  Next: ablate mechanism components (norm, β_t, decay) to find")
        print("  which piece carries the lift, then iterate.")
    else:
        print("DECISION: FAIL — mechanism does not clear the ceiling in this setup.")
        print("  Next: ablate initialization, training length, short-conv heads.")
    print("=" * 68)


if __name__ == "__main__":
    main()
