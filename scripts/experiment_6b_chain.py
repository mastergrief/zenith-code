"""Round 6b — capability test for PT+DeltaNet on multi-step chains.

Trains BOTH plain PT (R6a baseline-without-delta) and PT+DeltaNet
(R6a canonical) on the same multi-step-chain dataset. Compares
autoregressive accuracy per chain length.

HYPOTHESIS:
  PT+DeltaNet's recurrent state provides working memory for variable-
  reference resolution that plain PT lacks. At chain length ≥ 3 the
  hybrid should significantly outperform plain PT.

GATE (binary decision):
  PT+Delta > PT + 10pp at chain_length 3  →  capability gain confirmed
  PT+Delta ≈ PT at all chain lengths       →  R6a's backbone swap was
                                              equivalent perf, no new capability
  PT+Delta < PT                             →  DeltaNet backbone hurts
                                              on chain tasks

Data: mixed lengths 1/2/3; train on all, eval at each length separately.

Runtime: ~10-15 min on RTX 4070 (100 epochs × 2 models).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from calm.hrm.chain_data import ChainDataGenerator, ChainProblem, filter_by_length
from calm.hrm.data import VOCAB_SIZE, _CHAR_TO_ID, _ID_TO_CHAR
from calm.llm_computer.copy_augmented import build_copy_augmented_hrm
from calm.llm_computer.copy_augmented_delta import build_copy_augmented_delta
from scripts.train_substrate_hrm import SeqDataset


def _autoreg_eval(model, problems, device, max_gen=48) -> float:
    """Greedy autoregressive decode; exact expression match rate."""
    bos = _CHAR_TO_ID["<bos>"]
    sep = _CHAR_TO_ID["<sep>"]
    eos = _CHAR_TO_ID["<eos>"]
    model.eval()
    correct = 0
    for p in problems:
        ids = [bos] + [_CHAR_TO_ID[c] for c in p.question
                       if c in _CHAR_TO_ID] + [sep]
        gen = []
        for _ in range(max_gen):
            x = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                lp = model(x)
            nxt = int(lp[0, -1].argmax().item())
            if nxt == eos:
                break
            gen.append(nxt)
            ids.append(nxt)
        decoded = "".join(
            _ID_TO_CHAR.get(i, "") for i in gen
            if not _ID_TO_CHAR.get(i, "").startswith("<")
        ).strip().rstrip("=").strip()
        if decoded == p.expression.strip():
            correct += 1
    return correct / max(len(problems), 1)


def _evaluate_by_length(model, problems, device) -> Dict[int, float]:
    """Run autoregressive eval per chain length."""
    out: Dict[int, float] = {}
    for L in sorted({p.chain_length for p in problems}):
        subset = filter_by_length(problems, L)
        out[L] = _autoreg_eval(model, subset, device)
    return out


def train_and_eval(
    model, label: str, train_loader, val_loader, val_probs,
    epochs: int, lr: float, device: str,
    scheduled_sampling: bool, tf_ratio_start: float, tf_ratio_end: float,
    eval_every: int, nll_loss: bool,
) -> Tuple[float, Dict[int, float]]:
    """Train one model; return (best_autoreg_overall, per_length_autoreg_at_best)."""
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_autoreg = 0.0
    best_by_length: Dict[int, float] = {}
    t0 = time.time()

    loss_fn = F.nll_loss if nll_loss else F.cross_entropy

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        nb = 0
        tf_ratio = tf_ratio_start - (tf_ratio_start - tf_ratio_end) * (epoch / epochs)

        for b in train_loader:
            x = b["input_ids"].to(device)
            y = b["target_ids"].to(device)
            m = b["loss_mask"].to(device)
            B, S = x.shape

            if scheduled_sampling and tf_ratio < 0.99:
                with torch.no_grad():
                    pred = model(x).argmax(-1)
                swap = (torch.rand(B, S, device=device) > tf_ratio) & m
                swap_shifted = torch.zeros_like(swap)
                swap_shifted[:, 1:] = swap[:, :-1]
                preds_shifted = torch.zeros_like(x)
                preds_shifted[:, 1:] = pred[:, :-1]
                modified_x = torch.where(swap_shifted, preds_shifted, x)
                out = model(modified_x)
            else:
                out = model(x)

            lp = out.reshape(-1, VOCAB_SIZE)
            tf = y.reshape(-1)
            mf = m.reshape(-1)
            if not mf.any():
                continue
            loss = loss_fn(lp[mf], tf[mf])

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            nb += 1

        sched.step()

        if epoch % eval_every == 0 or epoch == 1:
            per_L = _evaluate_by_length(model, val_probs, device)
            overall = sum(per_L.values()) / len(per_L)
            print(f"[{label}] ep {epoch:4d}/{epochs} loss={total_loss/max(nb,1):.4f} "
                  f"overall={overall:.1%} "
                  + " ".join(f"L{L}={v:.1%}" for L, v in sorted(per_L.items()))
                  + f"  t={time.time()-t0:.0f}s", flush=True)
            if overall > best_autoreg:
                best_autoreg = overall
                best_by_length = per_L.copy()

    return best_autoreg, best_by_length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--problems", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=32)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--d-ffn", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--n-copy-heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--lengths", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--no-scheduled-sampling", action="store_true")
    ap.add_argument("--tf-ratio-end", type=float, default=0.3)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== Round 6b — PT vs PT+DeltaNet on chain data ===")
    print(f"  lengths={args.lengths}  n={args.problems}  epochs={args.epochs}  device={device}")
    sys.stdout.flush()

    gen = ChainDataGenerator(lengths=args.lengths, seed=args.seed)
    probs = gen.generate(args.problems)
    ds = SeqDataset(probs, max_len=args.max_len)
    val_size = max(len(args.lengths) * 30, len(ds) // 10)
    train_set, val_set = random_split(
        ds, [len(ds) - val_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    val_probs = [probs[i] for i in val_set.indices]

    print(f"  train={len(train_set)} val={len(val_set)}")
    per_L_in_val = {L: sum(1 for p in val_probs if p.chain_length == L) for L in args.lengths}
    print(f"  val per length: {per_L_in_val}")

    # --- Plain PT (R6a baseline-without-delta, softmax backbone) ---
    print(f"\n[1/2] plain PT (softmax backbone)")
    torch.manual_seed(args.seed)
    model_pt = build_copy_augmented_hrm(
        vocab_size=VOCAB_SIZE, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ffn=args.d_ffn, max_len=args.max_len,
        n_copy_heads=args.n_copy_heads, use_hard_max=False,
    )
    print(f"  params: {sum(p.numel() for p in model_pt.parameters()):,}")
    pt_overall, pt_by_L = train_and_eval(
        model_pt, "PT", train_loader, val_loader, val_probs,
        epochs=args.epochs, lr=args.lr, device=device,
        scheduled_sampling=not args.no_scheduled_sampling,
        tf_ratio_start=1.0, tf_ratio_end=args.tf_ratio_end,
        eval_every=args.eval_every, nll_loss=True,
    )

    # --- PT+DeltaNet (R6a canonical, delta backbone) ---
    print(f"\n[2/2] PT+DeltaNet (DeltaNet backbone)")
    torch.manual_seed(args.seed)
    model_delta = build_copy_augmented_delta(
        vocab_size=VOCAB_SIZE, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ffn=args.d_ffn, max_len=args.max_len,
        n_copy_heads=args.n_copy_heads, use_hard_max=False,
    )
    print(f"  params: {sum(p.numel() for p in model_delta.parameters()):,}")
    delta_overall, delta_by_L = train_and_eval(
        model_delta, "PT+Δ", train_loader, val_loader, val_probs,
        epochs=args.epochs, lr=args.lr, device=device,
        scheduled_sampling=not args.no_scheduled_sampling,
        tf_ratio_start=1.0, tf_ratio_end=args.tf_ratio_end,
        eval_every=args.eval_every, nll_loss=True,
    )

    # --- Comparison ---
    print("\n" + "=" * 64)
    print("Round 6b Results — chain-length breakdown")
    print("=" * 64)
    print(f"\n  chain_length   plain PT       PT+DeltaNet   Δ")
    print(f"  ------------   --------       -----------   ---------")
    for L in sorted(set(pt_by_L) | set(delta_by_L)):
        pt_v = pt_by_L.get(L, 0.0)
        d_v = delta_by_L.get(L, 0.0)
        delta = (d_v - pt_v) * 100
        print(f"  L={L}           {pt_v:>6.1%}        {d_v:>6.1%}        {delta:+5.1f} pp")
    print(f"  overall        {pt_overall:>6.1%}        {delta_overall:>6.1%}        "
          f"{(delta_overall - pt_overall)*100:+5.1f} pp")

    # --- Decision ---
    max_L = max(args.lengths)
    pt_max = pt_by_L.get(max_L, 0.0)
    d_max = delta_by_L.get(max_L, 0.0)
    gap = (d_max - pt_max) * 100
    print(f"\n  Capability gain at max chain length (L={max_L}):")
    print(f"    plain PT         = {pt_max:.1%}")
    print(f"    PT+DeltaNet      = {d_max:.1%}")
    print(f"    Δ                = {gap:+.1f} pp")
    print("\n" + "=" * 64)
    if gap >= 10.0:
        print(f"DECISION: CAPABILITY GAIN — PT+Delta > PT + 10pp at L={max_L}.")
        print(f"  DeltaNet's recurrent state meaningfully helps variable-reference")
        print(f"  resolution on multi-step chains. Hybrid is capability-distinct,")
        print(f"  not just equivalent architecture.")
    elif abs(gap) < 5.0:
        print(f"DECISION: PARITY — PT ≈ PT+Delta across all chain lengths.")
        print(f"  R6a's 100% on single-step NL math was equivalent perf to plain PT")
        print(f"  on structured tasks. DeltaNet state isn't buying new capability")
        print(f"  at these chain lengths. Commercially, hybrid is a substitute,")
        print(f"  not an upgrade, for single-step structure extraction.")
    elif gap < -5.0:
        print(f"DECISION: REGRESSION — PT+Delta < PT at L={max_L}.")
        print(f"  DeltaNet backbone hurts on chain tasks despite R6a's 100% on")
        print(f"  single-step. Hypothesis: softmax attention's parallel lookup")
        print(f"  handles long-range variable resolution better than DeltaNet's")
        print(f"  sequential state at this scale.")
    else:
        print(f"DECISION: MARGINAL — Δ = {gap:+.1f}pp, borderline. Run longer or")
        print(f"  increase chain length to sharpen the signal.")
    print("=" * 64)


if __name__ == "__main__":
    main()
