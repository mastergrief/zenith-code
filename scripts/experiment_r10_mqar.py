"""R10 — MQAR stress test: plain PT vs PT+DeltaNet at scaled N.

Hypothesis: at high N (many key-value bindings), DeltaNet's recurrent
state scales better than PT's softmax pointer attention. Expected gap
at N=15 or N=20.

Setup:
  Train one model per architecture on MIXED-N data (N ∈ {5, 10, 15, 20}).
  Eval autoregressive recall per-N on held-out problems.

Gate:
  PT+Δ > PT + 10pp at N=20  →  capability gain found
  PT+Δ ≈ PT at all N        →  softmax pointer matches DeltaNet state
                                on this task at substrate scale
  PT+Δ < PT                  →  DeltaNet backbone hurts on MQAR
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from calm.hrm.data import VOCAB_SIZE, _CHAR_TO_ID, _ID_TO_CHAR
from calm.hrm.memory_tasks import (
    MemProblem,
    gen_mqar_batch,
    gen_reassign_batch,
    gen_scratchpad_batch,
)

_TASK_GENERATORS = {
    "mqar": gen_mqar_batch,
    "reassign": gen_reassign_batch,
    "scratchpad": gen_scratchpad_batch,
}
from calm.llm_computer.copy_augmented import (
    CopyAugmentedConfig, CopyAugmentedTransformer, build_copy_augmented_hrm,
)
from calm.llm_computer.copy_augmented_delta import (
    CopyAugmentedDeltaConfig, CopyAugmentedDeltaNet, build_copy_augmented_delta,
)
from scripts.train_substrate_hrm import SeqDataset


def _build_pt_any_d_head(vocab_size, d_model, n_heads, n_layers, d_ffn,
                         max_len, n_copy_heads, use_hard_max):
    """PT with any d_head (bypasses substrate d_head=2 invariant for tests)."""
    cfg = CopyAugmentedConfig(
        vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ffn=d_ffn, max_len=max_len,
        n_copy_heads=n_copy_heads, use_hard_max=use_hard_max,
    )
    return CopyAugmentedTransformer(cfg)


def _build_delta_any_d_head(vocab_size, d_model, n_heads, n_layers, d_ffn,
                            max_len, n_copy_heads, use_hard_max):
    cfg = CopyAugmentedDeltaConfig(
        vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ffn=d_ffn, max_len=max_len,
        n_copy_heads=n_copy_heads, use_hard_max=use_hard_max,
        use_delta_net=True, use_softmax_attn=False,
    )
    return CopyAugmentedDeltaNet(cfg)


def _filter_by_difficulty(probs: List[MemProblem], diff: int) -> List[MemProblem]:
    return [p for p in probs if p.difficulty == diff]


def _autoreg_eval(model, problems, device, max_gen=8) -> float:
    """MQAR target is a single digit → tiny max_gen suffices."""
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


def _eval_per_N(model, val_probs, device) -> Dict[int, float]:
    out = {}
    for N in sorted({p.difficulty for p in val_probs}):
        subset = _filter_by_difficulty(val_probs, N)
        out[N] = _autoreg_eval(model, subset, device)
    return out


def train_and_measure(model, label, train_loader, val_probs, *,
                      epochs, lr, device, eval_every,
                      scheduled_sampling, tf_ratio_end,
                      nll_loss) -> Tuple[float, Dict[int, float]]:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = F.nll_loss if nll_loss else F.cross_entropy
    best_overall = 0.0
    best_by_N: Dict[int, float] = {}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        nb = 0
        tf_ratio = 1.0 - (1.0 - tf_ratio_end) * (epoch / epochs)

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
            per_N = _eval_per_N(model, val_probs, device)
            overall = sum(per_N.values()) / len(per_N)
            print(f"[{label}] ep {epoch:3d}/{epochs} loss={total_loss/max(nb,1):.4f} "
                  f"overall={overall:.1%} "
                  + " ".join(f"N{N}={v:.1%}" for N, v in sorted(per_N.items()))
                  + f"  t={time.time()-t0:.0f}s", flush=True)
            if overall > best_overall:
                best_overall = overall
                best_by_N = per_N.copy()
    return best_overall, best_by_N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--per-N-train", type=int, default=500,
                    help="training problems per N value")
    ap.add_argument("--per-N-val", type=int, default=100,
                    help="held-out problems per N value")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=32)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--d-ffn", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=80)
    ap.add_argument("--n-values", type=int, nargs="+", default=[5, 10, 15, 20])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--no-scheduled-sampling", action="store_true")
    ap.add_argument("--tf-ratio-end", type=float, default=0.3)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--task", type=str, default="mqar",
                    choices=list(_TASK_GENERATORS.keys()),
                    help="memory task generator: mqar | reassign | scratchpad")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    gen_fn = _TASK_GENERATORS[args.task]
    print(f"=== R10 memory-task stress: plain PT vs PT+DeltaNet ===")
    print(f"  task={args.task}")
    print(f"  N values: {args.n_values}")
    print(f"  per-N train={args.per_N_train} val={args.per_N_val}")
    print(f"  epochs={args.epochs} max_len={args.max_len} device={device}")
    sys.stdout.flush()

    # Generate mixed-N dataset using the selected task generator.
    train_probs, val_probs = [], []
    for i, N in enumerate(args.n_values):
        train_probs.extend(gen_fn(N, args.per_N_train, seed=args.seed + 1000 * i))
        val_probs.extend(gen_fn(N, args.per_N_val, seed=args.seed + 1000 * i + 500))
    print(f"  total train={len(train_probs)} val={len(val_probs)}")
    max_q_len = max(len(p.question) for p in train_probs + val_probs)
    print(f"  max question length: {max_q_len}")
    assert max_q_len + 10 < args.max_len, f"need max_len > {max_q_len + 10}"

    train_ds = SeqDataset(train_probs, max_len=args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)

    d_head = args.d_model // args.n_heads
    print(f"  d_head={d_head}")
    # --- Plain PT ---
    print(f"\n[1/2] plain PT")
    torch.manual_seed(args.seed)
    m_pt = _build_pt_any_d_head(
        vocab_size=VOCAB_SIZE, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ffn=args.d_ffn, max_len=args.max_len,
        n_copy_heads=4, use_hard_max=False,
    )
    print(f"  params: {sum(p.numel() for p in m_pt.parameters()):,}")
    pt_overall, pt_by_N = train_and_measure(
        m_pt, "PT", train_loader, val_probs,
        epochs=args.epochs, lr=args.lr, device=device,
        eval_every=args.eval_every,
        scheduled_sampling=not args.no_scheduled_sampling,
        tf_ratio_end=args.tf_ratio_end, nll_loss=True,
    )

    # --- PT+DeltaNet ---
    print(f"\n[2/2] PT+DeltaNet")
    torch.manual_seed(args.seed)
    m_d = _build_delta_any_d_head(
        vocab_size=VOCAB_SIZE, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ffn=args.d_ffn, max_len=args.max_len,
        n_copy_heads=4, use_hard_max=False,
    )
    print(f"  params: {sum(p.numel() for p in m_d.parameters()):,}")
    d_overall, d_by_N = train_and_measure(
        m_d, "PT+Δ", train_loader, val_probs,
        epochs=args.epochs, lr=args.lr, device=device,
        eval_every=args.eval_every,
        scheduled_sampling=not args.no_scheduled_sampling,
        tf_ratio_end=args.tf_ratio_end, nll_loss=True,
    )

    # --- Comparison ---
    print("\n" + "=" * 64)
    print("R10 MQAR Results")
    print("=" * 64)
    print(f"\n  N     plain PT    PT+Δ       Δ")
    print(f"  --    --------    ------     ------")
    for N in sorted(set(pt_by_N) | set(d_by_N)):
        pt_v = pt_by_N.get(N, 0.0)
        d_v = d_by_N.get(N, 0.0)
        delta = (d_v - pt_v) * 100
        print(f"  {N:>2}   {pt_v:>6.1%}     {d_v:>6.1%}    {delta:+5.1f} pp")
    print(f"  all  {pt_overall:>6.1%}     {d_overall:>6.1%}    {(d_overall - pt_overall)*100:+5.1f} pp")

    max_N = max(args.n_values)
    gap = (d_by_N.get(max_N, 0.0) - pt_by_N.get(max_N, 0.0)) * 100
    print(f"\n  Gap at N={max_N}: {gap:+.1f} pp")
    print("=" * 64)
    if gap >= 10.0:
        print(f"DECISION: CAPABILITY GAIN at N={max_N} — DeltaNet scales better.")
    elif abs(gap) < 5.0:
        print(f"DECISION: PARITY — softmax pointer matches DeltaNet state at this scale.")
    elif gap < -5.0:
        print(f"DECISION: REGRESSION — DeltaNet backbone hurts on MQAR.")
    else:
        print(f"DECISION: MARGINAL — borderline, Δ={gap:+.1f}pp.")
    print("=" * 64)


if __name__ == "__main__":
    main()
