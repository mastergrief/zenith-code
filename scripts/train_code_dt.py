"""Train a Delta-Transducer (DT) on (problem, skeleton) pairs mined from
CodeExampleDB. Produces `calm/hrm/checkpoints/dt_code_skel_best.pt`.

**DT (delta-transducer)** is the canonical product name for the
copy-augmented DeltaNet architecture — default meta across all new
training as of 2026-04-22. The underlying class `CopyAugmentedDeltaNet`
stays (implementation detail); DT is the product-level label.

Per `.claude/rules/delta_rule.md` §"MQAR data-scaling rule" — DeltaNet
fast-weight state is the right mechanism for large-key-space retrieval
(8970 code examples, hundreds of distinct function patterns). Plain PT's
softmax-over-d_head=2 caps at ~10 unique keys.

Training recipe (following `train_pt_delta_mqar.py`):
  - chunkwise UT transform (3-7× training speedup)
  - scheduled sampling tf 1.0 → 0.3 across epochs
  - F.nll_loss on log-probs (not F.cross_entropy) — DT returns log-probs
  - best-by-autoreg checkpoint selection

Scope this session: train, measure autoreg accuracy on held-out val.
Install on Gemma (via CardSlot) is deferred to a follow-up session
since daemon stability is a gate.

Usage:
    PYTHONPATH=. python3 -u scripts/train_code_dt.py \\
        --epochs 30 --batch-size 32 --lr 1e-3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from calm.hrm.code_dt_data import (
    CODE_VOCAB_SIZE,
    CodePairDataset,
    _CODE_CHAR_TO_ID,
    _CODE_ID_TO_CHAR,
    code_detokenize,
    extract_pairs_from_db,
    split_pairs,
)
from calm.llm_computer.copy_augmented_delta import build_copy_augmented_delta


CHECKPOINT_PATH = Path("calm/hrm/checkpoints/dt_code_skel_best.pt")
METRICS_PATH = Path("calm/hrm/checkpoints/dt_code_skel_metrics.json")


def autoreg_eval(model, val_pairs, device, max_gen=40):
    """Greedy decode from <bos>prob<sep>, measure exact-skeleton match."""
    bos = _CODE_CHAR_TO_ID["<bos>"]
    sep = _CODE_CHAR_TO_ID["<sep>"]
    eos = _CODE_CHAR_TO_ID["<eos>"]
    model.eval()
    n_correct = 0
    samples = []
    for p in val_pairs:
        from calm.hrm.code_dt_data import code_tokenize
        prefix = code_tokenize(p.question, add_bos=True, add_eos=False) + [sep]
        ids = list(prefix)
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
            if len(ids) >= getattr(model, "max_len", 256) - 1:
                break
        decoded = code_detokenize(gen)
        if decoded.strip() == p.expression.strip():
            n_correct += 1
        samples.append((p.question[:60], p.expression, decoded))
    acc = n_correct / max(len(val_pairs), 1)
    return acc, samples


def _scheduled_tf(model, input_ids, target_ids, tf_ratio):
    """Scheduled sampling: forward once to get predictions, swap some
    positions with the model's argmax, then forward again with gradient."""
    if tf_ratio >= 0.999:
        # Pure teacher forcing — single forward
        log_probs = model(input_ids)
        loss = F.nll_loss(
            log_probs.reshape(-1, log_probs.shape[-1]),
            target_ids.reshape(-1),
            ignore_index=_CODE_CHAR_TO_ID["<pad>"],
        )
        return loss
    with torch.no_grad():
        lp = model(input_ids)
        preds = lp.argmax(dim=-1)
    # Build mixed input: with prob (1-tf_ratio) swap to model prediction
    # Can't use predictions for position 0 (need bos). Shift and gate.
    bs, seq = input_ids.shape
    shifted = torch.cat(
        [input_ids[:, :1], preds[:, :-1]], dim=1
    )
    mask = torch.rand(bs, seq, device=input_ids.device) > tf_ratio
    # Preserve position 0 (bos); use the original input elsewhere conditionally
    mask[:, 0] = False
    mixed = torch.where(mask, shifted, input_ids)
    log_probs = model(mixed)
    return F.nll_loss(
        log_probs.reshape(-1, log_probs.shape[-1]),
        target_ids.reshape(-1),
        ignore_index=_CODE_CHAR_TO_ID["<pad>"],
    )


def train(
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    d_model: int = 64,
    n_heads: int = 32,
    n_layers: int = 4,
    d_ffn: int = 128,
    max_len: int = 256,
    n_copy_heads: int = 4,
    seed: int = 42,
    tf_ratio_start: float = 1.0,
    tf_ratio_end: float = 0.3,
    eval_every: int = 2,
    device: str | None = None,
    val_frac: float = 0.1,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    # --- Data ---
    print(f"[train] extracting pairs from CodeExampleDB...")
    pairs = extract_pairs_from_db()
    print(f"[train] total pairs: {len(pairs)}")
    train_pairs, val_pairs = split_pairs(pairs, val_frac=val_frac, seed=seed)
    print(f"[train] split: {len(train_pairs)} train / {len(val_pairs)} val")

    train_ds = CodePairDataset(train_pairs, max_len=max_len)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        num_workers=0)

    # --- Model ---
    model = build_copy_augmented_delta(
        vocab_size=CODE_VOCAB_SIZE,
        max_len=max_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ffn=d_ffn,
        n_copy_heads=n_copy_heads,
    ).to(device)
    # Enable chunkwise for training speed (set on config if supported)
    if hasattr(model, "config") and hasattr(model.config, "use_chunkwise"):
        model.config.use_chunkwise = True
    model.max_len = max_len   # for autoreg_eval len guard

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params: {n_params:,} ({n_params/1e3:.1f}K)")

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=epochs)

    best_acc = 0.0
    history: list[dict] = []
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        # tf_ratio linearly decays
        frac = (ep - 1) / max(epochs - 1, 1)
        tf_ratio = tf_ratio_start + frac * (tf_ratio_end - tf_ratio_start)

        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            input_ids, target_ids, _ = batch
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            loss = _scheduled_tf(model, input_ids, target_ids, tf_ratio)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        scheduler.step()

        elapsed = time.time() - t0
        rec = {"epoch": ep, "loss": avg_loss, "tf_ratio": round(tf_ratio, 3),
               "elapsed_s": round(elapsed, 1)}

        if ep % eval_every == 0 or ep == epochs:
            acc, samples = autoreg_eval(model, val_pairs, device)
            rec["val_autoreg"] = round(acc, 4)
            print(f"[train] ep{ep:3d} loss={avg_loss:.4f} tf={tf_ratio:.2f} "
                  f"val_autoreg={acc:.3f} elapsed={elapsed:.0f}s")
            if acc > best_acc:
                best_acc = acc
                torch.save({
                    "model_state": model.state_dict(),
                    "config": {
                        "vocab_size": CODE_VOCAB_SIZE,
                        "max_len": max_len,
                        "d_model": d_model,
                        "n_heads": n_heads,
                        "n_layers": n_layers,
                        "d_ffn": d_ffn,
                        "n_copy_heads": n_copy_heads,
                        "use_chunkwise": True,
                    },
                    "epoch": ep,
                    "val_autoreg": acc,
                    "n_train": len(train_pairs),
                    "n_val": len(val_pairs),
                }, CHECKPOINT_PATH)
                print(f"[train] ✓ saved (best autoreg={acc:.3f}) → {CHECKPOINT_PATH}")
                # Print 3 sample decodes
                for i, (q, tgt, out) in enumerate(samples[:3]):
                    mark = "✓" if out.strip() == tgt.strip() else "✗"
                    print(f"    {mark} tgt={tgt!r}")
                    print(f"      out={out!r}")
        else:
            print(f"[train] ep{ep:3d} loss={avg_loss:.4f} tf={tf_ratio:.2f} "
                  f"elapsed={elapsed:.0f}s")

        history.append(rec)

    # Final report
    print(f"\n[train] best val_autoreg: {best_acc:.3f}")
    METRICS_PATH.write_text(json.dumps({
        "best_val_autoreg": best_acc,
        "n_params": n_params,
        "epochs": epochs,
        "history": history,
    }, indent=2))
    print(f"[train] metrics → {METRICS_PATH}")
    return best_acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=32)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_len=args.max_len,
        seed=args.seed,
        device=args.device,
    )
