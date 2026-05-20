"""Train and persist a PT+Delta card on MQAR (R21).

Produces a deployable `copy_augmented_delta_mqar_best.pt` artifact
that solves MQAR at N=5-15 to ~99%. Trains with chunkwise + scheduled
sampling, saves best-by-autoreg checkpoint.

This is the card that gets installed on Gemma via CardSlot in R22.

Config chosen from R13/R14-b findings:
- 5K/N per-N training to cover N=15 cleanly
- N=[5, 10, 15] — covers sparse/dense key space for realistic recall
- chunkwise ON, n_delta_heads=1, n_iterations=1 (R20 consolidated defaults)
- 20 epochs is sufficient (R13-d converged at ep10-15)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from calm.hrm.data import VOCAB_SIZE, _CHAR_TO_ID, _ID_TO_CHAR
from calm.hrm.memory_tasks import gen_mqar_batch
from calm.llm_computer.copy_augmented_delta import build_copy_augmented_delta
from scripts.train_substrate_hrm import SeqDataset


CHECKPOINT_PATH = Path("calm/hrm/checkpoints/copy_augmented_delta_mqar_best.pt")


def _autoreg_eval(model, problems, device, max_gen=8):
    bos = _CHAR_TO_ID["<bos>"]
    sep = _CHAR_TO_ID["<sep>"]
    eos = _CHAR_TO_ID["<eos>"]
    model.eval()
    correct_by_n = {}
    count_by_n = {}
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
        n = p.difficulty
        count_by_n[n] = count_by_n.get(n, 0) + 1
        if decoded == p.expression.strip():
            correct_by_n[n] = correct_by_n.get(n, 0) + 1
    acc_by_n = {n: correct_by_n.get(n, 0) / count_by_n[n] for n in count_by_n}
    overall = sum(correct_by_n.values()) / max(sum(count_by_n.values()), 1)
    return overall, acc_by_n


def train(
    epochs=20, per_N_train=5000, per_N_val=100,
    n_values=(5, 10, 15),
    batch_size=64, lr=1e-3,
    d_model=64, n_heads=32, n_layers=4, d_ffn=128, max_len=128,
    n_copy_heads=4, seed=42, eval_every=2,
    tf_ratio_start=1.0, tf_ratio_end=0.3,
    noisy_frac=0.0,   # R-delta-22: >0 enables noise-augmented training
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    # Build corpus.
    train_probs, val_probs = [], []
    if noisy_frac > 0.0:
        from calm.hrm.memory_tasks import gen_mqar_batch_noisy
        gen_train = lambda N, count, seed: gen_mqar_batch_noisy(
            N, count, seed=seed, noisy_frac=noisy_frac)
    else:
        gen_train = gen_mqar_batch
    for i, N in enumerate(n_values):
        train_probs.extend(gen_train(N, per_N_train, seed=seed + 1000 * i))
        # Val stays clean — measures generalization FROM noisy training
        val_probs.extend(gen_mqar_batch(N, per_N_val, seed=seed + 1000 * i + 500))
    print(f"Corpus: train={len(train_probs)} val={len(val_probs)}  "
          f"N={list(n_values)}  noisy_frac={noisy_frac}")

    train_ds = SeqDataset(train_probs, max_len=max_len)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    m = build_copy_augmented_delta(
        vocab_size=VOCAB_SIZE, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ffn=d_ffn, max_len=max_len,
        n_copy_heads=n_copy_heads,
    )
    m.config.use_chunkwise = True
    m.config.chunk_size = 32
    m.to(device)
    print(f"Params: {sum(p.numel() for p in m.parameters()):,}")
    print(f"Config: chunkwise={m.config.use_chunkwise} "
          f"n_delta_heads={m.config.n_delta_heads} "
          f"n_iterations={m.config.n_iterations}")

    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_acc = -1.0
    best_ep = 0
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        m.train()
        tf_ratio = tf_ratio_start - (tf_ratio_start - tf_ratio_end) * (epoch / epochs)
        total_loss, total_n = 0.0, 0
        t0 = time.time()

        for batch in loader:
            x = batch["input_ids"].to(device)
            y = batch["target_ids"].to(device)
            mask = batch["loss_mask"].to(device)

            # Scheduled sampling: randomly replace input with prediction.
            if tf_ratio < 1.0:
                with torch.no_grad():
                    pred_logp = m(x)
                    preds = pred_logp.argmax(-1)
                    shifted = torch.cat(
                        [x[:, :1], preds[:, :-1]], dim=1,
                    )
                    rnd = torch.rand_like(x, dtype=torch.float)
                    swap = (rnd >= tf_ratio)
                    # Only swap in loss-mask region.
                    swap = swap & mask.bool()
                    x_input = torch.where(swap, shifted, x)
            else:
                x_input = x

            log_probs = m(x_input)
            loss = F.nll_loss(
                log_probs[mask].view(-1, log_probs.size(-1)),
                y[mask].view(-1),
                reduction="mean",
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * mask.sum().item()
            total_n += mask.sum().item()
        sched.step()

        if epoch % eval_every == 0 or epoch == epochs:
            acc, acc_by_n = _autoreg_eval(m, val_probs, device)
            elapsed = int(time.time() - t0)
            by_n_str = " ".join(f"N{n}={a:.1%}" for n, a in sorted(acc_by_n.items()))
            print(f"ep {epoch:2d}/{epochs} loss={total_loss/max(total_n,1):.4f} "
                  f"overall={acc:.1%} {by_n_str} tf={tf_ratio:.2f} t={elapsed}s")
            if acc > best_acc:
                best_acc = acc
                best_ep = epoch
                torch.save({
                    "model_state_dict": m.state_dict(),
                    "config": {
                        "vocab_size": VOCAB_SIZE,
                        "d_model": d_model, "n_heads": n_heads,
                        "n_layers": n_layers, "d_ffn": d_ffn,
                        "max_len": max_len, "n_copy_heads": n_copy_heads,
                        "use_chunkwise": True, "chunk_size": 32,
                        "n_delta_heads": 1,
                        # Read live values from model.config so flag-on training
                        # runs save the actual architecture (otherwise reload via
                        # load_dt_checkpoint would silently revert to defaults).
                        "n_iterations": getattr(m.config, "n_iterations", 1),
                        "use_loop_index": getattr(m.config, "use_loop_index", False),
                        "use_input_injection": getattr(m.config, "use_input_injection", False),
                        "use_gated_attention": getattr(m.config, "use_gated_attention", False),
                        "use_z_init": getattr(m.config, "use_z_init", False),
                        "use_lecun_init": getattr(m.config, "use_lecun_init", False),
                        "use_prefix_lm": getattr(m.config, "use_prefix_lm", False),
                        "h_cycles": getattr(m.config, "h_cycles", 1),
                        "use_h_rmsnorm": getattr(m.config, "use_h_rmsnorm", False),
                        "use_short_conv": getattr(m.config, "use_short_conv", False),
                    },
                    "epoch": epoch,
                    "autoreg_acc": acc,
                    "acc_by_N": acc_by_n,
                    "train_task": "mqar",
                    "n_values": list(n_values),
                    "per_N_train": per_N_train,
                }, CHECKPOINT_PATH)

    print(f"\nBest: epoch {best_ep} overall={best_acc:.1%}")
    print(f"Saved: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--per-N-train", type=int, default=5000)
    p.add_argument("--per-N-val", type=int, default=100)
    p.add_argument("--n-values", type=int, nargs="+", default=[5, 10, 15])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--d-ffn", type=int, default=128)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--noisy-frac", type=float, default=0.0,
                    help="R-delta-22: fraction of training examples drawn "
                    "from gen_mqar_batch_noisy (default 0 = clean-only, "
                    "reproduces R21). Try 0.5 for 50/50 clean/noisy mix.")
    args = p.parse_args()
    train(
        epochs=args.epochs,
        per_N_train=args.per_N_train,
        per_N_val=args.per_N_val,
        n_values=tuple(args.n_values),
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ffn=args.d_ffn,
        max_len=args.max_len,
        seed=args.seed, eval_every=args.eval_every,
        noisy_frac=args.noisy_frac,
    )
