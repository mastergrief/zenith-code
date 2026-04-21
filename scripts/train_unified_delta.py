"""Round 7 — unified DeltaNet (no separate copy mechanism).

Tests whether DeltaNet's Householder recurrence ALONE is sufficient for
NL→expression transduction that Round 6a hybrid solved via PT's separate
softmax copy path. No copy_gate, no copy_q_proj, no copy_k_proj —
DeltaNet's associative memory IS the copy mechanism.

Architecture:
  DeltaNetSmall2DTransformer (no changes from delta_rule.py)
  head(x) → vocab logits → cross-entropy (not NLL on blend)

Training mirrors Round 6a: same NL math data, same 100-epoch schedule,
same scheduled sampling tf_ratio 1.0→0.3, same AdamW.

Gate at best autoreg:
  ≥90% = unified works → ship as canonical (fewer params, same perf)
  50-90% = partial; copy mechanism did more than shield
  <50% = hybrid's copy path was load-bearing, unified caps at substrate scale
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from calm.hrm.data import VOCAB_SIZE, _CHAR_TO_ID, _ID_TO_CHAR
from calm.hrm.nl_data import NLMathDataGenerator
from calm.llm_computer.delta_rule import DeltaNetConfig, DeltaNetSmall2DTransformer
from scripts.train_substrate_hrm import SeqDataset


CHECKPOINT_PATH = Path("calm/hrm/checkpoints/unified_delta_best.pt")


def _autoreg_eval(model, problems, device, max_gen=30):
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
                logits = model(x)
            nxt = int(logits[0, -1].argmax().item())
            if nxt == eos:
                break
            gen.append(nxt)
            ids.append(nxt)
        decoded = "".join(
            _ID_TO_CHAR.get(i, "") for i in gen
            if not _ID_TO_CHAR.get(i, "").startswith("<")
        ).strip().rstrip("=").strip()
        expected = p.expression.strip()
        if decoded == expected:
            correct += 1
    return correct / max(len(problems), 1)


def _evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for b in loader:
            x = b["input_ids"].to(device)
            y = b["target_ids"].to(device)
            m = b["loss_mask"].to(device)
            logits = model(x)
            preds = logits.argmax(-1)
            correct += (preds[m] == y[m]).sum().item()
            total += m.sum().item()
    return correct / max(total, 1)


def train(epochs=100, problems=2000, batch_size=64, lr=1e-3,
          d_model=64, n_heads=32, n_layers=4, d_ffn=128,
          max_len=96, seed=42, eval_every=5,
          scheduled_sampling=True, tf_ratio_start=1.0, tf_ratio_end=0.3,
          device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    gen = NLMathDataGenerator(seed=seed)
    probs = gen.generate(problems)
    ds = SeqDataset(probs, max_len=max_len)
    val_size = max(1, len(ds) // 10)
    train_set, val_set = random_split(
        ds, [len(ds) - val_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    val_probs = [probs[i] for i in val_set.indices]

    cfg = DeltaNetConfig(
        vocab_size=VOCAB_SIZE, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ffn=d_ffn, max_len=max_len,
        use_hard_max=False,
        use_delta_net=True, use_softmax_attn=False,
    )
    model = DeltaNetSmall2DTransformer(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    beta_params = sum(p.numel() for p in model.beta_head.parameters())
    print(f"[unified-delta] model: {total_params:,} params "
          f"({beta_params:,} beta_head) on {device}", flush=True)
    if scheduled_sampling:
        print(f"[unified-delta] scheduled sampling: tf_ratio "
              f"{tf_ratio_start:.1f} → {tf_ratio_end:.1f} over {epochs} epochs",
              flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_autoreg = 0.0
    t0 = time.time()

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
                    pred_logits = model(x)
                    preds = pred_logits.argmax(-1)
                swap = (torch.rand(B, S, device=device) > tf_ratio) & m
                swap_shifted = torch.zeros_like(swap)
                swap_shifted[:, 1:] = swap[:, :-1]
                preds_shifted = torch.zeros_like(x)
                preds_shifted[:, 1:] = preds[:, :-1]
                modified_x = torch.where(swap_shifted, preds_shifted, x)
                logits = model(modified_x)
            else:
                logits = model(x)

            # Standard CE loss on raw logits (not log-probs; no copy blend).
            lg = logits.reshape(-1, VOCAB_SIZE)
            tf = y.reshape(-1)
            mf = m.reshape(-1)
            if not mf.any():
                continue
            loss = F.cross_entropy(lg[mf], tf[mf])

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            nb += 1

        sched.step()

        if epoch % eval_every == 0 or epoch == 1:
            va = _evaluate(model, val_loader, device)
            autoreg = _autoreg_eval(model, val_probs[:50], device)
            print(f"[unified-delta] epoch {epoch:4d}/{epochs}: "
                  f"loss={total_loss/max(nb,1):.4f}, val_acc={va:.1%}, "
                  f"autoreg={autoreg:.1%}, tf_ratio={tf_ratio:.2f}, "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
            if autoreg > best_autoreg:
                best_autoreg = autoreg
                CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "vocab_size": VOCAB_SIZE, "d_model": d_model,
                        "n_heads": n_heads, "n_layers": n_layers,
                        "d_ffn": d_ffn, "max_len": max_len,
                    },
                    "epoch": epoch, "val_acc": va,
                    "autoreg_acc": autoreg, "unified_delta": True,
                }, CHECKPOINT_PATH)

    print(f"[unified-delta] done: {time.time()-t0:.0f}s, "
          f"best autoreg={best_autoreg:.1%}", flush=True)
    print(f"[unified-delta] DECISION: ", end="", flush=True)
    if best_autoreg >= 0.90:
        print("PASS — unified design matches hybrid. Ship as canonical.")
    elif best_autoreg >= 0.50:
        print("PARTIAL — copy mechanism did more than shielding.")
    else:
        print("FAIL — hybrid's separate copy path is load-bearing.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--problems", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--d-ffn", type=int, default=128)
    p.add_argument("--max-len", type=int, default=96)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--no-scheduled-sampling", action="store_true")
    p.add_argument("--tf-ratio-end", type=float, default=0.3)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()
    train(
        epochs=args.epochs, problems=args.problems,
        batch_size=args.batch_size, lr=args.lr,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ffn=args.d_ffn,
        max_len=args.max_len, seed=args.seed,
        eval_every=args.eval_every,
        scheduled_sampling=not args.no_scheduled_sampling,
        tf_ratio_end=args.tf_ratio_end, device=args.device,
    )


if __name__ == "__main__":
    main()
