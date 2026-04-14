"""Router-HRM trainer — classify a query into {math, nl, word, gsm, meta}.

Target: ≥95% classification accuracy at h=16. At that budget the router
is ~8K params, runs on CPU, and its argmax can be trusted to dispatch
an incoming query to the right specialist.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from calm.hrm.router_data import LABELS, RouterDataset, RouterGenerator
from calm.hrm.router_model import RouterConfig, RouterHRM


CHECKPOINT_PATH = Path("calm/hrm/checkpoints/router_best.pt")


def train(epochs=500, problems=10000, batch_size=64, lr=1e-3,
          hidden=16, num_heads=4, l_layers=1, h_layers=1,
          max_len=384, seed=42, eval_every=25):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[router] generating {problems} balanced samples over {len(LABELS)} labels...")
    gen = RouterGenerator(seed=seed)
    samples = gen.generate(problems)
    by_label = [0] * len(LABELS)
    for s in samples:
        by_label[s.label_id] += 1
    print(f"[router] label distribution: "
          f"{dict(zip(LABELS, by_label))}")

    ds = RouterDataset(samples, max_len=max_len)
    val_size = max(1, len(ds) // 10)
    train_set, val_set = random_split(ds, [len(ds) - val_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    cfg = RouterConfig(vocab_size=80, hidden_size=hidden, num_heads=num_heads,
                       L_layers=l_layers, H_layers=h_layers,
                       max_seq_len=max_len, num_labels=len(LABELS))
    model = RouterHRM(cfg).to(device)
    print(f"[router] model: {model.param_count():,} params on {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best = 0.0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        nb = 0
        for b in train_loader:
            x = b["input_ids"].to(device)
            y = b["label"].to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); nb += 1
        sched.step()
        if epoch % eval_every == 0 or epoch == 1:
            va = _evaluate(model, val_loader, device)
            print(f"[router] epoch {epoch:4d}/{epochs}: loss={total/max(nb,1):.4f}, "
                  f"val_acc={va:.1%}, elapsed={time.time()-t0:.0f}s")
            if va > best:
                best = va
                CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "vocab_size": cfg.vocab_size, "hidden_size": cfg.hidden_size,
                        "num_heads": cfg.num_heads, "L_layers": cfg.L_layers,
                        "H_layers": cfg.H_layers, "max_seq_len": cfg.max_seq_len,
                        "num_labels": cfg.num_labels,
                    },
                    "epoch": epoch, "val_acc": va, "router": True,
                }, CHECKPOINT_PATH)
    print(f"[router] done: {time.time()-t0:.0f}s, best val_acc={best:.1%}")


def _evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for b in loader:
            x = b["input_ids"].to(device)
            y = b["label"].to(device)
            pred = model(x).argmax(-1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--problems", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--l-layers", type=int, default=1)
    p.add_argument("--h-layers", type=int, default=1)
    p.add_argument("--max-len", type=int, default=384)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=25)
    args = p.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
