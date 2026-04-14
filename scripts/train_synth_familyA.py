"""Family A IR synth trainer — learn template from 3 IO examples."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from calm.hrm.data import VOCAB_SIZE
from calm.hrm.model import HRMConfig, HRMSeq2Seq
from calm.llm_computer.synth.data import (
    SynthFamilyADataset, SynthFamilyAGenerator,
)


CHECKPOINT_PATH = Path("calm/hrm/checkpoints/synth_familyA_best.pt")


def train(epochs=300, problems=20000, batch_size=128, lr=1e-3, hidden=64,
          num_heads=4, l_layers=1, h_layers=1, dec_layers=1,
          max_enc=96, max_dec=16, seed=42, eval_every=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = SynthFamilyAGenerator(seed=seed)
    samples = gen.generate(problems)
    tmpls = {}
    for s in samples:
        tmpls[s.template] = tmpls.get(s.template, 0) + 1
    print(f"[synthA] template distribution ({len(tmpls)} unique): "
          f"{dict(list(sorted(tmpls.items()))[:6])}...")

    ds = SynthFamilyADataset(samples, max_enc_len=max_enc, max_dec_len=max_dec)
    val_size = max(1, len(ds) // 10)
    train_set, val_set = random_split(ds, [len(ds) - val_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    cfg = HRMConfig(vocab_size=VOCAB_SIZE, hidden_size=hidden, num_heads=num_heads,
                     L_layers=l_layers, H_layers=h_layers,
                     max_seq_len=max_enc, max_dec_len=max_dec,
                     decoder_layers=dec_layers)
    model = HRMSeq2Seq(cfg).to(device)
    print(f"[synthA] model: {model.param_count():,} params on {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = 0.0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        nb = 0
        for b in train_loader:
            enc = b["encoder_ids"].to(device)
            din = b["decoder_input_ids"].to(device)
            dt = b["decoder_target_ids"].to(device)
            m = b["loss_mask"].to(device)
            logits = model(enc, din)
            lf = logits.reshape(-1, cfg.vocab_size)
            tf = dt.reshape(-1)
            mf = m.reshape(-1)
            loss = F.cross_entropy(lf[mf], tf[mf])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); nb += 1
        sched.step()
        if epoch % eval_every == 0 or epoch == 1:
            va = _evaluate(model, val_loader, device)
            print(f"[synthA] epoch {epoch:4d}/{epochs}: loss={total/max(nb,1):.4f}, "
                  f"val_acc={va:.1%}, elapsed={time.time()-t0:.0f}s")
            if va > best:
                best = va
                CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "vocab_size": cfg.vocab_size, "hidden_size": cfg.hidden_size,
                        "num_heads": cfg.num_heads, "expansion": cfg.expansion,
                        "L_layers": cfg.L_layers, "H_layers": cfg.H_layers,
                        "L_cycles": cfg.L_cycles, "H_cycles": cfg.H_cycles,
                        "max_seq_len": cfg.max_seq_len,
                        "decoder_layers": cfg.decoder_layers,
                        "max_dec_len": cfg.max_dec_len,
                    },
                    "epoch": epoch, "val_acc": va, "synth_familyA": True,
                }, CHECKPOINT_PATH)
    print(f"[synthA] done: {time.time()-t0:.0f}s, best val_acc={best:.1%}")


def _evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for b in loader:
            enc = b["encoder_ids"].to(device)
            din = b["decoder_input_ids"].to(device)
            dt = b["decoder_target_ids"].to(device)
            m = b["loss_mask"].to(device)
            preds = model(enc, din).argmax(-1)
            correct += (preds[m] == dt[m]).sum().item()
            total += m.sum().item()
    return correct / max(total, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--problems", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--l-layers", type=int, default=1)
    p.add_argument("--h-layers", type=int, default=1)
    p.add_argument("--dec-layers", type=int, default=1)
    p.add_argument("--max-enc", type=int, default=96)
    p.add_argument("--max-dec", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=20)
    args = p.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
