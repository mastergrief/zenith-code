"""Train HRMSeq2Seq on word problems (CRLM scaling test).

Usage:
  PYTHONPATH=. python3 scripts/train_hrm_word.py \\
    --epochs 500 --problems 2000 --hidden 32 --max-enc 80 --max-dec 24
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from calm.hrm.data import VOCAB_SIZE
from calm.hrm.model import HRMConfig, HRMSeq2Seq
from calm.hrm.word_data import WordProblemDataset, WordProblemGenerator


CHECKPOINT_PATH = Path("calm/hrm/checkpoints/word_problem_best.pt")


def train(
    epochs: int = 500,
    problems: int = 2000,
    batch_size: int = 128,
    lr: float = 1e-3,
    hidden: int = 32,
    num_heads: int = 4,
    l_layers: int = 1,
    h_layers: int = 1,
    dec_layers: int = 1,
    max_enc: int = 80,
    max_dec: int = 24,
    seed: int = 42,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[word] generating {problems} word problems...")
    gen = WordProblemGenerator(seed=seed)
    word_probs = gen.generate(problems)
    print(f"[word] first 3 examples:")
    for p in word_probs[:3]:
        short = p.problem if len(p.problem) < 70 else p.problem[:67] + "..."
        print(f"  {short:70} → {p.expression}")

    dataset = WordProblemDataset(word_probs, max_enc_len=max_enc, max_dec_len=max_dec)
    val_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    config = HRMConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=hidden,
        num_heads=num_heads,
        L_layers=l_layers,
        H_layers=h_layers,
        max_seq_len=max_enc,
        max_dec_len=max_dec,
        decoder_layers=dec_layers,
    )
    model = HRMSeq2Seq(config).to(device)
    enc_p = sum(p.numel() for p in model.encoder.parameters())
    dec_p = sum(p.numel() for p in model.decoder.parameters())
    print(f"[word] model: {model.param_count():,} params "
          f"(enc {enc_p:,} + dec {dec_p:,}) on {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best = 0.0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        n_batch = 0
        for batch in train_loader:
            enc = batch["encoder_ids"].to(device)
            dec_in = batch["decoder_input_ids"].to(device)
            dec_tgt = batch["decoder_target_ids"].to(device)
            mask = batch["loss_mask"].to(device)
            logits = model(enc, dec_in)
            lf = logits.reshape(-1, config.vocab_size)
            tf = dec_tgt.reshape(-1)
            mf = mask.reshape(-1)
            if not mf.any():
                continue
            loss = F.cross_entropy(lf[mf], tf[mf])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            n_batch += 1
        scheduler.step()
        avg = total / max(n_batch, 1)
        if epoch % 50 == 0 or epoch == 1:
            va = _evaluate(model, val_loader, device)
            elapsed = time.time() - t0
            print(f"[word] epoch {epoch:4d}/{epochs}: loss={avg:.4f}, "
                  f"val_acc={va:.1%}, lr={scheduler.get_last_lr()[0]:.6f}, "
                  f"elapsed={elapsed:.0f}s")
            if va > best:
                best = va
                CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "vocab_size": config.vocab_size,
                        "hidden_size": config.hidden_size,
                        "num_heads": config.num_heads,
                        "expansion": config.expansion,
                        "L_layers": config.L_layers,
                        "H_layers": config.H_layers,
                        "L_cycles": config.L_cycles,
                        "H_cycles": config.H_cycles,
                        "max_seq_len": config.max_seq_len,
                        "decoder_layers": config.decoder_layers,
                        "max_dec_len": config.max_dec_len,
                    },
                    "epoch": epoch, "val_acc": va,
                    "word_problem": True,
                }, CHECKPOINT_PATH)

    print(f"[word] done: {time.time() - t0:.0f}s, best val_acc={best:.1%}")


def _evaluate(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            enc = batch["encoder_ids"].to(device)
            dec_in = batch["decoder_input_ids"].to(device)
            dec_tgt = batch["decoder_target_ids"].to(device)
            mask = batch["loss_mask"].to(device)
            preds = model(enc, dec_in).argmax(-1)
            correct += (preds[mask] == dec_tgt[mask]).sum().item()
            total += mask.sum().item()
    return correct / max(total, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--problems", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--l-layers", type=int, default=1)
    p.add_argument("--h-layers", type=int, default=1)
    p.add_argument("--dec-layers", type=int, default=1)
    p.add_argument("--max-enc", type=int, default=80)
    p.add_argument("--max-dec", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(epochs=args.epochs, problems=args.problems,
          batch_size=args.batch_size, lr=args.lr, hidden=args.hidden,
          num_heads=args.num_heads, l_layers=args.l_layers,
          h_layers=args.h_layers, dec_layers=args.dec_layers,
          max_enc=args.max_enc, max_dec=args.max_dec, seed=args.seed)


if __name__ == "__main__":
    main()
