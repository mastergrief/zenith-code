"""Train HRMSeq2Seq on NL → math-expression pairs (integration #3).

Same sweet-spot architecture as the math HRM (48K params, structure-only
loss), different encoder input (NL questions instead of math expressions).
The decoder target is the math expression + `=` terminator; at inference
the expression is fed through `parse_expression` + `interpret` to
recompute values analytically (same path as `eval_hrm_math.py --verified`).

Usage:
  PYTHONPATH=. python3 scripts/train_hrm_nl.py \\
    --epochs 500 --problems 2000 --hidden 32 --max-enc 48 --max-dec 24
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
from calm.hrm.nl_data import NLMathDataGenerator, NLMathSeq2SeqDataset


CHECKPOINT_PATH = Path("calm/hrm/checkpoints/nl_math_structure_best.pt")


def train_nl_math(
    epochs: int = 500,
    problems: int = 2000,
    batch_size: int = 128,
    lr: float = 1e-3,
    hidden: int = 32,
    num_heads: int = 4,
    l_layers: int = 1,
    h_layers: int = 1,
    dec_layers: int = 1,
    max_enc: int = 48,
    max_dec: int = 24,
    seed: int = 42,
    device: str | None = None,
) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[nl] generating {problems} NL→math pairs...")
    gen = NLMathDataGenerator(seed=seed)
    nl_problems = gen.generate(problems)
    print(f"[nl] first 3 examples:")
    for p in nl_problems[:3]:
        print(f"  {p.question!r:50} → {p.expression!r} = {p.answer}")

    dataset = NLMathSeq2SeqDataset(nl_problems, max_enc_len=max_enc,
                                    max_dec_len=max_dec)
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
    print(f"[nl] model: {model.param_count():,} params "
          f"(enc {enc_p:,} + dec {dec_p:,}) on {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
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
            logits_flat = logits.reshape(-1, config.vocab_size)
            tgt_flat = dec_tgt.reshape(-1)
            mask_flat = mask.reshape(-1)
            if not mask_flat.any():
                continue
            loss = F.cross_entropy(logits_flat[mask_flat], tgt_flat[mask_flat])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            n_batch += 1
        scheduler.step()
        avg = total / max(n_batch, 1)
        if epoch % 50 == 0 or epoch == 1:
            val_acc = _evaluate(model, val_loader, device)
            elapsed = time.time() - t0
            print(f"[nl] epoch {epoch:4d}/{epochs}: loss={avg:.4f}, "
                  f"val_acc={val_acc:.1%}, lr={scheduler.get_last_lr()[0]:.6f}, "
                  f"elapsed={elapsed:.0f}s")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
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
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "nl_math": True,
                }, CHECKPOINT_PATH)

    print(f"[nl] done: {time.time() - t0:.0f}s, best val_acc={best_val_acc:.1%}")
    print(f"[nl] checkpoint: {CHECKPOINT_PATH}")


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--problems", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--l-layers", type=int, default=1)
    parser.add_argument("--h-layers", type=int, default=1)
    parser.add_argument("--dec-layers", type=int, default=1)
    parser.add_argument("--max-enc", type=int, default=48)
    parser.add_argument("--max-dec", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_nl_math(
        epochs=args.epochs, problems=args.problems,
        batch_size=args.batch_size, lr=args.lr,
        hidden=args.hidden, num_heads=args.num_heads,
        l_layers=args.l_layers, h_layers=args.h_layers,
        dec_layers=args.dec_layers,
        max_enc=args.max_enc, max_dec=args.max_dec,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
