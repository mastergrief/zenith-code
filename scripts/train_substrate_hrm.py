"""Train SubstrateHRM — decoder-only Small2DTransformer for NL → expression.

Input format per sample:
    <bos> NL_tokens <sep> expression_tokens <eos>   → input_ids
Loss is computed only on the (expression + <eos>) positions, via a
per-position loss mask. Prefix tokens are teacher-forced but not
contributing to loss.

Target task: NL templates domain (nl_math_structure_best.pt). Smallest
structured NL, good baseline to see if Small2DTransformer primitives
can match the HRM's 97% on the same task.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from calm.hrm.data import VOCAB_SIZE, _CHAR_TO_ID
from calm.hrm.nl_data import NLMathDataGenerator
from calm.llm_computer.substrate_hrm import build_substrate_hrm


CHECKPOINT_PATH = Path("calm/hrm/checkpoints/substrate_hrm_nl_best.pt")


class SeqDataset(Dataset):
    """Pack (NL prompt, expression target) into a single token sequence with loss mask."""

    def __init__(self, problems, max_len=96):
        self.problems = problems
        self.max_len = max_len

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        p = self.problems[idx]
        pad = _CHAR_TO_ID["<pad>"]
        bos = _CHAR_TO_ID["<bos>"]
        eos = _CHAR_TO_ID["<eos>"]
        sep = _CHAR_TO_ID["<sep>"]

        prompt_ids = [_CHAR_TO_ID[c] for c in p.question if c in _CHAR_TO_ID]
        target_ids = [_CHAR_TO_ID[c] for c in p.expression if c in _CHAR_TO_ID]

        seq = [bos] + prompt_ids + [sep] + target_ids + [eos]
        target_start = 1 + len(prompt_ids) + 1  # after <bos> NL <sep>

        # Build input and target shifted by one (next-token prediction).
        input_ids = seq[:-1]
        target_seq = seq[1:]
        mask = [0] * len(input_ids)
        for i in range(len(input_ids)):
            # Position i predicts target_seq[i]. We want loss only on
            # positions where the target is part of the expression (or <eos>).
            if i >= target_start - 1:
                mask[i] = 1

        input_ids = input_ids[: self.max_len]
        target_seq = target_seq[: self.max_len]
        mask = mask[: self.max_len]
        while len(input_ids) < self.max_len:
            input_ids.append(pad); target_seq.append(pad); mask.append(0)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_seq, dtype=torch.long),
            "loss_mask": torch.tensor(mask, dtype=torch.bool),
        }


def train(epochs=200, problems=10000, batch_size=64, lr=1e-3,
          d_model=64, n_heads=32, n_layers=4, d_ffn=128,
          max_len=96, seed=42, eval_every=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = NLMathDataGenerator(seed=seed)
    probs = gen.generate(problems)
    ds = SeqDataset(probs, max_len=max_len)
    val_size = max(1, len(ds) // 10)
    train_set, val_set = random_split(ds, [len(ds) - val_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = build_substrate_hrm(
        vocab_size=VOCAB_SIZE, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ffn=d_ffn, max_len=max_len,
        use_hard_max=False,  # softmax for training gradients
    ).to(device)
    print(f"[substrate-nl] model: {sum(p.numel() for p in model.parameters()):,} "
          f"params on {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = 0.0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0; nb = 0
        for b in train_loader:
            x = b["input_ids"].to(device)
            y = b["target_ids"].to(device)
            m = b["loss_mask"].to(device)
            logits = model(x)
            lf = logits.reshape(-1, VOCAB_SIZE)
            tf = y.reshape(-1)
            mf = m.reshape(-1)
            if not mf.any():
                continue
            loss = F.cross_entropy(lf[mf], tf[mf])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); nb += 1
        sched.step()
        if epoch % eval_every == 0 or epoch == 1:
            va = _evaluate(model, val_loader, device)
            print(f"[substrate-nl] epoch {epoch:4d}/{epochs}: "
                  f"loss={total/max(nb,1):.4f}, val_acc={va:.1%}, "
                  f"elapsed={time.time()-t0:.0f}s")
            if va > best:
                best = va
                CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "vocab_size": VOCAB_SIZE, "d_model": d_model,
                        "n_heads": n_heads, "n_layers": n_layers,
                        "d_ffn": d_ffn, "max_len": max_len,
                    },
                    "epoch": epoch, "val_acc": va, "substrate": True,
                }, CHECKPOINT_PATH)
    print(f"[substrate-nl] done: {time.time()-t0:.0f}s, best val_acc={best:.1%}")


def _evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for b in loader:
            x = b["input_ids"].to(device)
            y = b["target_ids"].to(device)
            m = b["loss_mask"].to(device)
            preds = model(x).argmax(-1)
            correct += (preds[m] == y[m]).sum().item()
            total += m.sum().item()
    return correct / max(total, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--problems", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--d-ffn", type=int, default=128)
    p.add_argument("--max-len", type=int, default=96)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=20)
    args = p.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
