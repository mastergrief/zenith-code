"""Train copy-augmented HRM on multi-step reasoning.

9 categories: chained arithmetic, comparison, conditional, sequence cost,
percentage, ratio, max/min, transitivity, and explicit syllogism.
Longest inputs ~120 chars, expressions include function calls and
Python ternary conditionals.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from calm.hrm.data import VOCAB_SIZE, _CHAR_TO_ID, _ID_TO_CHAR
from calm.hrm.reasoning_data import FuncallReasoningGenerator as ReasoningDataGenerator
from calm.llm_computer.copy_augmented import build_copy_augmented_hrm


CHECKPOINT_PATH = Path("calm/hrm/checkpoints/copy_funcall_best.pt")


class ReasoningSeqDataset(Dataset):
    def __init__(self, problems, max_len=160):
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

        prompt_ids = [_CHAR_TO_ID[c] for c in p.problem if c in _CHAR_TO_ID]
        target_ids = [_CHAR_TO_ID[c] for c in p.expression if c in _CHAR_TO_ID]

        seq = [bos] + prompt_ids + [sep] + target_ids + [eos]
        target_start = 1 + len(prompt_ids) + 1

        input_ids = seq[:-1]
        target_seq = seq[1:]
        mask = [0] * len(input_ids)
        for i in range(len(input_ids)):
            if i >= target_start - 1:
                mask[i] = 1

        input_ids = input_ids[: self.max_len]
        target_seq = target_seq[: self.max_len]
        mask = mask[: self.max_len]
        while len(input_ids) < self.max_len:
            input_ids.append(pad)
            target_seq.append(pad)
            mask.append(0)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_seq, dtype=torch.long),
            "loss_mask": torch.tensor(mask, dtype=torch.bool),
        }


def _autoreg_eval(model, problems, device, max_gen=60):
    bos = _CHAR_TO_ID["<bos>"]
    sep = _CHAR_TO_ID["<sep>"]
    eos = _CHAR_TO_ID["<eos>"]
    pos_limit = model.config.max_len  # don't exceed positional embeddings
    model.eval()
    correct = 0
    for p in problems:
        ids = [bos] + [_CHAR_TO_ID[c] for c in p.problem
                       if c in _CHAR_TO_ID] + [sep]
        gen = []
        gen_budget = min(max_gen, pos_limit - len(ids) - 1)
        for _ in range(gen_budget):
            x = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                log_probs = model(x)
            nxt = int(log_probs[0, -1].argmax().item())
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
            preds = model(x).argmax(-1)
            correct += (preds[m] == y[m]).sum().item()
            total += m.sum().item()
    return correct / max(total, 1)


def train(epochs=500, problems=5000, batch_size=64, lr=1e-3,
          d_model=64, n_heads=32, n_layers=4, d_ffn=128,
          max_len=128, n_copy_heads=4, seed=42, eval_every=10,
          scheduled_sampling=True, tf_ratio_start=1.0, tf_ratio_end=0.3,
          device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    gen = ReasoningDataGenerator(seed=seed)
    probs = gen.generate(problems)
    print(f"[copy-fc] generated {len(probs)} reasoning problems")
    ds = ReasoningSeqDataset(probs, max_len=max_len)
    val_size = max(1, len(ds) // 10)
    train_set, val_set = random_split(ds, [len(ds) - val_size, val_size],
                                       generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    val_probs = [probs[i] for i in val_set.indices]

    model = build_copy_augmented_hrm(
        vocab_size=VOCAB_SIZE, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ffn=d_ffn, max_len=max_len,
        n_copy_heads=n_copy_heads, use_hard_max=False,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[copy-fc] model: {total_params:,} params on {device}")
    if scheduled_sampling:
        print(f"[copy-fc] scheduled sampling: tf_ratio {tf_ratio_start:.1f}"
              f" → {tf_ratio_end:.1f} over {epochs} epochs")

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
                    pred_lp = model(x)
                    preds = pred_lp.argmax(-1)
                swap = (torch.rand(B, S, device=device) > tf_ratio) & m
                swap_shifted = torch.zeros_like(swap)
                swap_shifted[:, 1:] = swap[:, :-1]
                preds_shifted = torch.zeros_like(x)
                preds_shifted[:, 1:] = preds[:, :-1]
                modified_x = torch.where(swap_shifted, preds_shifted, x)
                log_probs = model(modified_x)
            else:
                log_probs = model(x)

            lp = log_probs.reshape(-1, VOCAB_SIZE)
            tf = y.reshape(-1)
            mf = m.reshape(-1)
            if not mf.any():
                continue
            loss = F.nll_loss(lp[mf], tf[mf])

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
            print(f"[copy-fc] epoch {epoch:4d}/{epochs}: "
                  f"loss={total_loss/max(nb,1):.4f}, val_acc={va:.1%}, "
                  f"autoreg={autoreg:.1%}, tf_ratio={tf_ratio:.2f}, "
                  f"elapsed={time.time()-t0:.0f}s")
            if autoreg > best_autoreg:
                best_autoreg = autoreg
                CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "vocab_size": VOCAB_SIZE, "d_model": d_model,
                        "n_heads": n_heads, "n_layers": n_layers,
                        "d_ffn": d_ffn, "max_len": max_len,
                        "n_copy_heads": n_copy_heads,
                    },
                    "epoch": epoch, "val_acc": va,
                    "autoreg_acc": autoreg, "copy_augmented": True,
                    "domain": "reasoning",
                }, CHECKPOINT_PATH)

    print(f"[copy-fc] done: {time.time()-t0:.0f}s, best autoreg={best_autoreg:.1%}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--problems", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--d-ffn", type=int, default=128)
    p.add_argument("--max-len", type=int, default=208)
    p.add_argument("--n-copy-heads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--no-scheduled-sampling", action="store_true")
    p.add_argument("--tf-ratio-end", type=float, default=0.3)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()
    train(
        epochs=args.epochs, problems=args.problems,
        batch_size=args.batch_size, lr=args.lr,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ffn=args.d_ffn,
        max_len=args.max_len, n_copy_heads=args.n_copy_heads,
        seed=args.seed, eval_every=args.eval_every,
        scheduled_sampling=not args.no_scheduled_sampling,
        tf_ratio_end=args.tf_ratio_end, device=args.device,
    )


if __name__ == "__main__":
    main()
