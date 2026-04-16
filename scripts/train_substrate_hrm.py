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


def _autoreg_eval(model, problems, device, max_gen=30):
    """Autoregressive evaluation — the metric that ACTUALLY matters.
    Greedy-decodes the expression from the NL prompt and checks if
    the full expression matches. This is what inference does."""
    from calm.hrm.data import _CHAR_TO_ID, _ID_TO_CHAR
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


def train(epochs=500, problems=10000, batch_size=64, lr=1e-3,
          d_model=64, n_heads=32, n_layers=4, d_ffn=128,
          max_len=96, seed=42, eval_every=20, scheduled_sampling=True,
          tf_ratio_start=1.0, tf_ratio_end=0.3, device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = NLMathDataGenerator(seed=seed)
    probs = gen.generate(problems)
    ds = SeqDataset(probs, max_len=max_len)
    val_size = max(1, len(ds) // 10)
    train_set, val_set = random_split(ds, [len(ds) - val_size, val_size],
                                       generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    # Keep raw val problems for autoregressive eval
    val_probs = [probs[i] for i in val_set.indices]

    model = build_substrate_hrm(
        vocab_size=VOCAB_SIZE, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ffn=d_ffn, max_len=max_len,
        use_hard_max=False,
    ).to(device)
    print(f"[substrate-nl] model: {sum(p.numel() for p in model.parameters()):,} "
          f"params on {device}")
    if scheduled_sampling:
        print(f"[substrate-nl] scheduled sampling: tf_ratio {tf_ratio_start:.1f}"
              f" → {tf_ratio_end:.1f} over {epochs} epochs")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_autoreg = 0.0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0; nb = 0
        # Scheduled sampling: decay tf_ratio linearly
        tf_ratio = tf_ratio_start - (tf_ratio_start - tf_ratio_end) * (epoch / epochs)

        for b in train_loader:
            x = b["input_ids"].to(device)
            y = b["target_ids"].to(device)
            m = b["loss_mask"].to(device)
            B, S = x.shape

            if scheduled_sampling and tf_ratio < 0.99:
                # 2-pass scheduled sampling:
                # Pass 1: get model's predictions at each position (no grad)
                with torch.no_grad():
                    pred_logits = model(x)
                    preds = pred_logits.argmax(-1)  # (B, S)
                # For expression positions, randomly swap teacher input
                # with model prediction (shifts: pred at pos i → input at pos i+1)
                modified_x = x.clone()
                swap = (torch.rand(B, S, device=device) > tf_ratio) & m
                for pos in range(S - 1):
                    s = swap[:, pos]
                    if s.any():
                        modified_x[s, pos + 1] = preds[s, pos]
                # Pass 2: forward on modified input, compute loss
                logits = model(modified_x)
            else:
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
            # Autoregressive eval on 50 samples (slower but what matters)
            autoreg = _autoreg_eval(model, val_probs[:50], device)
            print(f"[substrate-nl] epoch {epoch:4d}/{epochs}: "
                  f"loss={total/max(nb,1):.4f}, val_acc={va:.1%}, "
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
                    },
                    "epoch": epoch, "val_acc": va,
                    "autoreg_acc": autoreg, "substrate": True,
                }, CHECKPOINT_PATH)
    print(f"[substrate-nl] done: {time.time()-t0:.0f}s, "
          f"best autoreg={best_autoreg:.1%} (teacher-forced best={best_autoreg:.1%})")


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
    p.add_argument("--epochs", type=int, default=500)
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
        tf_ratio_end=args.tf_ratio_end,
        device=args.device,
    )


if __name__ == "__main__":
    main()
