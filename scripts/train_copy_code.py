"""Train copy-augmented PT for code signature extraction (R53.5).

Maps NL problem statements → "def name(args)" or "class Name(bases)".

Training data: union of
  - agents/distill/data/generated/pt_*.jsonl (222 generator examples,
    signature stripped from "def x() | algo")
  - agents/distill/data/generated/pt_db_signature.jsonl (2674 DB
    extractions via CodeExampleDB.export_pt_signature_jsonl)

Architecture: CopyAugmentedTransformer at ~185K params (d_model=64,
n_heads=32, n_layers=4, d_ffn=128, max_len=288 — bumped from 208 for
longer code prompts).

Output: calm/hrm/checkpoints/copy_code_best.pt

Gate: ≥85% autoregressive accuracy on held-out 10% split.
If below, hypothesis: noise from advisory-Q examples in DB. Filter
pt_db_signature.jsonl to clean-source whitelist and retrain.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from calm.hrm.data import VOCAB_SIZE, _CHAR_TO_ID, _ID_TO_CHAR
from calm.llm_computer.copy_augmented import build_copy_augmented_hrm


CHECKPOINT_PATH = Path("calm/hrm/checkpoints/copy_code_best.pt")
DATA_GLOB = "agents/distill/data/generated/pt_*.jsonl"
DB_PT_PATH = "agents/distill/data/generated/pt_db_signature.jsonl"


@dataclass
class CodeProblem:
    """PT training pair. Mirrors what ReasoningDataGenerator returns
    so we can reuse the existing CopyAugmentedTransformer trainer
    structure verbatim."""
    problem: str
    expression: str       # the target signature; "expression" name kept
                          # for compatibility with autoreg eval helpers


class CodeSeqDataset(Dataset):
    def __init__(self, problems, max_len=288):
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


def _load_problems(max_len: int = 288) -> list[CodeProblem]:
    """Load union of generator pt_*.jsonl (signature-stripped) +
    DB pt_db_signature.jsonl. Filters examples whose
    prompt + target + 3 special tokens exceed max_len — silently
    truncating the input would destroy the target signal."""
    out: list[CodeProblem] = []
    seen_prompts: set[str] = set()
    n_skip_long = 0
    n_skip_vocab = 0

    def _maybe_add(prompt: str, sig: str) -> int:
        nonlocal n_skip_long, n_skip_vocab
        if any(c not in _CHAR_TO_ID for c in sig):
            n_skip_vocab += 1
            return 0
        prompt_len = sum(1 for c in prompt if c in _CHAR_TO_ID)
        if prompt_len + len(sig) + 3 > max_len:
            n_skip_long += 1
            return 0
        key = prompt.strip()
        if key in seen_prompts:
            return 0
        seen_prompts.add(key)
        out.append(CodeProblem(problem=prompt, expression=sig))
        return 1

    # 1) Generator pt_*.jsonl files (target may contain "| algo" — strip)
    for path in sorted(glob.glob(DATA_GLOB)):
        if path.endswith("/pt_db_signature.jsonl"):
            continue
        n_added = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                sig = obj["target"].split(" | ", 1)[0]
                n_added += _maybe_add(obj["prompt"], sig)
        print(f"[copy-code] loaded {n_added:>5} from {Path(path).name}",
              flush=True)

    # 2) DB-extracted signatures
    db_path = Path(DB_PT_PATH)
    if db_path.exists():
        n_added = 0
        with open(db_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                n_added += _maybe_add(obj["prompt"], obj["target"])
        print(f"[copy-code] loaded {n_added:>5} from {db_path.name}",
              flush=True)

    print(f"[copy-code] filtered: {n_skip_long} too long for max_len={max_len},"
          f" {n_skip_vocab} contain out-of-vocab chars", flush=True)
    return out


def _autoreg_eval(model, problems, device, max_gen=80):
    bos = _CHAR_TO_ID["<bos>"]
    sep = _CHAR_TO_ID["<sep>"]
    eos = _CHAR_TO_ID["<eos>"]
    pos_limit = model.config.max_len
    model.eval()
    correct = 0
    for p in problems:
        ids = [bos] + [_CHAR_TO_ID[c] for c in p.problem
                       if c in _CHAR_TO_ID] + [sep]
        gen = []
        gen_budget = min(max_gen, pos_limit - len(ids) - 1)
        for _ in range(gen_budget):
            x = torch.tensor([ids + gen], dtype=torch.long, device=device)
            with torch.no_grad():
                lp = model(x)
            tok = int(lp[0, -1].argmax().item())
            if tok == eos:
                break
            gen.append(tok)
        out = "".join(_ID_TO_CHAR[t] for t in gen)
        if out == p.expression:
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


def train(epochs=500, batch_size=64, lr=1e-3,
          d_model=64, n_heads=32, n_layers=4, d_ffn=128,
          max_len=288, n_copy_heads=4, seed=42, eval_every=10,
          scheduled_sampling=True, tf_ratio_start=1.0, tf_ratio_end=0.3,
          device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    probs = _load_problems(max_len=max_len)
    print(f"[copy-code] total problems: {len(probs)}", flush=True)
    if not probs:
        print("[copy-code] no training data — abort", flush=True)
        return

    ds = CodeSeqDataset(probs, max_len=max_len)
    val_size = max(1, len(ds) // 10)
    train_set, val_set = random_split(
        ds, [len(ds) - val_size, val_size],
        generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_set, batch_size=batch_size,
                               shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    val_probs = [probs[i] for i in val_set.indices]

    model = build_copy_augmented_hrm(
        vocab_size=VOCAB_SIZE, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ffn=d_ffn, max_len=max_len,
        n_copy_heads=n_copy_heads, use_hard_max=False,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[copy-code] model: {total_params:,} params on {device}",
          flush=True)
    print(f"[copy-code] train={len(train_set)} val={len(val_set)}",
          flush=True)
    if scheduled_sampling:
        print(f"[copy-code] scheduled sampling: tf_ratio "
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
        tf_ratio = (tf_ratio_start
                    - (tf_ratio_start - tf_ratio_end) * (epoch / epochs))

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
            print(f"[copy-code] epoch {epoch:4d}/{epochs}: "
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
                        "n_copy_heads": n_copy_heads,
                    },
                    "epoch": epoch, "val_acc": va,
                    "autoreg_acc": autoreg, "copy_augmented": True,
                    "domain": "code",
                }, CHECKPOINT_PATH)

    print(f"[copy-code] DONE: {time.time()-t0:.0f}s, "
          f"best autoreg={best_autoreg:.1%}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--d-ffn", type=int, default=128)
    p.add_argument("--max-len", type=int, default=288)
    p.add_argument("--n-copy-heads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--no-scheduled-sampling", action="store_true")
    p.add_argument("--tf-ratio-end", type=float, default=0.3)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()
    train(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ffn=args.d_ffn,
        max_len=args.max_len, n_copy_heads=args.n_copy_heads,
        seed=args.seed, eval_every=args.eval_every,
        scheduled_sampling=not args.no_scheduled_sampling,
        tf_ratio_end=args.tf_ratio_end, device=args.device,
    )


if __name__ == "__main__":
    main()
