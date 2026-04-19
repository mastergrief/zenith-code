"""R53.13 — PT small-sample test on aligned code signatures.

R53.8's broad-corpus PT plateaued at autoreg ~6% across 280 epochs
despite loss dropping cleanly (3.18 → 0.04). Diagnosis: many training
prompts (advisory questions, broad descriptions) don't contain the
target function name, so the copy mechanism has nothing to copy.

Mirrors the HRM→PT evolution discipline: identify the architecture's
sweet spot, build a SMALL dataset that fits it, train fast, measure
the gate. If ≥80% autoreg on aligned subset, scale up. If not, PT
is wrong for this task.

This script:
  1. Audits training data for prompt-target name alignment
  2. Filters to aligned subset (target name appears in prompt)
  3. Trains PT for shorter run on that subset
  4. Reports autoreg accuracy as the gate metric

Usage:
  PYTHONPATH=. python3 scripts/r53_pt_small_sample.py --audit
  PYTHONPATH=. python3 scripts/r53_pt_small_sample.py --train
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from calm.hrm.data import VOCAB_SIZE, _CHAR_TO_ID, _ID_TO_CHAR
from calm.llm_computer.copy_augmented import build_copy_augmented_hrm


CHECKPOINT_PATH = Path("calm/hrm/checkpoints/copy_code_aligned_best.pt")
DATA_GLOB = "agents/distill/data/generated/pt_*.jsonl"
DB_PT_PATH = "agents/distill/data/generated/pt_db_signature.jsonl"

# Regex to pull the function or class name from a target signature
_SIG_NAME_RE = re.compile(r"^(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")


@dataclass
class CodeProblem:
    problem: str
    expression: str   # the target signature


def extract_target_name(target: str) -> str | None:
    """Pull the function/class name from a 'def name(...)' or 'class Name'
    signature. Returns None if format unexpected."""
    sig = target.split(" | ", 1)[0]
    m = _SIG_NAME_RE.match(sig)
    return m.group(1) if m else None


def is_aligned(prompt: str, target_name: str) -> bool:
    """True if the target's function/class name appears verbatim in
    the prompt (case-sensitive). This is what makes the copy
    mechanism actually copyable from input."""
    return target_name in prompt


def load_all_pairs(max_len: int = 288) -> list[tuple[str, str, str]]:
    """Load all (prompt, target, source) triples from generator + DB
    pt_*.jsonl files. Same length filter as train_copy_code.py."""
    out: list[tuple[str, str, str]] = []
    seen = set()
    for path in sorted(glob.glob(DATA_GLOB)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                prompt = obj["prompt"]
                target = obj["target"].split(" | ", 1)[0]
                if any(c not in _CHAR_TO_ID for c in target):
                    continue
                p_len = sum(1 for c in prompt if c in _CHAR_TO_ID)
                if p_len + len(target) + 3 > max_len:
                    continue
                key = prompt.strip()
                if key in seen:
                    continue
                seen.add(key)
                out.append((prompt, target, Path(path).name))
    return out


def audit(max_len: int = 288):
    """Audit alignment + signature distribution."""
    pairs = load_all_pairs(max_len=max_len)
    print(f"Total pairs (length-filtered): {len(pairs)}")

    aligned = []
    unaligned = []
    no_name = []
    for prompt, target, source in pairs:
        name = extract_target_name(target)
        if name is None:
            no_name.append((prompt, target, source))
        elif is_aligned(prompt, name):
            aligned.append((prompt, target, source, name))
        else:
            unaligned.append((prompt, target, source, name))

    print(f"\nALIGNMENT BREAKDOWN:")
    print(f"  aligned (name appears in prompt):   {len(aligned)}")
    print(f"  unaligned (name NOT in prompt):     {len(unaligned)}")
    print(f"  no parseable signature:             {len(no_name)}")

    print(f"\nALIGNED by source:")
    from collections import Counter
    cnt = Counter(s for _, _, s, _ in aligned)
    for src, n in cnt.most_common():
        total = sum(1 for _, _, s in pairs if s == src)
        print(f"  {src:<35} {n:>4}/{total:<4} ({n/max(total,1)*100:>4.0f}%)")

    print(f"\nUNALIGNED samples (target name NOT in prompt):")
    for prompt, target, source, name in unaligned[:5]:
        print(f"  source={source}")
        print(f"    prompt: {prompt[:100]!r}")
        print(f"    target: {target!r}  (name={name!r})")
        print()

    print(f"\nALIGNED samples (target name IS in prompt):")
    for prompt, target, source, name in aligned[:5]:
        print(f"  source={source}")
        print(f"    prompt: {prompt[:100]!r}")
        print(f"    target: {target!r}  (name={name!r})")
        print()


def load_aligned_problems(max_len: int = 288) -> list[CodeProblem]:
    pairs = load_all_pairs(max_len=max_len)
    out: list[CodeProblem] = []
    for prompt, target, _source in pairs:
        name = extract_target_name(target)
        if name is None or not is_aligned(prompt, name):
            continue
        out.append(CodeProblem(problem=prompt, expression=target))
    return out


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
        target_ids = [_CHAR_TO_ID[c] for c in p.expression
                       if c in _CHAR_TO_ID]
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


def train(epochs=200, batch_size=32, lr=1e-3,
          d_model=64, n_heads=32, n_layers=4, d_ffn=128,
          max_len=288, n_copy_heads=4, seed=42, eval_every=10,
          tf_ratio_start=1.0, tf_ratio_end=0.3, device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    probs = load_aligned_problems(max_len=max_len)
    print(f"[copy-code-aligned] aligned problems: {len(probs)}",
          flush=True)
    if not probs:
        print("[copy-code-aligned] no aligned data — abort", flush=True)
        return

    ds = CodeSeqDataset(probs, max_len=max_len)
    val_size = max(1, len(ds) // 10)
    train_set, val_set = random_split(
        ds, [len(ds) - val_size, val_size],
        generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_probs = [probs[i] for i in val_set.indices]

    model = build_copy_augmented_hrm(
        vocab_size=VOCAB_SIZE, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ffn=d_ffn, max_len=max_len,
        n_copy_heads=n_copy_heads, use_hard_max=False,
    ).to(device)
    print(f"[copy-code-aligned] {sum(p.numel() for p in model.parameters()):,} "
          f"params on {device}", flush=True)
    print(f"[copy-code-aligned] train={len(train_set)} val={len(val_set)}",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_autoreg = 0.0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, nb = 0.0, 0
        tf_ratio = (tf_ratio_start
                    - (tf_ratio_start - tf_ratio_end) * (epoch / epochs))
        for b in train_loader:
            x = b["input_ids"].to(device)
            y = b["target_ids"].to(device)
            m = b["loss_mask"].to(device)
            B, S = x.shape
            if tf_ratio < 0.99:
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
            autoreg = _autoreg_eval(
                model, val_probs[:min(50, len(val_probs))], device)
            print(f"[copy-code-aligned] epoch {epoch:4d}/{epochs}: "
                  f"loss={total_loss/max(nb,1):.4f}, autoreg={autoreg:.1%}, "
                  f"tf={tf_ratio:.2f}, elapsed={time.time()-t0:.0f}s",
                  flush=True)
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
                    "epoch": epoch, "autoreg_acc": autoreg,
                    "domain": "code_aligned",
                }, CHECKPOINT_PATH)
    print(f"[copy-code-aligned] DONE: best autoreg={best_autoreg:.1%}",
          flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audit", action="store_true",
                   help="just audit alignment, don't train")
    p.add_argument("--train", action="store_true", help="run training")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()
    if args.audit:
        audit()
    elif args.train:
        train(epochs=args.epochs, batch_size=args.batch_size,
              device=args.device)
    else:
        audit()


if __name__ == "__main__":
    main()
