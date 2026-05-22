"""HRM-Text-1.58 Phase 1 Slice 2 — trainer.

Per task #51 + codex msg 1779452208756 (Phase 1 Slice 2 +1 implement
with corrections locked).

Custom training loop per D1.7. Source-faithful HRM-Text architecture
from `calm.hrm_text_158`, ported from sapientinc/HRM-Text SHA 056c4ec.

Deviations active per RESEARCH/HRM-Text-1.58/01_DEVIATIONS.md:
- D1.1: Tier A mini-capacity (hidden=256, n_layers=4 split, head_dim=128)
- D1.3: AdamW (lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
- D1.4: single-GPU (no FSDP/dist.all_reduce)
- D1.5: claw-code Gsm8kTokenizer + Gsm8kDataset (char-level, GSM8k corpus)
- D1.7: custom training loop (NOT vendored pretrain.py). The
  `compute_train_extra_args(step, total_steps)` interface is a
  simplification of upstream's `train_state` object — part of this
  custom-loop deviation.

Slice 13m carryover: only the repeatable multi-`--save-at-step` pattern,
per commit 38c3032 (TRM-1.58 Slice 13m, prior receipt msg
1779447055338-e1ee34dc). No trainer architecture or RDT logic inherited.

Label contract (source-faithful PrefixLM left-shift per Slice 1 test
`test_label_mask_tokenizer_tied`):
    inputs = ids[:-1]
    labels[:sep_pos] = IGNORE_LABEL_ID
    labels[sep_pos:] = ids[sep_pos+1:]
    labels at and after EOS position = IGNORE_LABEL_ID (padding)
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Reuse existing GSM8k tokenizer (read-only, RDT/DeltaNet-free).
from calm.llm_computer.gsm8k_tokenizer import Gsm8kTokenizer, NORMALIZER_VERSION

# HRM-Text-1.58 model
from calm.hrm_text_158 import (
    HierarchicalReasoningModel,
    HierarchicalReasoningModelConfig,
    LMHead,
    LMHeadConfig,
)
from calm.hrm_text_158.lm_head import IGNORE_LABEL_ID


# ----------------------------------------------------------------------------- #
# Neutral GSM8k splits loader (inlined to avoid RDT/Delta import leak)
# ----------------------------------------------------------------------------- #
# Provenance: logic copied verbatim from
# scripts/train_dt_gsm8k.py:57-89 (commit ancestry: feature/multi-agent-qwen).
# Inlined here so the HRM-Text-1.58 trainer does NOT transitively import
# `scripts.train_dt_gsm8k`, which pulls in `build_copy_augmented_delta`
# (DeltaNet) via its own top-level imports. Phase 1 guardrail: no
# RDT/Delta/copy imports anywhere in the HRM-Text-1.58 path. The
# multi-`--save-at-step` PATTERN is the only carryover; logic is fresh.

def load_gsm8k_splits(val_frac: float = 0.10) -> tuple[list[dict], list[dict], list[dict]]:
    """Load GSM8k via the `datasets` lib parquet backend.

    Returns (train, val, test). Train is 90% (deterministic head); val is
    10% (deterministic tail of train). Test is the full HF test split.

    Neutral inlined loader — no RDT/DeltaNet imports.
    """
    import re
    from datasets import load_dataset

    out: dict[str, list[dict]] = {"train": [], "test": []}
    for split in ("train", "test"):
        ds = load_dataset("openai/gsm8k", "main", split=split)
        for i, r in enumerate(ds):
            gt = r["answer"]
            m = re.search(r"####\s*(-?[\d,]+)", gt)
            if not m:
                continue
            try:
                expected = int(m.group(1).replace(",", "").strip())
            except ValueError:
                continue
            out[split].append({
                "id": f"gsm8k_{split}_{i}",
                "question": r["question"],
                "expected": expected,
                "answer_raw": gt,
            })
    full_train = out["train"]
    n_val = int(len(full_train) * val_frac)
    train = full_train[:-n_val] if n_val else full_train
    val = full_train[-n_val:] if n_val else []
    return train, val, out["test"]


SOURCE_PIN = {
    "repo": "github.com/sapientinc/HRM-Text",
    "sha": "056c4ecad217933b9db33dfb22e30a2f511315ed",
    "phase_0_audit": "RESEARCH/HRM-Text-1.58/00_ARCHITECTURE.md",
    "phase_0_deviations": "RESEARCH/HRM-Text-1.58/01_DEVIATIONS.md",
}


# ----------------------------------------------------------------------------- #
# Dataset wrapper (source-faithful shifted PrefixLM labels)
# ----------------------------------------------------------------------------- #

class HrmTextGsm8kDataset(Dataset):
    """Yields per-row dict matching LMHead's batch contract.

    Each row:
        inputs:        (L-1,) long  — ids[:-1] (drop EOS)
        labels:        (L-1,) long  — left-shifted, IGNORE on prefix + padding
        sep_position:  scalar long
        seq_len:       scalar long  — unpadded length (informational)

    Rows exceeding max_len are dropped (truncation rate same as
    Gsm8kDataset).
    """

    def __init__(self, rows: list[dict], tok: Gsm8kTokenizer, max_len: int):
        self.tok = tok
        self.max_len = max_len
        self.items: list[tuple[list[int], int]] = []  # (ids_full, sep_pos)
        n_dropped = 0
        for r in rows:
            ids, sep_pos = tok.encode_example(r["question"], r["expected"])
            if len(ids) > max_len:
                n_dropped += 1
                continue
            self.items.append((ids, sep_pos))
        self.n_dropped = n_dropped

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        ids_full, sep_pos = self.items[i]
        # Pad ids_full to max_len with pad_id
        pad_id = self.tok.pad_id
        ids_padded = list(ids_full) + [pad_id] * (self.max_len - len(ids_full))
        ids_padded = torch.tensor(ids_padded, dtype=torch.long)
        # Source-faithful shift: inputs = ids[:-1], labels = ids[1:] with prefix ignore
        inputs = ids_padded[:-1].contiguous()                 # (L-1,)
        labels = torch.full_like(inputs, IGNORE_LABEL_ID)
        # Real labels: positions sep_pos..eos_pos (inclusive of EOS)
        # ids_padded[sep_pos] = SEP, predicts ids_padded[sep_pos+1] = first target
        # labels[sep_pos] = ids_padded[sep_pos+1]
        labels[sep_pos:] = ids_padded[sep_pos + 1 :]
        # Pad positions (after EOS in original sequence): labels point to next pad,
        # which we don't want to train. Mask everything at-and-after the
        # position where ids_padded[i] == EOS (the EOS itself IS valid via the
        # left-shift; positions strictly after EOS are padding-predicting-padding).
        eos_id = self.tok.eos_id
        for pos in range(sep_pos + 1, len(ids_full)):
            if ids_full[pos] == eos_id:
                # In labels (length L-1, indices 0..L-2), labels[pos-1] = ids_full[pos] = EOS.
                # Anything at index pos..L-2 in labels predicts pad → ignore.
                if pos < labels.shape[0]:
                    labels[pos:] = IGNORE_LABEL_ID
                break
        return {
            "inputs": inputs,
            "labels": labels,
            "sep_position": torch.tensor(sep_pos, dtype=torch.long),
            "seq_len": torch.tensor(len(ids_full), dtype=torch.long),
        }


def _collate(batch: list[dict]) -> dict:
    return {
        "inputs": torch.stack([b["inputs"] for b in batch], dim=0),
        "labels": torch.stack([b["labels"] for b in batch], dim=0),
        "sep_positions": torch.stack([b["sep_position"] for b in batch], dim=0),
        # position_ids broadcasted from arange
    }


# ----------------------------------------------------------------------------- #
# LR schedule
# ----------------------------------------------------------------------------- #

def _lr_schedule(step: int, total_steps: int, warmup_steps: int, peak_lr: float,
                 min_lr: float = 1e-5) -> float:
    """Linear warmup + cosine decay."""
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    # Cosine decay
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine


# ----------------------------------------------------------------------------- #
# Train function
# ----------------------------------------------------------------------------- #

def train(
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 0.1,
    warmup_ratio: float = 0.1,
    # Tier A config (D1.1)
    hidden_size: int = 256,
    n_layers: int = 4,
    num_heads: int = 2,
    expansion: float = 4,
    H_cycles: int = 2,
    L_cycles: int = 3,
    half_layers: bool = True,
    bp_warmup_ratio: float = 0.2,
    bp_min_steps: int = 2,
    bp_max_steps: int = 5,
    max_len: int = 256,
    seed: int = 42,
    checkpoint_path: str = "calm/hrm/checkpoints/hrm_text_158_tier_a_best.pt",
    save_at_steps: list[int] | None = None,
    log_every: int = 50,
    n_train_cap: int | None = None,
    n_val_cap: int | None = None,
    device: str | None = None,
    splits_loader=load_gsm8k_splits,  # injectable for tests
) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    # Save-at-steps validation + dedupe (mirror Slice 13m pattern,
    # commit 38c3032, prior receipt msg 1779447055338-e1ee34dc)
    if save_at_steps is not None:
        for s in save_at_steps:
            if not isinstance(s, int) or s <= 0:
                raise ValueError(f"save_at_steps entries must be positive ints; got {s!r}")
        save_at_steps_set = frozenset(save_at_steps)
        print(f"[hrm158] save_at_steps ENABLED -> {sorted(save_at_steps_set)}", flush=True)
    else:
        save_at_steps_set = frozenset()

    # Load splits + build tokenizer
    print(f"[hrm158] loading GSM8k splits...", flush=True)
    full_train, full_val, test_rows = splits_loader(val_frac=0.10)
    print(f"[hrm158] splits: train={len(full_train)}  val={len(full_val)}  test={len(test_rows)}", flush=True)
    print(f"[hrm158] building tokenizer from full train+val (normalizer {NORMALIZER_VERSION})...", flush=True)
    tok = Gsm8kTokenizer.from_corpus(full_train + full_val)
    print(f"[hrm158] vocab: {tok.vocab_size} tokens", flush=True)
    tok.assert_corpus_covered(test_rows, label="test")

    train_rows = full_train[:n_train_cap] if n_train_cap is not None else full_train
    val_rows = full_val[:n_val_cap] if n_val_cap is not None else full_val

    train_ds = HrmTextGsm8kDataset(train_rows, tok, max_len=max_len)
    val_ds = HrmTextGsm8kDataset(val_rows, tok, max_len=max_len)
    print(f"[hrm158] usable rows after max_len={max_len} drop: "
          f"train={len(train_ds)} (dropped {train_ds.n_dropped}) "
          f"val={len(val_ds)} (dropped {val_ds.n_dropped})", flush=True)
    if len(train_ds) == 0:
        raise RuntimeError("No usable training rows after max_len drop.")

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate)

    # Build model
    cfg = HierarchicalReasoningModelConfig(
        max_seq_len=max_len,
        n_layers=n_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        expansion=expansion,
        H_cycles=H_cycles,
        L_cycles=L_cycles,
        half_layers=half_layers,
        bp_warmup_ratio=bp_warmup_ratio,
        bp_min_steps=bp_min_steps,
        bp_max_steps=bp_max_steps,
    )
    hrm = HierarchicalReasoningModel(cfg)
    m = LMHead(hrm, LMHeadConfig(vocab_size=tok.vocab_size)).to(device)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"[hrm158] params: {n_params:,}", flush=True)
    print(f"[hrm158] config: hidden={hidden_size} layers={n_layers} (half={half_layers}) "
          f"heads={num_heads} head_dim={hidden_size // num_heads} "
          f"H_cycles={H_cycles} L_cycles={L_cycles}", flush=True)

    # Optimizer + LR schedule
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    total_steps = epochs * len(loader)
    warmup_steps = int(total_steps * warmup_ratio)
    print(f"[hrm158] training: total_steps={total_steps} warmup_steps={warmup_steps} "
          f"lr={lr} weight_decay={weight_decay}", flush=True)

    # Train
    m.train()
    step = 0
    start_t = time.time()
    for ep in range(1, epochs + 1):
        for batch in loader:
            step += 1
            # Move to device + add position_ids
            inputs = batch["inputs"].to(device)
            labels = batch["labels"].to(device)
            sep_positions = batch["sep_positions"].to(device)
            B, L = inputs.shape
            position_ids = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)

            # LR schedule
            cur_lr = _lr_schedule(step, total_steps, warmup_steps, lr)
            for pg in opt.param_groups:
                pg["lr"] = cur_lr

            # bp_steps schedule via LMHead.compute_train_extra_args delegation
            extras = m.compute_train_extra_args(step, total_steps)

            # Forward + loss
            new_carry, loss, metrics = m(
                None,
                {"inputs": inputs, "labels": labels, "sep_positions": sep_positions,
                 "position_ids": position_ids},
                **extras,
            )

            if not torch.isfinite(loss):
                print(f"[NaN-DETECT] step={step} loss={loss.item()}", flush=True)
                sys.exit(2)

            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                print(f"[NaN-DETECT] step={step} grad_norm={grad_norm}", flush=True)
                sys.exit(2)
            opt.step()

            if step == 1 or step % log_every == 0:
                acc_count, acc_total = metrics["accuracy"]
                elapsed = time.time() - start_t
                print(f"[ep {ep:3d} step {step:5d}] loss={loss.item():.4f} "
                      f"grad_norm={float(grad_norm):.4f} lr={cur_lr:.6f} "
                      f"bp_steps={extras['bp_steps']} "
                      f"acc={int(acc_count)}/{int(acc_total)} t={elapsed:.1f}s",
                      flush=True)

            # Step-level save (Slice 13m pattern, multi)
            if step in save_at_steps_set:
                ckpt_path = Path(checkpoint_path).with_name(
                    Path(checkpoint_path).stem + f"_step{step:05d}.pt"
                )
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                ckpt_blob = {
                    "model_state": m.state_dict(),
                    "config": _build_ckpt_config(m, tok, cfg, max_len, batch_size),
                    "step": step,
                    "epoch": ep,
                    "source_pin": SOURCE_PIN,
                }
                torch.save(ckpt_blob, ckpt_path)
                print(f"[ep {ep:3d} step {step:5d}] save_at_step: saved {ckpt_path}", flush=True)

    print(f"[hrm158] training complete: {step} steps in {time.time() - start_t:.1f}s", flush=True)
    # Final save (best.pt — overwrite per Slice 13m pattern, name-only)
    final_path = Path(checkpoint_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_blob = {
        "model_state": m.state_dict(),
        "config": _build_ckpt_config(m, tok, cfg, max_len, batch_size),
        "step": step,
        "epoch": ep,
        "source_pin": SOURCE_PIN,
    }
    torch.save(ckpt_blob, final_path)
    print(f"[hrm158] final ckpt: {final_path}", flush=True)


def _build_ckpt_config(m, tok, cfg, max_len, batch_size) -> dict:
    """Single source of truth for ckpt config blob (per Slice 13m pattern)."""
    return {
        "vocab_size": tok.vocab_size,
        "gsm8k_char_vocab": tok.vocab_as_list(),
        "gsm8k_normalizer_version": tok.normalizer_version,
        "max_seq_len": cfg.max_seq_len,
        "n_layers": cfg.n_layers,
        "hidden_size": cfg.hidden_size,
        "num_heads": cfg.num_heads,
        "expansion": cfg.expansion,
        "H_cycles": cfg.H_cycles,
        "L_cycles": cfg.L_cycles,
        "half_layers": cfg.half_layers,
        "bp_warmup_ratio": cfg.bp_warmup_ratio,
        "bp_min_steps": cfg.bp_min_steps,
        "bp_max_steps": cfg.bp_max_steps,
        "norm_type": cfg.norm_type,
        "norm_eps": cfg.norm_eps,
        "rope_theta": cfg.rope_theta,
        "attn_type": cfg.attn_type,
        "init_type": cfg.init_type,
        "pos_emb_type": cfg.pos_emb_type,
        "max_len_runtime": max_len,
        "batch_size_runtime": batch_size,
    }


# ----------------------------------------------------------------------------- #
# CLI
# ----------------------------------------------------------------------------- #

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="HRM-Text-1.58 trainer (Phase 1 Slice 2). "
                    "Source-faithful port of sapientinc/HRM-Text SHA 056c4ec."
    )
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--num-heads", type=int, default=2)
    ap.add_argument("--expansion", type=float, default=4)
    ap.add_argument("--H-cycles", type=int, default=2)
    ap.add_argument("--L-cycles", type=int, default=3)
    ap.add_argument("--no-half-layers", action="store_true",
                    help="Disable half_layers (n_layers used as-is for both H and L)")
    ap.add_argument("--bp-warmup-ratio", type=float, default=0.2)
    ap.add_argument("--bp-min-steps", type=int, default=2)
    ap.add_argument("--bp-max-steps", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint-path", type=str,
                    default="calm/hrm/checkpoints/hrm_text_158_tier_a_best.pt")
    ap.add_argument("--save-at-step", type=int, action="append", default=None,
                    help="Repeatable. Pass multiple times (e.g. `--save-at-step 100 "
                         "--save-at-step 200`) to save at multiple step indices. "
                         "Pattern from TRM-1.58 Slice 13m commit 38c3032 "
                         "(prior receipt msg 1779447055338-e1ee34dc); pattern only, "
                         "not vendored logic.")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--n-train-cap", type=int, default=None)
    ap.add_argument("--n-val-cap", type=int, default=None)
    args = ap.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        expansion=args.expansion,
        H_cycles=args.H_cycles,
        L_cycles=args.L_cycles,
        half_layers=not args.no_half_layers,
        bp_warmup_ratio=args.bp_warmup_ratio,
        bp_min_steps=args.bp_min_steps,
        bp_max_steps=args.bp_max_steps,
        max_len=args.max_len,
        seed=args.seed,
        checkpoint_path=args.checkpoint_path,
        save_at_steps=args.save_at_step,
        log_every=args.log_every,
        n_train_cap=args.n_train_cap,
        n_val_cap=args.n_val_cap,
    )
