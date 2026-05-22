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
import random
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

def _stable_curriculum_seed(*parts) -> int:
    """Stable seed derivation for curriculum shuffle/sampling RNG (mirrors
    `calm.hrm_text_158.curriculum.generators._stable_seed`). Trainer-local
    copy so the import surface stays narrow."""
    import hashlib
    blob = repr(parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:4], "little")


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
    # Phase 2 D2.1: ternary bulk linears. When True, gqkv_proj/o_proj/
    # gate_up_proj/down_proj use BitLinear; lm_head/embd/norms/zL_init
    # stay FP per D2.2.
    use_ternary_bulk: bool = False,
    # Phase 3 Step 1 (codex msg 1779462307554-b57d8288):
    # Curriculum-mode replaces GSM8k corpus with synthetic per-rung
    # data. ALL fields optional; defaults preserve legacy GSM8k behavior.
    curriculum_rung: str | None = None,
    curriculum_seed: int = 42,
    curriculum_n_train: int = 4000,
    curriculum_n_heldout: int = 200,
    replay_ratio: float = 0.30,
    replay_rungs: str | None = None,
    use_broad_tokenizer: bool = False,
    load_from: str | None = None,
    dry_run: bool = False,
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

    # Curriculum-mode (Phase 3) vs GSM8k-mode (Phase 1/2) dispatcher
    prior_rungs: list[str] = []
    if curriculum_rung is not None:
        # Honest naming applies BEFORE any save (codex msg 1779463196431
        # secondary cleanup): rename `_best.pt` -> `_final.pt` once at top
        # so step-snapshot save-at-step files inherit the honest stem
        # (`..._final_step00100.pt`), not just the post-training final save.
        _ckpt_pre = Path(checkpoint_path)
        if _ckpt_pre.stem.endswith("_best"):
            _honest_stem = _ckpt_pre.stem[: -len("_best")] + "_final"
            checkpoint_path = str(_ckpt_pre.with_name(_honest_stem + _ckpt_pre.suffix))
            print(f"[hrm158] curriculum mode: --checkpoint-path stem _best -> _final "
                  f"({Path(checkpoint_path).name}) — no best-criterion selection runs",
                  flush=True)
        # Phase 3 curriculum corpus + broad tokenizer
        from calm.hrm_text_158.curriculum import (
            BroadTokenizer,
            RUNG_NAMES,
            make_rung_examples,
        )
        if curriculum_rung not in RUNG_NAMES:
            raise ValueError(f"--curriculum-rung must be one of {RUNG_NAMES}; got {curriculum_rung!r}")
        if not use_broad_tokenizer:
            raise ValueError(
                "--curriculum-rung requires --use-broad-tokenizer (Phase 3 design lock; "
                "byte-level UTF-8 fixed tokenizer across the rung chain). "
                "See 02_TOKENIZER_CONTRACT.md."
            )
        print(f"[hrm158] PHASE 3 curriculum mode: rung={curriculum_rung} replay_ratio={replay_ratio}", flush=True)
        tok = BroadTokenizer()
        print(f"[hrm158] BroadTokenizer (vocab={tok.vocab_size}, normalizer_version={tok.normalizer_version})", flush=True)

        # Resolve prior_rungs via shared helper (codex msg 1779475454122-1512da3b
        # structural fix). Helper validates explicit --replay-rungs (reject
        # unknown/current/future/R7/duplicate; allow diagnosis-only with WARN)
        # AND auto-excludes DIAGNOSIS_ONLY_RUNGS + R7 from positional default.
        from calm.hrm_text_158.curriculum.replay import (
            _resolve_prior_rungs,
            DIAGNOSIS_ONLY_RUNGS,
        )
        cur_idx = list(RUNG_NAMES).index(curriculum_rung)
        positional_full = list(RUNG_NAMES[:cur_idx])
        prior_rungs = _resolve_prior_rungs(curriculum_rung, replay_rungs)
        print(f"[hrm158] curriculum {curriculum_rung}: prior_rungs={prior_rungs} "
              f"(positional_full={positional_full}, "
              f"diagnosis_only={sorted(DIAGNOSIS_ONLY_RUNGS)}, "
              f"explicit_override={replay_rungs is not None})",
              flush=True)

        # Mandatory --load-from for R1+ (codex msg 1779463196431 rule 1):
        # curriculum builds via WEIGHTS continuity; random-init train at
        # R1+ breaks the checkpoint-chain contract. R0 is the only rung
        # permitted from random init.
        if prior_rungs and load_from is None:
            raise ValueError(
                f"--curriculum-rung {curriculum_rung!r} requires --load-from PATH "
                f"(prior rungs to chain from: {prior_rungs}). Curriculum builds via "
                f"weights continuity across the rung checkpoint chain; random-init "
                f"training at R1+ breaks the chain contract. R0 is the only rung "
                f"permitted from random init."
            )

        # Replay mix: (1 - replay_ratio) of train is the new rung; the
        # rest is uniformly split across prior rungs.
        #
        # R0 special case: no prior rungs means nothing to replay, so the
        # full curriculum_n_train budget goes to the new rung regardless
        # of CLI --replay-ratio. effective_replay_ratio is logged in the
        # ckpt config so the probe receipt reports what actually ran
        # (codex msg 1779462307554 rule 4).
        if prior_rungs:
            n_new = max(1, int(curriculum_n_train * (1.0 - replay_ratio)))
            n_replay_total = curriculum_n_train - n_new
            per_prior = max(1, n_replay_total // len(prior_rungs)) if n_replay_total > 0 else 0
            effective_replay_ratio = float(replay_ratio)
        else:
            n_new = curriculum_n_train
            n_replay_total = 0
            per_prior = 0
            effective_replay_ratio = 0.0
            if replay_ratio > 0:
                print(f"[hrm158] curriculum {curriculum_rung}: no prior rungs to replay; "
                      f"overriding --replay-ratio={replay_ratio} -> effective 0.0",
                      flush=True)

        train_rows: list[dict] = list(make_rung_examples(
            curriculum_rung, n=n_new, seed=curriculum_seed, split="train"
        ))
        replay_samples_by_rung: dict[str, int] = {}
        for pr in prior_rungs:
            if per_prior == 0:
                replay_samples_by_rung[pr] = 0
                continue
            replay_rows = make_rung_examples(pr, n=per_prior, seed=curriculum_seed, split="train")
            train_rows.extend(replay_rows)
            replay_samples_by_rung[pr] = len(replay_rows)

        # Held-out: new rung only (probe handles prior-rung retention separately via probe script)
        val_rows = list(make_rung_examples(
            curriculum_rung, n=curriculum_n_heldout, seed=curriculum_seed, split="held_out"
        ))

        # Deterministic shuffle of train corpus (so replay isn't tail-stacked)
        shuffle_rng = random.Random(_stable_curriculum_seed("shuffle", curriculum_rung, curriculum_seed))
        shuffle_rng.shuffle(train_rows)

        # Cap if requested (n_train_cap dominates; n_val_cap likewise)
        if n_train_cap is not None:
            train_rows = train_rows[:n_train_cap]
        if n_val_cap is not None:
            val_rows = val_rows[:n_val_cap]

        print(f"[hrm158] curriculum {curriculum_rung}: train={len(train_rows)} "
              f"({n_new} new + {sum(replay_samples_by_rung.values())} replay {replay_samples_by_rung}) "
              f"held_out={len(val_rows)}", flush=True)

        # Empty test split (curriculum has no separate test corpus; rung-cross retention is the eval)
        test_rows: list[dict] = []
        tok.assert_corpus_covered(train_rows + val_rows, label="curriculum")

    else:
        # Phase 1/2 GSM8k path (unchanged)
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
        use_ternary_bulk=use_ternary_bulk,
    )
    if use_ternary_bulk:
        print(f"[hrm158] Phase 2 D2.1: TERNARY BULK LINEARS ENABLED "
              f"(gqkv/o/gate_up/down → BitLinear; lm_head/embd/norms FP per D2.2)",
              flush=True)
    hrm = HierarchicalReasoningModel(cfg)
    m = LMHead(hrm, LMHeadConfig(vocab_size=tok.vocab_size)).to(device)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"[hrm158] params: {n_params:,}", flush=True)
    print(f"[hrm158] config: hidden={hidden_size} layers={n_layers} (half={half_layers}) "
          f"heads={num_heads} head_dim={hidden_size // num_heads} "
          f"H_cycles={H_cycles} L_cycles={L_cycles}", flush=True)

    # Phase 3 --load-from: compat-validate + load model_state ONLY (optimizer
    # state + LR schedule RESET per rung; curriculum builds primitives via
    # WEIGHTS continuity, not optimizer momentum). Per codex msg
    # 1779462307554 receipt requirement.
    if load_from is not None:
        from calm.hrm_text_158.curriculum import validate_load_from_ckpt_compat
        print(f"[hrm158] --load-from: {load_from}", flush=True)
        loaded_ckpt = torch.load(load_from, map_location="cpu", weights_only=False)
        loaded_cfg_blob = loaded_ckpt.get("config")
        if loaded_cfg_blob is None:
            raise ValueError(f"--load-from ckpt {load_from!r} missing 'config' field")
        validate_load_from_ckpt_compat(
            loaded_ckpt_config=loaded_cfg_blob,
            current_cfg=cfg,
            current_vocab_list=tok.vocab_as_list(),
            current_normalizer_version=tok.normalizer_version,
        )
        print(f"[hrm158] --load-from compat OK; loading model_state strict", flush=True)
        m.load_state_dict(loaded_ckpt["model_state"], strict=True)
        print(f"[hrm158] --load-from loaded; optimizer state + LR schedule will RESET per rung", flush=True)

    # Optimizer + LR schedule
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    total_steps = epochs * len(loader)
    warmup_steps = int(total_steps * warmup_ratio)
    print(f"[hrm158] training: total_steps={total_steps} warmup_steps={warmup_steps} "
          f"lr={lr} weight_decay={weight_decay}", flush=True)

    # Phase 3 --dry-run: build corpus + model + first batch + verify forward,
    # then exit BEFORE optimizer step. Used for Phase A receipt validation
    # without burning GPU time. Per codex msg 1779462307554 receipt requirement.
    if dry_run:
        # Corpus stats for throughput-relevant receipt (codex msg
        # 1779462666282-23cbaa3a gabe relay: "these smaller checkpoints are
        # faster to train too right" -- measure, don't claim).
        enc_lens = [len(items[0]) for items in train_ds.items]
        if enc_lens:
            avg_enc_len = sum(enc_lens) / len(enc_lens)
            max_enc_len_seen = max(enc_lens)
            total_tokens_est = int(avg_enc_len * len(train_ds) * epochs)
        else:
            avg_enc_len = 0.0
            max_enc_len_seen = 0
            total_tokens_est = 0
        print(f"[hrm158] --dry-run: corpus stats train_rows_usable={len(train_ds)} "
              f"val_rows_usable={len(val_ds)} avg_enc_len={avg_enc_len:.1f} "
              f"max_enc_len_seen={max_enc_len_seen} "
              f"total_tokens_est={total_tokens_est}", flush=True)
        print(f"[hrm158] --dry-run: validating first batch forward pass...", flush=True)
        first_batch = next(iter(loader))
        inputs = first_batch["inputs"].to(device)
        labels = first_batch["labels"].to(device)
        sep_positions = first_batch["sep_positions"].to(device)
        B, L = inputs.shape
        position_ids = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)
        extras = m.compute_train_extra_args(0, max(1, total_steps))
        with torch.no_grad():
            _new_carry, dry_loss, _metrics = m(
                None,
                {"inputs": inputs, "labels": labels, "sep_positions": sep_positions,
                 "position_ids": position_ids},
                **extras,
            )
        dry_finite = bool(torch.isfinite(dry_loss).item())
        print(f"[hrm158] --dry-run: first_batch shape inputs={tuple(inputs.shape)} "
              f"labels={tuple(labels.shape)} sep_positions={tuple(sep_positions.shape)}", flush=True)
        print(f"[hrm158] --dry-run: forward OK loss={dry_loss.item():.4f} finite={dry_finite}", flush=True)
        print(f"[hrm158] --dry-run: EXITING before optimizer step (no GPU training; "
              f"no ckpt written)", flush=True)
        return

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
                    "config": _build_ckpt_config(
                        m, tok, cfg, max_len, batch_size,
                        curriculum_rung=curriculum_rung,
                        curriculum_seed=curriculum_seed,
                        replay_ratio=effective_replay_ratio if curriculum_rung else 0.0,
                        prior_rungs=prior_rungs,
                    ),
                    "step": step,
                    "epoch": ep,
                    "source_pin": SOURCE_PIN,
                }
                torch.save(ckpt_blob, ckpt_path)
                print(f"[ep {ep:3d} step {step:5d}] save_at_step: saved {ckpt_path}", flush=True)

    print(f"[hrm158] training complete: {step} steps in {time.time() - start_t:.1f}s", flush=True)
    # Final save.
    #
    # Phase 1/2 GSM8k path: uses `checkpoint_path` as given (legacy naming
    # like `..._tier_a_best.pt`; existing Slice 13m / 13h pattern, no
    # best-criterion selection — file is the FINAL step's weights, not
    # "best" by any metric, but kept under the legacy name for backwards
    # compat).
    #
    # Phase 3 curriculum path (codex msg 1779462307554 rule 1 — honest
    # naming): rewrites `_best.pt` -> `_final.pt` so the on-disk name
    # accurately reflects "final-step weights, no best-criterion selection".
    # Pattern: e.g. `hrm_text_158_phase3_R0_best.pt` -> `..._R0_final.pt`.
    final_path = Path(checkpoint_path)
    if curriculum_rung is not None and final_path.stem.endswith("_best"):
        honest_stem = final_path.stem[: -len("_best")] + "_final"
        final_path = final_path.with_name(honest_stem + final_path.suffix)
        print(f"[hrm158] curriculum mode: renaming checkpoint to honest final "
              f"({final_path.name}) — no best-criterion selection ran", flush=True)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_blob = {
        "model_state": m.state_dict(),
        "config": _build_ckpt_config(
            m, tok, cfg, max_len, batch_size,
            curriculum_rung=curriculum_rung,
            curriculum_seed=curriculum_seed,
            replay_ratio=effective_replay_ratio if curriculum_rung else 0.0,
            prior_rungs=prior_rungs,
        ),
        "step": step,
        "epoch": ep,
        "source_pin": SOURCE_PIN,
    }
    torch.save(ckpt_blob, final_path)
    print(f"[hrm158] final ckpt: {final_path}", flush=True)


def _build_ckpt_config(
    m,
    tok,
    cfg,
    max_len,
    batch_size,
    *,
    curriculum_rung: str | None = None,
    curriculum_seed: int = 42,
    replay_ratio: float = 0.0,
    prior_rungs: list[str] | None = None,
) -> dict:
    """Single source of truth for ckpt config blob (per Slice 13m pattern).

    Phase 3 additions (curriculum_rung / replay_ratio / prior_rungs) are
    populated only when training in curriculum mode; absent on legacy
    GSM8k ckpts.
    """
    out: dict = {
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
        "use_ternary_bulk": cfg.use_ternary_bulk,
        "max_len_runtime": max_len,
        "batch_size_runtime": batch_size,
    }
    if curriculum_rung is not None:
        out["curriculum_rung"] = curriculum_rung
        out["curriculum_seed"] = curriculum_seed
        out["replay_ratio"] = replay_ratio
        out["prior_rungs"] = list(prior_rungs or [])
    return out


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
    ap.add_argument("--use-ternary-bulk", action="store_true",
                    help="Phase 2 D2.1: replace bulk LinearInit with BitLinear "
                         "(ternary master+STE) on gqkv_proj/o_proj/gate_up_proj/"
                         "down_proj. lm_head/embd/norms/zL_init stay FP per D2.2.")
    # Phase 3 Step 1 curriculum flags (codex msg 1779462307554 +1 implement Phase A)
    ap.add_argument("--curriculum-rung", type=str, default=None,
                    choices=["R0", "R1", "R1b1", "R1b2a", "R1b2", "R1b", "R2", "R3", "R4", "R5", "R6"],
                    help="Phase 3 curriculum mode. When set, swaps GSM8k corpus "
                         "for synthetic per-rung data + replay mix. Requires "
                         "--use-broad-tokenizer in Phase 3 design.")
    ap.add_argument("--curriculum-seed", type=int, default=42,
                    help="Deterministic seed for curriculum generator + shuffle (default 42).")
    ap.add_argument("--curriculum-n-train", type=int, default=4000,
                    help="Total train rows per rung (default 4000). Includes both "
                         "new-rung rows and replay-from-prior-rung rows.")
    ap.add_argument("--curriculum-n-heldout", type=int, default=200,
                    help="Held-out probe rows for current rung (default 200).")
    ap.add_argument("--replay-ratio", type=float, default=0.30,
                    help="Fraction of train mixed from prior rungs (default 0.30 per "
                         "codex msg 1779462307554 rule 4). Effective ratio is logged "
                         "into the ckpt config blob.")
    ap.add_argument("--replay-rungs", type=str, default=None,
                    help="Comma-separated explicit rung list to draw replay from. "
                         "Overrides positional RUNG_NAMES[:cur_idx] derivation. "
                         "Use to exclude diagnosis-only or failed rungs from replay "
                         "(e.g. --replay-rungs R0,R1,R1b1 when targeting R1b2 after "
                         "R1b2a failed and stays diagnosis-only). Validation rejects "
                         "unknown/current/future/R7/duplicate entries; diagnosis-only "
                         "in list emits WARN. Codex msg 1779475454122-1512da3b "
                         "structural fix.")
    ap.add_argument("--use-broad-tokenizer", action="store_true",
                    help="Use BroadTokenizer (byte-level UTF-8, vocab=260, "
                         "normalizer_version=byte_utf8_v1) instead of Gsm8kTokenizer. "
                         "Required for --curriculum-rung in Phase 3.")
    ap.add_argument("--load-from", type=str, default=None,
                    help="Path to prior-rung ckpt. validate_load_from_ckpt_compat "
                         "runs first (hard-fails on vocab/normalizer/ternary/arch "
                         "mismatch); then model_state loads strict; optimizer state "
                         "+ LR schedule RESET per rung.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build corpus + tokenizer + model + first batch + verify "
                         "forward pass, then exit BEFORE optimizer step. No ckpt "
                         "written. Used for Phase A receipt validation.")
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
        use_ternary_bulk=args.use_ternary_bulk,
        curriculum_rung=args.curriculum_rung,
        curriculum_seed=args.curriculum_seed,
        curriculum_n_train=args.curriculum_n_train,
        replay_rungs=args.replay_rungs,
        curriculum_n_heldout=args.curriculum_n_heldout,
        replay_ratio=args.replay_ratio,
        use_broad_tokenizer=args.use_broad_tokenizer,
        load_from=args.load_from,
        dry_run=args.dry_run,
    )
