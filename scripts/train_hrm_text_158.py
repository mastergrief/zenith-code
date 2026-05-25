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
from calm.hrm_text_158.curriculum.retention_anchors import (
    load_anchor_set,
    RETENTION_ANCHOR_SETS,
)


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

    def __init__(self, rows: list[dict], tok: Gsm8kTokenizer, max_len: int,
                 curriculum_rung: str | None = None):
        self.tok = tok
        self.max_len = max_len
        # (ids_full, sep_pos, is_prior). is_prior marks a retained-skill row
        # (prior rung or anchor) for the parent-consistency KL mask. A row is
        # prior when its generator `rung` field differs from the target
        # curriculum rung; anchors carry no `rung` (None != rung) so they
        # count as prior. Off-curriculum (curriculum_rung is None) → all False.
        self.items: list[tuple[list[int], int, bool]] = []
        n_dropped = 0
        for r in rows:
            ids, sep_pos = tok.encode_example(r["question"], r["expected"])
            if len(ids) > max_len:
                n_dropped += 1
                continue
            is_prior = (curriculum_rung is not None
                        and r.get("rung") != curriculum_rung)
            self.items.append((ids, sep_pos, is_prior))
        self.n_dropped = n_dropped

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        ids_full, sep_pos, is_prior = self.items[i]
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
            "is_prior": torch.tensor(is_prior, dtype=torch.bool),
        }


def _collate(batch: list[dict]) -> dict:
    return {
        "inputs": torch.stack([b["inputs"] for b in batch], dim=0),
        "labels": torch.stack([b["labels"] for b in batch], dim=0),
        "sep_positions": torch.stack([b["sep_position"] for b in batch], dim=0),
        # is_prior is trainer-only (parent-consistency KL mask). It is NOT put
        # in the dict passed to LMHead — that would route it into the model's
        # seq_info. The train loop reads batch["is_prior"] directly.
        "is_prior": torch.stack([b["is_prior"] for b in batch], dim=0),
        # position_ids broadcasted from arange
    }


def _build_train_loader(train_ds, batch_size, seed, legacy_loader_shuffle,
                        collate_fn=_collate):
    """Construct the curriculum training DataLoader.

    Default (`legacy_loader_shuffle=False`): an explicit `torch.Generator`
    seeded by `seed`, so the shuffle order depends ONLY on `--seed`, decoupled
    from however much global RNG model-init consumed — the post-1656ead
    deterministic path.

    Diagnostic (`legacy_loader_shuffle=True`, codex msg 1779652915624): NO
    explicit generator → the pre-1656ead global-RNG shuffle order. Isolation
    use ONLY (NOT recipe-default) — to test whether the seed-decoupled
    generator moved the fragile `10 minus 1` borrow-boundary row in F.2f.
    """
    if legacy_loader_shuffle:
        return DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          collate_fn=collate_fn)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                      collate_fn=collate_fn, generator=gen)


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


def _parent_consistency_kl(
    child_logits: "torch.Tensor",
    parent_logits: "torch.Tensor",
    labels: "torch.Tensor",
    is_prior: "torch.Tensor",
    temp: float = 1.0,
) -> "torch.Tensor":
    """Soft forward-KL(parent || child) on retained-skill rows.

    Penalizes the child for diverging from the frozen parent's output
    distribution on prior-rung/anchor rows (`is_prior`) at response positions
    (`labels != IGNORE_LABEL_ID`). Mode-covering direction: the child must keep
    probability mass where the parent (the retained skill) placed it. Computed
    in fp32; safe when a batch has no prior rows (denominator clamp -> 0.0).
    `temp` applies the standard distillation T with T^2 gradient scaling.
    """
    import torch.nn.functional as F
    resp_mask = labels != IGNORE_LABEL_ID                               # (B, L)
    mask = (is_prior.bool().unsqueeze(1) & resp_mask).to(torch.float32)  # (B, L)
    cl = child_logits.to(torch.float32) / temp
    pl = parent_logits.to(torch.float32) / temp
    log_child = F.log_softmax(cl, dim=-1)
    log_parent = F.log_softmax(pl, dim=-1)
    kl_pos = (log_parent.exp() * (log_parent - log_child)).sum(dim=-1)   # (B, L)
    denom = mask.sum().clamp(min=1.0)
    kl = (kl_pos * mask).sum() / denom
    return kl * (temp * temp)


def _compose_anchor_rows(
    retention_anchor_set: str, retention_anchor_repeat: int
) -> list[dict]:
    """Materialize the row-repeated anchor list for trainer composition.

    Slice B A1 row-repeat per codex msg 1779564576409-a7db0527. Each anchor
    row from the named set is replicated `retention_anchor_repeat` times.
    Rows include `anchor_id` so downstream multiplicity-floor accounting
    can exclude them from target-rung unique counts.

    Returns [] when set == "none" (default-off contract).
    """
    if retention_anchor_set == "none":
        return []

    def _rows(name: str) -> list[dict]:
        return [
            {
                "question": row.question,
                "expected": row.expected,
                "anchor_id": row.anchor_id,
                "source_rung": row.source_rung,
            }
            for row in load_anchor_set(name)
        ]

    # math_fragile_v2: per-subset repeat (codex msg 1779645719820). v1 rows are
    # fixed at repeat 3 (do NOT 5x v1 / weaken-or-overweight existing coverage);
    # the L0b hard-row guard takes the CLI --retention-anchor-repeat (5). This
    # is the "tiny code path composing v1 repeat3 + hard rows repeat5 explicitly".
    if retention_anchor_set == "math_fragile_v2":
        return _rows("math_fragile_v1") * 3 + _rows("l0b_hardrow_v1") * retention_anchor_repeat

    return _rows(retention_anchor_set) * retention_anchor_repeat


_RETAINED_SUPPORT_REGISTRY: tuple[str, ...] = ("L0b", "L0c", "math_a0", "math_r1b2_minus_one", "l0c_exhaustive")


def _retained_support(name: str, seed: int) -> tuple[list[tuple[str, int, str]], str]:
    """Canonical-ordered retained-support snapshot + 16-hex content hash.

    Generalizes the L0b-specific support (codex registry slice msg
    1779656084090) to a named registry of validated finite supports that
    become PROTECTED supports (soft forward-KL toward the frozen parent):

    - "L0b"     ← `_l0b_support(seed)` — 230 rows, `calculate <expr>.` surface.
                  SEED-DEPENDENT (two_digit picks seeded by `_stable_seed
                  ("L0b_partition", seed, ...)`) — build with curriculum_seed.
    - "L0c"     ← `_l0c_support(seed)` — 230 rows, `<expr> equals what?`
                  surface (canonical bounded L0c language support, same path as
                  `build_language_supports()["L0c"]`). SEED-DEPENDENT
                  (`_stable_seed("L0c_partition", seed, ...)`) — build with
                  curriculum_seed, mirrors L0b. F.4c: protects the L0c surface
                  F.4b left unprotected (replay-covered L0c1 → .917, but
                  unprotected L0c → .557 capped LANG-690 at .852).
    - "math_a0" ← `build_exhaustive_supports()` flattened — 1255 rows,
                  `what is <expr>?` surface, SEED-INDEPENDENT (exhaustive).
                  Contains `what is 10 minus 1?`->9 (R1b2), the row F.2f
                  regressed; protecting it via parent-KL is the broad fix.
    - "math_r1b2_minus_one" ← `build_exhaustive_supports()["R1b2"]` (the
                  `what is <a> minus 1?` class), SEED-INDEPENDENT. A
                  CONCENTRATED registry-derived subset of math_a0: F.2g showed
                  the broad 1255-row support is too dilute at K=8 to hold the
                  single high-pressure `10 minus 1` row against L0c1 CE on the
                  shared minus circuit. This class support gives that whole
                  rung dense parent-KL coverage (codex msg 1779659487346).
    - "l0c_exhaustive" ← `build_exhaustive_l0c_supports()` (the `<expr>
                  equals what?` wrapper over the full math-A0 set, 1255),
                  SEED-INDEPENDENT. DORMANT — registry-addressable for a
                  FUTURE math slice to replay once exhaustive-L0c banks; NOT
                  in any recipe default (codex msg 1779693537447).

    Rows are `(question, expected, source_rung)` sorted stably by
    `(source_rung, question, expected)` so repeated construction is
    byte-identical; the hash pins WHICH rows are protected into log + ckpt.
    """
    import hashlib
    if name == "L0b":
        from calm.hrm_text_158.curriculum.language_supports import _l0b_support
        rows = [(q, e, sr) for (q, e, sr) in _l0b_support(seed)]
    elif name == "L0c":
        # F.4c: canonical bounded L0c 230-row support (`<expr> equals what?`),
        # SEED-DEPENDENT, same language support path as
        # build_language_supports()["L0c"]; protects the L0c surface F.4b left
        # unprotected (no replay, no retained-support) which capped LANG-690.
        from calm.hrm_text_158.curriculum.language_supports import _l0c_support
        rows = [(q, e, sr) for (q, e, sr) in _l0c_support(seed)]
    elif name == "math_a0":
        from calm.hrm_text_158.curriculum.exhaustive_supports import build_exhaustive_supports
        rows = [(q, e, rung)
                for rung, pairs in build_exhaustive_supports().items()
                for (q, e) in pairs]
    elif name == "math_r1b2_minus_one":
        from calm.hrm_text_158.curriculum.exhaustive_supports import build_exhaustive_supports
        rows = [(q, e, "R1b2")
                for (q, e) in build_exhaustive_supports()["R1b2"]]
    elif name == "l0c_exhaustive":
        # Exhaustive L0c language-density support (codex msg 1779693537447):
        # the `<expr> equals what?` wrapper over the full math-A0 set (1255).
        # DORMANT — registry-addressable for a FUTURE math slice to replay as
        # broad low/mod retained support AFTER it banks; NOT in any recipe
        # default and NOT retained-KL'd in its own acquisition run.
        from calm.hrm_text_158.curriculum.language_supports import (
            build_exhaustive_l0c_supports,
        )
        rows = [(q, e, rung)
                for rung, pairs in build_exhaustive_l0c_supports().items()
                for (q, e) in pairs]
    else:
        raise ValueError(
            f"unknown retained support {name!r}; valid: {_RETAINED_SUPPORT_REGISTRY}"
        )
    rows = sorted(rows, key=lambda r: (r[2], r[0], r[1]))
    support_hash = hashlib.sha256(repr(rows).encode("utf-8")).hexdigest()[:16]
    return rows, support_hash


def _retained_sampler_seed(name: str, seed: int) -> int:
    """Per-support deterministic sampler seed. L0b keeps the legacy
    `"l0b_consistency"` namespace so its sampler is bit-identical to the
    pre-registry slice; other supports use `"retained:<name>"`."""
    ns = "l0b_consistency" if name == "L0b" else f"retained:{name}"
    return _stable_curriculum_seed(seed, ns)


def _l0b_consistency_support(seed: int) -> tuple[list[tuple[str, int, str]], str]:
    """Back-compat alias for `_retained_support("L0b", seed)`."""
    return _retained_support("L0b", seed)


class _RetainedSupportSampler:
    """Deterministic K-cyclic side-batch index sampler over a retained support.

    ONE seeded permutation (from `support_seed`), then a cyclic cursor walks it
    in fixed-size K batches, wrapping at the end. Pure index arithmetic — no
    DataLoader / worker randomness — so a rerun yields identical batches.
    K-cyclic (NOT full-N-every-step): even coverage, cheap. Per-support seed is
    derived by the caller (`_retained_sampler_seed`) so each support samples
    independently.
    """

    def __init__(self, n: int, support_seed: int, batch: int):
        if n <= 0:
            raise ValueError(f"_RetainedSupportSampler needs n > 0, got {n}")
        if batch <= 0:
            raise ValueError(f"_RetainedSupportSampler needs batch > 0, got {batch}")
        self.n = n
        self.batch = batch
        self.support_seed = support_seed
        g = torch.Generator()
        g.manual_seed(support_seed)
        self.perm = torch.randperm(n, generator=g).tolist()
        self.cursor = 0
        self.rows_seen = 0

    def next_indices(self) -> list[int]:
        out = []
        for _ in range(self.batch):
            out.append(self.perm[self.cursor])
            self.cursor = (self.cursor + 1) % self.n
        self.rows_seen += self.batch
        return out

    def coverage(self) -> dict:
        return {
            "rows_seen": self.rows_seen,
            "cursor": self.cursor,
            "full_cycles": self.rows_seen // self.n,
            "support_seed": self.support_seed,
        }


class _L0bConsistencySampler(_RetainedSupportSampler):
    """Back-compat: pre-registry L0b sampler. Constructs with (n, seed, batch),
    deriving the legacy `"l0b_consistency"` seed namespace internally — so
    existing L0b tests + bit-for-bit L0b behavior are preserved."""

    def __init__(self, n: int, seed: int, batch: int):
        super().__init__(n, _retained_sampler_seed("L0b", seed), batch)


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
    # TTrain-B: Triton fused-quantize STE-prep path for BitLinear training
    # forwards. Requires use_ternary_bulk=True. Default False preserves
    # current path bit-exact. Codex msg 1779538337913-2d79fa93.
    use_native_ternary_train: bool = False,
    # Phase 3 Step 1 (codex msg 1779462307554-b57d8288):
    # Curriculum-mode replaces GSM8k corpus with synthetic per-rung
    # data. ALL fields optional; defaults preserve legacy GSM8k behavior.
    curriculum_rung: str | None = None,
    curriculum_seed: int = 42,
    curriculum_n_train: int = 4000,
    curriculum_n_heldout: int = 200,
    replay_ratio: float = 0.30,
    replay_rungs: str | None = None,
    allow_future_replay: bool = False,
    use_broad_tokenizer: bool = False,
    load_from: str | None = None,
    # F.3f-a (codex msg 1779703363270): runtime per-row hard-weight override
    # for the L0c_exhaustive_2digit acquisition sampler. None -> keep spec
    # default 3.0 (F.3d-b behavior/tests unchanged). Valid ONLY with
    # --curriculum-rung L0c_exhaustive_2digit (fail-fast otherwise).
    l0c_hard_weight: float | None = None,
    # Retention-anchor V0 Slice B (codex msg 1779564576409-a7db0527).
    # A1 row-repeat: each anchor row appears `retention_anchor_repeat`
    # times in train_rows when set is non-'none'. Anchors append AFTER
    # the curriculum cap + log; NOT in the deterministic curriculum
    # shuffle. Defaults preserve byte-identical behavior to pre-Slice-B.
    retention_anchor_set: str = "none",
    retention_anchor_repeat: int = 2,
    # Parent-consistency loss (opt-in; default 0.0 = off, behavior-preserving).
    # Soft forward-KL(parent || child) on prior-rung/anchor rows penalizes the
    # child for drifting from the frozen --load-from parent's outputs on
    # retained skills. Requires --load-from when weight > 0.
    parent_consistency_weight: float = 0.0,
    parent_consistency_temp: float = 1.0,
    # L0b retained-support KL-only consistency (opt-in; default 0.0 = off — the
    # L0b SIDE PATH is skipped at weight 0. NOTE: this slice also adds an
    # always-on explicit DataLoader generator (separate determinism change), so
    # the weight-0 default path is deterministic-given-seed but NOT byte-
    # identical to pre-slice. Each step side-batches a deterministic K-cyclic
    # sample of the FULL 230-row L0b support (`_l0b_support(curriculum_seed)`,
    # train+held) and adds soft forward-KL(parent || child) on it (NO CE / no
    # target-task labels into the normal loss). Protects ALL L0b rows incl.
    # held rows that replay (train-only) + manual anchors never cover — the
    # broad fix for the F.2d/F.2e moving-held-hole whack-a-mole. Requires
    # --load-from + curriculum mode when weight > 0; reuses
    # --parent-consistency-temp. Codex +1 msg 1779647554279-522ba519.
    l0b_consistency_weight: float = 0.0,
    l0b_consistency_batch: int = 8,
    # Retained-support consistency registry (codex msg 1779656084090): the
    # generalization of l0b_consistency to a named registry of validated finite
    # supports (`_RETAINED_SUPPORT_REGISTRY`). `retained_support_profile` is a
    # list of (name, weight) pairs; each active support side-batches a K-cyclic
    # sample and adds soft forward-KL(parent || child) (NO CE), with its own
    # backward (sequential accumulation -> peak VRAM bounded). Legacy
    # `--l0b-consistency-weight` maps in as ("L0b", weight) with conflict
    # detection. `retained_support_batch` is the per-support side-batch K
    # (falls back to l0b_consistency_batch, then 8).
    retained_support_profile: list[tuple[str, float]] | None = None,
    retained_support_batch: int | None = None,
    # Diagnostic ONLY (codex msg 1779652915624): when True, build the training
    # DataLoader WITHOUT the explicit seeded generator (pre-1656ead global-RNG
    # shuffle order). Default False keeps the deterministic seeded generator.
    # NOT recipe-default — used to isolate whether 1656ead's loader-order change
    # caused the persistent F.2f `10 minus 1` value regression.
    legacy_loader_shuffle: bool = False,
    dry_run: bool = False,
) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    # Parent-consistency flag validation (codex pre-commit guard): reject
    # invalid weight/temp loudly before any work (covers --dry-run + tests +
    # programmatic callers).
    if parent_consistency_weight < 0.0:
        raise ValueError(
            f"--parent-consistency-weight must be >= 0 (got "
            f"{parent_consistency_weight}); a negative weight would reward "
            f"drift from the parent."
        )
    if parent_consistency_temp <= 0.0:
        raise ValueError(
            f"--parent-consistency-temp must be > 0 (got "
            f"{parent_consistency_temp}); <= 0 is an invalid distillation "
            f"temperature (division by zero)."
        )
    # Retained-support consistency guards + profile resolution (codex registry
    # slice msg 1779656084090). Fire BEFORE any data/model work so tests +
    # --dry-run + programmatic callers reject bad args loudly.
    if l0b_consistency_weight < 0.0:
        raise ValueError(
            f"--l0b-consistency-weight must be >= 0 (got "
            f"{l0b_consistency_weight}); a negative weight would reward drift "
            f"from the parent on retained L0b rows."
        )
    # Effective profile = explicit --retained-support pairs + legacy
    # --l0b-consistency-weight mapped to ("L0b", weight), with conflict detection.
    _profile: list[tuple[str, float]] = list(retained_support_profile or [])
    _profile_names = [n for (n, _w) in _profile]
    if l0b_consistency_weight > 0.0:
        if "L0b" in _profile_names:
            raise ValueError(
                "conflicting L0b config: both --l0b-consistency-weight and an "
                "explicit --retained-support L0b:<w> were given; specify L0b "
                "exactly once."
            )
        _profile.append(("L0b", l0b_consistency_weight))
        _profile_names.append("L0b")
    for nm, wt in _profile:
        if nm not in _RETAINED_SUPPORT_REGISTRY:
            raise ValueError(
                f"unknown retained support {nm!r}; valid: "
                f"{_RETAINED_SUPPORT_REGISTRY}"
            )
        if wt < 0.0:
            raise ValueError(
                f"retained-support weight for {nm!r} must be >= 0 (got {wt})"
            )
    if len(_profile_names) != len(set(_profile_names)):
        raise ValueError(
            f"duplicate retained-support names in profile: {_profile_names}"
        )
    # Only weight>0 supports are active.
    effective_retained_profile: list[tuple[str, float]] = [
        (nm, wt) for (nm, wt) in _profile if wt > 0.0
    ]
    effective_retained_batch = (
        retained_support_batch if retained_support_batch is not None
        else l0b_consistency_batch
    )
    if effective_retained_profile:
        if load_from is None:
            raise ValueError(
                "retained-support consistency (weight > 0) requires --load-from "
                "(the frozen parent reference checkpoint)."
            )
        if curriculum_rung is None:
            raise ValueError(
                "retained-support consistency (weight > 0) requires curriculum "
                "mode (--curriculum-rung); support snapshots are keyed by "
                "--curriculum-seed."
            )
        if effective_retained_batch < 1:
            raise ValueError(
                f"retained-support batch must be >= 1 when any support is "
                f"active; got {effective_retained_batch}"
            )

    # Slice B phase-gate (codex msg 1779565128372-c6872566 catch): anchors are
    # Phase 3 curriculum-only. In GSM8k mode no composition runs, but the
    # ckpt save path would otherwise falsely record retention_anchor_set as
    # active. Fail-fast here BEFORE either branch builds rows.
    if retention_anchor_set != "none" and curriculum_rung is None:
        raise ValueError(
            f"retention_anchor_set={retention_anchor_set!r} requires "
            f"curriculum_rung to be set (Phase 3 curriculum mode). "
            f"Retention anchors are not supported in GSM8k mode."
        )
    # Programmatic-call defense: argparse already rejects bad CLI values,
    # but train() may be called from tests or other scripts directly.
    if retention_anchor_set != "none" and retention_anchor_repeat < 1:
        raise ValueError(
            f"retention_anchor_repeat must be >= 1 when "
            f"retention_anchor_set != 'none'; "
            f"got {retention_anchor_repeat}"
        )

    # F.3f-a (codex msg 1779703363270 + 1779703935958): --l0c-hard-weight is
    # rung-specific AND must be > 0. Fail fast (before any model build / ckpt
    # load / data-gen). A non-positive weight would silently make TRAIN
    # all-easy (0) or fail late inside rng.choices (negative), bypassing this
    # fail-fast contract.
    if l0c_hard_weight is not None:
        if l0c_hard_weight <= 0:
            raise ValueError(
                f"--l0c-hard-weight must be > 0 (per-row hard weight vs easy=1.0); "
                f"got {l0c_hard_weight}."
            )
        if curriculum_rung != "L0c_exhaustive_2digit":
            raise ValueError(
                f"--l0c-hard-weight is only valid with "
                f"--curriculum-rung L0c_exhaustive_2digit; got rung={curriculum_rung!r}."
            )

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
        prior_rungs = _resolve_prior_rungs(
            curriculum_rung,
            replay_rungs,
            allow_future_replay=allow_future_replay,
        )
        print(f"[hrm158] curriculum {curriculum_rung}: prior_rungs={prior_rungs} "
              f"(positional_full={positional_full}, "
              f"diagnosis_only={sorted(DIAGNOSIS_ONLY_RUNGS)}, "
              f"explicit_override={replay_rungs is not None}, "
              f"allow_future_replay={allow_future_replay})",
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

        # F.3f-a (codex msg 1779703363270): apply the runtime hard-weight
        # override BEFORE current-rung data generation. Guarded above to fire
        # only for L0c_exhaustive_2digit. Default (None) leaves spec at 3.0.
        if l0c_hard_weight is not None:
            from calm.hrm_text_158.curriculum.generators import (
                set_l0c_exhaustive_2digit_hard_weight,
            )
            _eff_hw = set_l0c_exhaustive_2digit_hard_weight(l0c_hard_weight)
            print(f"[hrm158] L0c_exhaustive_2digit hard_weight override: {_eff_hw} "
                  f"(spec default 3.0); TRAIN per-row hard weight for this run",
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

        # Slice B retention-anchor V0 (codex msg 1779564576409-a7db0527):
        # A1 row-repeat appends anchor rows AFTER the curriculum cap+log.
        # Per codex correction: anchors do NOT enter the deterministic
        # curriculum shuffle at L385-387; interleaving relies on the
        # DataLoader(shuffle=True) at L424 below.
        #
        # Target-rung multiplicity math (n_new / unique_train_count) is
        # already computed PRE-composition at L354 / by the L0a generator's
        # `_enumerate_partition_l0a`. Anchor rows carry an `anchor_id`
        # field so any downstream code can exclude them.
        if retention_anchor_set != "none":
            anchor_rows = _compose_anchor_rows(
                retention_anchor_set, retention_anchor_repeat
            )
            anchor_unique = len(anchor_rows) // retention_anchor_repeat
            base_curriculum_train = len(train_rows)
            train_rows = train_rows + anchor_rows
            print(
                f"[hrm158] retention-anchor: set={retention_anchor_set} "
                f"repeat={retention_anchor_repeat} "
                f"anchor_rows_added={len(anchor_rows)} "
                f"anchor_unique={anchor_unique} "
                f"(base_curriculum_train={base_curriculum_train}, "
                f"anchor_inclusive_train={len(train_rows)})",
                flush=True,
            )
            tok.assert_corpus_covered(anchor_rows, label="retention_anchors")

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

    train_ds = HrmTextGsm8kDataset(train_rows, tok, max_len=max_len,
                                   curriculum_rung=curriculum_rung)
    val_ds = HrmTextGsm8kDataset(val_rows, tok, max_len=max_len,
                                 curriculum_rung=curriculum_rung)
    print(f"[hrm158] usable rows after max_len={max_len} drop: "
          f"train={len(train_ds)} (dropped {train_ds.n_dropped}) "
          f"val={len(val_ds)} (dropped {val_ds.n_dropped})", flush=True)
    if len(train_ds) == 0:
        raise RuntimeError("No usable training rows after max_len drop.")

    # DataLoader construction (codex determinism msg 1779647581438 + isolation
    # msg 1779652915624). Default: explicit Generator seeded by --seed (order
    # decoupled from model-init RNG). Diagnostic --legacy-loader-shuffle: the
    # pre-1656ead global-RNG order, to isolate whether the seed-decoupled
    # generator moved the fragile `10 minus 1` row.
    loader = _build_train_loader(train_ds, batch_size, seed, legacy_loader_shuffle)
    if legacy_loader_shuffle:
        print("[hrm158] --legacy-loader-shuffle ENABLED (DIAGNOSTIC): DataLoader uses "
              "global-RNG shuffle (pre-1656ead order); explicit seeded generator "
              "BYPASSED. NOT recipe-default.", flush=True)

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

    # Parent-consistency: frozen reference = the --load-from chain head. Built
    # after the dry-run early-return so dry runs skip it. Kept in eval (the
    # inference BitLinear forward) while the child may use the native-train
    # forward, so step-0 KL is ~0 within FP tolerance, not bitwise.
    parent_m = None
    if parent_consistency_weight > 0.0 or effective_retained_profile:
        if load_from is None:
            raise ValueError(
                "parent/retained-support consistency (weight > 0) requires "
                "--load-from (the frozen parent reference checkpoint)."
            )
        parent_hrm = HierarchicalReasoningModel(cfg)
        parent_m = LMHead(parent_hrm, LMHeadConfig(vocab_size=tok.vocab_size)).to(device)
        parent_m.load_state_dict(loaded_ckpt["model_state"], strict=True)
        parent_m.eval()
        for p in parent_m.parameters():
            p.requires_grad_(False)
        print(f"[hrm158] frozen parent reference LOADED from {load_from} "
              f"(parent_consistency_weight={parent_consistency_weight} "
              f"retained_support_profile={effective_retained_profile} "
              f"temp={parent_consistency_temp})", flush=True)

    # Retained-support consistency setup (codex registry slice msg 1779656084090).
    # For each active support: materialize the canonical-ordered snapshot, encode
    # it ONCE into a CPU cache, and arm an INDEPENDENT deterministic K-cyclic
    # sampler. Each support side-batches with its OWN backward in the train loop
    # (sequential accumulation -> peak VRAM bounded, same as the 2ce0da2 fix).
    # Skipped entirely when the profile is empty. Built after the dry-run
    # early-return so dry runs skip it.
    # active_supports: list of {name, weight, hash, count, cache, sampler}.
    active_supports: list[dict] = []
    for _name, _weight in effective_retained_profile:
        _rows, _hash = _retained_support(_name, curriculum_seed)
        _row_dicts = [
            {"question": q, "expected": e, "source_rung": sr}
            for (q, e, sr) in _rows
        ]
        tok.assert_corpus_covered(_row_dicts, label=f"retained:{_name}")
        _ds = HrmTextGsm8kDataset(_row_dicts, tok, max_len=max_len,
                                  curriculum_rung=None)
        if _ds.n_dropped != 0 or len(_ds) != len(_rows):
            raise RuntimeError(
                f"retained support {_name!r} lost rows to max_len={max_len} "
                f"(dropped {_ds.n_dropped}, kept {len(_ds)} of {len(_rows)}); "
                f"cannot align the deterministic sampler to canonical order."
            )
        _cache = [_ds[i] for i in range(len(_ds))]
        _sampler = _RetainedSupportSampler(
            n=len(_cache),
            support_seed=_retained_sampler_seed(_name, curriculum_seed),
            batch=effective_retained_batch,
        )
        active_supports.append({
            "name": _name, "weight": _weight, "hash": _hash,
            "count": len(_cache), "cache": _cache, "sampler": _sampler,
        })
        _first3 = _sampler.perm[: min(3, len(_sampler.perm))]
        _first_qs = [_rows[i][0] for i in _first3]
        print(
            f"[hrm158] retained-support ENABLED: name={_name} weight={_weight} "
            f"temp={parent_consistency_temp} batch={effective_retained_batch} "
            f"support_seed={curriculum_seed} support_count={len(_cache)} "
            f"support_hash={_hash} sampler_seed={_sampler.support_seed} "
            f"first3_perm={_first3} first3_q={_first_qs}", flush=True,
        )
    # L0b-only flag for ckpt-metadata back-compat (codex correction #2): keep
    # legacy l0b_consistency_* fields ONLY when the effective profile is exactly
    # single-support L0b; mixed profiles write retained_support_profile only.
    _retained_l0b_only = (
        len(active_supports) == 1 and active_supports[0]["name"] == "L0b"
    )
    retained_support_meta = [
        {"name": s["name"], "weight": s["weight"], "batch": effective_retained_batch,
         "count": s["count"], "hash": s["hash"]}
        for s in active_supports
    ]

    # Train
    m.train()
    # TTrain-B: enable native fused-quantize STE path on all BitLinear modules
    # AFTER m.train() so the eval-mode train() override doesn't immediately
    # clear the flag. Forward value bit-equivalent; backward STE-correct via
    # custom autograd.Function. Inference path unaffected. Codex msg
    # 1779538337913-2d79fa93 +1 implement Phase B.
    if use_native_ternary_train:
        if not use_ternary_bulk:
            print(f"[hrm158] --use-native-ternary-train requires --use-ternary-bulk; "
                  f"flag is a no-op (no BitLinear modules in model). Continuing.",
                  flush=True)
        else:
            from calm.hrm_text_158.bit_linear import enable_bitlinears_for_native_train
            n_enabled = enable_bitlinears_for_native_train(m)
            print(f"[hrm158] TTrain-B native-ternary-train: enabled {n_enabled} "
                  f"BitLinear modules (Triton fused-quantize + STE-correct backward)",
                  flush=True)
    step = 0
    # Reset CUDA peak-memory stats so the logged peak reflects the training
    # loop (post model+parent load), giving a clean F.2e-comparable peak.
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
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

            # Forward + loss with SEQUENTIAL backward to bound peak VRAM.
            # The main (CE [+ curriculum-PC]) graph is freed by ITS backward
            # BEFORE the L0b side graph is built, so peak activation memory is
            # max(main, side), NOT main+side. At bp_steps=5 (deep HRM
            # recurrence) holding BOTH return_logits graphs simultaneously
            # ~2x'd peak and thrashed the 8 GB allocator (F.2f run-1 OOM-thrash,
            # 0.7->13 s/step). Gradients accumulate across the two backwards —
            # identical to a single summed-loss backward. zero_grad ONCE up
            # front; step ONCE after both backwards.
            child_batch = {"inputs": inputs, "labels": labels,
                           "sep_positions": sep_positions, "position_ids": position_ids}
            opt.zero_grad()

            # --- Main backward: CE [+ curriculum parent-consistency KL] ---
            pc_kl_val = 0.0
            if parent_consistency_weight > 0.0:
                new_carry, loss_main, metrics = m(None, child_batch, return_logits=True, **extras)
                is_prior = batch["is_prior"].to(device)
                with torch.no_grad():
                    _, parent_logits = parent_m(
                        None,
                        {"inputs": inputs, "sep_positions": sep_positions,
                         "position_ids": position_ids},
                        **extras,
                    )
                pc_kl = _parent_consistency_kl(
                    metrics["logits"], parent_logits, labels, is_prior,
                    temp=parent_consistency_temp,
                )
                loss_main = loss_main + parent_consistency_weight * pc_kl
                pc_kl_val = float(pc_kl.detach())
            else:
                new_carry, loss_main, metrics = m(None, child_batch, **extras)
            if not torch.isfinite(loss_main):
                print(f"[NaN-DETECT] step={step} loss_main={loss_main.item()}", flush=True)
                sys.exit(2)
            loss_main.backward()                 # frees the main graph here
            disp_loss = float(loss_main.detach())
            acc_count, acc_total = metrics["accuracy"]

            # --- Retained-support KL-only side backwards (registry) ---
            # ONE side batch per active support, each built AFTER the prior
            # graph is freed -> peak VRAM stays max(one graph) (the 2ce0da2
            # sequential-backward discipline). NO CE: each side child-forward
            # computes a CE loss internally but it is DISCARDED (never
            # backpropped); only the KL term flows gradient. Protects every
            # validated finite support (L0b held rows; math A0 incl.
            # `what is 10 minus 1?`) via KL toward the frozen parent.
            retained_kl_vals: dict[str, float] = {}
            for _sup in active_supports:
                _idx = _sup["sampler"].next_indices()
                _picked = [_sup["cache"][i] for i in _idx]
                s_inputs = torch.stack([p["inputs"] for p in _picked], 0).to(device)
                s_labels = torch.stack([p["labels"] for p in _picked], 0).to(device)
                s_sep = torch.stack([p["sep_position"] for p in _picked], 0).to(device)
                sB, sL = s_inputs.shape
                s_pos = torch.arange(sL, dtype=torch.long, device=device).unsqueeze(0).expand(sB, -1)
                s_batch = {"inputs": s_inputs, "labels": s_labels,
                           "sep_positions": s_sep, "position_ids": s_pos}
                _sc, _sloss, s_metrics = m(None, s_batch, return_logits=True, **extras)
                with torch.no_grad():
                    _, s_parent_logits = parent_m(
                        None,
                        {"inputs": s_inputs, "sep_positions": s_sep,
                         "position_ids": s_pos},
                        **extras,
                    )
                s_is_prior = torch.ones(sB, dtype=torch.bool, device=device)
                s_kl = _parent_consistency_kl(
                    s_metrics["logits"], s_parent_logits, s_labels, s_is_prior,
                    temp=parent_consistency_temp,
                )
                s_loss = _sup["weight"] * s_kl
                if not torch.isfinite(s_loss):
                    print(f"[NaN-DETECT] step={step} support={_sup['name']} "
                          f"kl_loss={s_loss.item()}", flush=True)
                    sys.exit(2)
                s_loss.backward()            # accumulates grad, frees this side graph
                retained_kl_vals[_sup["name"]] = float(s_kl.detach())
                disp_loss += float(s_loss.detach())

            grad_norm = torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                print(f"[NaN-DETECT] step={step} grad_norm={grad_norm}", flush=True)
                sys.exit(2)
            opt.step()

            if step == 1 or step % log_every == 0:
                elapsed = time.time() - start_t
                pc_str = f" pc_kl={pc_kl_val:.6f}" if parent_consistency_weight > 0.0 else ""
                retained_str = "".join(
                    f" {nm}_kl={v:.6f}" for nm, v in retained_kl_vals.items()
                )
                # Peak CUDA memory (codex receipt msg 1779650973993): reserved
                # can stay high even when allocations are bounded, so report
                # both. max_memory_allocated is the true peak live tensors.
                mem_str = ""
                if device == "cuda":
                    mem_str = (f" peak_alloc={torch.cuda.max_memory_allocated()/1e9:.2f}GB"
                               f" peak_resv={torch.cuda.max_memory_reserved()/1e9:.2f}GB")
                print(f"[ep {ep:3d} step {step:5d}] loss={disp_loss:.4f} "
                      f"grad_norm={float(grad_norm):.4f} lr={cur_lr:.6f} "
                      f"bp_steps={extras['bp_steps']} "
                      f"acc={int(acc_count)}/{int(acc_total)}{pc_str}{retained_str}{mem_str} t={elapsed:.1f}s",
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
                        retention_anchor_set=retention_anchor_set,
                        retention_anchor_repeat=retention_anchor_repeat,
                        parent_consistency_weight=parent_consistency_weight,
                        parent_consistency_temp=parent_consistency_temp,
                        retained_support_meta=retained_support_meta,
                        retained_l0b_only=_retained_l0b_only,
                    ),
                    "step": step,
                    "epoch": ep,
                    "source_pin": SOURCE_PIN,
                }
                torch.save(ckpt_blob, ckpt_path)
                _cov = "".join(
                    f" {s['name']}_cov={s['sampler'].coverage()}" for s in active_supports
                )
                print(f"[ep {ep:3d} step {step:5d}] save_at_step: saved {ckpt_path}{_cov}", flush=True)

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
            retention_anchor_set=retention_anchor_set,
            retention_anchor_repeat=retention_anchor_repeat,
            parent_consistency_weight=parent_consistency_weight,
            parent_consistency_temp=parent_consistency_temp,
            retained_support_meta=retained_support_meta,
            retained_l0b_only=_retained_l0b_only,
        ),
        "step": step,
        "epoch": ep,
        "source_pin": SOURCE_PIN,
    }
    torch.save(ckpt_blob, final_path)
    _cov = "".join(
        f" {s['name']}_cov={s['sampler'].coverage()}" for s in active_supports
    )
    print(f"[hrm158] final ckpt: {final_path}{_cov}", flush=True)


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
    retention_anchor_set: str | None = None,
    retention_anchor_repeat: int | None = None,
    parent_consistency_weight: float = 0.0,
    parent_consistency_temp: float = 1.0,
    retained_support_meta: list[dict] | None = None,
    retained_l0b_only: bool = False,
) -> dict:
    """Single source of truth for ckpt config blob (per Slice 13m pattern).

    Phase 3 additions (curriculum_rung / replay_ratio / prior_rungs) are
    populated only when training in curriculum mode; absent on legacy
    GSM8k ckpts.

    Slice B additions (retention_anchor_set / retention_anchor_repeat) are
    populated only when retention-anchor V0 is enabled (set != 'none');
    absent when disabled, matching the default-off contract.
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
    if retention_anchor_set is not None and retention_anchor_set != "none":
        out["retention_anchor_set"] = retention_anchor_set
        out["retention_anchor_repeat"] = retention_anchor_repeat
    # Consistency-loss recipe (codex determinism msg 1779647581438 + registry
    # msg 1779656084090): record weights/temp + per-support hash/count so audit
    # naming + manifests pin the exact recipe. Recorded only when active.
    if parent_consistency_weight > 0.0:
        out["parent_consistency_weight"] = parent_consistency_weight
        out["parent_consistency_temp"] = parent_consistency_temp
    if retained_support_meta:
        # retained_support_profile is the source-of-truth for the active
        # supports (name/weight/batch/count/hash + the shared temp).
        out["retained_support_profile"] = [
            {**s, "temp": parent_consistency_temp} for s in retained_support_meta
        ]
        # Back-compat (codex correction msg 1779656084090): keep the legacy
        # l0b_consistency_* fields ONLY when the profile is exactly L0b-only.
        # Mixed profiles do NOT pretend to be old L0b-only checkpoints.
        if retained_l0b_only:
            _l0b = retained_support_meta[0]
            out["l0b_consistency_weight"] = _l0b["weight"]
            out["l0b_consistency_temp"] = parent_consistency_temp
            out["l0b_consistency_batch"] = _l0b["batch"]
            out["l0b_consistency_support_hash"] = _l0b["hash"]
            out["l0b_consistency_support_count"] = _l0b["count"]
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
    ap.add_argument("--use-native-ternary-train", action="store_true",
                    help="TTrain-B: enable Triton fused-quantize STE-prep path "
                         "for BitLinear training forwards (codex msg "
                         "1779538337913-2d79fa93). Forward value bit-equivalent "
                         "to default path; STE-correct backward via custom "
                         "autograd.Function. Inference path unchanged. Requires "
                         "--use-ternary-bulk (no-op otherwise).")
    # Phase 3 Step 1 curriculum flags (codex msg 1779462307554 +1 implement Phase A)
    ap.add_argument("--curriculum-rung", type=str, default=None,
                    choices=["R0", "R1", "R1b1", "R1b2a", "R1b2", "R1b3", "R1b4", "R1b4v2", "R1b5", "R1b6", "R1b7", "R1b8", "R1b9", "R1b10", "L0a", "L0b", "L0c1", "L0c2", "L0c2-K1", "L0c2-K2", "L0c2-K3", "L0c", "L0c_exhaustive", "L0c_exhaustive_2digit", "R1b", "R2a", "R2", "R3", "R4", "R5", "R6"],
                    help="Phase 3 curriculum mode. When set, swaps GSM8k corpus "
                         "for synthetic per-rung data + replay mix. Requires "
                         "--use-broad-tokenizer in Phase 3 design.")
    ap.add_argument("--l0c-hard-weight", type=float, default=None,
                    help="F.3f-a (codex msg 1779703363270): runtime per-row hard "
                         "weight for the L0c_exhaustive_2digit TRAIN sampler "
                         "(easy=1.0). Default None keeps spec 3.0 (F.3d-b "
                         "unchanged). VALID ONLY with --curriculum-rung "
                         "L0c_exhaustive_2digit (fail-fast otherwise). F.3e "
                         "rejected 3x (starved easy); F.3f uses ~1.5.")
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
    ap.add_argument("--allow-future-replay", action="store_true",
                    help="Allow EXPLICIT --replay-rungs to include later-rung "
                         "names (index >= current rung) for foundational-rung "
                         "repair passes. Default rejects future rungs. Use "
                         "ONLY when repairing a foundational primitive (e.g. "
                         "R1b2) while preserving later-rung mastery via "
                         "explicit replay of those rungs. Emits a WARN naming "
                         "each future rung accepted. All other validation "
                         "(unknown/R7/self/duplicate/empty/malformed) still "
                         "applies. Has NO effect on positional/default replay. "
                         "Codex msg 1779548482300-05680b9d Option G after "
                         "R1b6 commit 128b097 baseline revealed R1b2=0.78 "
                         "pre-existing gap; durable gabe provenance relay "
                         "1779547541812.")
    ap.add_argument("--use-broad-tokenizer", action="store_true",
                    help="Use BroadTokenizer (byte-level UTF-8, vocab=260, "
                         "normalizer_version=byte_utf8_v1) instead of Gsm8kTokenizer. "
                         "Required for --curriculum-rung in Phase 3.")
    ap.add_argument("--load-from", type=str, default=None,
                    help="Path to prior-rung ckpt. validate_load_from_ckpt_compat "
                         "runs first (hard-fails on vocab/normalizer/ternary/arch "
                         "mismatch); then model_state loads strict; optimizer state "
                         "+ LR schedule RESET per rung.")
    # Retention-anchor V0 Slice B (codex msg 1779564576409-a7db0527 +1 A1
    # row-repeat implementation). Default-off; ckpt config records anchor
    # metadata only when enabled. Anchors are excluded from target-rung
    # unique-count / multiplicity math. n_train_cap applies to base
    # curriculum BEFORE anchor composition; anchors append after cap.
    # Anchors do NOT enter the deterministic curriculum shuffle at L385-387;
    # interleaving comes from the existing DataLoader(shuffle=True) at L424.
    ap.add_argument("--retention-anchor-set", type=str, default="none",
                    choices=["none", *sorted(RETENTION_ANCHOR_SETS)],
                    help="Retention-anchor V0 sentinel set. Default 'none' = "
                         "no composition change. When enabled, anchor rows "
                         "are appended after curriculum cap + log (NOT in the "
                         "pre-cap shuffle); interleaving relies on DataLoader."
                         " Recorded in ckpt config when enabled, absent when "
                         "disabled.")
    ap.add_argument("--retention-anchor-repeat", type=int, default=2,
                    help="Row-repeat multiplier for A1 anchor composition. "
                         "Each anchor row appears N times in train_rows. "
                         "Default 2. Integer-only (argparse type=int rejects "
                         "non-integers loudly); must be >= 1 (rejected at "
                         "parse time). Ignored when --retention-anchor-set "
                         "is 'none'.")
    ap.add_argument("--parent-consistency-weight", type=float, default=0.0,
                    help="Opt-in parent-consistency loss weight (lambda). When "
                         ">0, adds soft forward-KL(parent||child) on prior-rung/"
                         "anchor rows, penalizing drift from the frozen "
                         "--load-from parent on retained skills. Requires "
                         "--load-from. Default 0.0 (off, behavior-preserving).")
    ap.add_argument("--parent-consistency-temp", type=float, default=1.0,
                    help="Softmax temperature for the parent-consistency KL "
                         "(standard distillation T with T^2 grad scaling). "
                         "Default 1.0. Also applies to the L0b-consistency KL.")
    ap.add_argument("--l0b-consistency-weight", type=float, default=0.0,
                    help="Weight on the L0b retained-support KL-only side batch. "
                         ">0 side-batches a deterministic K-cyclic sample of the "
                         "full 230-row L0b support (_l0b_support(--curriculum-seed), "
                         "train+held) each step and adds soft forward-KL(parent||"
                         "child) on it (NO CE), protecting held L0b rows that "
                         "replay/anchors never cover. Requires --load-from + "
                         "curriculum mode. Default 0.0 (off).")
    ap.add_argument("--l0b-consistency-batch", type=int, default=8,
                    help="K rows per L0b-consistency side batch (K-cyclic "
                         "sampler). At ~1500 steps, K=8 -> ~52x coverage of the "
                         "230-row support. Default 8.")
    ap.add_argument("--retained-support", action="append", default=None,
                    metavar="NAME:WEIGHT",
                    help="Repeatable. Add a validated finite support to the "
                         "retained-support consistency profile (soft forward-KL "
                         "toward the frozen parent, NO CE). NAME in {L0b, math_a0}; "
                         "WEIGHT float >= 0. E.g. --retained-support L0b:1.0 "
                         "--retained-support math_a0:1.0. Legacy "
                         "--l0b-consistency-weight maps to L0b:<weight> (errors if "
                         "both set L0b).")
    ap.add_argument("--retained-support-batch", type=int, default=None,
                    help="K rows per retained-support side batch (per support, "
                         "K-cyclic). Falls back to --l0b-consistency-batch, then 8.")
    ap.add_argument("--legacy-loader-shuffle", action="store_true",
                    help="DIAGNOSTIC ONLY (not recipe-default): build the training "
                         "DataLoader without the explicit seeded generator, restoring "
                         "the pre-1656ead global-RNG shuffle order. Used to isolate "
                         "whether the seed-decoupled generator moved a fragile boundary "
                         "row. Default off (explicit seeded generator).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build corpus + tokenizer + model + first batch + verify "
                         "forward pass, then exit BEFORE optimizer step. No ckpt "
                         "written. Used for Phase A receipt validation.")
    args = ap.parse_args()

    # Slice B: integer-and->=1 sanity bound for --retention-anchor-repeat.
    # argparse type=int already rejects non-integer; this catches zero/neg.
    if args.retention_anchor_repeat < 1:
        ap.error(
            f"--retention-anchor-repeat must be >= 1; "
            f"got {args.retention_anchor_repeat}"
        )

    # Parse --retained-support NAME:WEIGHT pairs into (name, weight) tuples.
    # Registry membership / duplicate / conflict validation happens in train()
    # so programmatic callers + tests get the same guards.
    _retained_profile = None
    if args.retained_support:
        _retained_profile = []
        for _spec in args.retained_support:
            if ":" not in _spec:
                ap.error(f"--retained-support expects NAME:WEIGHT; got {_spec!r}")
            _nm, _, _wt = _spec.partition(":")
            try:
                _wtf = float(_wt)
            except ValueError:
                ap.error(f"--retained-support WEIGHT must be a float; got {_spec!r}")
            _retained_profile.append((_nm.strip(), _wtf))

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
        use_native_ternary_train=args.use_native_ternary_train,
        curriculum_rung=args.curriculum_rung,
        curriculum_seed=args.curriculum_seed,
        curriculum_n_train=args.curriculum_n_train,
        replay_rungs=args.replay_rungs,
        allow_future_replay=args.allow_future_replay,
        l0c_hard_weight=args.l0c_hard_weight,
        curriculum_n_heldout=args.curriculum_n_heldout,
        replay_ratio=args.replay_ratio,
        use_broad_tokenizer=args.use_broad_tokenizer,
        load_from=args.load_from,
        retention_anchor_set=args.retention_anchor_set,
        retention_anchor_repeat=args.retention_anchor_repeat,
        parent_consistency_weight=args.parent_consistency_weight,
        parent_consistency_temp=args.parent_consistency_temp,
        l0b_consistency_weight=args.l0b_consistency_weight,
        l0b_consistency_batch=args.l0b_consistency_batch,
        retained_support_profile=_retained_profile,
        retained_support_batch=args.retained_support_batch,
        legacy_loader_shuffle=args.legacy_loader_shuffle,
        dry_run=args.dry_run,
    )
