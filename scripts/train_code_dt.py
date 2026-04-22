"""Train a Delta-Transducer (DT) on (problem, skeleton) pairs mined from
CodeExampleDB. Produces `calm/hrm/checkpoints/dt_code_skel_best.pt`.

**DT (delta-transducer)** is the canonical product name for the
copy-augmented DeltaNet architecture — default meta across all new
training as of 2026-04-22. The underlying class `CopyAugmentedDeltaNet`
stays (implementation detail); DT is the product-level label.

Per `.claude/rules/delta_rule.md` §"MQAR data-scaling rule" — DeltaNet
fast-weight state is the right mechanism for large-key-space retrieval
(8970 code examples, hundreds of distinct function patterns). Plain PT's
softmax-over-d_head=2 caps at ~10 unique keys.

Training recipe (following `train_pt_delta_mqar.py`):
  - chunkwise UT transform (3-7× training speedup)
  - scheduled sampling tf 1.0 → 0.3 across epochs
  - F.nll_loss on log-probs (not F.cross_entropy) — DT returns log-probs
  - best-by-autoreg checkpoint selection

Scope this session: train, measure autoreg accuracy on held-out val.
Install on Gemma (via CardSlot) is deferred to a follow-up session
since daemon stability is a gate.

Usage:
    PYTHONPATH=. python3 -u scripts/train_code_dt.py \\
        --epochs 30 --batch-size 32 --lr 1e-3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from calm.hrm.code_dt_data import (
    CODE_VOCAB_SIZE,
    CodePairDataset,
    CodeProblem,
    _CODE_CHAR_TO_ID,
    _CODE_ID_TO_CHAR,
    _paraphrase_augment,
    build_balanced_sampler_weights,
    code_detokenize,
    dedupe_ambiguous_prompts,
    extract_pairs_from_db,
    filter_rare_classes,
    normalize_skeleton,
    split_pairs,
)
from calm.hrm.rare_class_synth import synthesize_rare_class_pairs
from calm.llm_computer.copy_augmented_delta import build_copy_augmented_delta


CHECKPOINT_PATH = Path("calm/hrm/checkpoints/dt_code_skel_best.pt")
METRICS_PATH = Path("calm/hrm/checkpoints/dt_code_skel_metrics.json")


def autoreg_eval(model, val_pairs, device, max_gen=40):
    """Greedy decode from <bos>prob<sep>, measure exact-skeleton match."""
    bos = _CODE_CHAR_TO_ID["<bos>"]
    sep = _CODE_CHAR_TO_ID["<sep>"]
    eos = _CODE_CHAR_TO_ID["<eos>"]
    model.eval()
    n_correct = 0
    samples = []
    for p in val_pairs:
        from calm.hrm.code_dt_data import code_tokenize
        prefix = code_tokenize(p.question, add_bos=True, add_eos=False) + [sep]
        ids = list(prefix)
        gen = []
        for _ in range(max_gen):
            x = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                lp = model(x)
            nxt = int(lp[0, -1].argmax().item())
            if nxt == eos:
                break
            gen.append(nxt)
            ids.append(nxt)
            if len(ids) >= getattr(model, "max_len", 256) - 1:
                break
        decoded = code_detokenize(gen)
        if decoded.strip() == p.expression.strip():
            n_correct += 1
        samples.append((p.question[:60], p.expression, decoded))
    acc = n_correct / max(len(val_pairs), 1)
    return acc, samples


def _scheduled_tf(model, input_ids, target_ids, tf_ratio):
    """Scheduled sampling: forward once to get predictions, swap some
    positions with the model's argmax, then forward again with gradient."""
    if tf_ratio >= 0.999:
        # Pure teacher forcing — single forward
        log_probs = model(input_ids)
        loss = F.nll_loss(
            log_probs.reshape(-1, log_probs.shape[-1]),
            target_ids.reshape(-1),
            ignore_index=_CODE_CHAR_TO_ID["<pad>"],
        )
        return loss
    with torch.no_grad():
        lp = model(input_ids)
        preds = lp.argmax(dim=-1)
    # Build mixed input: with prob (1-tf_ratio) swap to model prediction
    # Can't use predictions for position 0 (need bos). Shift and gate.
    bs, seq = input_ids.shape
    shifted = torch.cat(
        [input_ids[:, :1], preds[:, :-1]], dim=1
    )
    mask = torch.rand(bs, seq, device=input_ids.device) > tf_ratio
    # Preserve position 0 (bos); use the original input elsewhere conditionally
    mask[:, 0] = False
    mixed = torch.where(mask, shifted, input_ids)
    log_probs = model(mixed)
    return F.nll_loss(
        log_probs.reshape(-1, log_probs.shape[-1]),
        target_ids.reshape(-1),
        ignore_index=_CODE_CHAR_TO_ID["<pad>"],
    )


def train(
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    d_model: int = 64,
    n_heads: int = 32,
    n_layers: int = 4,
    d_ffn: int = 128,
    max_len: int = 256,
    n_copy_heads: int = 4,
    seed: int = 42,
    tf_ratio_start: float = 1.0,
    tf_ratio_end: float = 0.3,
    eval_every: int = 2,
    device: str | None = None,
    val_frac: float = 0.1,
    augment: bool = True,
    augment_factor: int = 8,
    plateau_patience: int = 5,    # val evals with no improvement → [PLATEAU]
    plateau_min_delta: float = 0.005,   # improvement threshold
    balanced_sampler: str = "none",     # "none", "inverse", "sqrt_inverse", "capped"
    copy_gate_bias_init: float = -2.0,  # -2.0 v4 default (favors gen); 0.0 neutral; +1.0 favors copy
    normalize_skeletons: bool = False,  # R6: collapse spacing variants (FN(a, b) ≡ FN(a,b))
    drop_rare_count: int = 0,           # R6: drop training classes with count < N (0 = keep all)
    extract_all_defs: bool = False,     # R8: emit ALL top-level defs per solution (+19% raw)
    synth_rare: int = 0,                # R9: synthesize N pairs per rare class (0=off, 30 recommended)
    synth_rare_min: int = 3,            # R9: min raw count to synthesize
    synth_rare_max: int = 20,           # R9: max raw count to synthesize
    dedupe_ambiguous: bool = False,     # R19: drop/resolve same-prompt→multiple-skeletons
    num_workers: int = 0,               # R20: DataLoader parallelism
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    # --- Data ---
    if synth_rare > 0:
        # R9 path: extract raw (no aug), normalize, synthesize rare classes,
        # THEN aug so synthetic prompts also get paraphrased.
        print(f"[train] R9 path: extract raw + synthesize rare + aug")
        raw_pairs = extract_pairs_from_db(augment=False,
                                            extract_all_defs=extract_all_defs)
        print(f"[train] raw pairs: {len(raw_pairs)}")
        if normalize_skeletons:
            raw_pairs = [CodeProblem(question=p.question,
                                       expression=normalize_skeleton(p.expression))
                         for p in raw_pairs]
            n_classes = len(set(p.expression for p in raw_pairs))
            print(f"[train] normalized raw → {n_classes} classes")
        if dedupe_ambiguous:
            n_before = len(raw_pairs)
            raw_pairs = dedupe_ambiguous_prompts(raw_pairs)
            n_classes_after = len(set(p.expression for p in raw_pairs))
            print(f"[train] R19 dedup ambiguous: "
                  f"{n_before} → {len(raw_pairs)} pairs ({n_classes_after} classes)")
        synth_pairs = synthesize_rare_class_pairs(
            raw_pairs, min_count=synth_rare_min, max_count=synth_rare_max,
            target_per_class=synth_rare, seed=seed,
        )
        print(f"[train] R9 synthetic: {len(synth_pairs)} pairs across "
              f"{len(set(p.expression for p in synth_pairs))} rare classes")
        combined = raw_pairs + synth_pairs
        if augment:
            pairs = _paraphrase_augment(combined, factor=augment_factor, seed=seed)
        else:
            pairs = combined
        print(f"[train] after aug (factor={augment_factor}): {len(pairs)} pairs, "
              f"{len(set(p.expression for p in pairs))} classes")
    else:
        # Original path: aug at extract-time
        print(f"[train] extracting pairs from CodeExampleDB "
              f"(augment={augment}, factor={augment_factor})...")
        pairs = extract_pairs_from_db(augment=augment,
                                        augment_factor=augment_factor,
                                        extract_all_defs=extract_all_defs)
        print(f"[train] total pairs: {len(pairs)}")
        if normalize_skeletons:
            n_classes_before = len(set(p.expression for p in pairs))
            pairs = [CodeProblem(question=p.question,
                                  expression=normalize_skeleton(p.expression))
                     for p in pairs]
            n_classes_after = len(set(p.expression for p in pairs))
            print(f"[train] normalized skeletons: "
                  f"{n_classes_before} → {n_classes_after} classes "
                  f"(-{n_classes_before - n_classes_after} spacing variants)")

    train_pairs, val_pairs = split_pairs(pairs, val_frac=val_frac, seed=seed)

    if drop_rare_count > 0:
        n_before = len(train_pairs)
        train_pairs = filter_rare_classes(train_pairs, min_count=drop_rare_count)
        n_train_classes = len(set(p.expression for p in train_pairs))
        print(f"[train] dropped rare classes (min_count={drop_rare_count}): "
              f"{n_before} → {len(train_pairs)} train pairs, "
              f"{n_train_classes} classes")

    print(f"[train] split: {len(train_pairs)} train / {len(val_pairs)} val")

    train_ds = CodePairDataset(train_pairs, max_len=max_len)
    # R20: num_workers + pin_memory overlap CPU data prep with GPU compute
    use_pin_memory = device.startswith("cuda")
    common_loader_kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=(num_workers > 0),
    )
    if balanced_sampler != "none":
        from torch.utils.data import WeightedRandomSampler
        weights = build_balanced_sampler_weights(
            train_pairs, strategy=balanced_sampler,
        )
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(train_pairs),
            replacement=True,
        )
        loader = DataLoader(train_ds, sampler=sampler, **common_loader_kw)
        print(f"[train] balanced sampler: strategy={balanced_sampler!r}, "
              f"unique skeletons={len(set(p.expression for p in train_pairs))}")
    else:
        loader = DataLoader(train_ds, shuffle=True, **common_loader_kw)
    print(f"[train] DataLoader: batch={batch_size}, num_workers={num_workers}, "
          f"pin_memory={use_pin_memory}")

    # --- Model ---
    model = build_copy_augmented_delta(
        vocab_size=CODE_VOCAB_SIZE,
        max_len=max_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ffn=d_ffn,
        n_copy_heads=n_copy_heads,
        copy_gate_bias_init=copy_gate_bias_init,
    ).to(device)
    print(f"[train] copy_gate_bias_init={copy_gate_bias_init} "
          f"(sigmoid = {torch.sigmoid(torch.tensor(copy_gate_bias_init)).item():.3f})")
    # Enable chunkwise for training speed (set on config if supported)
    if hasattr(model, "config") and hasattr(model.config, "use_chunkwise"):
        model.config.use_chunkwise = True
    model.max_len = max_len   # for autoreg_eval len guard

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params: {n_params:,} ({n_params/1e3:.1f}K)")

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=epochs)

    best_acc = 0.0
    history: list[dict] = []
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Plateau detection state
    evals_since_improvement = 0
    plateau_triggered = False

    t0 = time.time()
    for ep in range(1, epochs + 1):
        # tf_ratio linearly decays
        frac = (ep - 1) / max(epochs - 1, 1)
        tf_ratio = tf_ratio_start + frac * (tf_ratio_end - tf_ratio_start)

        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            input_ids, target_ids, _ = batch
            input_ids = input_ids.to(device, non_blocking=use_pin_memory)
            target_ids = target_ids.to(device, non_blocking=use_pin_memory)
            loss = _scheduled_tf(model, input_ids, target_ids, tf_ratio)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        scheduler.step()

        elapsed = time.time() - t0
        rec = {"epoch": ep, "loss": avg_loss, "tf_ratio": round(tf_ratio, 3),
               "elapsed_s": round(elapsed, 1)}

        if ep % eval_every == 0 or ep == epochs:
            acc, samples = autoreg_eval(model, val_pairs, device)
            rec["val_autoreg"] = round(acc, 4)
            print(f"[train] ep{ep:3d} loss={avg_loss:.4f} tf={tf_ratio:.2f} "
                  f"val_autoreg={acc:.3f} elapsed={elapsed:.0f}s")
            if acc > best_acc + plateau_min_delta:
                best_acc = acc
                evals_since_improvement = 0
                torch.save({
                    "model_state": model.state_dict(),
                    "config": {
                        "vocab_size": CODE_VOCAB_SIZE,
                        "max_len": max_len,
                        "d_model": d_model,
                        "n_heads": n_heads,
                        "n_layers": n_layers,
                        "d_ffn": d_ffn,
                        "n_copy_heads": n_copy_heads,
                        "use_chunkwise": True,
                        "copy_gate_bias_init": copy_gate_bias_init,
                    },
                    "epoch": ep,
                    "val_autoreg": acc,
                    "n_train": len(train_pairs),
                    "n_val": len(val_pairs),
                }, CHECKPOINT_PATH)
                print(f"[train] ✓ saved (best autoreg={acc:.3f}) → {CHECKPOINT_PATH}")
                # Print 3 sample decodes
                for i, (q, tgt, out) in enumerate(samples[:3]):
                    mark = "✓" if out.strip() == tgt.strip() else "✗"
                    print(f"    {mark} tgt={tgt!r}")
                    print(f"      out={out!r}")
            else:
                evals_since_improvement += 1
                if evals_since_improvement >= plateau_patience:
                    print(f"[train] [PLATEAU] {evals_since_improvement} evals "
                          f"without improvement > {plateau_min_delta} — "
                          f"best={best_acc:.3f} ep={ep} — stopping early.")
                    plateau_triggered = True
        else:
            print(f"[train] ep{ep:3d} loss={avg_loss:.4f} tf={tf_ratio:.2f} "
                  f"elapsed={elapsed:.0f}s")

        history.append(rec)

        if plateau_triggered:
            break

    # Final report
    print(f"\n[train] best val_autoreg: {best_acc:.3f}")
    METRICS_PATH.write_text(json.dumps({
        "best_val_autoreg": best_acc,
        "n_params": n_params,
        "epochs": epochs,
        "history": history,
    }, indent=2))
    print(f"[train] metrics → {METRICS_PATH}")
    return best_acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=32)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--balanced-sampler", type=str, default="none",
                    choices=["none", "inverse", "sqrt_inverse", "capped"],
                    help="Upsample rare skeleton classes during training "
                         "(Round 3 lever). Default none (uniform shuffle).")
    ap.add_argument("--copy-gate-bias-init", type=float, default=-2.0,
                    help="Initial bias for copy gate (Round 5 lever). "
                         "-2.0 (v4 default, favors gen); 0.0 (neutral); "
                         "+1.0 (favors copy).")
    ap.add_argument("--normalize-skeletons", action="store_true",
                    help="R6 lever: collapse skeleton spacing variants "
                         "(FN(a, b) ≡ FN(a,b)) to canonical form before split.")
    ap.add_argument("--drop-rare-count", type=int, default=0,
                    help="R6 lever: drop training classes with count < N "
                         "(0 = keep all). Val set unaffected.")
    ap.add_argument("--extract-all-defs", action="store_true",
                    help="R8 lever: emit ALL top-level defs per solution "
                         "(not just last). +19%% raw pairs.")
    ap.add_argument("--synth-rare", type=int, default=0,
                    help="R9 lever: synthesize N pairs per rare class "
                         "(0=off, 30 recommended).")
    ap.add_argument("--synth-rare-min", type=int, default=3,
                    help="R9: min raw count to target (default 3).")
    ap.add_argument("--synth-rare-max", type=int, default=20,
                    help="R9: max raw count to target (default 20).")
    ap.add_argument("--dedupe-ambiguous", action="store_true",
                    help="R19: drop conceptual prompts with 3+ distinct "
                         "target skeletons; majority-vote on 2-skeleton cases.")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="R20: DataLoader worker processes (2 recommended).")
    args = ap.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_len=args.max_len,
        seed=args.seed,
        device=args.device,
        balanced_sampler=args.balanced_sampler,
        copy_gate_bias_init=args.copy_gate_bias_init,
        normalize_skeletons=args.normalize_skeletons,
        drop_rare_count=args.drop_rare_count,
        extract_all_defs=args.extract_all_defs,
        synth_rare=args.synth_rare,
        synth_rare_min=args.synth_rare_min,
        synth_rare_max=args.synth_rare_max,
        dedupe_ambiguous=args.dedupe_ambiguous,
        num_workers=args.num_workers,
    )
