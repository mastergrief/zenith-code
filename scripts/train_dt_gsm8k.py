"""Train a (RDT-v2 or baseline) Delta-Transducer card on real GSM8k.

S0b2 of the rdt-v2 first-flag-enabled-card arc (board task
`1779311831769-1d1e02e5`). Codex audit chain `1779311799556` →
`1779312982222` → `1779313349390` → `1779313584790` locked the contract:

- Corpus: real GSM8k via `datasets` library parquet backend
  (HF datasets-server rate-limits paged fetches, see scripts/preflight).
- Split: train+val from `train` (last 10% deterministic held-out); `test`
  is OOV check + final A/B only.
- Tokenizer: `calm.llm_computer.gsm8k_tokenizer.Gsm8kTokenizer.from_corpus`
  on train+val only. 98-token vocab; normalizer v2 applied at train,
  eval, inference.
- Hard-fail at startup if any corpus char is OOV vs declared vocab.
- Target: `<bos> question <sep> {integer} <eos>` (final-integer-only).
- Loss: F.nll_loss on log-probs, masked to positions `> sep_pos`.
- max_len=512 (1.46% truncation tail).
- Tier-A+B flag bundle exposed via the S0a CLI plumbing pattern.
- Checkpoint shape compatible with `dt_install.load_dt_checkpoint`:
  `model_state` + `config` (incl. `gsm8k_char_vocab` and
  `gsm8k_normalizer_version` metadata).

Usage:
    PYTHONPATH=. python3 -u scripts/train_dt_gsm8k.py \\
        --epochs 30 --batch-size 32 --max-len 512 \\
        --d-model 64 --n-heads 32 --n-layers 4 --d-ffn 128 \\
        --use-loop-index --use-input-injection --use-z-init \\
        --use-lecun-init --use-gated-attention --use-short-conv \\
        --use-h-rmsnorm --use-h-layer-stack \\
        --n-iterations 2 --h-cycles 2

Baseline (flags off) and Core-H/L (codex's locked first-card config)
share this entry-point; the `--use-*` flags pick the variant.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from calm.llm_computer.copy_augmented_delta import build_copy_augmented_delta
from calm.llm_computer.gsm8k_tokenizer import (
    NORMALIZER_VERSION,
    Gsm8kTokenizer,
)


DEFAULT_CHECKPOINT = Path("calm/hrm/checkpoints/dt_gsm8k_best.pt")


def load_gsm8k_splits(val_frac: float = 0.10) -> tuple[list[dict], list[dict], list[dict]]:
    """Load GSM8k via the `datasets` lib parquet backend.

    Returns (train, val, test). Train is 90% (deterministic head); val is
    10% (deterministic tail of train). Test is the full HF test split.
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


class Gsm8kDataset(Dataset):
    """Yields `(ids, sep_pos, length)` per row.

    Rows exceeding `max_len` are dropped (truncation rate measured by the
    preflight; ~1.46% at max_len=512). Dropping > truncating prevents the
    last-chars-of-question being silently amputated.
    """

    def __init__(self, rows: list[dict], tok: Gsm8kTokenizer, max_len: int):
        self.tok = tok
        self.max_len = max_len
        self.items: list[tuple[list[int], int]] = []
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

    def __getitem__(self, i: int):
        return self.items[i]


def collate(batch, pad_id: int, max_len: int):
    """Right-pad to the batch's longest sequence; build a target-position
    mask (1 for positions whose NEXT-token prediction loss counts).
    """
    seq_lens = [len(ids) for ids, _ in batch]
    max_L = max(seq_lens)
    B = len(batch)
    pad = torch.full((B, max_L), pad_id, dtype=torch.long)
    sep_positions = torch.zeros(B, dtype=torch.long)
    # Loss is on positions where the model PREDICTS a target token. The
    # model produces log-probs at positions [0..L-1] predicting input[1..L].
    # So we want mask[t] = True iff (t+1) is a target-side position
    # (sep_pos < t+1 < L, i.e. t >= sep_pos).
    loss_mask = torch.zeros(B, max_L, dtype=torch.bool)
    for i, (ids, sep_pos) in enumerate(batch):
        L = len(ids)
        pad[i, :L] = torch.tensor(ids, dtype=torch.long)
        sep_positions[i] = sep_pos
        # Loss positions: sep_pos <= t < L-1 (predicts ids[t+1] which is
        # within the target span, including the final <eos>).
        if L > sep_pos + 1:
            loss_mask[i, sep_pos:L - 1] = True
    return pad, loss_mask, sep_positions


def autoreg_decode_integer(model, tok: Gsm8kTokenizer, question: str,
                           max_new: int = 16, device: str = "cuda") -> str:
    """Greedy autoreg from `<bos> question <sep>` to first `<eos>`.

    Returns the decoded target string (post-decode; no normalization).
    """
    model.eval()
    q_ids = tok.encode(question)
    prefix = [tok.bos_id] + q_ids + [tok.sep_id]
    ids = torch.tensor([prefix], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(max_new):
            if ids.shape[1] >= model.config.max_len:
                break
            log_probs = model(ids)
            next_id = int(log_probs[0, -1].argmax().item())
            if next_id == tok.eos_id:
                break
            ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
    return tok.decode(ids[0, len(prefix):].tolist(), stop_at_eos=True)


def autoreg_eval(model, tok: Gsm8kTokenizer, val_rows: list[dict],
                 cap: int, device: str) -> tuple[float, int, int]:
    """Returns (accuracy, n_correct, n_evaluated). Scores via
    `surface_gsm8k.score_row` for parity with the b-v Step 2 A/B harness.
    """
    from scripts.bv_step2.surface_gsm8k import score_row

    n_eval = min(cap, len(val_rows))
    n_correct = 0
    for r in val_rows[:n_eval]:
        generated = autoreg_decode_integer(model, tok, r["question"], device=device)
        _, correct = score_row(generated, r)
        if correct:
            n_correct += 1
    acc = n_correct / max(n_eval, 1)
    return acc, n_correct, n_eval


def train(
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    d_model: int = 64,
    n_heads: int = 32,
    n_layers: int = 4,
    d_ffn: int = 128,
    max_len: int = 512,
    n_copy_heads: int = 4,
    seed: int = 42,
    eval_every: int = 1,
    eval_cap: int = 100,
    device: str | None = None,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    n_train_cap: int | None = None,
    n_val_cap: int | None = None,
    # rdt-v2 Tier A+B build-time flags (S0a CLI plumbing pattern).
    use_chunkwise: bool = True,
    n_iterations: int = 1,
    use_loop_index: bool = False,
    use_input_injection: bool = False,
    use_gated_attention: bool = False,
    use_z_init: bool = False,
    use_lecun_init: bool = False,
    use_prefix_lm: bool = False,
    h_cycles: int = 1,
    use_h_rmsnorm: bool = False,
    use_short_conv: bool = False,
    use_h_layer_stack: bool = False,
    use_halt_head: bool = False,
    use_carry: bool = False,
    chunk_size: int = 32,
) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    print(f"[gsm8k] loading splits via `datasets` lib...")
    full_train, full_val, test_rows = load_gsm8k_splits(val_frac=0.10)
    print(f"[gsm8k] splits: train={len(full_train)}  val={len(full_val)}  test={len(test_rows)}")

    # Vocab MUST be built from the full train+val so the OOV gate (and
    # checkpoint metadata) is locked at the canonical 98-token shape
    # regardless of smoke-test caps. Caps apply to training data only.
    print(f"[gsm8k] building tokenizer from full train+val (normalizer {NORMALIZER_VERSION})...")
    tok = Gsm8kTokenizer.from_corpus(full_train + full_val)
    print(f"[gsm8k] vocab: {tok.vocab_size} tokens")

    train_rows = full_train[:n_train_cap] if n_train_cap is not None else full_train
    val_rows = full_val[:n_val_cap] if n_val_cap is not None else full_val
    if n_train_cap is not None or n_val_cap is not None:
        print(f"[gsm8k] applied caps: train={len(train_rows)}/{len(full_train)}  "
              f"val={len(val_rows)}/{len(full_val)}")

    # Hard-fail at startup on OOV (codex's smallest-S0 gate, mirrored here).
    print(f"[gsm8k] OOV check on test split (must pass; declared vocab is "
          f"locked at train time)...")
    tok.assert_corpus_covered(test_rows, label="test")
    print(f"[gsm8k] OOV check PASS — test split covered by train+val vocab")

    train_ds = Gsm8kDataset(train_rows, tok, max_len=max_len)
    val_ds = Gsm8kDataset(val_rows, tok, max_len=max_len)
    print(f"[gsm8k] usable rows after max_len={max_len} drop: "
          f"train={len(train_ds)} (dropped {train_ds.n_dropped}) "
          f"val={len(val_ds)} (dropped {val_ds.n_dropped})")

    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        collate_fn=lambda b: collate(b, tok.pad_id, max_len),
    )

    print(f"[gsm8k] building model (d_model={d_model}, layers={n_layers}, "
          f"vocab={tok.vocab_size})...")
    m = build_copy_augmented_delta(
        vocab_size=tok.vocab_size,
        d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ffn=d_ffn, max_len=max_len,
        n_copy_heads=n_copy_heads,
        sep_token_id=tok.sep_id,
        use_chunkwise=use_chunkwise,
        n_iterations=n_iterations,
        use_loop_index=use_loop_index,
        use_input_injection=use_input_injection,
        use_gated_attention=use_gated_attention,
        use_z_init=use_z_init,
        use_lecun_init=use_lecun_init,
        use_prefix_lm=use_prefix_lm,
        h_cycles=h_cycles,
        use_h_rmsnorm=use_h_rmsnorm,
        use_short_conv=use_short_conv,
        use_h_layer_stack=use_h_layer_stack,
        use_halt_head=use_halt_head,
        use_carry=use_carry,
    ).to(device)
    m.config.chunk_size = chunk_size
    m.max_len = max_len
    print(f"[gsm8k] params: {sum(p.numel() for p in m.parameters()):,}")
    print(f"[gsm8k] config: n_iter={m.config.n_iterations}  h_cycles={m.config.h_cycles}  "
          f"chunkwise={m.config.use_chunkwise}  layer_stack={m.config.use_h_layer_stack}")

    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_acc = -1.0
    best_ep = -1
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, epochs + 1):
        m.train()
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0
        for ids, mask, _sep in loader:
            ids = ids.to(device)
            mask = mask.to(device)
            log_probs = m(ids)
            # Shift: predict ids[:, 1:] from log_probs[:, :-1].
            log_probs = log_probs[:, :-1].contiguous()
            targets = ids[:, 1:].contiguous()
            mask = mask[:, :-1].contiguous()
            # F.nll_loss expects log-probs flattened over class dim.
            B, L, V = log_probs.shape
            loss_per = F.nll_loss(
                log_probs.reshape(B * L, V),
                targets.reshape(B * L),
                reduction="none",
            ).reshape(B, L)
            denom = mask.float().sum().clamp(min=1.0)
            loss = (loss_per * mask.float()).sum() / denom
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        sched.step()
        avg_loss = total_loss / max(n_batches, 1)
        epoch_secs = time.time() - t0
        print(f"[ep {ep:3d}] loss={avg_loss:.4f}  lr={sched.get_last_lr()[0]:.2e}  "
              f"time={epoch_secs:.1f}s")

        if ep % eval_every == 0 or ep == epochs:
            acc, n_c, n_e = autoreg_eval(m, tok, val_rows, cap=eval_cap, device=device)
            print(f"[ep {ep:3d}] val_acc={acc:.3f} ({n_c}/{n_e})")
            if acc > best_acc:
                best_acc = acc
                best_ep = ep
                torch.save({
                    "model_state": m.state_dict(),
                    "config": {
                        "vocab_size": tok.vocab_size,
                        "max_len": max_len,
                        "d_model": d_model,
                        "n_heads": n_heads,
                        "n_layers": n_layers,
                        "d_ffn": d_ffn,
                        "n_copy_heads": n_copy_heads,
                        "copy_gate_bias_init": -2.0,
                        # rdt-v2 flags — live values from m.config so reload
                        # rebuilds the trained architecture exactly.
                        "use_chunkwise": getattr(m.config, "use_chunkwise", True),
                        "chunk_size": getattr(m.config, "chunk_size", 32),
                        "n_iterations": getattr(m.config, "n_iterations", 1),
                        "use_loop_index": getattr(m.config, "use_loop_index", False),
                        "use_input_injection": getattr(m.config, "use_input_injection", False),
                        "use_gated_attention": getattr(m.config, "use_gated_attention", False),
                        "use_z_init": getattr(m.config, "use_z_init", False),
                        "use_lecun_init": getattr(m.config, "use_lecun_init", False),
                        "use_prefix_lm": getattr(m.config, "use_prefix_lm", False),
                        "h_cycles": getattr(m.config, "h_cycles", 1),
                        "use_h_rmsnorm": getattr(m.config, "use_h_rmsnorm", False),
                        "use_short_conv": getattr(m.config, "use_short_conv", False),
                        "use_h_layer_stack": getattr(m.config, "use_h_layer_stack", False),
                        "use_halt_head": getattr(m.config, "use_halt_head", False),
                        "use_carry": getattr(m.config, "use_carry", False),
                        # GSM8k-specific metadata (locked S0b2 contract).
                        "gsm8k_char_vocab": tok.vocab_as_list(),
                        "gsm8k_normalizer_version": tok.normalizer_version,
                    },
                    "epoch": ep,
                    "val_acc": acc,
                    "n_train": len(train_ds),
                    "n_val": len(val_ds),
                }, checkpoint_path)
                print(f"[ep {ep:3d}] saved best to {checkpoint_path}")

    print(f"\nBest: epoch {best_ep}  val_acc={best_acc:.3f}")
    print(f"Saved: {checkpoint_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=32)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--d-ffn", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--n-copy-heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--eval-cap", type=int, default=100)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--checkpoint-path", type=str, default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--n-train-cap", type=int, default=None,
                    help="Cap training rows (smoke test). None = full set.")
    ap.add_argument("--n-val-cap", type=int, default=None)
    # rdt-v2 Tier A+B flag bundle. Defaults preserve baseline behavior.
    ap.add_argument("--no-chunkwise", dest="use_chunkwise",
                    action="store_false", default=True)
    ap.add_argument("--n-iterations", type=int, default=1)
    ap.add_argument("--use-loop-index", action="store_true")
    ap.add_argument("--use-input-injection", action="store_true")
    ap.add_argument("--use-gated-attention", action="store_true")
    ap.add_argument("--use-z-init", action="store_true")
    ap.add_argument("--use-lecun-init", action="store_true")
    ap.add_argument("--use-prefix-lm", action="store_true")
    ap.add_argument("--h-cycles", type=int, default=1)
    ap.add_argument("--use-h-rmsnorm", action="store_true")
    ap.add_argument("--use-short-conv", action="store_true")
    ap.add_argument("--use-h-layer-stack", action="store_true")
    ap.add_argument("--use-halt-head", action="store_true")
    ap.add_argument("--use-carry", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=32)
    args = ap.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ffn=args.d_ffn,
        max_len=args.max_len,
        n_copy_heads=args.n_copy_heads,
        seed=args.seed,
        eval_every=args.eval_every,
        eval_cap=args.eval_cap,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        n_train_cap=args.n_train_cap,
        n_val_cap=args.n_val_cap,
        use_chunkwise=args.use_chunkwise,
        n_iterations=args.n_iterations,
        use_loop_index=args.use_loop_index,
        use_input_injection=args.use_input_injection,
        use_gated_attention=args.use_gated_attention,
        use_z_init=args.use_z_init,
        use_lecun_init=args.use_lecun_init,
        use_prefix_lm=args.use_prefix_lm,
        h_cycles=args.h_cycles,
        use_h_rmsnorm=args.use_h_rmsnorm,
        use_short_conv=args.use_short_conv,
        use_h_layer_stack=args.use_h_layer_stack,
        use_halt_head=args.use_halt_head,
        use_carry=args.use_carry,
        chunk_size=args.chunk_size,
    )
