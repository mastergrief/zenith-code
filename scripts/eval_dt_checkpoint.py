"""Standalone eval for a trained DT checkpoint.

Loads `dt_code_skel_best.pt` (or any compatible), runs autoregressive
eval on the canonical val split, and reports:
  - val_autoreg (exact skeleton match)
  - per-skeleton-class accuracy (how well does DT discriminate common
    shapes vs rare ones)
  - copy-gate usage diagnostic (is DT actually using its copy mechanism,
    or defaulting to pure generation?)

Usage:
    PYTHONPATH=. python3 -u scripts/eval_dt_checkpoint.py \\
        [--checkpoint calm/hrm/checkpoints/dt_code_skel_best.pt] \\
        [--n-samples 200]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from calm.hrm.code_dt_data import (
    _CODE_CHAR_TO_ID,
    _CODE_ID_TO_CHAR,
    code_detokenize,
    code_tokenize,
    extract_pairs_from_db,
    split_pairs,
)
from calm.llm_computer.dt_install import load_dt_checkpoint


def autoreg_decode_with_gate(model, prompt: str, max_gen: int = 40):
    """Greedy decode + return (decoded_str, avg_copy_gate)."""
    bos = _CODE_CHAR_TO_ID["<bos>"]
    sep = _CODE_CHAR_TO_ID["<sep>"]
    eos = _CODE_CHAR_TO_ID["<eos>"]
    device = next(model.parameters()).device
    prefix = code_tokenize(prompt, add_bos=True, add_eos=False) + [sep]
    ids = list(prefix)
    gen = []
    copy_gates = []
    for _ in range(max_gen):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            lp = model(x)
            # Try to grab copy_gate value from model internals if exposed.
            # Many CopyAugmented* expose `last_p_copy` after a forward.
            gate = getattr(model, "last_p_copy", None)
            if gate is not None:
                # last position's copy gate
                g = gate[0, -1].mean().item() if gate.dim() > 1 else float(gate)
                copy_gates.append(g)
        nxt = int(lp[0, -1].argmax().item())
        if nxt == eos:
            break
        gen.append(nxt)
        ids.append(nxt)
        if len(ids) >= getattr(model, "max_len", 256) - 1:
            break
    decoded = code_detokenize(gen)
    avg_gate = sum(copy_gates) / len(copy_gates) if copy_gates else None
    return decoded, avg_gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="calm/hrm/checkpoints/dt_code_skel_best.pt")
    ap.add_argument("--n-samples", type=int, default=200,
                    help="How many val problems to eval (cap)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--augment", action="store_true",
                    help="Use augmented data (match training distribution)")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"[eval] loading {args.checkpoint}...")
    model, ckpt = load_dt_checkpoint(args.checkpoint, device=device)
    print(f"[eval] loaded: epoch={ckpt.get('epoch')} "
          f"reported val_autoreg={ckpt.get('val_autoreg'):.3f}")
    print(f"[eval] params: {sum(p.numel() for p in model.parameters()):,}")

    # Load val data — same split seed as training for reproducibility
    print(f"[eval] extracting pairs (augment={args.augment})...")
    pairs = extract_pairs_from_db(augment=args.augment)
    _, val_pairs = split_pairs(pairs, val_frac=0.1, seed=42)
    val_pairs = val_pairs[:args.n_samples]
    print(f"[eval] evaluating on {len(val_pairs)} val problems")

    # Per-class accuracy tracker
    class_total: dict = defaultdict(int)
    class_correct: dict = defaultdict(int)
    n_correct = 0
    gates = []

    # Sample decode tracking
    hits_by_shape: dict = defaultdict(list)
    misses_by_shape: dict = defaultdict(list)

    for p in val_pairs:
        decoded, gate = autoreg_decode_with_gate(model, p.question)
        exact = decoded.strip() == p.expression.strip()
        if exact:
            n_correct += 1
            class_correct[p.expression] += 1
            hits_by_shape[p.expression].append(p.question[:60])
        else:
            misses_by_shape[p.expression].append(
                (p.question[:60], decoded))
        class_total[p.expression] += 1
        if gate is not None:
            gates.append(gate)

    acc = n_correct / max(len(val_pairs), 1)
    print(f"\n[eval] overall val_autoreg: {acc:.4f} "
          f"({n_correct}/{len(val_pairs)})")

    # Per-class breakdown — bucket by frequency in val set
    sorted_classes = sorted(class_total.items(), key=lambda x: -x[1])
    print(f"\n[eval] per-class accuracy (top-15 by val frequency):")
    print(f"{'skeleton':<35} {'val_count':>10} {'correct':>10} {'acc':>8}")
    for skel, total in sorted_classes[:15]:
        c = class_correct[skel]
        ac = c / total if total else 0
        print(f"  {skel!r:<35} {total:>10} {c:>10} {ac:>8.2%}")

    # Copy-gate usage
    if gates:
        avg_gate = sum(gates) / len(gates)
        print(f"\n[eval] avg copy-gate over {len(gates)} decodes: {avg_gate:.3f}")
        print("  (0 = pure generation, 1 = pure copy; typical useful: 0.2-0.5)")
    else:
        print(f"\n[eval] copy-gate not exposed by model (no `last_p_copy`)")

    # Majority-class prior check
    val_skel_counts = Counter(p.expression for p in val_pairs)
    majority_skel, majority_count = val_skel_counts.most_common(1)[0]
    print(f"\n[eval] majority class in val: {majority_skel!r} "
          f"({majority_count}/{len(val_pairs)} = "
          f"{majority_count/len(val_pairs):.2%})")
    print(f"  if DT only predicted mode, would score "
          f"{majority_count/len(val_pairs):.2%}")
    print(f"  DT is {'ABOVE' if acc > majority_count/len(val_pairs) else 'AT/BELOW'} "
          "that baseline")

    # Dump detailed miss samples for the top-5 miss-heavy classes
    miss_counts = sorted(
        [(s, len(misses_by_shape[s])) for s in misses_by_shape],
        key=lambda x: -x[1]
    )
    print(f"\n[eval] 5 most-missed classes (showing 2 examples each):")
    for skel, n in miss_counts[:5]:
        print(f"  ✗ tgt={skel!r}  ({n} misses)")
        for q, out in misses_by_shape[skel][:2]:
            print(f"     Q: {q!r}")
            print(f"     O: {out!r}")


if __name__ == "__main__":
    main()
