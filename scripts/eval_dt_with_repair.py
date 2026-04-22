"""Round 4 A/B: DT skeleton repair post-decode.

Hypothesis: ~15-25% of baseline misses are one-char-off malformations
(d FN(n):, def FN(x,:, def m, n):, def FN(sel:). repair_skeleton()
rewrites them via deterministic regex — zero training, inference-only.

Usage:
    PYTHONPATH=. python3 -u scripts/eval_dt_with_repair.py \\
        --checkpoint calm/hrm/checkpoints/dt_code_skel_v4_mid_0184.pt
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import torch

from calm.hrm.code_dt_data import (
    _CODE_CHAR_TO_ID, code_detokenize, code_tokenize,
    extract_pairs_from_db, split_pairs,
)
from calm.hrm.dt_skeleton_repair import repair_skeleton, _is_valid
from calm.llm_computer.dt_install import load_dt_checkpoint


def decode(model, prompt: str, device, max_gen: int = 40) -> str:
    sep = _CODE_CHAR_TO_ID["<sep>"]
    eos = _CODE_CHAR_TO_ID["<eos>"]
    prefix = code_tokenize(prompt, add_bos=True, add_eos=False) + [sep]
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
    return code_detokenize(gen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="calm/hrm/checkpoints/dt_code_skel_v4_mid_0184.pt")
    ap.add_argument("--n-samples", type=int, default=157)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[r4-eval] loading {args.checkpoint}...")
    model, ckpt = load_dt_checkpoint(args.checkpoint, device=device)
    print(f"[r4-eval] epoch={ckpt.get('epoch')} "
          f"reported val_autoreg={ckpt.get('val_autoreg'):.3f}")

    pairs = extract_pairs_from_db(augment=False)
    _, val_pairs = split_pairs(pairs, val_frac=0.1, seed=42)
    val_pairs = val_pairs[:args.n_samples]

    n_baseline = 0
    n_repair = 0
    n_modified = 0
    n_invalid_baseline = 0
    n_invalid_after_repair = 0
    class_base: dict = defaultdict(lambda: [0, 0])  # [correct, total]
    class_rep: dict = defaultdict(lambda: [0, 0])
    repair_flip_examples = []  # baseline-miss → repair-hit

    for p in val_pairs:
        out_raw = decode(model, p.question, device).strip()
        out_rep = repair_skeleton(out_raw)

        base_exact = out_raw == p.expression.strip()
        rep_exact = out_rep == p.expression.strip()

        class_base[p.expression][1] += 1
        class_rep[p.expression][1] += 1
        if base_exact:
            n_baseline += 1
            class_base[p.expression][0] += 1
        if rep_exact:
            n_repair += 1
            class_rep[p.expression][0] += 1
        if out_rep != out_raw:
            n_modified += 1
        if not _is_valid(out_raw):
            n_invalid_baseline += 1
        if not _is_valid(out_rep):
            n_invalid_after_repair += 1

        if rep_exact and not base_exact:
            repair_flip_examples.append((p.question[:60], out_raw, out_rep))

    N = len(val_pairs)
    print(f"\n{'='*60}")
    print(f"Round 4 A/B — post-decode skeleton repair")
    print(f"{'='*60}")
    print(f"val_autoreg baseline:     {n_baseline}/{N} = {n_baseline/N:.4f}")
    print(f"val_autoreg + repair:     {n_repair}/{N} = {n_repair/N:.4f}")
    print(f"Δ:                        {(n_repair-n_baseline)/N:+.4f} "
          f"(+{n_repair-n_baseline} correct)")
    print(f"outputs modified:         {n_modified}/{N} ({n_modified/N:.2%})")
    print(f"invalid before repair:    {n_invalid_baseline}/{N} "
          f"({n_invalid_baseline/N:.2%})")
    print(f"invalid after repair:     {n_invalid_after_repair}/{N} "
          f"({n_invalid_after_repair/N:.2%})")
    print(f"\nbase-miss → repair-hit examples (up to 5):")
    for q, b, r in repair_flip_examples[:5]:
        print(f"  Q: {q!r}")
        print(f"  raw:    {b!r}")
        print(f"  rep:    {r!r}")

    # Per-class delta for classes with n>=3
    print(f"\n[r4-eval] per-class accuracy delta (n>=3):")
    sorted_classes = sorted(
        [(s, t[1]) for s, t in class_base.items() if t[1] >= 3],
        key=lambda x: -x[1],
    )
    print(f"{'skeleton':<30} {'base':>6} {'rep':>6} {'Δ':>6}")
    for skel, total in sorted_classes:
        b = class_base[skel][0]
        r = class_rep[skel][0]
        print(f"  {skel!r:<30} {b:>5}/{total:<2} {r:>5}/{total:<2} "
              f"{(r-b)/max(total,1):+.2f}")


if __name__ == "__main__":
    main()
