"""R24: comprehensive post-training eval for DT checkpoints.

Combines all inference-time levers on one checkpoint:
  1. Full unaug val set (no eval_cap subsample) — honest number
  2. Greedy baseline
  3. + R4 skeleton repair (post-decode regex rewrites)
  4. + R22 beam search with validity preference
  5. + combined beam + repair
  6. Per-class accuracy breakdown
  7. Copy-gate diagnostic (R1)

Usage:
    PYTHONPATH=. python3 -u scripts/eval_dt_final.py \\
        --checkpoint calm/hrm/checkpoints/dt_code_skel_best.pt \\
        --beam 4

Produces the authoritative "what can this DT checkpoint actually do"
report. Expected to land 0.02-0.07 above the trainer's reported
best_autoreg once levers compose.
"""
from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict

import torch

from calm.hrm.code_dt_data import (
    _CODE_CHAR_TO_ID, code_detokenize, code_tokenize,
    extract_pairs_from_db, split_pairs,
)
from calm.hrm.dt_skeleton_repair import _is_valid, repair_skeleton
from calm.llm_computer.dt_install import load_dt_checkpoint

# Reuse beam decoding from R22
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from eval_dt_beam import beam_decode, greedy_decode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="calm/hrm/checkpoints/dt_code_skel_best.pt")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--n-samples", type=int, default=0,
                    help="0 = full val, else cap at N")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[final-eval] loading {args.checkpoint}...")
    model, ckpt = load_dt_checkpoint(args.checkpoint, device=device)
    print(f"[final-eval] epoch={ckpt.get('epoch')} "
          f"train-reported={ckpt.get('val_autoreg'):.3f}")

    pairs = extract_pairs_from_db(augment=False, extract_all_defs=True)
    _, val_pairs = split_pairs(pairs, val_frac=0.1, seed=42)
    if args.n_samples > 0:
        val_pairs = val_pairs[:args.n_samples]
    N = len(val_pairs)
    print(f"[final-eval] val set: {N} unaug problems\n")

    # Four conditions:
    #   A. greedy
    #   B. greedy + R4 repair
    #   C. beam (B=args.beam, prefer_valid=True)
    #   D. beam + R4 repair
    cond_correct = {"greedy": 0, "greedy+repair": 0,
                    "beam": 0, "beam+repair": 0}
    class_correct: dict = {k: defaultdict(lambda: [0, 0]) for k in cond_correct}
    gates = []
    t0 = time.time()

    for i, p in enumerate(val_pairs):
        tgt = p.expression.strip()

        # A. greedy
        g = greedy_decode(model, p.question, device)
        # B. greedy + repair
        gr = repair_skeleton(g)
        # C. beam with validity bias
        b, _ = beam_decode(model, p.question, device,
                            beam=args.beam, prefer_valid=True)
        # D. beam + repair
        br = repair_skeleton(b)

        for cond, out in [("greedy", g), ("greedy+repair", gr),
                           ("beam", b), ("beam+repair", br)]:
            class_correct[cond][p.expression][1] += 1
            if out == tgt:
                cond_correct[cond] += 1
                class_correct[cond][p.expression][0] += 1

        # Gate diagnostic: read last_p_copy from last forward
        g_attr = getattr(model, "last_p_copy", None)
        if g_attr is not None:
            gates.append(g_attr[0, -1].mean().item())

        if i == 0 or (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (N - i - 1) / rate
            print(f"  [{i+1}/{N}] elapsed={elapsed:.0f}s "
                  f"rate={rate:.1f}/s eta={eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"R24 comprehensive eval — n={N} unaug val, beam={args.beam}")
    print(f"{'='*60}")
    print(f"elapsed:              {elapsed:.0f}s")
    for cond, cnt in cond_correct.items():
        print(f"{cond:<18} {cnt:>4}/{N}  = {cnt/N:.4f}")

    best = max(cond_correct.items(), key=lambda kv: kv[1])
    print(f"\nBest condition: {best[0]!r} @ {best[1]/N:.4f}")
    if gates:
        print(f"avg copy-gate: {sum(gates)/len(gates):.3f}")

    # Per-class breakdown on BEST condition
    print(f"\n[final-eval] per-class accuracy ({best[0]}, n>=3):")
    cb = class_correct[best[0]]
    sorted_classes = sorted(
        [(s, t[1]) for s, t in cb.items() if t[1] >= 3],
        key=lambda x: -x[1],
    )
    print(f"{'skeleton':<35} {'val_n':>6} {'correct':>8} {'acc':>8}")
    for skel, total in sorted_classes:
        c = cb[skel][0]
        print(f"  {skel!r:<35} {total:>6} {c:>8} {c/total:>8.2%}")

    # Still-failing-completely classes
    zero_classes = [s for s, t in cb.items() if t[1] >= 3 and cb[s][0] == 0]
    print(f"\n[final-eval] classes with 0% accuracy (n>=3): {len(zero_classes)}")
    for s in zero_classes[:20]:
        print(f"  0/{cb[s][1]}  {s!r}")


if __name__ == "__main__":
    main()
