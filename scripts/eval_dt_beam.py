"""R22: Beam search decode for DT checkpoints.

Greedy argmax (v9 default) may emit wrong token at an early step even
when the right token is top-k. Beam search keeps B hypotheses, extends
each, and returns the best-scoring (or highest-validity) completion.

Adds a skeleton-validity filter on final beams: a well-formed
`def FN(<args>):` beam wins over a malformed one even if the malformed
has higher log-prob — matches R4's post-decode repair philosophy but
catches more cases because it prevents commitment to bad paths early.

Usage:
    PYTHONPATH=. python3 -u scripts/eval_dt_beam.py \\
        --checkpoint calm/hrm/checkpoints/dt_code_skel_best.pt \\
        --n-samples 200 --beam 4

Reports: greedy vs beam autoreg, delta, per-class breakdown.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple

import torch

from calm.hrm.code_dt_data import (
    _CODE_CHAR_TO_ID, code_detokenize, code_tokenize,
    extract_pairs_from_db, split_pairs,
)
from calm.hrm.dt_skeleton_repair import _is_valid
from calm.llm_computer.dt_install import load_dt_checkpoint


@dataclass
class Beam:
    ids: List[int]  # generated token ids
    logp: float     # cumulative log prob
    done: bool = False


def beam_decode(model, prompt: str, device, beam: int = 4,
                 max_gen: int = 40,
                 prefer_valid: bool = True) -> Tuple[str, float]:
    """Beam decode, prefer valid skeletons among final beams.

    Returns (best_decoded_string, best_logp).
    """
    sep = _CODE_CHAR_TO_ID["<sep>"]
    eos = _CODE_CHAR_TO_ID["<eos>"]
    prefix = code_tokenize(prompt, add_bos=True, add_eos=False) + [sep]
    max_len = getattr(model, "max_len", 256)
    if len(prefix) > max_len - max_gen - 1:
        prefix = [prefix[0]] + prefix[-(max_len - max_gen - 2):]

    beams: List[Beam] = [Beam(ids=[], logp=0.0, done=False)]

    for step in range(max_gen):
        if all(b.done for b in beams):
            break
        new_beams: List[Beam] = []
        for b in beams:
            if b.done:
                new_beams.append(b)
                continue
            x = torch.tensor([prefix + b.ids], dtype=torch.long,
                              device=device)
            with torch.no_grad():
                lp = model(x)  # (1, S, V) log-probs
            last_lp = lp[0, -1]  # (V,)
            topk = torch.topk(last_lp, k=beam)
            top_vals = topk.values.tolist()
            top_ids = topk.indices.tolist()
            for tid, tlp in zip(top_ids, top_vals):
                nb = Beam(ids=b.ids + [tid], logp=b.logp + tlp,
                          done=(tid == eos))
                new_beams.append(nb)

        # Prune to top-B by logp
        new_beams.sort(key=lambda b: b.logp, reverse=True)
        beams = new_beams[:beam]

    # Decode candidates
    candidates: List[Tuple[str, float, bool]] = []
    for b in beams:
        # Strip trailing eos if present
        ids = b.ids
        if ids and ids[-1] == eos:
            ids = ids[:-1]
        s = code_detokenize(ids).strip()
        candidates.append((s, b.logp, _is_valid(s)))

    # Prefer highest-logp VALID beam; fall back to highest-logp otherwise.
    if prefer_valid:
        valid = [c for c in candidates if c[2]]
        if valid:
            best = max(valid, key=lambda c: c[1])
            return best[0], best[1]
    best = max(candidates, key=lambda c: c[1])
    return best[0], best[1]


def greedy_decode(model, prompt: str, device, max_gen: int = 40) -> str:
    sep = _CODE_CHAR_TO_ID["<sep>"]
    eos = _CODE_CHAR_TO_ID["<eos>"]
    prefix = code_tokenize(prompt, add_bos=True, add_eos=False) + [sep]
    ids = list(prefix)
    gen: List[int] = []
    max_len = getattr(model, "max_len", 256)
    for _ in range(max_gen):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            lp = model(x)
        nxt = int(lp[0, -1].argmax().item())
        if nxt == eos:
            break
        gen.append(nxt)
        ids.append(nxt)
        if len(ids) >= max_len - 1:
            break
    return code_detokenize(gen).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="calm/hrm/checkpoints/dt_code_skel_best.pt")
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-prefer-valid", action="store_true",
                    help="Do NOT bias toward valid-skeleton beams "
                         "(raw highest-logp only).")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[beam-eval] loading {args.checkpoint}...")
    model, ckpt = load_dt_checkpoint(args.checkpoint, device=device)
    print(f"[beam-eval] epoch={ckpt.get('epoch')} "
          f"reported val_autoreg={ckpt.get('val_autoreg'):.3f}")

    pairs = extract_pairs_from_db(augment=False, extract_all_defs=True)
    _, val_pairs = split_pairs(pairs, val_frac=0.1, seed=42)
    val_pairs = val_pairs[:args.n_samples]
    print(f"[beam-eval] evaluating on {len(val_pairs)} unaug val problems "
          f"(beam={args.beam}, prefer_valid={not args.no_prefer_valid})")

    n_greedy = 0
    n_beam = 0
    class_g: dict = defaultdict(lambda: [0, 0])
    class_b: dict = defaultdict(lambda: [0, 0])
    flips = []  # greedy-miss → beam-hit

    for i, p in enumerate(val_pairs):
        g_out = greedy_decode(model, p.question, device)
        b_out, _ = beam_decode(model, p.question, device,
                                beam=args.beam,
                                prefer_valid=not args.no_prefer_valid)
        g_ex = g_out == p.expression.strip()
        b_ex = b_out == p.expression.strip()
        class_g[p.expression][1] += 1
        class_b[p.expression][1] += 1
        if g_ex:
            n_greedy += 1
            class_g[p.expression][0] += 1
        if b_ex:
            n_beam += 1
            class_b[p.expression][0] += 1
        if b_ex and not g_ex:
            flips.append((p.question[:60], g_out, b_out))
        if i < 5:
            print(f"  [{i}] Q: {p.question[:55]!r}  tgt: {p.expression!r}")
            print(f"      g: {g_out!r}  {'✓' if g_ex else '✗'}")
            print(f"      b: {b_out!r}  {'✓' if b_ex else '✗'}")

    N = len(val_pairs)
    print(f"\n{'='*60}")
    print(f"Round 22 A/B — beam search decode")
    print(f"{'='*60}")
    print(f"greedy autoreg: {n_greedy}/{N} = {n_greedy/N:.4f}")
    print(f"beam autoreg:   {n_beam}/{N} = {n_beam/N:.4f}")
    print(f"Δ:              {(n_beam-n_greedy)/N:+.4f} "
          f"(+{n_beam-n_greedy} correct)")
    print(f"\ngreedy-miss → beam-hit examples (first 5):")
    for q, g, b in flips[:5]:
        print(f"  Q: {q!r}")
        print(f"  g: {g!r}")
        print(f"  b: {b!r}")

    # Per-class delta
    print(f"\n[beam-eval] per-class delta (n>=3):")
    sorted_classes = sorted(
        [(s, t[1]) for s, t in class_g.items() if t[1] >= 3],
        key=lambda x: -x[1],
    )
    print(f"{'skeleton':<30} {'greedy':>7} {'beam':>6} {'Δ':>6}")
    for skel, total in sorted_classes:
        g = class_g[skel][0]
        b = class_b[skel][0]
        print(f"  {skel!r:<30} {g:>5}/{total:<2} {b:>4}/{total:<2} "
              f"{(b-g)/max(total,1):+.2f}")


if __name__ == "__main__":
    main()
