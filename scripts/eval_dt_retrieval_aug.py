"""Round 2: retrieval-augmented prompt A/B for DT.

Hypothesis: injecting top-k retrieved skeletons into the DT prompt
gives the copy path strong structural evidence. Even at gate=0.193
(Round 1 measurement on v4_mid_0184), the retrieved answer sitting
in the copyable prefix should lift val_autoreg meaningfully without
retraining.

Zero training cost — pure inference-time lever.

Leakage guard: val pairs come from the same CodeExampleDB the
retriever indexes, so naive top-1 finds the val problem ITSELF.
Drop near-duplicates by normalized-string equality before taking
top-k.

Usage:
    PYTHONPATH=. python3 -u scripts/eval_dt_retrieval_aug.py \\
        --checkpoint calm/hrm/checkpoints/dt_code_skel_v4_mid_0184.pt \\
        --n-samples 157 --top-k 3
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import torch

from calm.hrm.code_dt_data import (
    _CODE_CHAR_TO_ID,
    code_detokenize,
    code_tokenize,
    extract_pairs_from_db,
    split_pairs,
)
from calm.llm_computer.dt_install import load_dt_checkpoint
from calm.llm_computer.facades.code_example_db import CodeExampleDB


_ALLOWED = set(
    "0123456789+-*/()=.,:; "
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "_><"
)

_DEF_RE = re.compile(
    r"^\s*def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*[^:]+)?:",
    re.MULTILINE,
)


def _norm(s: str) -> str:
    """Normalize for near-duplicate comparison."""
    return " ".join(s.lower().split())[:120]


def _extract_skeleton_str(solution: str) -> str | None:
    """Same logic as code_dt_data._extract_skeleton, returns skel str only."""
    sol = solution.replace("\r", "")
    matches = list(_DEF_RE.finditer(sol))
    if not matches:
        return None
    top = [m for m in matches if m.group(0).startswith("def ")]
    m = top[-1] if top else matches[-1]
    args = m.group(2).strip()
    skeleton = f"def FN({args}):"
    if len(skeleton) > 80:
        return None
    if not all(c in _ALLOWED for c in skeleton):
        return None
    return skeleton


def _build_augmented_prompt(
    original: str, top_skels: list[str], max_total: int = 220,
) -> str:
    """Format: 'original; similar: skel1; skel2; skel3'. All chars in
    DT vocab. Sep <sep> is added by tokenizer call downstream.
    """
    if not top_skels:
        return original
    similar = "; ".join(top_skels)
    aug = f"{original}; similar: {similar}"
    if len(aug) > max_total:
        # Trim trailing skeletons to fit
        while len(aug) > max_total and len(top_skels) > 1:
            top_skels = top_skels[:-1]
            similar = "; ".join(top_skels)
            aug = f"{original}; similar: {similar}"
        if len(aug) > max_total:
            aug = aug[:max_total]
    return aug


def decode(model, prompt: str, device, max_gen: int = 40) -> tuple[str, float | None]:
    sep = _CODE_CHAR_TO_ID["<sep>"]
    eos = _CODE_CHAR_TO_ID["<eos>"]
    prefix = code_tokenize(prompt, add_bos=True, add_eos=False) + [sep]
    # Clamp to max_len - 40 for decode room
    max_len = getattr(model, "max_len", 256)
    if len(prefix) > max_len - 40:
        prefix = [prefix[0]] + prefix[-(max_len - 41):]
    ids = list(prefix)
    gen = []
    gates = []
    for _ in range(max_gen):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            lp = model(x)
            g = getattr(model, "last_p_copy", None)
            if g is not None:
                gates.append(g[0, -1].mean().item())
        nxt = int(lp[0, -1].argmax().item())
        if nxt == eos:
            break
        gen.append(nxt)
        ids.append(nxt)
        if len(ids) >= max_len - 1:
            break
    avg_gate = sum(gates) / len(gates) if gates else None
    return code_detokenize(gen), avg_gate


def retrieve_skeletons(
    db: CodeExampleDB, query: str, top_k: int = 3, search_k: int = 10,
) -> list[str]:
    """Retrieve top-k UNIQUE skeletons, filtering near-duplicate problems."""
    q_norm = _norm(query)
    hits = db.retrieve(query, k=search_k, mode="tfidf")
    seen: set[str] = set()
    out: list[str] = []
    for h in hits:
        ex_norm = _norm(h.example.problem)
        if ex_norm == q_norm:
            continue  # leakage guard
        # Levenshtein-like char-set overlap proxy
        if len(set(ex_norm.split()) & set(q_norm.split())) > 0.9 * min(
            len(ex_norm.split()), len(q_norm.split()), 1
        ) and len(ex_norm) > 20 and abs(len(ex_norm) - len(q_norm)) < 10:
            # Very similar text — likely same problem with minor edit
            continue
        skel = _extract_skeleton_str(h.example.solution)
        if skel is None:
            continue
        if skel in seen:
            continue
        seen.add(skel)
        out.append(skel)
        if len(out) >= top_k:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="calm/hrm/checkpoints/dt_code_skel_v4_mid_0184.pt")
    ap.add_argument("--n-samples", type=int, default=157)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[aug-eval] loading {args.checkpoint}...")
    model, ckpt = load_dt_checkpoint(args.checkpoint, device=device)
    print(f"[aug-eval] loaded: epoch={ckpt.get('epoch')} "
          f"reported val_autoreg={ckpt.get('val_autoreg'):.3f}")

    print(f"[aug-eval] loading CodeExampleDB + TF-IDF index...")
    db = CodeExampleDB.load_default()
    db.load_indices(Path(".cache/r53_code_db"))

    print(f"[aug-eval] extracting val pairs (augment=False, seed=42)...")
    pairs = extract_pairs_from_db(augment=False)
    _, val_pairs = split_pairs(pairs, val_frac=0.1, seed=42)
    val_pairs = val_pairs[:args.n_samples]
    print(f"[aug-eval] evaluating on {len(val_pairs)} val problems")

    # A/B loop
    base_correct = 0
    aug_correct = 0
    base_class_acc: dict = defaultdict(lambda: [0, 0])  # [correct, total]
    aug_class_acc: dict = defaultdict(lambda: [0, 0])
    aug_gate_sum = 0.0
    base_gate_sum = 0.0
    n_base_gates = 0
    n_aug_gates = 0
    n_retrieval_hits = 0
    retrieval_skeleton_matches_target = 0

    for i, p in enumerate(val_pairs):
        # Baseline
        base_out, base_g = decode(model, p.question, device)
        base_exact = base_out.strip() == p.expression.strip()
        if base_exact:
            base_correct += 1
            base_class_acc[p.expression][0] += 1
        base_class_acc[p.expression][1] += 1
        if base_g is not None:
            base_gate_sum += base_g
            n_base_gates += 1

        # Aug
        top_skels = retrieve_skeletons(db, p.question, top_k=args.top_k)
        if top_skels:
            n_retrieval_hits += 1
            if p.expression in top_skels:
                retrieval_skeleton_matches_target += 1
        aug_prompt = _build_augmented_prompt(p.question, top_skels)
        aug_out, aug_g = decode(model, aug_prompt, device)
        aug_exact = aug_out.strip() == p.expression.strip()
        if aug_exact:
            aug_correct += 1
            aug_class_acc[p.expression][0] += 1
        aug_class_acc[p.expression][1] += 1
        if aug_g is not None:
            aug_gate_sum += aug_g
            n_aug_gates += 1

        if i < 5:
            print(f"  [{i}] Q: {p.question[:60]!r}")
            print(f"      tgt:   {p.expression!r}")
            print(f"      base:  {base_out.strip()!r}  "
                  f"{'✓' if base_exact else '✗'}")
            print(f"      retr:  {top_skels}")
            print(f"      aug:   {aug_out.strip()!r}  "
                  f"{'✓' if aug_exact else '✗'}")

    N = len(val_pairs)
    print(f"\n{'='*60}")
    print(f"Round 2 A/B — retrieval-augmented prompt")
    print(f"{'='*60}")
    print(f"val_autoreg baseline:    {base_correct}/{N} = {base_correct/N:.4f}")
    print(f"val_autoreg retr-aug:    {aug_correct}/{N} = {aug_correct/N:.4f}")
    print(f"Δ:                       {(aug_correct - base_correct)/N:+.4f} "
          f"(+{aug_correct - base_correct} correct)")
    print(f"retrieval hit rate:      {n_retrieval_hits}/{N} "
          f"({n_retrieval_hits/N:.2%})")
    print(f"retrieved skel == target: {retrieval_skeleton_matches_target}/{N} "
          f"({retrieval_skeleton_matches_target/N:.2%})")
    if n_base_gates:
        print(f"avg copy-gate baseline:  {base_gate_sum/n_base_gates:.3f}")
    if n_aug_gates:
        print(f"avg copy-gate retr-aug:  {aug_gate_sum/n_aug_gates:.3f}")

    # Per-class delta for rare classes
    print(f"\n[aug-eval] per-class accuracy delta (n>=3 in val):")
    print(f"{'skeleton':<30} {'base':>6} {'aug':>6} {'Δ':>6}")
    combined = sorted(
        [(s, t[1]) for s, t in base_class_acc.items() if t[1] >= 3],
        key=lambda x: -x[1],
    )
    for skel, total in combined:
        b = base_class_acc[skel][0]
        a = aug_class_acc[skel][0]
        print(f"  {skel!r:<30} {b:>5}/{total:<2} {a:>5}/{total:<2} "
              f"{(a-b)/max(total,1):+.2f}")


if __name__ == "__main__":
    main()
