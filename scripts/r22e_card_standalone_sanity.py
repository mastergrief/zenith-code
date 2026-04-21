"""R22e sanity — test PT+Delta MQAR card STANDALONE on adapter-extracted
strings from the r22b corpus.

Goal: disambiguate the 67% fired-precision (2W 1R on 3 fires) from r22b r7.
Three hypotheses:
  (a) card standalone fails on adapter inputs → noise training needed
  (b) card is ~100% standalone → integration (gate/hook) issue
  (c) small sample fluke

Method:
  1. Regenerate r22b's 60-prompt pooled corpus (same seeds).
  2. For each prompt, run parse_mqar_prompt to get the MQAR string.
  3. Run card standalone (no Gemma) on the MQAR tokens.
  4. Compare card's argmax (first char after <sep>) to expected digit.
  5. Report: card standalone accuracy + margin distribution.

If standalone acc is ~67% → (a) wins, noise training is right lever.
If standalone acc is >95% → (b) wins, integration needs more work.
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
assert "m" in globals() and "tok" in globals(), (  # type: ignore[name-defined]
    "run via bin/gemma-run scripts/r22e_card_standalone_sanity.py"
)

sys.path.insert(0, str(ROOT))
from calm.hrm.data import _CHAR_TO_ID, _ID_TO_CHAR
from calm.llm_computer.facades.retrieval import _monkey_patch_fast_encode

_monkey_patch_fast_encode(tok)  # type: ignore[name-defined]

# Reuse r22b round 2's corpus generator
import importlib.util as _ilu
_r2_src = (ROOT / "scripts" / "r22b_round2.py").read_text()
_r2_src = _r2_src.split("main()\nprint(\"R22B_R2_DONE\")")[0]
_ns = {"__name__": "_r22b_r2", "__file__": str(ROOT / "scripts" / "r22b_round2.py"),
       "m": m, "tok": tok}  # type: ignore[name-defined]
exec(compile(_r2_src, str(ROOT / "scripts" / "r22b_round2.py"), "exec"), _ns)
make_prompt = _ns["make_prompt"]
load_mqar_card = _ns["load_mqar_card"]
parse_mqar_prompt = _ns["parse_mqar_prompt"]
mqar_to_ids = _ns["mqar_to_ids"]


def main():
    # Regenerate r22b round 5/6/7's pooled corpus
    seeds = [2026_04_22, 2026_04_23]
    cells = [
        (5,  500, "confusing"),
        (5,  1500, "confusing_long"),
        (10, 500, "confusing"),
        (10, 1500, "confusing_long"),
        (15, 500, "confusing"),
        (15, 1500, "confusing_long"),
    ]
    REPLICAS = 5

    candidates = []
    for seed in seeds:
        rng = random.Random(seed)
        for (n_pairs, dist_tok, mode) in cells:
            for r in range(REPLICAS):
                prompt, q_key, expected = make_prompt(
                    n_pairs, dist_tok, mode, rng)
                candidates.append({
                    "seed": seed, "n_pairs": n_pairs,
                    "distractor_tokens": dist_tok, "mode": mode,
                    "replica": r, "prompt": prompt,
                    "query_key": q_key, "expected": expected,
                })
    print(f"[r22e] {len(candidates)} candidates (same corpus as r22b-5/6/7)")

    # Load card
    print("[r22e] loading MQAR card...")
    card = load_mqar_card(ROOT / "calm/hrm/checkpoints/copy_augmented_delta_mqar_best.pt")

    # Extract MQAR strings + run standalone
    print("[r22e] running card standalone on adapter-extracted strings...")
    n_parse_ok = 0
    n_correct = 0
    margins_correct = []
    margins_wrong = []
    wrong_cases = []
    from collections import defaultdict
    by_n = defaultdict(lambda: {"correct": 0, "total": 0})

    t0 = time.time()
    for c in candidates:
        mqar_str = parse_mqar_prompt(c["prompt"])
        if mqar_str is None:
            continue
        n_parse_ok += 1
        ids_list = mqar_to_ids(mqar_str)
        ids = torch.tensor([ids_list], device="cuda")
        with torch.no_grad():
            out = card(ids)  # (1, S, vocab)
        last = out[0, -1].float()
        card_argmax = int(last.argmax().item())
        card_char = _ID_TO_CHAR.get(card_argmax, "?")
        peak = last.max().item()
        med = last.median().item()
        margin = peak - med

        expected_char = c["expected"]
        is_correct = (card_char == expected_char)

        if is_correct:
            n_correct += 1
            margins_correct.append(margin)
        else:
            margins_wrong.append(margin)
            wrong_cases.append({
                "seed": c["seed"], "n_pairs": c["n_pairs"],
                "distractor": c["distractor_tokens"],
                "mqar": mqar_str[:60] + ("..." if len(mqar_str) > 60 else ""),
                "expected": expected_char, "got": card_char,
                "margin": margin,
            })

        by_n[c["n_pairs"]]["total"] += 1
        if is_correct:
            by_n[c["n_pairs"]]["correct"] += 1

    elapsed = time.time() - t0

    print(f"\n=== CARD STANDALONE ACCURACY ===")
    print(f"  elapsed: {elapsed:.1f}s ({len(candidates)} prompts)")
    print(f"  parse_ok: {n_parse_ok}/{len(candidates)}")
    print(f"  correct:  {n_correct}/{n_parse_ok} "
          f"({100 * n_correct / max(n_parse_ok, 1):.1f}%)")

    print(f"\n=== BY N_PAIRS ===")
    for n in sorted(by_n.keys()):
        s = by_n[n]
        pct = 100 * s["correct"] / max(s["total"], 1)
        print(f"  N={n:>2}:  {s['correct']:>2}/{s['total']:<2}  ({pct:.1f}%)")

    if margins_correct:
        print(f"\n=== MARGIN DISTRIBUTION ===")
        print(f"  correct: n={len(margins_correct)}  "
              f"mean={sum(margins_correct)/len(margins_correct):.2f}  "
              f"min={min(margins_correct):.2f}  "
              f"max={max(margins_correct):.2f}")
    if margins_wrong:
        print(f"  wrong:   n={len(margins_wrong)}  "
              f"mean={sum(margins_wrong)/len(margins_wrong):.2f}  "
              f"min={min(margins_wrong):.2f}  "
              f"max={max(margins_wrong):.2f}")

    if wrong_cases:
        print(f"\n=== WRONG CASES (first 10) ===")
        for w in wrong_cases[:10]:
            print(f"  N={w['n_pairs']:>2} dist={w['distractor']} "
                  f"margin={w['margin']:>6.2f}  "
                  f"{w['mqar']:<63}  exp={w['expected']!r} got={w['got']!r}")


main()
print("R22E_DONE")
