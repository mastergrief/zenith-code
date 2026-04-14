"""NEW held-out OOD for the multi10 HRM.

Format VARIATIONS of the 6 extended training categories. Each variation
is RELATED to but DIFFERENT from the training surface form — so success
requires format-invariance, not template memorization.

Trained format    → Held-out variation
-----------------   --------------------------------------------------
code_var           function-call syntax: `f(17, 23, op=mul)`
prefix_op          phrasal verbs: `take 8 and 14 and sum them`
distractor         past-tense narratives: `a farmer had 5 apples...`
units              new unit types: `3 degrees plus 7 degrees` (degrees
                   not in training; trained on meters/kg/liters/etc.)
let_bound          alt variable declaration: `let a be 5, let b be 12`
eq_complete        reworded: `what is the result of 7 * 9`

18 cases, 3 per category. If multi10 training teaches format-invariance,
these should lift substantially over h=32 baseline's 17%.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from calm.hrm.data import _CHAR_TO_ID, _ID_TO_CHAR
from calm.hrm.inference import HRMSeq2SeqReasoner
from calm.llm_computer.interpret import InterpreterError, interpret
from calm.llm_computer.parse import ParseError, extract_problem_from_trace, parse_expression


_HELDOUT_OOD = [
    # Category 1: function-call syntax (not in training code_var format)
    ("f(17, 23, op=mul)",                            "391", "fn-call"),
    ("compute(50, 30, +)",                           "80",  "fn-call"),
    ("apply(100, 25, subtract)",                     "75",  "fn-call"),

    # Category 2: phrasal verbs (not in training prefix_op format)
    ("take 8 and 14 and sum them",                   "22",  "phrasal"),
    ("take 50 and 30 and multiply",                  "1500","phrasal"),
    ("take 100 and 25 and find the difference",      "75",  "phrasal"),

    # Category 3: past-tense narratives (not in training distractor format)
    ("a farmer had 5 apples. he found 3 more. how many apples does he have",
                                                     "8",   "past-narr"),
    ("she had 100 dollars. she gave away 25. how much is left",
                                                     "75",  "past-narr"),
    ("there were 12 birds. 4 flew away. how many remain",
                                                     "8",   "past-narr"),

    # Category 4: unit types NOT in training (training: meters/kg/liters/grams/sec/feet/dollars)
    ("3 degrees plus 7 degrees equals",              "10",  "new-units"),
    ("4 pounds times 5 equals",                      "20",  "new-units"),
    ("9 miles minus 2 miles equals",                 "7",   "new-units"),

    # Category 5: alternative let phrasing (not `if a=X and b=Y`)
    ("let a be 5, let b be 12, find a+b",            "17",  "alt-let"),
    ("set x to 20, set y to 8, compute x-y",         "12",  "alt-let"),
    ("define m=6, n=7, evaluate m*n",                "42",  "alt-let"),

    # Category 6: eq-complete variations (not `X op Y = ?`)
    ("what is the result of 7 * 9",                  "63",  "eq-var"),
    ("compute 50 + 30",                              "80",  "eq-var"),
    ("evaluate 20 - 8",                              "12",  "eq-var"),
]


def _decode(reasoner, q: str) -> str:
    pad = _CHAR_TO_ID["<pad>"]; bos = _CHAR_TO_ID["<bos>"]; eos = _CHAR_TO_ID["<eos>"]
    enc = [bos] + [_CHAR_TO_ID[c] for c in q if c in _CHAR_TO_ID] + [eos]
    enc = enc[:reasoner.config.max_seq_len]
    while len(enc) < reasoner.config.max_seq_len:
        enc.append(pad)
    enc_t = torch.tensor([enc], dtype=torch.long, device=reasoner.device)
    with torch.no_grad():
        mem = reasoner.model.encode(enc_t)
        dec = [bos]
        for _ in range(reasoner.config.max_dec_len - 1):
            padded = dec + [pad] * (reasoner.config.max_dec_len - len(dec))
            dt = torch.tensor([padded], dtype=torch.long, device=reasoner.device)
            logits = reasoner.model.decode_step(dt, mem)
            nid = int(logits[0, len(dec) - 1, :].argmax().item())
            if nid == eos:
                break
            dec.append(nid)
    out = ""
    for tid in dec[1:]:
        if tid in (pad, bos, eos):
            continue
        out += _ID_TO_CHAR.get(tid, "?")
    return out


def _verified(reasoner, q):
    emit = _decode(reasoner, q)
    expr = extract_problem_from_trace(emit)
    try:
        ans = interpret(parse_expression(expr))
    except (ParseError, InterpreterError):
        return None, emit
    if isinstance(ans, float) and ans == int(ans):
        ans = int(ans)
    return str(ans), emit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="calm/hrm/checkpoints/multi10_best.pt")
    args = p.parse_args()
    if not Path(args.ckpt).exists():
        print(f"ERROR: ckpt not found: {args.ckpt}", file=sys.stderr)
        sys.exit(1)
    reasoner = HRMSeq2SeqReasoner(args.ckpt)
    print(reasoner.info())
    print("\n# NEW held-out OOD — format variations of trained categories\n")

    per_category = {}
    for prompt, expected, category in _HELDOUT_OOD:
        got, emit = _verified(reasoner, prompt)
        ok = got == expected
        per_category.setdefault(category, [0, 0])
        per_category[category][0] += int(ok)
        per_category[category][1] += 1
        marker = "ok" if ok else "FAIL"
        pshort = prompt if len(prompt) < 55 else prompt[:52] + "..."
        print(f"  [{marker}] [{category:10}] {pshort:58} → {emit!r:22} got={got!r:6} (exp {expected})")

    print(f"\n# Per-category")
    print(f"  {'category':<12} {'rate':>10}")
    total_correct = 0
    total = 0
    for cat, (ok, n) in per_category.items():
        rate = ok / n
        total_correct += ok
        total += n
        print(f"  {cat:<12} {ok}/{n} = {rate:.0%}")
    print(f"\n  TOTAL       {total_correct}/{total} = {total_correct/total:.1%}")


if __name__ == "__main__":
    main()
