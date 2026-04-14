"""Out-of-distribution test for the multi-task HRM.

Vector 2 Level-4 generalization test: domains the model has NEVER seen,
but sharing the target language (math expression) and the underlying
skill (extract operands + operator, emit as structured expression).

Hand-crafted prompts in 6 novel formats:
  1. Code-style variable binding ("x = ...; y = ...; result = x op y")
  2. Reversed prefix notation ("add 8 and 14", "subtract 5 from 20")
  3. Distractor-heavy narratives (story + irrelevant info + question)
  4. Unit-bearing inputs ("3 meters plus 7 meters equals")
  5. Let-bound phrasing ("if a=5 and b=12, what is a+b")
  6. Simple equation completion ("50 + 30 = ?")

Each prompt targets a math expression. The HRM has to ignore the
surface variation and emit the correct operands + operator. The
interpreter does the arithmetic.

Usage:
  PYTHONPATH=. python3 scripts/eval_hrm_ood.py
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


# 18 out-of-distribution prompts across 6 novel formats, 3 per format.
# Each entry: (prompt, expected_answer_string, category).
_OOD_CASES = [
    # Category 1: code-style variable binding (NEW — never in training)
    ("x = 17; y = 23; result = x * y",               "391", "code-var"),
    ("a = 50; b = 30; sum = a + b",                  "80",  "code-var"),
    ("a = 100; b = 25; diff = a - b",                "75",  "code-var"),

    # Category 2: reversed/prefix notation
    ("add 8 and 14",                                 "22",  "prefix"),
    ("subtract 12 from 50",                          "38",  "prefix"),
    ("multiply 7 and 9",                             "63",  "prefix"),

    # Category 3: distractor-heavy narratives
    ("on saturday tom went to the market. he bought 5 apples and then 3 more. how many apples",
                                                     "8",   "distractor"),
    ("lisa drove for 2 hours at 60 mph. what distance did she cover",
                                                     "120", "distractor"),
    ("a library has 100 books. yesterday 15 were borrowed. how many remain",
                                                     "85",  "distractor"),

    # Category 4: unit-bearing inputs
    ("3 meters plus 7 meters equals",                "10",  "units"),
    ("12 kilograms minus 5 kilograms equals",        "7",   "units"),
    ("5 liters times 4 equals",                      "20",  "units"),

    # Category 5: let-bound / variable algebraic
    ("if a=5 and b=12, what is a+b",                 "17",  "let"),
    ("if x equals 20 and y equals 8, what is x - y", "12",  "let"),
    ("given p=6 and q=7, compute p * q",             "42",  "let"),

    # Category 6: equation completion
    ("50 + 30 = ?",                                  "80",  "eq-complete"),
    ("144 / 12 = ?",                                 "12",  "eq-complete"),
    ("7 * 9 = ?",                                    "63",  "eq-complete"),
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
    p.add_argument("--ckpt", default="calm/hrm/checkpoints/multi_task_best.pt")
    args = p.parse_args()
    if not Path(args.ckpt).exists():
        print(f"ERROR: ckpt not found: {args.ckpt}", file=sys.stderr)
        sys.exit(1)
    reasoner = HRMSeq2SeqReasoner(args.ckpt)
    print(reasoner.info())
    print("\n# OOD cases — none of these formats appear in any training domain\n")

    per_category = {}
    for prompt, expected, category in _OOD_CASES:
        got, emit = _verified(reasoner, prompt)
        ok = got == expected
        per_category.setdefault(category, [0, 0])
        per_category[category][0] += int(ok)
        per_category[category][1] += 1
        marker = "ok" if ok else "FAIL"
        pshort = prompt if len(prompt) < 55 else prompt[:52] + "..."
        print(f"  [{marker}] [{category:12}] {pshort:58} → {emit!r:22} got={got!r:6} (exp {expected})")

    print(f"\n# Per-category breakdown")
    print(f"  {'category':<15} {'rate':>10}")
    total_correct = 0
    total = 0
    for cat, (ok, n) in per_category.items():
        rate = ok / n
        total_correct += ok
        total += n
        print(f"  {cat:<15} {ok}/{n} = {rate:.0%}")
    print(f"\n  TOTAL          {total_correct}/{total} = {total_correct/total:.1%}")


if __name__ == "__main__":
    main()
