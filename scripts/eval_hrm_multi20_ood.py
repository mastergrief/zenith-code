"""NEW held-out OOD for the multi20 HRM.

Test formats designed to NOT match any of the 20 trained formats:
rhyming/conversational, math-as-prose, comparative, implicit-op,
double-negated, percent-of, and genuinely weird phrasings. If the
scaling curve continues, we see another +20-30pp lift. If it plateaus,
format-invariance has a distribution-independent ceiling and meta-
learning is the next lever.
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
    # Category 1: conversational / second-person
    ("how many is five plus seven",                               "12",  "conv"),  # words-as-digits (may fail — alphanum)
    ("hey what do you get when you add 17 and 23",                "40",  "conv"),
    ("could you tell me what 50 minus 15 is",                     "35",  "conv"),

    # Category 2: comparative / implicit
    ("which is larger, 7 times 8 or 50",                          "56",  "compare"),  # harder — requires both computation+compare
    ("by how much does 100 exceed 37",                            "63",  "compare"),
    ("how much more is 80 than 45",                               "35",  "compare"),

    # Category 3: math-as-prose
    ("starting from 50 and adding 30 gives us what",              "80",  "prose"),
    ("beginning with 100 and removing 25 leaves",                 "75",  "prose"),
    ("taking 7 groups of 9 amounts to",                           "63",  "prose"),

    # Category 4: implicit-op (no explicit operator word)
    ("50 and 30 combined",                                        "80",  "implicit"),
    ("100 reduced by 25",                                         "75",  "implicit"),
    ("7 repeated 9 times",                                        "63",  "implicit"),

    # Category 5: percent / ratio framing
    ("what is 50 percent of 80",                                  "40",  "percent"),  # may fail — needs / then *
    ("double of 25",                                              "50",  "percent"),  # *2 trick
    ("half of 80",                                                "40",  "percent"),  # /2 trick

    # Category 6: question embedded in context
    ("if a bakery sells 12 loaves every hour for 5 hours, total loaves sold is",  "60",  "embed"),
    ("a box weighing 3 kilograms contains 4 items, combined weight is 3 * 4, which is",  "12",  "embed"),
    ("given 25 and 75 as two values, their sum is",               "100", "embed"),
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
    p.add_argument("--ckpt", default="calm/hrm/checkpoints/multi20_best.pt")
    args = p.parse_args()
    if not Path(args.ckpt).exists():
        print(f"ERROR: ckpt not found: {args.ckpt}", file=sys.stderr)
        sys.exit(1)
    reasoner = HRMSeq2SeqReasoner(args.ckpt)
    print(reasoner.info())
    print("\n# NEW held-out OOD — formats not in any of 20 trained categories\n")

    per_category = {}
    for prompt, expected, category in _HELDOUT_OOD:
        got, emit = _verified(reasoner, prompt)
        ok = got == expected
        per_category.setdefault(category, [0, 0])
        per_category[category][0] += int(ok)
        per_category[category][1] += 1
        marker = "ok" if ok else "FAIL"
        pshort = prompt if len(prompt) < 55 else prompt[:52] + "..."
        print(f"  [{marker}] [{category:9}] {pshort:58} → {emit!r:22} got={got!r:6} (exp {expected})")

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
