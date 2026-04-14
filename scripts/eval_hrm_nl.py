"""NL→math HRM eval — integration #3 verified-mode measurement.

Pipeline: NL question → HRM encoder → HRM decoder emits math expression
+ `=` terminator → parse_expression → interpret → answer.

Scoring:
  - smoke: 5 canonical NL questions with known answers.
  - held-out: 30 fresh NL questions (seed != training), full-expression
    accuracy via the verified path.

Usage:
  PYTHONPATH=. python3 scripts/eval_hrm_nl.py --verified --n 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from calm.expression import safe_eval
from calm.hrm.data import _CHAR_TO_ID, _ID_TO_CHAR
from calm.hrm.inference import HRMSeq2SeqReasoner
from calm.hrm.nl_data import NLMathDataGenerator
from calm.llm_computer.interpret import InterpreterError, interpret
from calm.llm_computer.parse import ParseError, extract_problem_from_trace, parse_expression


_SMOKE_QUESTIONS = [
    ("what is 17 plus 23",          "40"),
    ("what is 347 times 289",       "100283"),
    ("gcd of 48 and 180",           "12"),
    ("factorial of 7",              "5040"),
    ("product of 12 and 11",        "132"),
]


def _hrm_decode(reasoner: HRMSeq2SeqReasoner, question: str) -> str:
    pad_id = _CHAR_TO_ID["<pad>"]
    bos_id = _CHAR_TO_ID["<bos>"]
    eos_id = _CHAR_TO_ID["<eos>"]
    enc_ids = [bos_id] + [_CHAR_TO_ID[c] for c in question if c in _CHAR_TO_ID] + [eos_id]
    enc_ids = enc_ids[: reasoner.config.max_seq_len]
    while len(enc_ids) < reasoner.config.max_seq_len:
        enc_ids.append(pad_id)
    enc_t = torch.tensor([enc_ids], dtype=torch.long, device=reasoner.device)
    with torch.no_grad():
        memory = reasoner.model.encode(enc_t)
        dec_ids = [bos_id]
        for _ in range(reasoner.config.max_dec_len - 1):
            padded = dec_ids + [pad_id] * (reasoner.config.max_dec_len - len(dec_ids))
            dec_t = torch.tensor([padded], dtype=torch.long, device=reasoner.device)
            logits = reasoner.model.decode_step(dec_t, memory)
            nid = int(logits[0, len(dec_ids) - 1, :].argmax().item())
            if nid == eos_id:
                break
            dec_ids.append(nid)
    out = ""
    for tid in dec_ids[1:]:
        if tid in (pad_id, bos_id, eos_id):
            continue
        out += _ID_TO_CHAR.get(tid, "?")
    return out


def _verified(reasoner, question):
    emit = _hrm_decode(reasoner, question)
    expr = extract_problem_from_trace(emit)
    try:
        graph = parse_expression(expr)
        ans = interpret(graph)
    except (ParseError, InterpreterError):
        return None, emit, False
    if isinstance(ans, float) and ans == int(ans):
        ans = int(ans)
    return str(ans), emit, True


def eval_nl(ckpt: str, n: int, seed: int) -> None:
    reasoner = HRMSeq2SeqReasoner(ckpt)
    print(reasoner.info())

    print("\nHeld-out NL questions:")
    gen = NLMathDataGenerator(seed=seed)
    problems = gen.generate(n)
    correct = 0
    struct_correct = 0
    for p in problems:
        got, emit, ok_parse = _verified(reasoner, p.question)
        expected = p.answer
        ok = got is not None and str(got) == expected
        correct += int(ok)
        if ok_parse and emit.rstrip("=").strip() == p.expression.strip():
            struct_correct += 1
        marker = "ok" if ok else "FAIL"
        emit_short = emit if len(emit) < 40 else emit[:37] + "..."
        print(f"  [{marker}] {p.question!r:50} → expr={emit_short!r} got={got!r} (expected {expected})")
    print(f"\nHeld-out (seed={seed}, n={n}): {correct}/{n} = {correct/n:.1%}")
    print(f"  structural match: {struct_correct}/{n} ({struct_correct/n:.0%})")

    print("\nSmoke cases:")
    smoke_correct = 0
    for q, exp in _SMOKE_QUESTIONS:
        got, emit, _ = _verified(reasoner, q)
        ok = got is not None and str(got) == exp
        smoke_correct += int(ok)
        marker = "ok" if ok else "FAIL"
        print(f"  [{marker}] {q!r} → expr={emit!r} got={got!r} (expected {exp})")
    print(f"Smoke: {smoke_correct}/{len(_SMOKE_QUESTIONS)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="calm/hrm/checkpoints/nl_math_structure_best.pt")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=9999)
    parser.add_argument("--verified", action="store_true",
                        help="Alias for the default behavior; kept for API parity.")
    args = parser.parse_args()
    if not Path(args.ckpt).exists():
        print(f"ERROR: checkpoint not found: {args.ckpt}", file=sys.stderr)
        sys.exit(1)
    eval_nl(args.ckpt, args.n, args.seed)


if __name__ == "__main__":
    main()
