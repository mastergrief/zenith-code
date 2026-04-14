"""Held-out eval for the math HRM.

Generates N fresh problems (seed != training seed) and scores:
  - full-expression accuracy: HRM output string == str(safe_eval(expr))
  - smoke cases: fixed set of canonical expressions

Usage:
  PYTHONPATH=. python3 scripts/eval_hrm_math.py
  PYTHONPATH=. python3 scripts/eval_hrm_math.py --n 30 --seed 9999
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from calm.hrm.data import MathDataGenerator, _CHAR_TO_ID, _ID_TO_CHAR
from calm.hrm.inference import HRMSeq2SeqReasoner
from calm.expression import safe_eval
from calm.llm_computer.interpret import InterpreterError, interpret
from calm.llm_computer.parse import ParseError, extract_problem_from_trace, parse_expression
import torch


_SMOKE_CASES = [
    "17 * 23",
    "347 * 289",
    "gcd(48, 180)",
    "factorial(7)",
    "fibonacci(12)",
]


def _norm(s: str) -> str:
    return s.strip()


def _expected(expr: str) -> str:
    ans = safe_eval(expr)
    if isinstance(ans, float) and ans == int(ans):
        ans = int(ans)
    return str(ans)


def _hrm_raw_emit(reasoner: HRMSeq2SeqReasoner, expression: str) -> str:
    """Decode HRM's raw scratchpad string for an input (no answer extraction)."""
    pad_id = _CHAR_TO_ID["<pad>"]
    bos_id = _CHAR_TO_ID["<bos>"]
    eos_id = _CHAR_TO_ID["<eos>"]
    enc_ids = [bos_id] + [_CHAR_TO_ID[c] for c in expression if c in _CHAR_TO_ID] + [eos_id]
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
        if tid == _CHAR_TO_ID["<call>"]:
            out += "<call>"
        elif tid == _CHAR_TO_ID["<end_call>"]:
            out += "<end_call>"
        else:
            out += _ID_TO_CHAR.get(tid, "?")
    return out


def _verified_answer(reasoner: HRMSeq2SeqReasoner, expression: str) -> tuple[str, str, bool]:
    """HRM's emission → parser → interpreter pipeline.

    Returns `(answer_str, hrm_emission, used_hrm_emission)`.
    `used_hrm_emission` is False when HRM's emission failed to parse and we
    fell back to the input. Lets us measure HRM's actual contribution.

    The split:
      - HRM emits a scratchpad trace (its learned structure).
      - Parser extracts the pre-`=` segment as a problem expression.
      - Interpreter walks the parsed GateGraph, recomputing every value.
    """
    trace = _hrm_raw_emit(reasoner, expression)
    hrm_expr = extract_problem_from_trace(trace)
    used_hrm = True
    try:
        graph = parse_expression(hrm_expr)
    except ParseError:
        # HRM emitted unparseable garbage — fall back to the input.
        used_hrm = False
        graph = parse_expression(expression)
    try:
        ans = interpret(graph)
    except InterpreterError:
        used_hrm = False
        graph = parse_expression(expression)
        ans = interpret(graph)
    if isinstance(ans, float) and ans == int(ans):
        ans = int(ans)
    return str(ans), trace, used_hrm


def eval_held_out(ckpt: str, n: int, seed: int, verified: bool = False) -> None:
    gen = MathDataGenerator(seed=seed)
    probs = gen.generate(n)
    reasoner = HRMSeq2SeqReasoner(ckpt)
    print(reasoner.info())
    if verified:
        print("[mode] verified: HRM trace ignored, parse+interpret recomputes answers")

    correct = 0
    hrm_used = 0
    hrm_correct = 0
    by_diff: dict[int, list[int]] = {d: [0, 0] for d in range(1, 6)}

    for p in probs:
        if verified:
            got, hrm_emit, used_hrm = _verified_answer(reasoner, p.expression)
            hrm_used += int(used_hrm)
            hrm_marker = ""
            if used_hrm:
                # Did HRM's structure happen to match the input?
                if extract_problem_from_trace(hrm_emit).strip() == p.expression.strip():
                    hrm_correct += 1
                    hrm_marker = " [hrm-struct-ok]"
                else:
                    hrm_marker = f" [hrm-emitted={extract_problem_from_trace(hrm_emit)!r}]"
        else:
            got = reasoner.reason(p.expression)
            hrm_marker = ""
        expected = _norm(p.answer)
        ok = got is not None and _norm(got) == expected
        correct += int(ok)
        by_diff[p.difficulty][0] += int(ok)
        by_diff[p.difficulty][1] += 1
        marker = "ok" if ok else "FAIL"
        print(f"  [{marker}] d={p.difficulty} {p.expression} = {expected!r}  got={got!r}{hrm_marker}")

    total = len(probs)
    print(f"\nHeld-out ({seed=}, n={total}): {correct}/{total} = {correct/total:.1%}")
    for d, (c, t) in by_diff.items():
        if t:
            print(f"  difficulty {d}: {c}/{t} = {c/t:.1%}")
    if verified:
        print(f"  HRM emission used: {hrm_used}/{total} ({hrm_used/total:.0%}); "
              f"of those, structurally matched input: {hrm_correct}/{hrm_used if hrm_used else 1} "
              f"({hrm_correct/hrm_used:.0%} if hrm_used else 0%)")

    print("\nSmoke cases:")
    smoke_correct = 0
    for expr in _SMOKE_CASES:
        if verified:
            got, hrm_emit, used_hrm = _verified_answer(reasoner, expr)
            extra = f" [hrm={extract_problem_from_trace(hrm_emit)!r}]"
        else:
            got = reasoner.reason(expr)
            extra = ""
        expected = _expected(expr)
        ok = got is not None and _norm(got) == expected
        smoke_correct += int(ok)
        marker = "ok" if ok else "FAIL"
        print(f"  [{marker}] {expr} = {expected!r}  got={got!r}{extra}")
    print(f"Smoke: {smoke_correct}/{len(_SMOKE_CASES)}")

    full_expr_pct = correct / total
    smoke_pct = smoke_correct / len(_SMOKE_CASES)
    print(f"\nGates: full-expression {full_expr_pct:.1%} (≥ 90%?), smoke {smoke_pct:.0%} (5/5?)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="calm/hrm/checkpoints/math_seq2seq_best.pt")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=9999)
    parser.add_argument("--verified", action="store_true",
                        help="Use HRM for reasoning but interpret the input via "
                             "LLM-Computer for analytically-correct values.")
    args = parser.parse_args()

    if not Path(args.ckpt).exists():
        print(f"ERROR: checkpoint not found: {args.ckpt}", file=sys.stderr)
        sys.exit(1)

    eval_held_out(args.ckpt, args.n, args.seed, verified=args.verified)


if __name__ == "__main__":
    main()
