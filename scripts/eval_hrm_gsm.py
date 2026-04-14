"""GSM HRM eval — verified mode."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from calm.hrm.data import _CHAR_TO_ID, _ID_TO_CHAR
from calm.hrm.gsm_data import GSMDataGenerator
from calm.hrm.inference import HRMSeq2SeqReasoner
from calm.llm_computer.interpret import InterpreterError, interpret
from calm.llm_computer.parse import ParseError, extract_problem_from_trace, parse_expression


_SMOKE = [
    ("jane had 50 dollars. she spent 12 on lunch. later she found 30 more. how much does she have now", "68"),
    ("tom had 20 apples. after he gave 5 to lisa, he bought 15 more. how many apples does he have now", "30"),
    ("sam earns 8 dollars per day. he worked 10 days. he already had 25. how much does he have total", "105"),
    ("alice has 5 boxes of cookies. each box holds 12 cookies. she sold 8. how many cookies does she have left", "52"),
    ("a shop has 3 pencils at 5 dollars each and 4 pencils at 8 dollars each. total revenue", "47"),
]


def _decode(reasoner, q):
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
        graph = parse_expression(expr)
        ans = interpret(graph)
    except (ParseError, InterpreterError):
        return None, emit, False
    if isinstance(ans, float) and ans == int(ans):
        ans = int(ans)
    return str(ans), emit, True


def evalit(ckpt, n, seed):
    r = HRMSeq2SeqReasoner(ckpt)
    print(r.info())

    print("\nHeld-out GSM problems:")
    gen = GSMDataGenerator(seed=seed)
    probs = gen.generate(n)
    correct, struct = 0, 0
    for p in probs:
        got, emit, ok_parse = _verified(r, p.problem)
        ok = got is not None and str(got) == p.answer
        correct += int(ok)
        if ok_parse and emit.rstrip("=").strip() == p.expression.strip():
            struct += 1
        marker = "ok" if ok else "FAIL"
        short = p.problem[:60] + "..." if len(p.problem) > 60 else p.problem
        print(f"  [{marker}] {short:63} → {emit!r:25} got={got!r} (exp {p.answer})")
    print(f"\nHeld-out (seed={seed}, n={n}): {correct}/{n} = {correct/n:.1%}")
    print(f"  structural match: {struct}/{n} ({struct/n:.0%})")

    print("\nSmoke:")
    sc = 0
    for q, exp in _SMOKE:
        got, emit, _ = _verified(r, q)
        ok = got is not None and str(got) == exp
        sc += int(ok)
        marker = "ok" if ok else "FAIL"
        short = q[:60] + "..." if len(q) > 60 else q
        print(f"  [{marker}] {short:63} → {emit!r:25} got={got!r} (exp {exp})")
    print(f"Smoke: {sc}/{len(_SMOKE)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="calm/hrm/checkpoints/gsm_best.pt")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=9999)
    args = p.parse_args()
    if not Path(args.ckpt).exists():
        print(f"ERROR: ckpt not found: {args.ckpt}", file=sys.stderr)
        sys.exit(1)
    evalit(args.ckpt, args.n, args.seed)


if __name__ == "__main__":
    main()
