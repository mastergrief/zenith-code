"""Per-domain eval for the multi-task HRM.

Loads one checkpoint, runs each domain's held-out set through it,
reports full-expression accuracy per domain. The question: does the
single 48K model match the per-domain HRMs (100%, 97%, 100%, 93%)?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from calm.expression import safe_eval
from calm.hrm.data import _CHAR_TO_ID, _ID_TO_CHAR, MathDataGenerator
from calm.hrm.gsm_data import GSMDataGenerator
from calm.hrm.inference import HRMSeq2SeqReasoner
from calm.hrm.nl_data import NLMathDataGenerator
from calm.hrm.word_data import WordProblemGenerator
from calm.llm_computer.interpret import InterpreterError, interpret
from calm.llm_computer.parse import ParseError, extract_problem_from_trace, parse_expression


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


def _verified(reasoner, query):
    emit = _decode(reasoner, query)
    expr = extract_problem_from_trace(emit)
    try:
        ans = interpret(parse_expression(expr))
    except (ParseError, InterpreterError):
        return None, emit
    if isinstance(ans, float) and ans == int(ans):
        ans = int(ans)
    return str(ans), emit


def _eval_domain(label: str, reasoner, queries, answers):
    print(f"\n# {label} ({len(queries)} cases)")
    correct = 0
    struct = 0
    for q, exp, struct_exp in queries:
        got, emit = _verified(reasoner, q)
        if got == exp:
            correct += 1
        if emit.rstrip("=").strip() == struct_exp:
            struct += 1
    print(f"  full-expression: {correct}/{len(queries)} = {correct/len(queries):.1%}")
    print(f"  structural:      {struct}/{len(queries)} = {struct/len(queries):.0%}")
    return correct / len(queries), struct / len(queries)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="calm/hrm/checkpoints/multi_task_best.pt")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=9999)
    args = p.parse_args()
    if not Path(args.ckpt).exists():
        print(f"ERROR: ckpt not found: {args.ckpt}", file=sys.stderr)
        sys.exit(1)

    reasoner = HRMSeq2SeqReasoner(args.ckpt)
    print(reasoner.info())

    # Math domain: encoder input = expression, expected = expression.
    math_gen = MathDataGenerator(seed=args.seed)
    math_probs = math_gen.generate(args.n)
    math_queries = [(p.expression, str(p.answer), p.expression) for p in math_probs]
    # Replace answer with computed-from-safe-eval form for consistency.
    for i, p in enumerate(math_probs):
        ans = safe_eval(p.expression)
        if isinstance(ans, float) and ans == int(ans):
            ans = int(ans)
        math_queries[i] = (p.expression, str(ans), p.expression)

    # NL, word, GSM all have `expression` field already.
    nl_probs = NLMathDataGenerator(seed=args.seed).generate(args.n)
    nl_queries = [(p.question, p.answer, p.expression) for p in nl_probs]

    word_probs = WordProblemGenerator(seed=args.seed).generate(args.n)
    word_queries = [(p.problem, p.answer, p.expression) for p in word_probs]

    gsm_probs = GSMDataGenerator(seed=args.seed).generate(args.n)
    gsm_queries = [(p.problem, p.answer, p.expression) for p in gsm_probs]

    results = {}
    for label, queries in [
        ("math-echo", math_queries),
        ("nl-template", nl_queries),
        ("word-problem", word_queries),
        ("gsm-style", gsm_queries),
    ]:
        full, struct = _eval_domain(label, reasoner, queries, [])
        results[label] = (full, struct)

    print("\n# Summary vs per-domain baselines")
    print(f"  {'domain':<15} {'multi-task':>12} {'per-domain':>12}")
    baselines = {"math-echo": 1.00, "nl-template": 0.97, "word-problem": 1.00, "gsm-style": 0.93}
    for dom, (full, struct) in results.items():
        print(f"  {dom:<15} {full:>11.1%} {baselines[dom]:>11.0%}")


if __name__ == "__main__":
    main()
