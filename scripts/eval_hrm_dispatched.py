"""End-to-end eval for the router + dispatcher pipeline.

For each domain's held-out pool, the dispatcher must:
  1. Classify the query to the correct specialist (router accuracy).
  2. Get a correct verified answer via that specialist (end-to-end accuracy).

Compared to each specialist's standalone eval (multi-task), the per-
domain accuracy should be within ~2pp. Router misclassification makes
the gap wider; stable classification closes it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch

from calm.hrm.data import MathDataGenerator
from calm.hrm.dispatcher import DEFAULT_SPECIALIST_CKPTS, Dispatcher
from calm.hrm.gsm_data import GSMDataGenerator
from calm.hrm.meta_data import MetaGenerator, TEST_FORMATS
from calm.hrm.nl_data import NLMathDataGenerator
from calm.hrm.router_data import _make_meta_text
from calm.hrm.word_data import WordProblemGenerator
from calm.llm_computer.interpret import InterpreterError, interpret
from calm.llm_computer.parse import ParseError, parse_expression


def _expected_value(expr: str):
    try:
        v = interpret(parse_expression(expr))
        if isinstance(v, float) and v == int(v):
            v = int(v)
        return str(v)
    except (ParseError, InterpreterError):
        return None


def _eval_domain(d: Dispatcher, domain: str, samples: List[Tuple[str, str]],
                 verbose: bool):
    """samples: list of (query_text_with_sep, expected_expression)."""
    routing_correct = 0
    verified_correct = 0
    for text, expected_expr in samples:
        expected = _expected_value(expected_expr)
        if expected is None:
            continue
        result = d.run(text)
        router_ok = (result.label == domain)
        verified_ok = (result.answer == expected)
        routing_correct += int(router_ok)
        verified_correct += int(verified_ok)
        if verbose and not verified_ok:
            tshort = text if len(text) < 50 else text[:47] + "..."
            print(f"  [FAIL] [{domain}] route={result.label} emit={result.emit!r:30}"
                  f" answer={result.answer} (expected {expected}): {tshort}")
    return routing_correct, verified_correct, len(samples)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=60, help="samples per domain")
    p.add_argument("--seed", type=int, default=9999)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    # Missing meta checkpoint is OK — just skip that branch.
    meta_available = Path(DEFAULT_SPECIALIST_CKPTS["meta"]).exists()

    d = Dispatcher()

    math_samples = [(p.expression, p.expression)
                    for p in MathDataGenerator(seed=args.seed).generate(args.n)]
    nl_samples = [(p.question, p.expression)
                  for p in NLMathDataGenerator(seed=args.seed).generate(args.n)]
    word_samples = [(p.problem, p.expression)
                    for p in WordProblemGenerator(seed=args.seed).generate(args.n)]
    gsm_samples = [(p.problem, p.expression)
                   for p in GSMDataGenerator(seed=args.seed).generate(args.n)]

    print(f"\n# Dispatcher end-to-end eval ({args.n} per domain)\n")
    print(f"  {'domain':<8} {'route':>10} {'verified':>14}")
    total_r, total_v, total_n = 0, 0, 0
    for name, samples in (("math", math_samples),
                          ("nl", nl_samples),
                          ("word", word_samples),
                          ("gsm", gsm_samples)):
        r, v, n = _eval_domain(d, name, samples, args.verbose)
        total_r += r; total_v += v; total_n += n
        print(f"  {name:<8} {r}/{n} = {r/n:>5.0%}   {v}/{n} = {v/n:>5.0%}")

    if meta_available:
        meta_gen = MetaGenerator(seed=args.seed, formats=TEST_FORMATS)
        samples = meta_gen.generate(args.n)
        formatted = [(_make_meta_text(s), s.query_expr) for s in samples]
        r, v, n = _eval_domain(d, "meta", formatted, args.verbose)
        total_r += r; total_v += v; total_n += n
        print(f"  {'meta':<8} {r}/{n} = {r/n:>5.0%}   {v}/{n} = {v/n:>5.0%}")
    else:
        print("  meta     (skipped: meta_best.pt not on disk)")

    print(f"\n  {'TOTAL':<8} {total_r}/{total_n} = {total_r/total_n:>5.0%}   "
          f"{total_v}/{total_n} = {total_v/total_n:>5.0%}")


if __name__ == "__main__":
    main()
