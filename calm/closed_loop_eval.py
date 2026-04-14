"""Closed-loop effectiveness harness.

Quantifies: as AutoLearner accumulates corrections, does the precompute
hit rate on held-out prompts climb?

The loop:
  Round 1: feed K prompts through the "error simulator" (each produces a
           wrong claim). AutoLearner records patterns.
  Round 2: feed K DIFFERENT prompts (held-out). Measure how many match
           stored patterns and produce the correct precomputed value.
  Round 3: expose more error types. Re-measure.

If the loop is working, the hit rate on held-out prompts should climb
as the pattern DB grows. If not, we have learning but no generalization.

Usage:
  PYTHONPATH=. python3 -m calm.closed_loop_eval

This does NOT run Gemma — it uses the AutoLearner API directly. The
"error simulator" is a deterministic mock: certain prompt shapes get
recorded as errors, which is what Auto-CALM's verifier would have
caught at real inference time.
"""

from __future__ import annotations

import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from calm.auto_learn import AutoLearner


@dataclass
class _Scenario:
    """A single (prompt, expression-as-would-be-emitted-wrong) pair.
    In reality Gemma would emit a wrong claim like 'X = Y'; here we
    represent the claim by the expression that needs learning.
    """
    prompt: str
    error_expression: str   # what the verifier would have caught
    expected_value: object  # what a correct precompute produces


def _arithmetic_scenarios(n: int, rng: random.Random) -> List[_Scenario]:
    scenarios = []
    ops = ["+", "-", "*"]
    for _ in range(n):
        a = rng.randint(10, 999)
        b = rng.randint(10, 999)
        op = rng.choice(ops)
        expr = f"{a} {op} {b}"
        # "what is X * Y" is a typical Gemma failure mode on 3-digit.
        scenarios.append(_Scenario(
            prompt=f"what is {a} {op} {b}",
            error_expression=expr,
            expected_value=eval(expr, {"__builtins__": {}}),
        ))
    return scenarios


def _function_scenarios(n: int, rng: random.Random) -> List[_Scenario]:
    from calm.expression import safe_eval
    scenarios = []
    templates = [
        ("is_prime({n})",   lambda r: {"n": r.randint(2, 500)}),
        ("gcd({a}, {b})",   lambda r: {"a": r.randint(2, 500), "b": r.randint(2, 500)}),
        ("factorial({n})",  lambda r: {"n": r.randint(1, 8)}),
        ("fibonacci({n})",  lambda r: {"n": r.randint(1, 15)}),
    ]
    for _ in range(n):
        tmpl, values = rng.choice(templates)
        vals = values(rng)
        expr = tmpl.format(**vals)
        nice_args = ", ".join(str(v) for v in vals.values())
        fn = tmpl.split("(")[0]
        scenarios.append(_Scenario(
            prompt=f"compute {fn}({nice_args})",
            error_expression=expr,
            expected_value=safe_eval(expr),
        ))
    return scenarios


def _measure_hit_rate(learner: AutoLearner, scenarios: List[_Scenario]) -> Tuple[int, int]:
    """Return (hits, total). A hit = a precompute fired AND value matched."""
    hits = 0
    for s in scenarios:
        precomputes = learner.suggest_precomputes(s.prompt)
        # Hit if the exact expression made it into the precompute set AND
        # the value matches.
        for expr, val in precomputes.items():
            if _expressions_equivalent(expr, s.error_expression):
                if _value_matches(val, s.expected_value):
                    hits += 1
                    break
    return hits, len(scenarios)


def _expressions_equivalent(a: str, b: str) -> bool:
    """Normalise whitespace/ordering and compare."""
    def norm(e: str) -> str:
        return "".join(e.split())
    return norm(a) == norm(b)


def _value_matches(got, exp) -> bool:
    try:
        if isinstance(got, bool) or isinstance(exp, bool):
            return bool(got) == bool(exp)
        return float(got) == float(exp)
    except (ValueError, TypeError):
        return str(got) == str(exp)


@dataclass
class _RoundResult:
    round_idx: int
    patterns_in_db: int
    seeded_errors: int    # how many errors fed in this round
    holdout_hits: int
    holdout_total: int

    @property
    def holdout_rate(self) -> float:
        return self.holdout_hits / max(1, self.holdout_total)


def run_closed_loop(
    rounds: int = 3,
    errors_per_round: int = 20,
    holdout_per_round: int = 30,
    seed: int = 42,
) -> List[_RoundResult]:
    """Exercise the loop for `rounds` rounds.

    Each round:
      1. Generate `errors_per_round` error scenarios (half arithmetic,
         half function).
      2. Feed them as corrections to the learner.
      3. Generate `holdout_per_round` fresh held-out scenarios.
      4. Measure precompute hit rate.
    """
    results: List[_RoundResult] = []
    rng = random.Random(seed)

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "loop_eval.jsonl"
        learner = AutoLearner(db_path=db_path)

        for r in range(rounds):
            # Seed errors — mix of arithmetic and function corrections.
            arith_errors = _arithmetic_scenarios(errors_per_round // 2, rng)
            func_errors = _function_scenarios(errors_per_round - errors_per_round // 2, rng)
            all_errors = arith_errors + func_errors
            for s in all_errors:
                fake_claim = _FakeClaim(expression=s.error_expression)
                learner.learn_from_correction(fake_claim)

            # Measure on a fresh held-out mix.
            arith_holdout = _arithmetic_scenarios(holdout_per_round // 2, rng)
            func_holdout = _function_scenarios(holdout_per_round - holdout_per_round // 2, rng)
            holdout = arith_holdout + func_holdout
            hits, total = _measure_hit_rate(learner, holdout)

            results.append(_RoundResult(
                round_idx=r + 1,
                patterns_in_db=len(learner._patterns),
                seeded_errors=len(all_errors),
                holdout_hits=hits,
                holdout_total=total,
            ))

    return results


@dataclass
class _FakeClaim:
    expression: str
    actual_value: Optional[object] = 0
    correct: bool = False


def main():
    print("# Closed-loop effectiveness eval")
    print("# Fresh DB, 3 rounds of 20 corrections each, 30 held-out measures.\n")
    results = run_closed_loop(rounds=3, errors_per_round=20, holdout_per_round=30)

    print(f"{'round':>6} {'patterns':>10} {'seeded':>8} {'hits':>6} {'total':>6} {'rate':>8}")
    print("-" * 50)
    for r in results:
        print(f"{r.round_idx:>6} {r.patterns_in_db:>10} {r.seeded_errors:>8} "
              f"{r.holdout_hits:>6} {r.holdout_total:>6} {r.holdout_rate:>7.1%}")

    if len(results) >= 2:
        first, last = results[0].holdout_rate, results[-1].holdout_rate
        delta = last - first
        print(f"\nLoop effectiveness: round 1 = {first:.1%}, round {len(results)} = {last:.1%}, "
              f"delta = {delta:+.1%}")
        if delta > 0:
            print("Loop is CLOSING — hit rate improves with more corrections.")
        elif delta == 0:
            print("Loop at saturation — first-round patterns already cover subsequent errors.")
        else:
            print("WARNING: hit rate regressed — pattern interference or bug.")


if __name__ == "__main__":
    main()
