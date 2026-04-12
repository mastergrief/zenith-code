"""
CALM v0.1 benchmark — automated evaluation across problem categories.

Runs 50 problems through the reasoning engine and scores:
- Correctness: does the VM output / final stack match expected?
- Iterations: how many engine loops needed?
- Errors: any CALM block failures?

Categories:
1. Arithmetic (basic + compound)
2. Number theory (primes, GCD, factors)
3. Sequences (Fibonacci, Collatz)
4. Algebra (quadratic, expressions)
5. Reasoning chains (multi-step, conditional)

Usage:
    python3 -m calm.benchmark              # run all
    python3 -m calm.benchmark --category 2 # number theory only
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from calm.engine import CalmEngine


@dataclass
class Problem:
    id: int
    category: str
    prompt: str
    expected: Any           # expected answer (checked in final response text)
    keywords: List[str]     # keywords that must appear in the response


@dataclass
class Result:
    problem: Problem
    passed: bool = False
    response: str = ""
    calm_blocks: int = 0
    iterations: int = 0
    errors: int = 0
    time_s: float = 0.0
    reason: str = ""        # why it failed


PROBLEMS = [
    # --- Category 1: Arithmetic ---
    Problem(1, "arithmetic", "What is 17 * 23?", None, ["391"]),
    Problem(2, "arithmetic", "What is 42 * 19?", None, ["798"]),
    Problem(3, "arithmetic", "What is (17 * 23) + (42 * 19) - 100?", None, ["1089"]),
    Problem(4, "arithmetic", "What is 2^10?", None, ["1024"]),
    Problem(5, "arithmetic", "What is 144 / 12?", None, ["12"]),
    Problem(6, "arithmetic", "What is 999 + 1?", None, ["1000"]),
    Problem(7, "arithmetic", "What is 100 - 37?", None, ["63"]),
    Problem(8, "arithmetic", "What is 7 * 8 + 3?", None, ["59"]),
    Problem(9, "arithmetic", "What is factorial(10)?", None, ["3628800"]),
    Problem(10, "arithmetic", "What is the sum of all integers from 1 to 100?", None, ["5050"]),

    # --- Category 2: Number Theory ---
    Problem(11, "number_theory", "Is 391 prime?", None, ["not prime|not a prime|composite|False|no"]),
    Problem(12, "number_theory", "What are the prime factors of 391?", None, ["17", "23"]),
    Problem(13, "number_theory", "What is the GCD of 391 and 782?", None, ["391"]),
    Problem(14, "number_theory", "What is the smallest prime greater than 1000?", None, ["1009"]),
    Problem(15, "number_theory", "Is 1000003 prime?", None, ["prime|True|yes"]),
    Problem(16, "number_theory", "What is the 25th prime number?", None, ["97"]),
    Problem(17, "number_theory", "How many divisors does 28 have?", None, ["6"]),
    Problem(18, "number_theory", "Is 28 a perfect number?", None, ["yes|True|perfect"]),
    Problem(19, "number_theory", "What is the LCM of 12 and 8?", None, ["24"]),
    Problem(20, "number_theory", "What is the digit sum of 12345?", None, ["15"]),

    # --- Category 3: Sequences ---
    Problem(21, "sequences", "What is the 10th Fibonacci number?", None, ["55"]),
    Problem(22, "sequences", "What is the 20th Fibonacci number?", None, ["6765"]),
    Problem(23, "sequences", "How long is the Collatz sequence starting from 27?", None, ["112"]),
    Problem(24, "sequences", "What is fibonacci(30)?", None, ["832040"]),
    Problem(25, "sequences", "What is the digital root of 12345?", None, ["6"]),

    # --- Category 4: Algebra ---
    Problem(26, "algebra", "Solve x^2 - 5x + 6 = 0", None, ["2", "3"]),
    Problem(27, "algebra", "Solve x^2 - 1 = 0", None, ["1", "-1"]),
    Problem(28, "algebra", "What is sqrt(1764)?", None, ["42"]),
    Problem(29, "algebra", "What is 2^20?", None, ["1048576"]),
    Problem(30, "algebra", "What is log2(1024)?", None, ["10"]),

    # --- Category 5: Reasoning Chains ---
    Problem(31, "reasoning", "What is 17 * 23? Is the result prime?",
            None, ["391", "not prime|not a prime|composite|no"]),
    Problem(32, "reasoning", "Find the GCD of 391 and 782. Is it prime?",
            None, ["391", "not prime|not a prime|composite|no"]),
    Problem(33, "reasoning",
            "What is the smallest prime > 1000? What is its digit sum?",
            None, ["1009", "10"]),
    Problem(34, "reasoning",
            "Solve x^2 - 5x + 6 = 0. Verify by computing x1 * x2.",
            None, ["2", "3", "6"]),
    Problem(35, "reasoning",
            "What is fibonacci(15)? Is it prime? How many divisors does it have?",
            None, ["610", "not prime|not a prime|composite|no|False"]),

    # --- Category 6: Multi-step with branching ---
    Problem(36, "multi_step",
            "Find all primes between 1 and 20. Which ones are twin primes?",
            None, ["twin"]),
    Problem(37, "multi_step",
            "Compute 2^8, check if it's prime, and find its number of divisors.",
            None, ["256", "not prime|not a prime|composite|no", "9"]),
    Problem(38, "multi_step",
            "What is 17 * 23 + 42 * 19 - 100? Is the result divisible by 3?",
            None, ["1089", "divisible|yes|True"]),
    Problem(39, "multi_step",
            "Find the next prime after 100 and the next prime after 200. What is their GCD?",
            None, ["101", "211", "1"]),
    Problem(40, "multi_step",
            "What is factorial(7)? What is the digit sum of the result?",
            None, ["5040", "9"]),
]


def _check_keywords(response: str, keywords: List[str]) -> tuple:
    """
    Check if all keywords appear in the response. Each keyword entry
    can contain alternatives separated by |.
    E.g. "not prime|not a prime|composite" means any of those suffice.
    Returns (passed, missing).
    """
    response_lower = response.lower()
    missing = []
    for kw in keywords:
        alternatives = [a.strip().lower() for a in kw.split("|")]
        if not any(alt in response_lower for alt in alternatives):
            missing.append(kw)
    return len(missing) == 0, missing


def run_benchmark(
    categories: Optional[List[str]] = None,
    verbose: bool = False,
    thinking_budget: int = 8192,
) -> List[Result]:
    """Run the benchmark. Returns a list of Results."""
    engine = CalmEngine(thinking_budget=thinking_budget)
    results: List[Result] = []

    problems = PROBLEMS
    if categories:
        problems = [p for p in problems if p.category in categories]

    print(f"Running {len(problems)} problems...\n")

    for prob in problems:
        t0 = time.time()
        try:
            er = engine.run(prob.prompt, verbose=False)
            elapsed = time.time() - t0

            passed, missing = _check_keywords(er.response, prob.keywords)

            r = Result(
                problem=prob,
                passed=passed,
                response=er.response,
                calm_blocks=er.calm_blocks,
                iterations=er.iterations,
                errors=len(er.training_log),
                time_s=elapsed,
                reason=f"missing: {missing}" if missing else "",
            )
        except Exception as e:
            elapsed = time.time() - t0
            r = Result(
                problem=prob,
                passed=False,
                time_s=elapsed,
                reason=f"exception: {e}",
            )

        results.append(r)
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] #{prob.id:2d} {prob.category:15s} "
              f"{prob.prompt[:50]:50s} "
              f"blocks={r.calm_blocks} iters={r.iterations} "
              f"{r.time_s:.1f}s"
              + (f"  ({r.reason})" if r.reason else ""))

    # Summary.
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS: {passed}/{total} passed ({100*passed/total:.0f}%)")

    # Per-category breakdown.
    cats = sorted(set(r.problem.category for r in results))
    for cat in cats:
        cat_results = [r for r in results if r.problem.category == cat]
        cat_passed = sum(1 for r in cat_results if r.passed)
        avg_iters = sum(r.iterations for r in cat_results) / len(cat_results)
        avg_blocks = sum(r.calm_blocks for r in cat_results) / len(cat_results)
        avg_time = sum(r.time_s for r in cat_results) / len(cat_results)
        print(f"  {cat:15s}: {cat_passed}/{len(cat_results)} "
              f"(avg {avg_iters:.1f} iters, {avg_blocks:.1f} blocks, {avg_time:.1f}s)")

    total_time = sum(r.time_s for r in results)
    print(f"\nTotal time: {total_time:.0f}s")

    return results


if __name__ == "__main__":
    import sys
    cats = None
    if "--category" in sys.argv:
        idx = sys.argv.index("--category")
        cat_map = {
            "1": "arithmetic", "2": "number_theory", "3": "sequences",
            "4": "algebra", "5": "reasoning", "6": "multi_step",
        }
        cats = [cat_map.get(sys.argv[idx + 1], sys.argv[idx + 1])]
    results = run_benchmark(categories=cats)
