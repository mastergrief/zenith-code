"""Generate multi-step code-reasoning data for R53.

Produces a JSONL corpus where each assistant turn walks through the
full code-reasoning chain:

    <think>
    STEP 1 DECOMPOSE — what is the function signature? inputs? output?
    STEP 2 PLAN — what's the algorithm? complexity? edge cases?
    STEP 3 IMPLEMENT — write the code
    STEP 4 VERIFY — mentally test on cases (incl. edges)
    STEP 5 ANSWER — final code block
    </think>

    ```python
    def ...
    ```

    Verified test cases:
    - f(inputs) -> expected  ✓

Each example is generated programmatically from a template + a
parameter sweep, then verified via CALM's sandbox/ast backends so only
passing solutions end up in the corpus. This is the corpus the code PT
will train on (R53.5) and the DB KnowledgeStore will index at install
time (R53.6).

Usage:
    PYTHONPATH=. python3 scripts/generate_multi_step_code_data.py \
        [--out agents/distill/data/multi_step_code.jsonl] [--count 500]
"""

from __future__ import annotations

import argparse
import json
import random
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple


# ----- template framework -----

@dataclass
class CodeProblem:
    """One generated (problem, solution) pair with verification data."""
    category: str
    problem: str
    signature: str
    solution: str                # function body, no fences
    test_cases: List[Tuple[Any, Any]]     # [(input, expected), ...]
    algorithm: str               # 1-line algorithm description
    complexity: str              # e.g. "O(√n)"
    edge_cases: List[str]        # what could go wrong
    # Sandbox can't run certain imports (urllib, socket, etc). Mark
    # such problems so they ship on AST-only verification. The solutions
    # are hand-written and correct; the flag just routes around
    # sandbox policy, not code review.
    skip_sandbox: bool = False


TemplateFn = Callable[[random.Random], Optional[CodeProblem]]


# ----- arithmetic / number-theory templates -----

def tpl_is_divisible(rng: random.Random) -> CodeProblem:
    k = rng.choice([2, 3, 5, 7, 11, 13])
    return CodeProblem(
        category="number_theory",
        problem=f"Write a Python function `is_divisible_by_{k}(n)` that returns True if n is divisible by {k}, False otherwise. Handle negative n.",
        signature=f"def is_divisible_by_{k}(n):",
        solution=f"def is_divisible_by_{k}(n):\n    return n % {k} == 0\n",
        test_cases=[(0, True), (k, True), (k*2, True), (1, False), (-k*3, True), (-k+1, False)],
        algorithm=f"integer modulo by {k}",
        complexity="O(1)",
        edge_cases=["zero", "negatives", "multiples past integer overflow (Python has none)"],
    )


def tpl_is_even_odd(rng: random.Random) -> CodeProblem:
    flavour = rng.choice(["even", "odd"])
    ret = "n % 2 == 0" if flavour == "even" else "n % 2 != 0"
    expected_zero = True if flavour == "even" else False
    return CodeProblem(
        category="number_theory",
        problem=f"Write a Python function `is_{flavour}(n)` that returns True if n is {flavour}, False otherwise.",
        signature=f"def is_{flavour}(n):",
        solution=f"def is_{flavour}(n):\n    return {ret}\n",
        test_cases=[(0, expected_zero), (1, not expected_zero), (2, expected_zero), (-3, not expected_zero)],
        algorithm=f"bitmask/modulo for parity",
        complexity="O(1)",
        edge_cases=["zero (is even)", "negative numbers"],
    )


def tpl_min_of(rng: random.Random) -> CodeProblem:
    n = rng.choice([2, 3, 4])
    args_names = [chr(ord("a") + i) for i in range(n)]
    args_sig = ", ".join(args_names)
    body_lines = [f"def min_of_{n}({args_sig}):"]
    body_lines.append(f"    m = {args_names[0]}")
    for a in args_names[1:]:
        body_lines.append(f"    if {a} < m: m = {a}")
    body_lines.append("    return m")
    solution = "\n".join(body_lines) + "\n"
    # Generate test cases deterministically based on n
    bases = [(1, 2, 3, 4), (3, 2, 1, 0), (-1, -5, -3, -2), (10, 20, 30, 40)]
    test_cases = []
    for base in bases:
        vals = tuple(base[:n])
        test_cases.append(vals + (min(vals),))
    return CodeProblem(
        category="number_theory",
        problem=f"Write a Python function `min_of_{n}({args_sig})` that returns the smallest of {n} values without using min().",
        signature=f"def min_of_{n}({args_sig}):",
        solution=solution,
        test_cases=test_cases,
        algorithm="pairwise comparison running-min",
        complexity="O(1) (fixed arity)",
        edge_cases=["all equal", "negatives", "mixed signs"],
    )


def tpl_sum_squares(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="number_theory",
        problem="Write a Python function `sum_squares(n)` that returns 1**2 + 2**2 + ... + n**2 for non-negative n. sum_squares(0) = 0.",
        signature="def sum_squares(n):",
        solution=(
            "def sum_squares(n):\n"
            "    total = 0\n"
            "    for i in range(1, n + 1):\n"
            "        total += i * i\n"
            "    return total\n"
        ),
        test_cases=[(0, 0), (1, 1), (2, 5), (3, 14), (5, 55), (10, 385)],
        algorithm="iterate 1..n, accumulate i*i",
        complexity="O(n)",
        edge_cases=["n = 0 returns 0", "closed form n(n+1)(2n+1)/6 also valid"],
    )


def tpl_digit_sum(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="number_theory",
        problem="Write a Python function `digit_sum(n)` that returns the sum of the decimal digits of a non-negative integer.",
        signature="def digit_sum(n):",
        solution=(
            "def digit_sum(n):\n"
            "    total = 0\n"
            "    while n:\n"
            "        total += n % 10\n"
            "        n //= 10\n"
            "    return total\n"
        ),
        test_cases=[(0, 0), (5, 5), (12, 3), (99, 18), (1000, 1), (12345, 15)],
        algorithm="repeated modulo-10 extract, then integer divide",
        complexity="O(log₁₀ n)",
        edge_cases=["n = 0 (returns 0)", "single digit", "trailing zeros"],
    )


def tpl_reverse_int(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="number_theory",
        problem="Write a Python function `reverse_int(n)` that returns n with its decimal digits reversed. Preserve sign. reverse_int(-123) = -321, reverse_int(120) = 21.",
        signature="def reverse_int(n):",
        solution=(
            "def reverse_int(n):\n"
            "    sign = -1 if n < 0 else 1\n"
            "    n = abs(n)\n"
            "    r = 0\n"
            "    while n:\n"
            "        r = r * 10 + n % 10\n"
            "        n //= 10\n"
            "    return sign * r\n"
        ),
        test_cases=[(0, 0), (1, 1), (12, 21), (123, 321), (-123, -321), (120, 21), (1000, 1)],
        algorithm="pull off last digit, shift accumulator, repeat; restore sign",
        complexity="O(log₁₀ n)",
        edge_cases=["zero", "single digit", "trailing zeros drop", "negative"],
    )


def tpl_count_set_bits(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="bitwise",
        problem="Write a Python function `popcount(n)` that returns the number of 1-bits in the binary representation of a non-negative integer. Don't use bin().count.",
        signature="def popcount(n):",
        solution=(
            "def popcount(n):\n"
            "    c = 0\n"
            "    while n:\n"
            "        c += n & 1\n"
            "        n >>= 1\n"
            "    return c\n"
        ),
        test_cases=[(0, 0), (1, 1), (2, 1), (3, 2), (255, 8), (1024, 1), (1023, 10)],
        algorithm="mask lowest bit, right-shift, repeat",
        complexity="O(bits)",
        edge_cases=["zero", "powers of two (1 bit)", "all-ones (popcount = bits)"],
    )


def tpl_min_max_tuple(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="collections",
        problem="Write a Python function `min_max(xs)` that returns the tuple (minimum, maximum) of a non-empty list in a single pass. Raise ValueError for empty input.",
        signature="def min_max(xs):",
        solution=(
            "def min_max(xs):\n"
            "    if not xs:\n"
            "        raise ValueError('empty sequence')\n"
            "    lo = hi = xs[0]\n"
            "    for x in xs[1:]:\n"
            "        if x < lo: lo = x\n"
            "        elif x > hi: hi = x\n"
            "    return (lo, hi)\n"
        ),
        test_cases=[([1], (1, 1)), ([1, 2, 3], (1, 3)), ([3, 2, 1], (1, 3)), ([-5, 10, 0, -3], (-5, 10))],
        algorithm="single-pass running lo/hi with pairwise comparison",
        complexity="O(n)",
        edge_cases=["single element (lo == hi)", "all same (lo == hi)", "empty raises"],
    )


def tpl_transpose(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="collections",
        problem="Write a Python function `transpose(matrix)` that returns the transpose of a 2D list (list of lists). All rows assumed equal length. transpose([]) returns [].",
        signature="def transpose(matrix):",
        solution=(
            "def transpose(matrix):\n"
            "    if not matrix:\n"
            "        return []\n"
            "    return [list(row) for row in zip(*matrix)]\n"
        ),
        test_cases=[
            ([], []),
            ([[1, 2, 3]], [[1], [2], [3]]),
            ([[1], [2], [3]], [[1, 2, 3]]),
            ([[1, 2], [3, 4]], [[1, 3], [2, 4]]),
            ([[1, 2, 3], [4, 5, 6]], [[1, 4], [2, 5], [3, 6]]),
        ],
        algorithm="zip(*matrix) creates column-wise tuples; wrap as lists",
        complexity="O(rows × cols)",
        edge_cases=["empty matrix", "single row", "single column", "rectangular"],
    )


def tpl_dict_invert(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="collections",
        problem="Write a Python function `invert_dict(d)` that returns a new dict with keys and values swapped. If duplicate values cause collisions, the last key wins.",
        signature="def invert_dict(d):",
        solution=(
            "def invert_dict(d):\n"
            "    return {v: k for k, v in d.items()}\n"
        ),
        test_cases=[
            ({}, {}),
            ({"a": 1}, {1: "a"}),
            ({"a": 1, "b": 2}, {1: "a", 2: "b"}),
            ({"a": 1, "b": 1}, {1: "b"}),  # collision — last wins
        ],
        algorithm="dict comprehension with swapped (k, v) pairs",
        complexity="O(n)",
        edge_cases=["empty dict", "value collisions (order-dependent)", "values must be hashable"],
    )


def tpl_group_by(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="collections",
        problem="Write a Python function `group_by(xs, key)` that returns a dict mapping key(x) → list of all xs with that key, preserving input order.",
        signature="def group_by(xs, key):",
        solution=(
            "def group_by(xs, key):\n"
            "    out = {}\n"
            "    for x in xs:\n"
            "        k = key(x)\n"
            "        if k not in out:\n"
            "            out[k] = []\n"
            "        out[k].append(x)\n"
            "    return out\n"
        ),
        test_cases=[
            ([], (lambda x: x), {}),
            ([1, 2, 3], (lambda x: x % 2), {1: [1, 3], 0: [2]}),
            (["apple", "ant", "bee"], (lambda s: s[0]), {"a": ["apple", "ant"], "b": ["bee"]}),
        ],
        algorithm="iterate, compute key, setdefault + append",
        complexity="O(n)",
        edge_cases=["empty input", "all same key", "all distinct keys"],
    )


def tpl_chunk(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="collections",
        problem="Write a Python function `chunk(xs, n)` that splits xs into lists of at most n elements. Last chunk may be shorter. Raise ValueError for n <= 0.",
        signature="def chunk(xs, n):",
        solution=(
            "def chunk(xs, n):\n"
            "    if n <= 0:\n"
            "        raise ValueError('chunk size must be positive')\n"
            "    return [xs[i:i + n] for i in range(0, len(xs), n)]\n"
        ),
        test_cases=[
            ([], 3, []),
            ([1, 2, 3], 1, [[1], [2], [3]]),
            ([1, 2, 3, 4], 2, [[1, 2], [3, 4]]),
            ([1, 2, 3, 4, 5], 2, [[1, 2], [3, 4], [5]]),
            ([1, 2, 3], 10, [[1, 2, 3]]),
        ],
        algorithm="slice from 0, step n, build list of slices",
        complexity="O(n)",
        edge_cases=["empty input", "n larger than len(xs)", "n = 1", "n <= 0 raises"],
    )


def tpl_zip_longest(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="collections",
        problem="Write a Python function `zip_fill(a, b, fill=None)` that zips two lists, padding the shorter with fill to match length. Returns a list of tuples.",
        signature="def zip_fill(a, b, fill=None):",
        solution=(
            "def zip_fill(a, b, fill=None):\n"
            "    n = max(len(a), len(b))\n"
            "    out = []\n"
            "    for i in range(n):\n"
            "        va = a[i] if i < len(a) else fill\n"
            "        vb = b[i] if i < len(b) else fill\n"
            "        out.append((va, vb))\n"
            "    return out\n"
        ),
        test_cases=[
            ([], [], None, []),
            ([1, 2], [3, 4], None, [(1, 3), (2, 4)]),
            ([1], [2, 3], 0, [(1, 2), (0, 3)]),
            ([1, 2, 3], [4], None, [(1, 4), (2, None), (3, None)]),
        ],
        algorithm="iterate max-length, index both lists or substitute fill",
        complexity="O(max(|a|, |b|))",
        edge_cases=["both empty", "one empty", "same length (equivalent to zip)"],
    )


def tpl_running_average(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="collections",
        problem="Write a Python function `running_avg(xs)` that returns a list where each element is the cumulative mean of xs[:i+1]. Return [] for empty input.",
        signature="def running_avg(xs):",
        solution=(
            "def running_avg(xs):\n"
            "    out = []\n"
            "    total = 0\n"
            "    for i, x in enumerate(xs, 1):\n"
            "        total += x\n"
            "        out.append(total / i)\n"
            "    return out\n"
        ),
        test_cases=[
            ([], []),
            ([10], [10.0]),
            ([1, 2, 3], [1.0, 1.5, 2.0]),
            ([2, 4, 6, 8], [2.0, 3.0, 4.0, 5.0]),
        ],
        algorithm="running sum / running count",
        complexity="O(n)",
        edge_cases=["empty", "single element (same as value)", "integer/float output"],
    )


def tpl_absolute_value(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="number_theory",
        problem="Write a Python function `absval(x)` that returns the absolute value of x without using the `abs` builtin.",
        signature="def absval(x):",
        solution="def absval(x):\n    return -x if x < 0 else x\n",
        test_cases=[(0, 0), (5, 5), (-5, 5), (-0.0, 0.0), (3.14, 3.14), (-2.718, 2.718)],
        algorithm="conditional negation",
        complexity="O(1)",
        edge_cases=["zero", "negative zero (float)", "negatives"],
    )


def tpl_square(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="number_theory",
        problem="Write a Python function `square(x)` that returns x squared.",
        signature="def square(x):",
        solution="def square(x):\n    return x * x\n",
        test_cases=[(0, 0), (1, 1), (-3, 9), (4, 16), (1.5, 2.25)],
        algorithm="multiply by self",
        complexity="O(1)",
        edge_cases=["zero", "negatives (square is positive)", "floats"],
    )


def tpl_sum_list(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="collections",
        problem="Write a Python function `sum_list(xs)` that returns the sum of a list of numbers without using the `sum` builtin.",
        signature="def sum_list(xs):",
        solution="def sum_list(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total\n",
        test_cases=[([], 0), ([1], 1), ([1, 2, 3], 6), ([-1, 1], 0), ([1.5, 2.5], 4.0)],
        algorithm="iterate and accumulate",
        complexity="O(n) time, O(1) space",
        edge_cases=["empty list (returns 0)", "negatives cancelling", "floats"],
    )


def tpl_max_list(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="collections",
        problem="Write a Python function `max_list(xs)` that returns the maximum element of a non-empty list, without using the `max` builtin. Raise ValueError on empty input.",
        signature="def max_list(xs):",
        solution="def max_list(xs):\n    if not xs:\n        raise ValueError('empty sequence')\n    m = xs[0]\n    for x in xs[1:]:\n        if x > m:\n            m = x\n    return m\n",
        test_cases=[([1], 1), ([1, 2, 3], 3), ([3, 2, 1], 3), ([-5, -1, -3], -1), ([0, -0.0], 0)],
        algorithm="single-pass running max",
        complexity="O(n)",
        edge_cases=["single element", "all-negative", "duplicates", "empty (raise)"],
    )


def tpl_reverse_string(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="strings",
        problem="Write a Python function `reverse_str(s)` that returns the reversed string.",
        signature="def reverse_str(s):",
        solution="def reverse_str(s):\n    return s[::-1]\n",
        test_cases=[("", ""), ("a", "a"), ("ab", "ba"), ("hello", "olleh"), ("  ", "  ")],
        algorithm="slice with negative step",
        complexity="O(n)",
        edge_cases=["empty", "single char (palindrome)", "whitespace"],
    )


def tpl_count_vowels(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="strings",
        problem="Write a Python function `count_vowels(s)` that counts vowels (a, e, i, o, u, case-insensitive) in a string.",
        signature="def count_vowels(s):",
        solution="def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')\n",
        test_cases=[("", 0), ("aeiou", 5), ("AEIOU", 5), ("hello", 2), ("xyz", 0), ("Python", 1)],
        algorithm="case-fold then count membership in vowel set",
        complexity="O(n)",
        edge_cases=["empty", "all vowels", "no vowels", "mixed case"],
    )


def tpl_is_palindrome(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="strings",
        problem="Write a Python function `is_palindrome(s)` that returns True if s reads the same forwards and backwards (ignore case, ignore non-alphanumeric).",
        signature="def is_palindrome(s):",
        solution="def is_palindrome(s):\n    t = ''.join(c.lower() for c in s if c.isalnum())\n    return t == t[::-1]\n",
        test_cases=[("", True), ("a", True), ("racecar", True), ("Race Car", True), ("hello", False), ("A man a plan a canal Panama", True), ("12321", True)],
        algorithm="normalize then compare with reversed",
        complexity="O(n)",
        edge_cases=["empty (True)", "single char (True)", "non-alphanumeric chars", "mixed case"],
    )


def tpl_is_anagram(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="strings",
        problem="Write a Python function `is_anagram(a, b)` that returns True if a and b are anagrams (same letters, any order). Case-insensitive. Ignore spaces.",
        signature="def is_anagram(a, b):",
        solution=(
            "def is_anagram(a, b):\n"
            "    a = a.replace(' ', '').lower()\n"
            "    b = b.replace(' ', '').lower()\n"
            "    return sorted(a) == sorted(b)\n"
        ),
        test_cases=[("listen", "silent", True), ("a", "A", True), ("", "", True),
                    ("abc", "cab", True), ("abc", "abd", False), ("rail safety", "fairy tales", True)],
        algorithm="normalize then compare sorted char sequences",
        complexity="O(n log n)",
        edge_cases=["case differences", "spaces", "empty pair", "same-length but different letters"],
    )


def tpl_count_occurrences(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="collections",
        problem="Write a Python function `count_occurrences(xs, target)` that returns how many times target appears in xs.",
        signature="def count_occurrences(xs, target):",
        solution="def count_occurrences(xs, target):\n    return sum(1 for x in xs if x == target)\n",
        test_cases=[([], 1, 0), ([1, 2, 3], 2, 1), ([1, 1, 1], 1, 3), (['a', 'b', 'a'], 'a', 2)],
        algorithm="linear count via comprehension",
        complexity="O(n)",
        edge_cases=["empty list (0)", "target absent (0)", "all matches"],
    )


def tpl_unique_elements(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="collections",
        problem="Write a Python function `unique(xs)` that returns a list of unique elements preserving order of first occurrence.",
        signature="def unique(xs):",
        solution=(
            "def unique(xs):\n"
            "    seen = set()\n"
            "    out = []\n"
            "    for x in xs:\n"
            "        if x not in seen:\n"
            "            seen.add(x)\n"
            "            out.append(x)\n"
            "    return out\n"
        ),
        test_cases=[([], []), ([1, 2, 1, 3], [1, 2, 3]), ([1, 1, 1], [1]), (['a', 'b', 'a', 'c'], ['a', 'b', 'c'])],
        algorithm="seen-set + ordered emit",
        complexity="O(n) time, O(n) space",
        edge_cases=["empty", "all same", "preserve first-seen order"],
    )


def tpl_gcd(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="number_theory",
        problem="Write a Python function `gcd(a, b)` that returns the greatest common divisor using the Euclidean algorithm. Handle zeros.",
        signature="def gcd(a, b):",
        solution=(
            "def gcd(a, b):\n"
            "    a, b = abs(a), abs(b)\n"
            "    while b:\n"
            "        a, b = b, a % b\n"
            "    return a\n"
        ),
        test_cases=[(12, 18, 6), (100, 75, 25), (17, 5, 1), (0, 7, 7), (7, 0, 7), (0, 0, 0), (-48, 18, 6)],
        algorithm="Euclidean algorithm via repeated modulo",
        complexity="O(log min(a, b))",
        edge_cases=["gcd(0, 0) = 0", "one zero", "negative inputs (use abs)"],
    )


def tpl_fibonacci(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="number_theory",
        problem="Write an iterative Python function `fib(n)` that returns the n-th Fibonacci number. fib(0) = 0, fib(1) = 1.",
        signature="def fib(n):",
        solution=(
            "def fib(n):\n"
            "    if n < 0:\n"
            "        raise ValueError('n must be non-negative')\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a\n"
        ),
        test_cases=[(0, 0), (1, 1), (2, 1), (3, 2), (10, 55), (15, 610), (20, 6765)],
        algorithm="bottom-up two-variable iteration",
        complexity="O(n) time, O(1) space",
        edge_cases=["n=0 returns 0", "n=1 returns 1", "large n (Python handles big ints)"],
    )


def tpl_is_prime(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="number_theory",
        problem="Write a Python function `is_prime(n)` that returns True if n is a prime number, False otherwise. Handle n <= 1 correctly (not prime).",
        signature="def is_prime(n):",
        solution=(
            "def is_prime(n):\n"
            "    if n < 2:\n"
            "        return False\n"
            "    if n < 4:\n"
            "        return True\n"
            "    if n % 2 == 0:\n"
            "        return False\n"
            "    i = 3\n"
            "    while i * i <= n:\n"
            "        if n % i == 0:\n"
            "            return False\n"
            "        i += 2\n"
            "    return True\n"
        ),
        test_cases=[(2, True), (3, True), (4, False), (1, False), (0, False), (-3, False), (17, True), (15, False), (97, True), (100, False)],
        algorithm="trial division up to sqrt(n), skip evens after 2",
        complexity="O(√n)",
        edge_cases=["n < 2 is not prime", "n = 2 is smallest prime", "even numbers > 2 aren't prime"],
    )


def tpl_factorial(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="number_theory",
        problem="Write a Python function `factorial(n)` that returns n!. Raise ValueError for negative n.",
        signature="def factorial(n):",
        solution=(
            "def factorial(n):\n"
            "    if n < 0:\n"
            "        raise ValueError('n must be non-negative')\n"
            "    result = 1\n"
            "    for i in range(2, n + 1):\n"
            "        result *= i\n"
            "    return result\n"
        ),
        test_cases=[(0, 1), (1, 1), (2, 2), (5, 120), (10, 3628800)],
        algorithm="iterative product from 2 to n",
        complexity="O(n)",
        edge_cases=["0! = 1", "1! = 1", "negative raises ValueError"],
    )


def tpl_power(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="number_theory",
        problem="Write a Python function `power(base, exp)` that computes base**exp for non-negative integer exp using repeated squaring. Do not use the ** operator or math.pow.",
        signature="def power(base, exp):",
        solution=(
            "def power(base, exp):\n"
            "    if exp < 0:\n"
            "        raise ValueError('exp must be non-negative')\n"
            "    result = 1\n"
            "    while exp:\n"
            "        if exp & 1:\n"
            "            result *= base\n"
            "        base *= base\n"
            "        exp >>= 1\n"
            "    return result\n"
        ),
        test_cases=[(2, 0, 1), (2, 1, 2), (2, 10, 1024), (3, 4, 81), (5, 3, 125), (7, 2, 49), (0, 5, 0), (1, 100, 1)],
        algorithm="binary exponentiation / repeated squaring",
        complexity="O(log exp)",
        edge_cases=["exp = 0 returns 1", "base = 0 (except 0^0)", "large exp"],
    )


def tpl_binary_search(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="algorithms",
        problem="Write a Python function `bsearch(arr, target)` that returns the index of target in a sorted list, or -1 if not found.",
        signature="def bsearch(arr, target):",
        solution=(
            "def bsearch(arr, target):\n"
            "    lo, hi = 0, len(arr) - 1\n"
            "    while lo <= hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        if arr[mid] < target:\n"
            "            lo = mid + 1\n"
            "        else:\n"
            "            hi = mid - 1\n"
            "    return -1\n"
        ),
        test_cases=[([1, 3, 5, 7, 9], 7, 3), ([1, 3, 5, 7, 9], 1, 0), ([1, 3, 5, 7, 9], 9, 4),
                    ([1, 3, 5, 7, 9], 4, -1), ([1, 3, 5, 7, 9], 0, -1), ([], 5, -1), ([5], 5, 0)],
        algorithm="classic binary search with inclusive bounds",
        complexity="O(log n)",
        edge_cases=["empty list", "target at first index", "target at last index", "target absent"],
    )


def tpl_balanced_parens(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="algorithms",
        problem="Write a Python function `balanced(s)` that returns True if the string's parentheses/brackets/braces are balanced (matched + correctly nested). Ignore non-bracket chars.",
        signature="def balanced(s):",
        solution=(
            "def balanced(s):\n"
            "    pairs = {')': '(', ']': '[', '}': '{'}\n"
            "    opens = set(pairs.values())\n"
            "    stack = []\n"
            "    for c in s:\n"
            "        if c in opens:\n"
            "            stack.append(c)\n"
            "        elif c in pairs:\n"
            "            if not stack or stack.pop() != pairs[c]:\n"
            "                return False\n"
            "    return not stack\n"
        ),
        test_cases=[("", True), ("()", True), ("()[]", True), ("(]", False), ("({[]})", True), ("(((", False), ("a(b)c", True), ("(]()", False)],
        algorithm="stack of opens, match on close",
        complexity="O(n)",
        edge_cases=["empty string (balanced)", "unmatched open at end", "wrong-type close", "non-bracket chars ignored"],
    )


def tpl_flatten(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="algorithms",
        problem="Write a Python function `flatten(x)` that flattens arbitrarily nested lists into a single flat list. Non-list elements are kept as-is.",
        signature="def flatten(x):",
        solution=(
            "def flatten(x):\n"
            "    out = []\n"
            "    for item in x:\n"
            "        if isinstance(item, list):\n"
            "            out.extend(flatten(item))\n"
            "        else:\n"
            "            out.append(item)\n"
            "    return out\n"
        ),
        test_cases=[([1, [2, 3]], [1, 2, 3]), ([[1, 2], [3, [4, [5]]]], [1, 2, 3, 4, 5]), ([], []), ([1, 2, 3], [1, 2, 3]), ([[], []], [])],
        algorithm="recursive descent, extend on list, append on scalar",
        complexity="O(total elements)",
        edge_cases=["empty list", "already flat", "deeply nested", "empty sub-lists"],
    )


def tpl_two_sum(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="algorithms",
        problem="Write a Python function `two_sum(nums, target)` that returns a tuple (i, j) with nums[i] + nums[j] == target, i < j. Return None if no solution.",
        signature="def two_sum(nums, target):",
        solution=(
            "def two_sum(nums, target):\n"
            "    seen = {}\n"
            "    for j, x in enumerate(nums):\n"
            "        need = target - x\n"
            "        if need in seen:\n"
            "            return (seen[need], j)\n"
            "        seen[x] = j\n"
            "    return None\n"
        ),
        test_cases=[([2, 7, 11, 15], 9, (0, 1)), ([3, 2, 4], 6, (1, 2)), ([3, 3], 6, (0, 1)), ([1, 2, 3], 100, None)],
        algorithm="hash-map complement lookup in single pass",
        complexity="O(n) time, O(n) space",
        edge_cases=["no pair sums to target (None)", "duplicate values", "pair includes first and last"],
    )


def tpl_caesar(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="strings",
        problem="Write a Python function `caesar(text, shift)` that applies a Caesar cipher. Preserves case; leaves non-letters unchanged. Shift may be negative or > 26.",
        signature="def caesar(text, shift):",
        solution=(
            "def caesar(text, shift):\n"
            "    out = []\n"
            "    for c in text:\n"
            "        if 'a' <= c <= 'z':\n"
            "            out.append(chr((ord(c) - ord('a') + shift) % 26 + ord('a')))\n"
            "        elif 'A' <= c <= 'Z':\n"
            "            out.append(chr((ord(c) - ord('A') + shift) % 26 + ord('A')))\n"
            "        else:\n"
            "            out.append(c)\n"
            "    return ''.join(out)\n"
        ),
        test_cases=[("abc", 1, "bcd"), ("xyz", 3, "abc"), ("Hello, World!", 13, "Uryyb, Jbeyq!"), ("abc", 0, "abc"), ("abc", -1, "zab"), ("abc", 27, "bcd")],
        algorithm="per-letter offset with mod 26, preserve case, pass through non-letters",
        complexity="O(n)",
        edge_cases=["shift of 0 (identity)", "negative shift", "shift > 26 (use modulo)", "non-alpha chars unchanged"],
    )


def tpl_roman_to_int(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="strings",
        problem="Write a Python function `roman_to_int(s)` that converts a Roman numeral to an integer. Handle subtractive notation (IV=4, IX=9, XL=40, XC=90, CD=400, CM=900).",
        signature="def roman_to_int(s):",
        solution=(
            "def roman_to_int(s):\n"
            "    v = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n"
            "    total = 0\n"
            "    prev = 0\n"
            "    for c in reversed(s):\n"
            "        cur = v[c]\n"
            "        if cur < prev:\n"
            "            total -= cur\n"
            "        else:\n"
            "            total += cur\n"
            "        prev = cur\n"
            "    return total\n"
        ),
        test_cases=[("III", 3), ("IV", 4), ("IX", 9), ("LVIII", 58), ("MCMXCIV", 1994), ("XL", 40), ("CD", 400), ("MMMCMXCIX", 3999)],
        algorithm="right-to-left scan: subtract if smaller than previous, else add",
        complexity="O(n)",
        edge_cases=["simple additive (III)", "subtractive (IV, IX)", "4-digit (MCMXCIV)", "boundary (MMMCMXCIX = 3999)"],
    )


def tpl_levenshtein(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="algorithms",
        problem="Write a Python function `levenshtein(a, b)` that returns the edit distance between two strings.",
        signature="def levenshtein(a, b):",
        solution=(
            "def levenshtein(a, b):\n"
            "    if not a:\n"
            "        return len(b)\n"
            "    if not b:\n"
            "        return len(a)\n"
            "    prev = list(range(len(b) + 1))\n"
            "    for i, ca in enumerate(a, 1):\n"
            "        cur = [i]\n"
            "        for j, cb in enumerate(b, 1):\n"
            "            ins = cur[j - 1] + 1\n"
            "            dele = prev[j] + 1\n"
            "            sub = prev[j - 1] + (ca != cb)\n"
            "            cur.append(min(ins, dele, sub))\n"
            "        prev = cur\n"
            "    return prev[-1]\n"
        ),
        test_cases=[("kitten", "sitting", 3), ("", "abc", 3), ("abc", "", 3), ("same", "same", 0), ("ab", "ba", 2), ("cat", "cats", 1)],
        algorithm="bottom-up DP with two rows (space-optimized)",
        complexity="O(|a| * |b|) time, O(|b|) space",
        edge_cases=["either string empty", "identical strings (0)", "transposition (2 edits: 1 del + 1 ins)"],
    )


def tpl_rle_encode(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="strings",
        problem="Write a Python function `rle_encode(s)` that run-length encodes a string. 'aaabb' -> 'a3b2'. Single chars still get a count of 1.",
        signature="def rle_encode(s):",
        solution=(
            "def rle_encode(s):\n"
            "    if not s:\n"
            "        return ''\n"
            "    out = []\n"
            "    cur = s[0]\n"
            "    count = 1\n"
            "    for c in s[1:]:\n"
            "        if c == cur:\n"
            "            count += 1\n"
            "        else:\n"
            "            out.append(cur + str(count))\n"
            "            cur = c\n"
            "            count = 1\n"
            "    out.append(cur + str(count))\n"
            "    return ''.join(out)\n"
        ),
        test_cases=[("aaabb", "a3b2"), ("a", "a1"), ("", ""), ("abcd", "a1b1c1d1"), ("aabbaa", "a2b2a2")],
        algorithm="single pass with run counter; emit on change",
        complexity="O(n)",
        edge_cases=["empty string (empty result)", "single char", "non-consecutive same chars"],
    )


def tpl_safe_url(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="security",
        problem="Write a Python function `is_safe_url(url)` that returns True only if the URL uses http(s) AND its hostname is not private/loopback/link-local. Block: 127.0.0.0/8, 10.0.0.0/8, 192.168.0.0/16, 172.16.0.0/12, 169.254.0.0/16, 0.0.0.0, localhost, metadata.google.internal, metadata.internal. Return False for non-http(s) schemes.",
        signature="def is_safe_url(url):",
        solution=(
            "def is_safe_url(url):\n"
            "    from urllib.parse import urlparse\n"
            "    import ipaddress\n"
            "    BLOCKED_HOSTS = {'localhost', 'metadata.google.internal', 'metadata.internal', '0.0.0.0'}\n"
            "    try:\n"
            "        p = urlparse(url)\n"
            "    except Exception:\n"
            "        return False\n"
            "    if p.scheme not in ('http', 'https'):\n"
            "        return False\n"
            "    host = (p.hostname or '').lower()\n"
            "    if not host:\n"
            "        return False\n"
            "    if host in BLOCKED_HOSTS:\n"
            "        return False\n"
            "    try:\n"
            "        ip = ipaddress.ip_address(host)\n"
            "    except ValueError:\n"
            "        return True\n"
            "    return not (ip.is_private or ip.is_loopback or ip.is_link_local\n"
            "                or ip.is_reserved or ip.is_multicast or ip.is_unspecified)\n"
        ),
        test_cases=[
            ("https://example.com", True),
            ("http://127.0.0.1/admin", False),
            ("http://localhost/", False),
            ("http://10.0.0.5/", False),
            ("http://192.168.1.1/", False),
            ("http://169.254.169.254/", False),
            ("https://metadata.google.internal/", False),
            ("file:///etc/passwd", False),
            ("javascript:alert(1)", False),
            ("https://8.8.8.8/", True),
        ],
        algorithm="urlparse → scheme check → ipaddress categorization against private/loopback/link-local/reserved/multicast ranges + explicit deny list",
        complexity="O(1) — constant-size deny list, single parse",
        edge_cases=[
            "non-http(s) schemes (file://, javascript:)",
            "hostname of 'localhost' (no IP match but dangerous)",
            "metadata service hostnames (cloud SSRF)",
            "reserved/link-local addresses (169.254.x)",
            "IPv6 private ranges (::1, fc00::/7)",
        ],
        skip_sandbox=True,  # sandbox blocks urllib/ipaddress imports
    )


def tpl_parse_int_safe(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="parsing",
        problem="Write a Python function `parse_int_safe(s, default)` that returns int(s) if parseable (accepts optional leading +/-, whitespace), else returns default. Must not raise.",
        signature="def parse_int_safe(s, default):",
        solution=(
            "def parse_int_safe(s, default):\n"
            "    try:\n"
            "        return int(str(s).strip())\n"
            "    except (ValueError, TypeError):\n"
            "        return default\n"
        ),
        test_cases=[("42", 0, 42), ("  -7 ", 0, -7), ("+3", 0, 3), ("abc", 99, 99), ("", 5, 5), (None, -1, -1), ("3.14", 0, 0)],
        algorithm="int() with try/except on ValueError, TypeError",
        complexity="O(|s|)",
        edge_cases=["whitespace padding", "empty string", "None", "float strings (int doesn't parse)", "non-string"],
    )


def tpl_strip_html(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="security",
        problem="Write a Python function `strip_html_tags(s)` that returns the input with all HTML tags removed. Uses `html.parser` (stdlib) — not regex — for correctness on malformed input.",
        signature="def strip_html_tags(s):",
        solution=(
            "def strip_html_tags(s):\n"
            "    from html.parser import HTMLParser\n"
            "    class _Stripper(HTMLParser):\n"
            "        def __init__(self):\n"
            "            super().__init__()\n"
            "            self.chunks = []\n"
            "        def handle_data(self, d):\n"
            "            self.chunks.append(d)\n"
            "    p = _Stripper()\n"
            "    p.feed(s)\n"
            "    return ''.join(p.chunks)\n"
        ),
        test_cases=[
            ("<p>hello</p>", "hello"),
            ("<a href='x'>link</a>", "link"),
            ("plain text", "plain text"),
            ("<b>bold</b> and <i>italic</i>", "bold and italic"),
            ("", ""),
            ("<script>alert(1)</script>ok", "alert(1)ok"),
        ],
        algorithm="HTMLParser subclass; collect handle_data chunks",
        complexity="O(n)",
        edge_cases=["nested tags", "malformed HTML (parser is forgiving)", "tags with attributes", "empty input"],
        skip_sandbox=True,  # sandbox blocks html.parser via import filter
    )


def tpl_word_frequency(rng: random.Random) -> CodeProblem:
    return CodeProblem(
        category="collections",
        problem="Write a Python function `word_freq(text)` that returns a dict of lowercased word counts. Words are sequences of alphabetic chars; everything else separates.",
        signature="def word_freq(text):",
        solution=(
            "def word_freq(text):\n"
            "    import re\n"
            "    words = re.findall(r'[A-Za-z]+', text.lower())\n"
            "    freq = {}\n"
            "    for w in words:\n"
            "        freq[w] = freq.get(w, 0) + 1\n"
            "    return freq\n"
        ),
        test_cases=[
            ("", {}),
            ("hello world", {"hello": 1, "world": 1}),
            ("Hello HELLO hello", {"hello": 3}),
            ("one, two; two. three three three!", {"one": 1, "two": 2, "three": 3}),
        ],
        algorithm="regex extract words, lowercase, counter dict",
        complexity="O(n)",
        edge_cases=["empty string", "case folding", "punctuation splits words", "digits excluded"],
    )


# Registry ordered for deterministic iteration
TEMPLATES: List[TemplateFn] = [
    # Number theory
    tpl_is_divisible, tpl_is_even_odd, tpl_absolute_value, tpl_square,
    tpl_min_of, tpl_sum_squares, tpl_digit_sum, tpl_reverse_int,
    tpl_count_set_bits, tpl_gcd, tpl_fibonacci, tpl_is_prime,
    tpl_factorial, tpl_power,
    # Collections
    tpl_sum_list, tpl_max_list, tpl_count_occurrences, tpl_unique_elements,
    tpl_min_max_tuple, tpl_transpose, tpl_dict_invert,
    tpl_chunk, tpl_zip_longest, tpl_running_average,
    # (tpl_group_by excluded — lambda args don't serialize for sandbox verify)
    # Strings
    tpl_reverse_string, tpl_count_vowels, tpl_is_palindrome, tpl_is_anagram,
    tpl_word_frequency, tpl_caesar, tpl_roman_to_int, tpl_rle_encode,
    # Algorithms
    tpl_binary_search, tpl_balanced_parens, tpl_flatten, tpl_two_sum,
    tpl_levenshtein,
    # Security / parsing
    tpl_safe_url, tpl_parse_int_safe, tpl_strip_html,
]


# ----- assistant reasoning formatter -----

THINK_TEMPLATE = textwrap.dedent("""\
    <think>
    STEP 1 — DECOMPOSE
    The user wants: {problem}
    Required signature: `{signature}`
    Category: {category}

    STEP 2 — PLAN
    Algorithm: {algorithm}
    Complexity: {complexity}
    Edge cases to handle:
    {edge_bullets}

    STEP 3 — IMPLEMENT
    I'll write the function directly. Core logic corresponds to the algorithm above.

    STEP 4 — VERIFY (mental test)
    Tracing through representative inputs:
    {trace_bullets}

    STEP 5 — ANSWER
    The implementation below handles all edge cases above.
    </think>

    ```python
    {solution_body}```

    **Verified test cases:**
    {test_bullets}
    """)


def _format_test_case(tc: tuple) -> str:
    """Render a test-case tuple as `f(args) -> expected ✓`. Handles
    multi-arg inputs by treating all but the last element as args."""
    *args, expected = tc
    args_s = ", ".join(repr(a) for a in args)
    return f"  - `f({args_s}) -> {expected!r}`  ✓"


def _mental_trace(tc: tuple, signature: str) -> str:
    *args, expected = tc
    args_s = ", ".join(repr(a) for a in args)
    fname = signature.split("def ", 1)[1].split("(")[0].strip()
    return f"  - `{fname}({args_s})` → expected `{expected!r}`"


def build_messages(prob: CodeProblem,
                   system_prompt: str = "You are a careful, correct coding assistant.",
                   max_trace: int = 4) -> dict:
    """Serialize one CodeProblem as a messages-schema chat record.

    The assistant turn bakes in:
      - 5-step reasoning inside <think>
      - the code as a fenced ```python block
      - an explicit verified-test-cases list
    """
    # Limit mental-trace lines — verbose for problems with many tests
    trace_items = prob.test_cases[:max_trace]
    assistant = THINK_TEMPLATE.format(
        problem=prob.problem,
        signature=prob.signature,
        category=prob.category,
        algorithm=prob.algorithm,
        complexity=prob.complexity,
        edge_bullets="\n".join(f"  - {e}" for e in prob.edge_cases),
        trace_bullets="\n".join(_mental_trace(tc, prob.signature) for tc in trace_items),
        solution_body=prob.solution,
        test_bullets="\n".join(_format_test_case(tc) for tc in prob.test_cases),
    )
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prob.problem},
            {"role": "assistant", "content": assistant},
        ],
    }


# ----- verification + output -----

def verify_problem(prob: CodeProblem) -> Tuple[bool, Optional[str]]:
    """Run each stored test case through the sandbox using the stored
    solution. Returns (ok, error_summary). Ensures we only ship examples
    that actually pass.

    For problems marked `skip_sandbox=True`, falls back to AST-only
    validation (syntax valid + expected function name present). These
    are hand-written known-correct solutions that happen to use modules
    blocked by the sandbox import filter (urllib, html, etc)."""
    if prob.skip_sandbox:
        from calm.backends.ast_ops import ast_parse
        parsed = ast_parse(prob.solution)
        if not parsed.get("valid"):
            return False, "AST: " + ", ".join(parsed.get("errors", []))
        fname = prob.signature.split("def ", 1)[1].split("(")[0].strip()
        names = {f["name"] for f in parsed.get("functions", [])}
        if fname not in names:
            return False, f"missing function `{fname}`"
        return True, "AST-verified (sandbox skipped)"

    # Local import — keeps this module free of substrate imports
    # when generation is being tested on a clean machine.
    from calm.sandbox import run_python

    fname = prob.signature.split("def ", 1)[1].split("(")[0].strip()
    body = prob.solution
    # Build the test harness without interpolating `expr` into any
    # quoted print — f-strings don't escape quotes inside interpolated
    # values, so `reverse_str('')` inside a 'FAIL ...' literal would
    # break the string. Use `repr(_got)` + an integer tag to identify
    # failing cases.
    test_lines: List[str] = []
    for i, tc in enumerate(prob.test_cases):
        *args, expected = tc
        args_s = ", ".join(repr(a) for a in args)
        exp_r = repr(expected)
        test_lines.append(
            f"try:\n"
            f"    _got = {fname}({args_s})\n"
            f"    if _got == {exp_r}:\n"
            f"        print('PASS')\n"
            f"    else:\n"
            f"        print('FAIL idx={i} got=' + repr(_got))\n"
            f"except Exception as _e:\n"
            f"    print('FAIL idx={i} raised=' + type(_e).__name__)"
        )
    script = body + "\n\n" + "\n".join(test_lines) + "\npass\n"
    result = run_python(script, timeout=5.0)
    if result.error:
        return False, str(result.error)
    out = result.stdout or ""
    if "FAIL" in out:
        return False, out.strip().splitlines()[0]
    passed = out.count("PASS")
    expected = len(test_lines)
    return passed == expected, f"passed {passed}/{expected}"


def generate(count: int, seed: int = 0) -> List[dict]:
    """Main generation loop.

    Walks TEMPLATES round-robin, using variant seeds so parameterized
    templates produce different concrete problems each visit. Skips any
    problem whose stored solution doesn't pass its own test cases, and
    deduplicates by problem statement (templates returning constants
    only contribute one example).
    """
    rng = random.Random(seed)
    out: List[dict] = []
    seen_problems: set[str] = set()
    skipped_verify = 0
    skipped_dup = 0
    attempts = 0
    # Allow substantial loop room — constant templates dedup fast and
    # the loop exits early once no new variants are produced for
    # several full passes.
    max_attempts = count * 8
    consecutive_no_new = 0
    while len(out) < count and attempts < max_attempts:
        tpl = TEMPLATES[attempts % len(TEMPLATES)]
        prob = tpl(rng)
        attempts += 1
        if prob is None:
            continue
        if prob.problem in seen_problems:
            skipped_dup += 1
            consecutive_no_new += 1
            # Hit every template twice with no new variants — stop
            if consecutive_no_new > len(TEMPLATES) * 2:
                break
            continue
        ok, info = verify_problem(prob)
        if not ok:
            skipped_verify += 1
            print(f"  [skip] {tpl.__name__}: {info}", flush=True)
            continue
        seen_problems.add(prob.problem)
        out.append(build_messages(prob))
        consecutive_no_new = 0
    print(
        f"generated {len(out)} unique / skipped {skipped_verify} verify / "
        f"{skipped_dup} dup / attempts {attempts}",
        flush=True,
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", type=Path,
        default=Path("agents/distill/data/multi_step_code.jsonl"))
    ap.add_argument("--count", type=int, default=len(TEMPLATES) * 3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = generate(args.count, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} examples to {args.out}", flush=True)


if __name__ == "__main__":
    main()
