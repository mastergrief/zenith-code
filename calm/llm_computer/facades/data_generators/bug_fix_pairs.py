"""BugFixPairsGenerator — broken code + <think> diagnosis + fixed code.

Each example teaches the DB (and PT) how to recognize a common bug
pattern and the canonical fix. The problem statement presents buggy
code; the solution walks through diagnosis and emits corrected code.

The `problem` field embeds the buggy snippet inside a fenced code
block so retrieval by symptom (ZeroDivisionError, mutable default,
etc.) finds it via both Jaccard and TF-IDF indexes.

Verification strategy:
  - The fixed code (the `solution`) is AST-parsed + sandbox-tested
    against its test cases — same as other generators.
  - The buggy code is NOT executed (would raise by design).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

from calm.llm_computer.facades.data_generators import register_generator
from calm.llm_computer.facades.data_generators.base import (
    DomainDataGenerator,
    VerifiedExample,
)


@dataclass
class BugFixSpec:
    bug_name: str         # e.g. "mutable_default_argument"
    bug_symptom: str      # NL description: "returns wrong result for repeated calls"
    buggy_code: str       # the broken snippet
    fix_explanation: str  # <think>-style diagnosis
    fixed_signature: str  # the fix's def line
    fixed_code: str       # full corrected function
    test_cases: List[Tuple]
    category: str = "bug_fix"
    algorithm: str = ""
    complexity: str = ""
    # Set True for fixes that use sandbox-blocked modules (threading,
    # subprocess, socket, urllib) or produce non-deterministic output
    # (random, secrets, datetime.now). Such fixes ship AST-verified
    # only; their correctness is by construction / code review.
    skip_sandbox: bool = False


def _specs() -> List[BugFixSpec]:
    out: List[BugFixSpec] = []

    out.append(BugFixSpec(
        bug_name="mutable_default_argument",
        bug_symptom="The function returns growing results across repeated calls because Python evaluates default arguments ONCE at definition time.",
        buggy_code=(
            "def append_item(item, bucket=[]):\n"
            "    bucket.append(item)\n"
            "    return bucket\n"
        ),
        fix_explanation=(
            "STEP 1 — SYMPTOM: `append_item(1)` returns `[1]`, then\n"
            "`append_item(2)` returns `[1, 2]` instead of `[2]`.\n"
            "STEP 2 — ROOT CAUSE: the `bucket=[]` default is created\n"
            "ONCE when the function is defined. Subsequent calls share\n"
            "the same list object.\n"
            "STEP 3 — FIX: use `None` sentinel and create a fresh list\n"
            "inside the function body."
        ),
        fixed_signature="def append_item(item, bucket=None):",
        fixed_code=(
            "def append_item(item, bucket=None):\n"
            "    if bucket is None:\n"
            "        bucket = []\n"
            "    bucket.append(item)\n"
            "    return bucket\n"
        ),
        test_cases=[
            (1, None, [1]),
            (2, None, [2]),              # NOT [1, 2]
            (3, [10], [10, 3]),
            ('a', None, ['a']),
        ],
        algorithm="None sentinel for mutable default",
        complexity="O(1)",
    ))

    out.append(BugFixSpec(
        bug_name="late_binding_closure",
        bug_symptom="A list of lambdas from a loop all return the same value because they capture the loop variable by reference, not value.",
        buggy_code=(
            "def make_multipliers(n):\n"
            "    return [lambda x: x * i for i in range(n)]\n"
            # When called: multipliers[0](5) returns 5*(n-1) not 5*0
        ),
        fix_explanation=(
            "STEP 1 — SYMPTOM: for `n=3`, each returned lambda multiplies\n"
            "by 2 (the final value of `i`), not by 0, 1, 2 respectively.\n"
            "STEP 2 — ROOT CAUSE: closures in Python capture variables\n"
            "by reference. When the lambdas are called later, they all\n"
            "see the same `i` bound to its final loop value.\n"
            "STEP 3 — FIX: bind `i` as a default argument, forcing early\n"
            "evaluation at lambda-definition time."
        ),
        fixed_signature="def make_multipliers(n):",
        fixed_code=(
            "def make_multipliers(n):\n"
            "    return [lambda x, _i=i: x * _i for i in range(n)]\n"
        ),
        test_cases=[
            # (n, test_tuple_of_inputs, expected_output)
            # Evaluate: for each multiplier at idx i, call it with x, expect i*x
            (3, 5, [0, 5, 10]),
            (4, 2, [0, 2, 4, 6]),
            (1, 7, [0]),
            (0, 5, []),
        ],
        algorithm="default-argument early binding (_i=i idiom)",
        complexity="O(n)",
    ))

    out.append(BugFixSpec(
        bug_name="off_by_one_range",
        bug_symptom="A loop misses the last element because range(1, n) goes up to but NOT including n.",
        buggy_code=(
            "def sum_up_to(n):\n"
            "    total = 0\n"
            "    for i in range(1, n):\n"
            "        total += i\n"
            "    return total\n"
            # For n=5: sums 1+2+3+4 = 10, NOT 1+2+3+4+5 = 15
        ),
        fix_explanation=(
            "STEP 1 — SYMPTOM: `sum_up_to(5)` returns 10 but should return 15.\n"
            "STEP 2 — ROOT CAUSE: `range(1, n)` is [1, n), exclusive of\n"
            "the upper bound. This is Python's standard half-open range.\n"
            "STEP 3 — FIX: use `range(1, n + 1)` to include n, OR rewrite\n"
            "bounds semantics clearly in the function name + docstring."
        ),
        fixed_signature="def sum_up_to(n):",
        fixed_code=(
            "def sum_up_to(n):\n"
            "    total = 0\n"
            "    for i in range(1, n + 1):\n"
            "        total += i\n"
            "    return total\n"
        ),
        test_cases=[(0, 0), (1, 1), (2, 3), (5, 15), (10, 55), (100, 5050)],
        algorithm="inclusive upper-bound range",
        complexity="O(n)",
    ))

    out.append(BugFixSpec(
        bug_name="dict_modify_during_iteration",
        bug_symptom="Raises RuntimeError: dictionary changed size during iteration because deleting keys from a live dict is illegal.",
        buggy_code=(
            "def drop_zeros(d):\n"
            "    for k, v in d.items():\n"
            "        if v == 0:\n"
            "            del d[k]\n"
            "    return d\n"
        ),
        fix_explanation=(
            "STEP 1 — SYMPTOM: `drop_zeros({'a': 1, 'b': 0})` raises\n"
            "RuntimeError: dictionary changed size during iteration.\n"
            "STEP 2 — ROOT CAUSE: `dict.items()` returns a live view.\n"
            "Modifying `d` during iteration invalidates it.\n"
            "STEP 3 — FIX: materialize the keys to delete first, then\n"
            "delete (OR build a new dict via comprehension)."
        ),
        fixed_signature="def drop_zeros(d):",
        fixed_code=(
            "def drop_zeros(d):\n"
            "    return {k: v for k, v in d.items() if v != 0}\n"
        ),
        test_cases=[
            ({}, {}),
            ({"a": 1, "b": 0, "c": 2}, {"a": 1, "c": 2}),
            ({"a": 0, "b": 0}, {}),
            ({"a": 1}, {"a": 1}),
        ],
        algorithm="dict comprehension with filter (immutable snapshot)",
        complexity="O(n)",
    ))

    out.append(BugFixSpec(
        bug_name="integer_division_vs_float",
        bug_symptom="Returns 0 for `average([1, 2])` in Python 2-habit code because `/` used to be integer division.",
        buggy_code=(
            "def average(xs):\n"
            "    total = 0\n"
            "    for x in xs:\n"
            "        total += x\n"
            "    return total // len(xs)\n"
            # // is integer division; average([1, 2, 3]) returns 2, not 2.0
        ),
        fix_explanation=(
            "STEP 1 — SYMPTOM: `average([1, 2])` returns 1 instead of 1.5.\n"
            "STEP 2 — ROOT CAUSE: `//` is integer floor division. We\n"
            "want `/` for float division in Python 3.\n"
            "STEP 3 — FIX: replace `//` with `/`. Also guard against\n"
            "empty input (ZeroDivisionError)."
        ),
        fixed_signature="def average(xs):",
        fixed_code=(
            "def average(xs):\n"
            "    if not xs:\n"
            "        raise ValueError('empty sequence')\n"
            "    total = 0\n"
            "    for x in xs:\n"
            "        total += x\n"
            "    return total / len(xs)\n"
        ),
        test_cases=[
            ([1, 2], 1.5),
            ([1, 2, 3], 2.0),
            ([10], 10.0),
            ([-1, 1], 0.0),
        ],
        algorithm="float division with empty-guard",
        complexity="O(n)",
    ))

    out.append(BugFixSpec(
        bug_name="string_immutability_mistake",
        bug_symptom="Tries to modify a string in place — strings are immutable in Python.",
        buggy_code=(
            "def replace_spaces(s):\n"
            "    for i in range(len(s)):\n"
            "        if s[i] == ' ':\n"
            "            s[i] = '_'\n"
            "    return s\n"
            # TypeError: 'str' object does not support item assignment
        ),
        fix_explanation=(
            "STEP 1 — SYMPTOM: TypeError: 'str' object does not support\n"
            "item assignment.\n"
            "STEP 2 — ROOT CAUSE: strings are immutable in Python. You\n"
            "cannot assign to individual indices.\n"
            "STEP 3 — FIX: build a new string via join + genexp, or use\n"
            "`str.replace`. `replace` is clearer here."
        ),
        fixed_signature="def replace_spaces(s):",
        fixed_code=(
            "def replace_spaces(s):\n"
            "    return s.replace(' ', '_')\n"
        ),
        test_cases=[
            ("", ""),
            ("hello world", "hello_world"),
            ("no_spaces", "no_spaces"),
            ("  ", "__"),
        ],
        algorithm="str.replace (immutable-safe)",
        complexity="O(n)",
    ))

    out.append(BugFixSpec(
        bug_name="shallow_copy_nested",
        bug_symptom="Modifying a nested list via a copy also modifies the original because `list.copy()` is shallow.",
        buggy_code=(
            "def duplicate(rows):\n"
            "    out = rows.copy()\n"
            "    out[0][0] = 'X'\n"
            "    return out\n"
            # rows[0][0] is ALSO modified because inner lists are shared
        ),
        fix_explanation=(
            "STEP 1 — SYMPTOM: after calling duplicate, the ORIGINAL\n"
            "rows also has 'X' at [0][0].\n"
            "STEP 2 — ROOT CAUSE: `list.copy()` is a shallow copy. The\n"
            "outer list is new, but inner lists are shared references.\n"
            "STEP 3 — FIX: use `copy.deepcopy` for fully independent\n"
            "nested structures."
        ),
        fixed_signature="def duplicate(rows):",
        fixed_code=(
            "def duplicate(rows):\n"
            "    import copy\n"
            "    out = copy.deepcopy(rows)\n"
            "    if out and out[0]:\n"
            "        out[0][0] = 'X'\n"
            "    return out\n"
        ),
        test_cases=[
            ([[1, 2], [3, 4]], [['X', 2], [3, 4]]),
            ([[0]], [['X']]),
            ([], []),
        ],
        algorithm="copy.deepcopy for nested mutables",
        complexity="O(total elements)",
    ))

    out.append(BugFixSpec(
        bug_name="boolean_coercion_string",
        bug_symptom="Treating the string 'False' as falsy. Any non-empty string is truthy in Python.",
        buggy_code=(
            "def is_enabled(flag):\n"
            "    if flag:\n"
            "        return True\n"
            "    return False\n"
            # is_enabled('False') returns True
        ),
        fix_explanation=(
            "STEP 1 — SYMPTOM: `is_enabled('False')` returns True,\n"
            "surprising users who expect 'False' to mean off.\n"
            "STEP 2 — ROOT CAUSE: any non-empty string is truthy in\n"
            "Python. Only empty string, 0, None, empty collections are\n"
            "falsy.\n"
            "STEP 3 — FIX: explicitly parse known string values rather\n"
            "than relying on implicit truthiness."
        ),
        fixed_signature="def is_enabled(flag):",
        fixed_code=(
            "def is_enabled(flag):\n"
            "    if isinstance(flag, str):\n"
            "        return flag.strip().lower() in ('true', '1', 'yes', 'on')\n"
            "    return bool(flag)\n"
        ),
        test_cases=[
            (True, True),
            (False, False),
            (1, True),
            (0, False),
            ("true", True),
            ("True", True),
            ("FALSE", False),
            ("", False),
            ("0", False),
            ("yes", True),
        ],
        algorithm="explicit string whitelist + fallback to bool()",
        complexity="O(|flag|) for strings, O(1) otherwise",
    ))

    out.append(BugFixSpec(
        bug_name="float_equality_comparison",
        bug_symptom="Returns False for what looks like equal floats because of binary floating-point precision.",
        buggy_code=(
            "def close_to_point_three(x):\n"
            "    return x == 0.1 + 0.2\n"
            # 0.1 + 0.2 == 0.30000000000000004 in IEEE 754
        ),
        fix_explanation=(
            "STEP 1 — SYMPTOM: `close_to_point_three(0.3)` returns False.\n"
            "STEP 2 — ROOT CAUSE: IEEE 754 floats cannot represent 0.1,\n"
            "0.2, or 0.3 exactly. `0.1 + 0.2 == 0.30000000000000004`.\n"
            "STEP 3 — FIX: compare floats with a tolerance using\n"
            "`math.isclose` (relative + absolute tolerance)."
        ),
        fixed_signature="def close_to_point_three(x):",
        fixed_code=(
            "def close_to_point_three(x):\n"
            "    import math\n"
            "    return math.isclose(x, 0.3, rel_tol=1e-9, abs_tol=1e-12)\n"
        ),
        test_cases=[
            (0.3, True),
            (0.1 + 0.2, True),
            (0.30000001, False),
            (0.0, False),
            (0.3000000000000001, True),   # within tolerance
        ],
        algorithm="math.isclose for float equality",
        complexity="O(1)",
    ))

    out.append(BugFixSpec(
        bug_name="type_check_isinstance_vs_type",
        bug_symptom="Uses `type(x) == int` which rejects bool (a subclass of int) erroneously — or fails to recognize subclasses at all.",
        buggy_code=(
            "def is_integer(x):\n"
            "    return type(x) == int\n"
            # Misses True/False which ARE ints (isinstance(True, int) is True)
            # Also: rejects MyIntSubclass
        ),
        fix_explanation=(
            "STEP 1 — SYMPTOM: `is_integer(True)` returns False, though\n"
            "bool is a subclass of int. Any int subclass also returns False.\n"
            "STEP 2 — ROOT CAUSE: `type(x) == int` checks exact type.\n"
            "`isinstance(x, int)` respects the class hierarchy.\n"
            "STEP 3 — FIX: use isinstance. If bool should NOT count as\n"
            "int (common for numeric code), explicitly exclude it."
        ),
        fixed_signature="def is_integer(x):",
        fixed_code=(
            "def is_integer(x):\n"
            "    # Reject bool to avoid silent True/False confusion.\n"
            "    return isinstance(x, int) and not isinstance(x, bool)\n"
        ),
        test_cases=[
            (1, True),
            (-5, True),
            (0, True),
            (True, False),
            (False, False),
            (1.0, False),
            ("1", False),
        ],
        algorithm="isinstance with explicit bool exclusion",
        complexity="O(1)",
    ))

    out.append(BugFixSpec(
        bug_name="except_bare_except",
        bug_symptom="Bare `except:` swallows SystemExit and KeyboardInterrupt, making the program unstoppable.",
        buggy_code=(
            "def safe_divide(a, b):\n"
            "    try:\n"
            "        return a / b\n"
            "    except:\n"
            "        return None\n"
            # Catches EVERYTHING including Ctrl-C and sys.exit
        ),
        fix_explanation=(
            "STEP 1 — SYMPTOM: program won't respond to Ctrl-C during\n"
            "this call; sys.exit() calls get silently swallowed.\n"
            "STEP 2 — ROOT CAUSE: `except:` with no type catches BaseException,\n"
            "which includes SystemExit + KeyboardInterrupt.\n"
            "STEP 3 — FIX: catch specific exceptions. For divide, only\n"
            "ZeroDivisionError + TypeError are plausible — name them."
        ),
        fixed_signature="def safe_divide(a, b):",
        fixed_code=(
            "def safe_divide(a, b):\n"
            "    try:\n"
            "        return a / b\n"
            "    except (ZeroDivisionError, TypeError):\n"
            "        return None\n"
        ),
        test_cases=[
            (10, 2, 5.0),
            (10, 0, None),
            (0, 0, None),
            (10, "a", None),
            (-5, 2, -2.5),
        ],
        algorithm="narrow except clause over ZeroDivisionError + TypeError",
        complexity="O(1)",
    ))

    out.append(BugFixSpec(
        bug_name="chained_comparison_and",
        bug_symptom="Uses `x == y == z` expecting pairwise equality, but (for non-equality) `x < y < z` chains correctly — while `x == y and y == z` is clearer for most readers.",
        buggy_code=(
            "def all_equal_three(a, b, c):\n"
            "    return a == b and b == c or c == a\n"
            # Operator precedence: `and` binds tighter than `or`
            # This parses as (a == b and b == c) or (c == a)
            # For (1, 2, 1): first clause False, second True → True (wrong!)
        ),
        fix_explanation=(
            "STEP 1 — SYMPTOM: `all_equal_three(1, 2, 1)` returns True\n"
            "though the three values are not all equal.\n"
            "STEP 2 — ROOT CAUSE: operator precedence. `and` binds tighter\n"
            "than `or`, so the expression is `(a == b and b == c) or (c == a)`,\n"
            "which is satisfied by ANY two-way equality.\n"
            "STEP 3 — FIX: use chained comparison `a == b == c` OR parenthesize\n"
            "`(a == b) and (b == c) and (a == c)`. Python's chained == is both\n"
            "clearer and correct."
        ),
        fixed_signature="def all_equal_three(a, b, c):",
        fixed_code=(
            "def all_equal_three(a, b, c):\n"
            "    return a == b == c\n"
        ),
        test_cases=[
            (1, 1, 1, True),
            (1, 1, 2, False),
            (1, 2, 1, False),
            (2, 1, 1, False),
            ('a', 'a', 'a', True),
            (0, 0.0, 0, True),       # int/float equal by value
        ],
        algorithm="Python chained equality `a == b == c`",
        complexity="O(1)",
    ))

    out.append(BugFixSpec(
        bug_name="iter_once_exhaustion",
        bug_symptom="Generator exhausts after first iteration; subsequent iterations see an empty sequence.",
        buggy_code=(
            "def count_and_sum(xs):\n"
            "    n = sum(1 for _ in xs)\n"
            "    total = sum(xs)\n"
            "    return n, total\n"
            # xs might be a generator — after first sum it's exhausted
        ),
        fix_explanation=(
            "STEP 1 — SYMPTOM: when called with a generator, returns\n"
            "(n, 0) because the second sum iterates an empty iterator.\n"
            "STEP 2 — ROOT CAUSE: generators can only be iterated once.\n"
            "After the first pass, they are exhausted.\n"
            "STEP 3 — FIX: materialize the input to a list first, or\n"
            "stream both metrics in a single pass."
        ),
        fixed_signature="def count_and_sum(xs):",
        fixed_code=(
            "def count_and_sum(xs):\n"
            "    n = 0\n"
            "    total = 0\n"
            "    for x in xs:\n"
            "        n += 1\n"
            "        total += x\n"
            "    return n, total\n"
        ),
        test_cases=[
            ([], (0, 0)),
            ([1, 2, 3], (3, 6)),
            ([5], (1, 5)),
            (iter([1, 2, 3, 4]), (4, 10)),   # generator input still works
        ],
        algorithm="single-pass iteration accumulates both metrics",
        complexity="O(n)",
    ))

    out.append(BugFixSpec(
        bug_name="division_zero_no_guard",
        bug_symptom="Raises ZeroDivisionError when input list is empty because `sum(xs) / len(xs)` divides by zero.",
        buggy_code=(
            "def mean(xs):\n"
            "    return sum(xs) / len(xs)\n"
        ),
        fix_explanation=(
            "STEP 1 — SYMPTOM: `mean([])` raises ZeroDivisionError.\n"
            "STEP 2 — ROOT CAUSE: empty list → `len(xs) == 0` → division\n"
            "by zero.\n"
            "STEP 3 — FIX: guard with an explicit empty-check. Choose\n"
            "one of: raise ValueError, return None, return 0.0. Here\n"
            "we raise ValueError because 'mean of nothing' is undefined."
        ),
        fixed_signature="def mean(xs):",
        fixed_code=(
            "def mean(xs):\n"
            "    if not xs:\n"
            "        raise ValueError('mean() requires non-empty sequence')\n"
            "    return sum(xs) / len(xs)\n"
        ),
        test_cases=[
            ([1, 2, 3], 2.0),
            ([5], 5.0),
            ([-1, 1], 0.0),
            ([10, 20, 30, 40], 25.0),
        ],
        algorithm="empty-guard + float division",
        complexity="O(n)",
    ))

    out.append(BugFixSpec(
        bug_name="json_dumps_set",
        bug_symptom="TypeError: Object of type set is not JSON serializable when serializing data that contains a set.",
        buggy_code=(
            "def to_json(data):\n"
            "    import json\n"
            "    return json.dumps(data)\n"
            # json can't serialize set or frozenset
        ),
        fix_explanation=(
            "STEP 1 — SYMPTOM: `to_json({1, 2, 3})` raises TypeError.\n"
            "STEP 2 — ROOT CAUSE: JSON has no set type. `json.dumps` by\n"
            "default raises on sets.\n"
            "STEP 3 — FIX: convert sets to sorted lists via a `default`\n"
            "callback. Sorted for stable output; raw list ordering\n"
            "depends on insertion + hash."
        ),
        fixed_signature="def to_json(data):",
        fixed_code=(
            "def to_json(data):\n"
            "    import json\n"
            "    def _default(obj):\n"
            "        if isinstance(obj, (set, frozenset)):\n"
            "            return sorted(obj, key=lambda x: (str(type(x)), repr(x)))\n"
            "        raise TypeError(f'Not JSON serializable: {type(obj).__name__}')\n"
            "    return json.dumps(data, default=_default)\n"
        ),
        test_cases=[
            ([1, 2, 3], "[1, 2, 3]"),
            ({"a": 1}, '{\"a\": 1}'),
            ({1, 2, 3}, "[1, 2, 3]"),
            ({"xs": {3, 1, 2}}, '{\"xs\": [1, 2, 3]}'),
        ],
        algorithm="json.dumps with default= callback for sets",
        complexity="O(|data|)",
    ))

    out.append(BugFixSpec(
        bug_name="list_concat_in_loop_quadratic",
        bug_symptom="Quadratic slowdown building a string/list by += inside a loop because each operation creates a new object and copies everything.",
        buggy_code=(
            "def concat_strings(items):\n"
            "    result = ''\n"
            "    for s in items:\n"
            "        result += s\n"
            "    return result\n"
            # Strings are immutable — each += creates a new string
        ),
        fix_explanation=(
            "STEP 1 — SYMPTOM: O(n²) instead of O(n) total work. Painful\n"
            "on 100K+ items.\n"
            "STEP 2 — ROOT CAUSE: strings are immutable. Each `+=`\n"
            "allocates a new string and copies all existing chars.\n"
            "STEP 3 — FIX: use `str.join` or accumulate into a list and\n"
            "join once. (Note: CPython has an optimization that detects\n"
            "`s += t` on a str with refcount==1 and mutates in place,\n"
            "but this is fragile and doesn't apply cross-impl.)"
        ),
        fixed_signature="def concat_strings(items):",
        fixed_code=(
            "def concat_strings(items):\n"
            "    return ''.join(items)\n"
        ),
        test_cases=[
            ([], ""),
            (["a"], "a"),
            (["a", "b", "c"], "abc"),
            (["hello", " ", "world"], "hello world"),
            ([""], ""),
        ],
        algorithm="str.join (O(n) allocation once)",
        complexity="O(total chars)",
    ))

    out.append(BugFixSpec(
        bug_name="counter_increment_not_atomic",
        bug_symptom="Multi-threaded counter loses increments because `count += 1` is three operations (read, add, write), not atomic.",
        buggy_code=(
            "class Counter:\n"
            "    def __init__(self):\n"
            "        self.count = 0\n"
            "    def increment(self):\n"
            "        self.count += 1\n"
            # Under threads: read-modify-write race
        ),
        fix_explanation=(
            "STEP 1 — SYMPTOM: 10 threads each increment 1000 times but\n"
            "final count < 10000.\n"
            "STEP 2 — ROOT CAUSE: `count += 1` decomposes into load,\n"
            "add, store. Interleaved threads can read the same value.\n"
            "STEP 3 — FIX: hold a lock around the increment. Alternative:\n"
            "use `itertools.count` or `threading.Lock` explicitly."
        ),
        fixed_signature="def make_counter():",
        fixed_code=(
            "def make_counter():\n"
            "    import threading\n"
            "    lock = threading.Lock()\n"
            "    count = [0]      # mutable closure cell\n"
            "    def increment():\n"
            "        with lock:\n"
            "            count[0] += 1\n"
            "            return count[0]\n"
            "    return increment\n"
        ),
        test_cases=[],
        algorithm="threading.Lock around read-modify-write",
        complexity="O(1) per increment, plus lock contention",
        skip_sandbox=True,      # threading-dependent
    ))

    out.append(BugFixSpec(
        bug_name="sort_in_place_reassign",
        bug_symptom="Reassigns the result of sort() to a variable, getting None because list.sort() mutates in place and returns None.",
        buggy_code=(
            "def sort_ascending(xs):\n"
            "    return xs.sort()\n"
            # sort() returns None; sorted() returns a new list
        ),
        fix_explanation=(
            "STEP 1 — SYMPTOM: `sort_ascending([3, 1, 2])` returns None.\n"
            "STEP 2 — ROOT CAUSE: `list.sort()` sorts in place and\n"
            "returns None (Python convention for mutating methods).\n"
            "STEP 3 — FIX: use built-in `sorted()` for a new list, OR\n"
            "sort in place and return the list explicitly."
        ),
        fixed_signature="def sort_ascending(xs):",
        fixed_code=(
            "def sort_ascending(xs):\n"
            "    return sorted(xs)\n"
        ),
        test_cases=[
            ([], []),
            ([1], [1]),
            ([3, 1, 2], [1, 2, 3]),
            ([3, 3, 1, 2, 2], [1, 2, 2, 3, 3]),
            (['c', 'a', 'b'], ['a', 'b', 'c']),
        ],
        algorithm="sorted() returns new list (sort() mutates)",
        complexity="O(n log n)",
    ))

    out.append(BugFixSpec(
        bug_name="dict_get_default_mutable",
        bug_symptom="Using dict.get(key, []) + appending to the result silently drops the append because `[]` is a new empty list each time, not stored.",
        buggy_code=(
            "def append_to_key(d, k, v):\n"
            "    d.get(k, []).append(v)\n"
            "    return d\n"
            # If key not in d, we append to a throwaway list
        ),
        fix_explanation=(
            "STEP 1 — SYMPTOM: `append_to_key({}, 'x', 1)` returns `{}`\n"
            "but user expected `{'x': [1]}`.\n"
            "STEP 2 — ROOT CAUSE: `.get(k, [])` returns a fresh `[]`\n"
            "when k is missing. The append modifies that temporary.\n"
            "STEP 3 — FIX: use `setdefault(k, []).append(v)` to create\n"
            "AND store the list when missing."
        ),
        fixed_signature="def append_to_key(d, k, v):",
        fixed_code=(
            "def append_to_key(d, k, v):\n"
            "    d.setdefault(k, []).append(v)\n"
            "    return d\n"
        ),
        test_cases=[
            ({}, 'x', 1, {'x': [1]}),
            ({'x': [1]}, 'x', 2, {'x': [1, 2]}),
            ({'y': [9]}, 'x', 1, {'y': [9], 'x': [1]}),
            ({}, 'a', 'hello', {'a': ['hello']}),
        ],
        algorithm="setdefault(k, []).append(v)",
        complexity="O(1) amortized",
    ))

    return out


class BugFixPairsGenerator(DomainDataGenerator):
    """Bug diagnosis → fix pairs. Each example trains the DB/PT to
    recognize common Python pitfalls and emit canonical corrections."""

    name = "bug_fix"

    def __init__(self, rng=None):
        super().__init__(rng)
        self._specs = _specs()

    def generate_raw(self, n: int) -> List[VerifiedExample]:
        out: List[VerifiedExample] = []
        self.rng.shuffle(self._specs)
        for s in self._specs[:n]:
            problem = (
                f"The following Python code has a bug: {s.bug_symptom}\n\n"
                f"```python\n{s.buggy_code}```\n\n"
                f"Explain the bug and provide a corrected version."
            )
            out.append(VerifiedExample(
                problem=problem,
                signature=s.fixed_signature,
                solution=s.fixed_code,
                test_cases=list(s.test_cases),
                reasoning=s.fix_explanation,
                algorithm=s.algorithm,
                complexity=s.complexity,
                edge_cases=[f"bug: {s.bug_name}"],
                category="bug_fix",
                generator_name=self.name,
                skip_sandbox=s.skip_sandbox,
                metadata={"bug_name": s.bug_name},
            ))
        return out


register_generator("bug_fix", BugFixPairsGenerator)
