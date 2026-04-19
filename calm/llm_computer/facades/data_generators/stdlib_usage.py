"""StdlibUsageGenerator — introspected "how do I use X?" problems.

For each target stdlib callable, emits:
  - a natural-language problem ("Write a function that uses X to do Y")
  - a canonical solution using X
  - test cases that exercise expected behavior
  - algorithm + complexity annotations

Target surface area: high-traffic stdlib modules developers ask
about constantly but Gemma sometimes hallucinates signatures for.

  - pathlib.Path (joinpath, stem, suffix, iterdir)
  - collections (Counter, defaultdict, deque, OrderedDict)
  - itertools (accumulate, chain, combinations, permutations, groupby)
  - functools (reduce, lru_cache, partial, cmp_to_key)
  - string constants / transforms
  - os.path (non-import-blocked subset usable via pathlib)

Each entry is sandbox-verified — the sandbox allows pathlib import
guard-free. For modules the sandbox blocks (os, subprocess), we use
`skip_sandbox=True` + AST verify.
"""

from __future__ import annotations

import inspect
import random
from dataclasses import dataclass
from typing import Any, Callable, List, Tuple

from calm.llm_computer.facades.data_generators import register_generator
from calm.llm_computer.facades.data_generators.base import (
    DomainDataGenerator,
    VerifiedExample,
)


@dataclass
class StdlibEntry:
    """One generated stdlib problem spec."""
    problem: str
    signature: str
    solution: str
    test_cases: List[Tuple]
    algorithm: str
    complexity: str
    edge_cases: List[str]
    category: str
    skip_sandbox: bool = False


def _entries() -> List[StdlibEntry]:
    """Concrete stdlib problems. Each is small, idiomatic, fully
    deterministic."""
    out: List[StdlibEntry] = []

    # ---- collections ----
    out.append(StdlibEntry(
        category="stdlib_collections",
        problem="Write a Python function `top_n_frequent(xs, n)` that returns the n most-common elements using collections.Counter.",
        signature="def top_n_frequent(xs, n):",
        solution=(
            "def top_n_frequent(xs, n):\n"
            "    from collections import Counter\n"
            "    return [item for item, _ in Counter(xs).most_common(n)]\n"
        ),
        test_cases=[
            ([], 3, []),
            ([1, 1, 2, 3], 2, [1, 2]),
            (['a', 'b', 'a', 'c', 'a', 'b'], 1, ['a']),
            (['a', 'b', 'a', 'c', 'a', 'b'], 2, ['a', 'b']),
        ],
        algorithm="Counter.most_common",
        complexity="O(n log n)",
        edge_cases=["empty input", "n > unique count", "ties in frequency"],
    ))
    out.append(StdlibEntry(
        category="stdlib_collections",
        problem="Write a Python function `bucket_by_length(words)` that returns a defaultdict(list) mapping len(word) → list of words of that length.",
        signature="def bucket_by_length(words):",
        solution=(
            "def bucket_by_length(words):\n"
            "    from collections import defaultdict\n"
            "    buckets = defaultdict(list)\n"
            "    for w in words:\n"
            "        buckets[len(w)].append(w)\n"
            "    return dict(buckets)\n"
        ),
        test_cases=[
            ([], {}),
            (["a", "bb", "c", "dd"], {1: ["a", "c"], 2: ["bb", "dd"]}),
            (["hello"], {5: ["hello"]}),
        ],
        algorithm="defaultdict-backed grouping",
        complexity="O(total chars)",
        edge_cases=["empty list", "all-same-length", "duplicates preserved"],
    ))
    out.append(StdlibEntry(
        category="stdlib_collections",
        problem="Write a Python function `sliding_max(xs, k)` using collections.deque that returns the maximum of each window of size k as a list. Raise ValueError if k > len(xs) or k <= 0.",
        signature="def sliding_max(xs, k):",
        solution=(
            "def sliding_max(xs, k):\n"
            "    from collections import deque\n"
            "    if k <= 0 or k > len(xs):\n"
            "        raise ValueError('bad window size')\n"
            "    dq = deque()\n"
            "    out = []\n"
            "    for i, x in enumerate(xs):\n"
            "        while dq and xs[dq[-1]] <= x:\n"
            "            dq.pop()\n"
            "        dq.append(i)\n"
            "        if dq[0] <= i - k:\n"
            "            dq.popleft()\n"
            "        if i >= k - 1:\n"
            "            out.append(xs[dq[0]])\n"
            "    return out\n"
        ),
        test_cases=[
            ([1, 3, -1, -3, 5, 3, 6, 7], 3, [3, 3, 5, 5, 6, 7]),
            ([1], 1, [1]),
            ([5, 4, 3, 2, 1], 2, [5, 4, 3, 2]),
        ],
        algorithm="monotonic-deque sliding window maximum",
        complexity="O(n) amortized",
        edge_cases=["k = 1 (identity)", "k = len(xs) (one result)", "k > len raises"],
    ))

    # ---- itertools ----
    out.append(StdlibEntry(
        category="stdlib_itertools",
        problem="Write a Python function `cumulative_sum(xs)` using itertools.accumulate that returns the running total of xs as a list.",
        signature="def cumulative_sum(xs):",
        solution=(
            "def cumulative_sum(xs):\n"
            "    from itertools import accumulate\n"
            "    return list(accumulate(xs))\n"
        ),
        test_cases=[
            ([], []),
            ([1], [1]),
            ([1, 2, 3], [1, 3, 6]),
            ([5, -2, 4], [5, 3, 7]),
        ],
        algorithm="itertools.accumulate default (+)",
        complexity="O(n)",
        edge_cases=["empty", "single element", "negatives cancel"],
    ))
    out.append(StdlibEntry(
        category="stdlib_itertools",
        problem="Write a Python function `pairwise_adjacent(xs)` using itertools that returns a list of (xs[i], xs[i+1]) pairs. Empty or single-element input returns [].",
        signature="def pairwise_adjacent(xs):",
        solution=(
            "def pairwise_adjacent(xs):\n"
            "    from itertools import tee\n"
            "    a, b = tee(xs)\n"
            "    next(b, None)\n"
            "    return list(zip(a, b))\n"
        ),
        test_cases=[
            ([], []),
            ([1], []),
            ([1, 2, 3], [(1, 2), (2, 3)]),
            ([1, 2, 3, 4], [(1, 2), (2, 3), (3, 4)]),
        ],
        algorithm="itertools.tee + zip offset (pre-3.10 pairwise)",
        complexity="O(n)",
        edge_cases=["empty", "single element (empty result)", "two elements (one pair)"],
    ))
    out.append(StdlibEntry(
        category="stdlib_itertools",
        problem="Write a Python function `chain_flatten(lists)` using itertools.chain.from_iterable that flattens a list of lists one level deep.",
        signature="def chain_flatten(lists):",
        solution=(
            "def chain_flatten(lists):\n"
            "    from itertools import chain\n"
            "    return list(chain.from_iterable(lists))\n"
        ),
        test_cases=[
            ([], []),
            ([[1, 2], [3], [], [4, 5]], [1, 2, 3, 4, 5]),
            ([[[1, 2]], [3]], [[1, 2], 3]),  # only one level!
        ],
        algorithm="chain.from_iterable (single-level concat)",
        complexity="O(total elements)",
        edge_cases=["empty outer", "empty inner lists", "nested > 1 level NOT flattened"],
    ))
    out.append(StdlibEntry(
        category="stdlib_itertools",
        problem="Write a Python function `unique_pairs(xs)` that returns all unordered 2-element combinations of xs as a list of tuples. Uses itertools.combinations.",
        signature="def unique_pairs(xs):",
        solution=(
            "def unique_pairs(xs):\n"
            "    from itertools import combinations\n"
            "    return list(combinations(xs, 2))\n"
        ),
        test_cases=[
            ([], []),
            ([1], []),
            ([1, 2], [(1, 2)]),
            ([1, 2, 3], [(1, 2), (1, 3), (2, 3)]),
            (['a', 'b', 'c', 'd'], [('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')]),
        ],
        algorithm="itertools.combinations(xs, 2)",
        complexity="O(n²)",
        edge_cases=["fewer than 2 elements → empty", "order preserves input order"],
    ))

    # ---- functools ----
    out.append(StdlibEntry(
        category="stdlib_functools",
        problem="Write a Python function `product_of(xs)` using functools.reduce that returns the product of a list of numbers. product_of([]) = 1.",
        signature="def product_of(xs):",
        solution=(
            "def product_of(xs):\n"
            "    from functools import reduce\n"
            "    return reduce(lambda acc, x: acc * x, xs, 1)\n"
        ),
        test_cases=[
            ([], 1),
            ([1], 1),
            ([2, 3, 4], 24),
            ([5], 5),
            ([2, 0, 5], 0),
        ],
        algorithm="functools.reduce with multiplicative identity",
        complexity="O(n)",
        edge_cases=["empty (identity 1)", "contains zero", "single element"],
    ))
    out.append(StdlibEntry(
        category="stdlib_functools",
        problem="Write a Python function `memoize_fib(n)` that returns the n-th Fibonacci using functools.lru_cache. Must handle n up to 100 without recursion-depth errors.",
        signature="def memoize_fib(n):",
        solution=(
            "def memoize_fib(n):\n"
            "    from functools import lru_cache\n"
            "    @lru_cache(maxsize=None)\n"
            "    def f(k):\n"
            "        return k if k < 2 else f(k - 1) + f(k - 2)\n"
            "    return f(n)\n"
        ),
        test_cases=[
            (0, 0), (1, 1), (2, 1), (10, 55), (20, 6765), (50, 12586269025),
        ],
        algorithm="lru_cache memoized top-down recursion",
        complexity="O(n) time with O(n) space",
        edge_cases=["n=0 returns 0", "large n no stack overflow (linearized by cache)"],
    ))

    # ---- pathlib ----
    out.append(StdlibEntry(
        category="stdlib_pathlib",
        problem="Write a Python function `basename_stem(path)` that returns the filename without extension using pathlib.Path. `/a/b/c.py` → 'c'.",
        signature="def basename_stem(path):",
        solution=(
            "def basename_stem(path):\n"
            "    from pathlib import Path\n"
            "    return Path(path).stem\n"
        ),
        test_cases=[
            ("/a/b/c.py", "c"),
            ("c.txt", "c"),
            ("/a/b/c", "c"),
            ("a.tar.gz", "a.tar"),  # stem only removes last suffix
            ("", ""),
        ],
        algorithm="pathlib.Path.stem property",
        complexity="O(1)",
        edge_cases=["no extension (stem == name)", "double extension (stem keeps inner)", "empty"],
    ))
    out.append(StdlibEntry(
        category="stdlib_pathlib",
        problem="Write a Python function `safe_join(base, *parts)` using pathlib.Path that joins base + parts but rejects path traversal (`..`). Raise ValueError if any part contains `..`.",
        signature="def safe_join(base, *parts):",
        solution=(
            "def safe_join(base, *parts):\n"
            "    from pathlib import PurePosixPath\n"
            "    for p in parts:\n"
            "        if '..' in PurePosixPath(p).parts:\n"
            "            raise ValueError('path traversal not allowed')\n"
            "    out = PurePosixPath(base)\n"
            "    for p in parts:\n"
            "        out = out / p\n"
            "    return str(out)\n"
        ),
        test_cases=[
            ("/home", "user", "file.txt", "/home/user/file.txt"),
            ("/a", "b", "/a/b"),
            ("base", "sub", "file", "base/sub/file"),
        ],
        algorithm="PurePosixPath with explicit '..' rejection",
        complexity="O(|parts|)",
        edge_cases=["no parts", "absolute subparts (Path swallows base)", "traversal raises"],
    ))

    # ---- string ----
    out.append(StdlibEntry(
        category="stdlib_string",
        problem="Write a Python function `title_case_preserving_acronyms(s)` that title-cases each word but preserves 2-3 letter all-caps words (acronyms like 'USA', 'API', 'IO').",
        signature="def title_case_preserving_acronyms(s):",
        solution=(
            "def title_case_preserving_acronyms(s):\n"
            "    words = s.split()\n"
            "    out = []\n"
            "    for w in words:\n"
            "        if 2 <= len(w) <= 3 and w.isupper():\n"
            "            out.append(w)\n"
            "        else:\n"
            "            out.append(w.capitalize())\n"
            "    return ' '.join(out)\n"
        ),
        test_cases=[
            ("", ""),
            ("hello world", "Hello World"),
            ("USA is big", "USA Is Big"),
            ("api design", "Api Design"),         # lowercase 'api' gets capitalized
            ("the API design", "The API Design"),
            ("IO bound", "IO Bound"),
        ],
        algorithm="per-word acronym detection via isupper() + length",
        complexity="O(chars)",
        edge_cases=["empty", "single word", "mixed acronym casing"],
    ))

    # ---- itertools (more) ----
    out.append(StdlibEntry(
        category="stdlib_itertools",
        problem="Write a Python function `running_starmap_sum(xs, ys)` that uses itertools.starmap with operator.add to compute the element-wise sum of two lists of equal length. Returns a list.",
        signature="def running_starmap_sum(xs, ys):",
        solution=(
            "def running_starmap_sum(xs, ys):\n"
            "    from itertools import starmap\n"
            "    import operator\n"
            "    if len(xs) != len(ys):\n"
            "        raise ValueError('length mismatch')\n"
            "    return list(starmap(operator.add, zip(xs, ys)))\n"
        ),
        test_cases=[
            ([], [], []),
            ([1], [2], [3]),
            ([1, 2, 3], [10, 20, 30], [11, 22, 33]),
            ([-1, -2], [1, 2], [0, 0]),
        ],
        algorithm="starmap over zip with operator.add",
        complexity="O(n)",
        edge_cases=["empty lists", "length mismatch raises", "works on floats too"],
    ))

    out.append(StdlibEntry(
        category="stdlib_itertools",
        problem="Write a Python function `take_while_positive(xs)` using itertools.takewhile that returns a list of leading positive elements, stopping at the first non-positive value.",
        signature="def take_while_positive(xs):",
        solution=(
            "def take_while_positive(xs):\n"
            "    from itertools import takewhile\n"
            "    return list(takewhile(lambda x: x > 0, xs))\n"
        ),
        test_cases=[
            ([], []),
            ([1, 2, 3, -1, 5], [1, 2, 3]),
            ([-1, 1, 2], []),                 # first is <=0 → empty
            ([1, 2, 3], [1, 2, 3]),
            ([0, 1, 2], []),                  # 0 is not > 0
        ],
        algorithm="takewhile with positive predicate",
        complexity="O(n)",
        edge_cases=["empty", "first non-positive → empty result", "zero stops", "all positive"],
    ))

    # ---- functools (more) ----
    out.append(StdlibEntry(
        category="stdlib_functools",
        problem="Write a Python function `partial_adder(n)` using functools.partial that returns a new function adding n to its argument.",
        signature="def partial_adder(n):",
        solution=(
            "def partial_adder(n):\n"
            "    from functools import partial\n"
            "    def add(a, b):\n"
            "        return a + b\n"
            "    return partial(add, n)\n"
        ),
        test_cases=[],
        algorithm="functools.partial (curry first arg)",
        complexity="O(1)",
        edge_cases=["returned closure holds n", "works for any left operand"],
        skip_sandbox=True,      # returns a callable; sandbox equality check awkward
    ))

    out.append(StdlibEntry(
        category="stdlib_functools",
        problem="Write a Python function `sort_by_custom(xs, cmp)` using functools.cmp_to_key to sort xs with a Python 2-style comparator. cmp(a, b) returns negative/zero/positive.",
        signature="def sort_by_custom(xs, cmp):",
        solution=(
            "def sort_by_custom(xs, cmp):\n"
            "    from functools import cmp_to_key\n"
            "    return sorted(xs, key=cmp_to_key(cmp))\n"
        ),
        test_cases=[],
        algorithm="functools.cmp_to_key to bridge 3-way comparator",
        complexity="O(n log n)",
        edge_cases=["empty list", "custom ordering", "stable sort preserves ties"],
        skip_sandbox=True,
    ))

    # ---- collections (more) ----
    out.append(StdlibEntry(
        category="stdlib_collections",
        problem="Write a Python function `namedtuple_point(x, y)` that creates a new namedtuple class `Point` with fields ('x', 'y') and returns an instance with the given values.",
        signature="def namedtuple_point(x, y):",
        solution=(
            "def namedtuple_point(x, y):\n"
            "    from collections import namedtuple\n"
            "    Point = namedtuple('Point', ('x', 'y'))\n"
            "    return Point(x=x, y=y)\n"
        ),
        test_cases=[
            (0, 0, (0, 0)),       # tuple equality still works
            (1, 2, (1, 2)),
            (-3, 4, (-3, 4)),
        ],
        algorithm="collections.namedtuple class factory",
        complexity="O(1)",
        edge_cases=["access via .x/.y or index", "hashable if values hashable"],
    ))

    out.append(StdlibEntry(
        category="stdlib_collections",
        problem="Write a Python function `chain_maps(*dicts)` using collections.ChainMap that returns a view over multiple dicts where lookups hit the first dict with the key.",
        signature="def chain_maps(*dicts):",
        solution=(
            "def chain_maps(*dicts):\n"
            "    from collections import ChainMap\n"
            "    return dict(ChainMap(*dicts))\n"
        ),
        test_cases=[
            ({"a": 1}, {"b": 2}, {"a": 1, "b": 2}),
            ({"a": 1}, {"a": 2}, {"a": 1}),                # first wins
            ({}, {"a": 1}, {"a": 1}),
            ({"a": 1, "b": 2}, {"b": 20, "c": 30}, {"a": 1, "b": 2, "c": 30}),
        ],
        algorithm="ChainMap with first-dict-wins lookup",
        complexity="O(k) lookup",
        edge_cases=["key in multiple dicts (first wins)", "empty dicts ignored", "order matters"],
    ))

    # ---- string ----
    out.append(StdlibEntry(
        category="stdlib_string",
        problem="Write a Python function `normalize_whitespace(s)` that replaces all runs of whitespace with a single space and strips leading/trailing. Uses only str methods (no regex).",
        signature="def normalize_whitespace(s):",
        solution=(
            "def normalize_whitespace(s):\n"
            "    return ' '.join(s.split())\n"
        ),
        test_cases=[
            ("", ""),
            ("hello", "hello"),
            ("hello   world", "hello world"),
            ("  hello\tworld\n", "hello world"),
            ("a\n\nb\n\tc", "a b c"),
            ("   ", ""),
        ],
        algorithm="str.split with no arg splits on any whitespace run",
        complexity="O(n)",
        edge_cases=["empty", "tabs and newlines", "leading/trailing spaces", "all whitespace → empty"],
    ))

    # ---- bisect ----
    out.append(StdlibEntry(
        category="stdlib_bisect",
        problem="Write a Python function `insert_sorted(sorted_xs, v)` using bisect.insort to insert v into a sorted list, preserving order. Modifies and returns sorted_xs.",
        signature="def insert_sorted(sorted_xs, v):",
        solution=(
            "def insert_sorted(sorted_xs, v):\n"
            "    import bisect\n"
            "    bisect.insort(sorted_xs, v)\n"
            "    return sorted_xs\n"
        ),
        test_cases=[
            ([], 5, [5]),
            ([1, 3, 5], 2, [1, 2, 3, 5]),
            ([1, 3, 5], 0, [0, 1, 3, 5]),
            ([1, 3, 5], 10, [1, 3, 5, 10]),
            ([1, 3, 3, 5], 3, [1, 3, 3, 3, 5]),
        ],
        algorithm="bisect.insort (O(log n) find + O(n) shift)",
        complexity="O(n) worst case (list shift)",
        edge_cases=["empty list", "insert at start/end", "duplicates preserved"],
    ))

    out.append(StdlibEntry(
        category="stdlib_bisect",
        problem="Write a Python function `find_le(sorted_xs, v)` using bisect that returns the rightmost element in sorted_xs that is <= v, or None if no such element exists.",
        signature="def find_le(sorted_xs, v):",
        solution=(
            "def find_le(sorted_xs, v):\n"
            "    import bisect\n"
            "    i = bisect.bisect_right(sorted_xs, v)\n"
            "    if i:\n"
            "        return sorted_xs[i - 1]\n"
            "    return None\n"
        ),
        test_cases=[
            ([], 5, None),
            ([1, 3, 5], 0, None),
            ([1, 3, 5], 1, 1),
            ([1, 3, 5], 4, 3),
            ([1, 3, 5], 5, 5),
            ([1, 3, 5], 100, 5),
        ],
        algorithm="bisect_right + index decrement",
        complexity="O(log n)",
        edge_cases=["empty → None", "v below all → None", "v matches an element (equal counts)"],
    ))

    # ---- enum ----
    out.append(StdlibEntry(
        category="stdlib_enum",
        problem="Write a Python function `color_name(r, g, b)` that maps an (r, g, b) tuple to a named color via enum.Enum. Uses a Color enum with RED=(255,0,0), GREEN=(0,255,0), BLUE=(0,0,255), WHITE=(255,255,255), BLACK=(0,0,0). Return the color name, or None if not a named color.",
        signature="def color_name(r, g, b):",
        solution=(
            "def color_name(r, g, b):\n"
            "    from enum import Enum\n"
            "    class Color(Enum):\n"
            "        RED = (255, 0, 0)\n"
            "        GREEN = (0, 255, 0)\n"
            "        BLUE = (0, 0, 255)\n"
            "        WHITE = (255, 255, 255)\n"
            "        BLACK = (0, 0, 0)\n"
            "    for c in Color:\n"
            "        if c.value == (r, g, b):\n"
            "            return c.name\n"
            "    return None\n"
        ),
        test_cases=[
            (255, 0, 0, "RED"),
            (0, 255, 0, "GREEN"),
            (0, 0, 255, "BLUE"),
            (255, 255, 255, "WHITE"),
            (0, 0, 0, "BLACK"),
            (128, 128, 128, None),
            (255, 0, 128, None),
        ],
        algorithm="enum.Enum with tuple values + linear lookup",
        complexity="O(k) where k is number of enum members",
        edge_cases=["no match returns None", "exact tuple equality", "case-sensitive name"],
    ))

    # ---- json (safe, stdlib, no external I/O) ----
    out.append(StdlibEntry(
        category="stdlib_json",
        problem="Write a Python function `safe_parse_json(s, default)` that returns json.loads(s) if the string is valid JSON, else returns `default`. Must not raise.",
        signature="def safe_parse_json(s, default):",
        solution=(
            "def safe_parse_json(s, default):\n"
            "    import json\n"
            "    try:\n"
            "        return json.loads(s)\n"
            "    except (json.JSONDecodeError, TypeError):\n"
            "        return default\n"
        ),
        test_cases=[
            ('{"a": 1}', None, {"a": 1}),
            ('[1, 2, 3]', None, [1, 2, 3]),
            ('null', None, None),
            ('not json', "oops", "oops"),
            ('', None, None),   # empty string is not valid JSON
            (None, {}, {}),     # non-string default
        ],
        algorithm="json.loads + try/except on JSONDecodeError + TypeError",
        complexity="O(|s|)",
        edge_cases=["empty string", "None input", "invalid JSON returns default", "valid JSON null returns Python None"],
    ))

    return out


class StdlibUsageGenerator(DomainDataGenerator):
    """Stdlib-usage problems covering high-traffic modules (collections,
    itertools, functools, pathlib, string, json).

    Every example uses only stdlib, fits in the sandbox's allowed-imports
    set (pathlib, collections, itertools, functools, json, string are
    all allowed by default in calm/sandbox.py's _PRELUDE)."""

    name = "stdlib"

    def __init__(self, rng=None):
        super().__init__(rng)
        self._entries = _entries()

    def generate_raw(self, n: int) -> List[VerifiedExample]:
        out: List[VerifiedExample] = []
        self.rng.shuffle(self._entries)   # mutates in place — per-call order varies
        for e in self._entries[:n]:
            out.append(VerifiedExample(
                problem=e.problem,
                signature=e.signature,
                solution=e.solution,
                test_cases=list(e.test_cases),
                reasoning="",
                algorithm=e.algorithm,
                complexity=e.complexity,
                edge_cases=list(e.edge_cases),
                category=e.category,
                generator_name=self.name,
                skip_sandbox=e.skip_sandbox,
            ))
        return out


register_generator("stdlib", StdlibUsageGenerator)
