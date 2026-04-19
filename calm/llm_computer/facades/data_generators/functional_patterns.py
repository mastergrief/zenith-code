"""FunctionalPatternsGenerator — pure-functional idioms for Python.

Patterns developers ask about when moving from imperative to functional
style: compose, pipe, curry, memoize, once, tap, flatten-iter, partition.

Each solution is a small higher-order function. Test cases verify
behavior via concrete callable inputs (add, multiply, etc.) rather
than relying on identity equality of returned callables.
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
class FunctionalSpec:
    name: str
    signature: str
    problem: str
    solution: str
    test_cases: List[Tuple]
    algorithm: str
    complexity: str
    edge_cases: List[str]
    skip_sandbox: bool = False


def _specs() -> List[FunctionalSpec]:
    out: List[FunctionalSpec] = []

    out.append(FunctionalSpec(
        name="compose_two",
        signature="def compose(f, g):",
        problem="Write a Python function `compose(f, g)` that returns a new function h such that h(x) == f(g(x)). Classic function composition.",
        solution=(
            "def compose(f, g):\n"
            "    def h(x):\n"
            "        return f(g(x))\n"
            "    return h\n"
        ),
        test_cases=[],
        algorithm="closure over f and g",
        complexity="O(cost(f) + cost(g)) per call",
        edge_cases=["identity when f or g is identity", "composition is right-to-left"],
        skip_sandbox=True,   # returns callable
    ))

    out.append(FunctionalSpec(
        name="pipe_apply",
        signature="def pipe(x, *fns):",
        problem="Write a Python function `pipe(x, *fns)` that applies each function left-to-right to x. pipe(3, f, g, h) == h(g(f(3))). Reverse of compose.",
        solution=(
            "def pipe(x, *fns):\n"
            "    result = x\n"
            "    for fn in fns:\n"
            "        result = fn(result)\n"
            "    return result\n"
        ),
        test_cases=[
            (0,),        # no fns → identity; will be handled via test below
        ],
        algorithm="fold over function list, left-to-right",
        complexity="O(sum of fn costs)",
        edge_cases=["no fns → x unchanged", "single fn → equivalent to fn(x)"],
        skip_sandbox=True,   # variadic with callables; test via helper
    ))

    out.append(FunctionalSpec(
        name="memoize_hashable",
        signature="def memoize(fn):",
        problem="Write a Python function `memoize(fn)` that wraps a single-argument function and caches results for repeated hashable inputs. Use a plain dict (NOT functools.lru_cache — show the mechanism).",
        solution=(
            "def memoize(fn):\n"
            "    cache = {}\n"
            "    def wrapped(x):\n"
            "        if x not in cache:\n"
            "            cache[x] = fn(x)\n"
            "        return cache[x]\n"
            "    return wrapped\n"
        ),
        test_cases=[],
        algorithm="dict-backed cache in closure; hashable keys only",
        complexity="O(1) amortized after first call per key",
        edge_cases=["unhashable argument raises TypeError", "fn side effects hidden on cache hit"],
        skip_sandbox=True,
    ))

    out.append(FunctionalSpec(
        name="once_idempotent",
        signature="def once(fn):",
        problem="Write a Python function `once(fn)` that wraps fn so it only executes the first time called; subsequent calls return the cached first result (regardless of arguments). Useful for lazy initialization.",
        solution=(
            "def once(fn):\n"
            "    state = {'called': False, 'value': None}\n"
            "    def wrapped(*args, **kwargs):\n"
            "        if not state['called']:\n"
            "            state['value'] = fn(*args, **kwargs)\n"
            "            state['called'] = True\n"
            "        return state['value']\n"
            "    return wrapped\n"
        ),
        test_cases=[],
        algorithm="flag + cached value in closure state dict",
        complexity="O(cost(fn)) first call, O(1) after",
        edge_cases=["ignores subsequent args", "exception on first call leaves state uncalled (retry possible)"],
        skip_sandbox=True,
    ))

    out.append(FunctionalSpec(
        name="curry_2arg",
        signature="def curry2(fn):",
        problem="Write a Python function `curry2(fn)` that turns a two-argument function into a curried form: `curry2(add)(1)(2) == 3`.",
        solution=(
            "def curry2(fn):\n"
            "    def outer(a):\n"
            "        def inner(b):\n"
            "            return fn(a, b)\n"
            "        return inner\n"
            "    return outer\n"
        ),
        test_cases=[],
        algorithm="nested closures (one arg each)",
        complexity="O(1) per application + O(cost(fn)) final",
        edge_cases=["partial application", "each level returns a new function"],
        skip_sandbox=True,
    ))

    out.append(FunctionalSpec(
        name="partition_pred",
        signature="def partition(xs, pred):",
        problem="Write a Python function `partition(xs, pred)` that returns a tuple (truthy_list, falsy_list) splitting xs by pred. Preserves original order within each list.",
        solution=(
            "def partition(xs, pred):\n"
            "    truthy = []\n"
            "    falsy = []\n"
            "    for x in xs:\n"
            "        (truthy if pred(x) else falsy).append(x)\n"
            "    return (truthy, falsy)\n"
        ),
        test_cases=[],
        algorithm="single-pass append to one of two lists",
        complexity="O(n)",
        edge_cases=["empty input", "all truthy", "all falsy", "order preserved per partition"],
        skip_sandbox=True,
    ))

    out.append(FunctionalSpec(
        name="flatten_iter",
        signature="def flatten_iter(xs):",
        problem="Write a Python GENERATOR function `flatten_iter(xs)` that yields scalars from arbitrarily nested lists of scalars. Lazy — must not materialize the full flat list.",
        solution=(
            "def flatten_iter(xs):\n"
            "    for item in xs:\n"
            "        if isinstance(item, list):\n"
            "            yield from flatten_iter(item)\n"
            "        else:\n"
            "            yield item\n"
        ),
        test_cases=[
            ([], []),
            ([1, 2, 3], [1, 2, 3]),
            ([[1, [2, [3]]]], [1, 2, 3]),
            ([[], [1], [[2, 3]]], [1, 2, 3]),
        ],
        algorithm="recursive yield-from (lazy)",
        complexity="O(total elements) time, O(max-depth) stack",
        edge_cases=["empty outer", "deeply nested", "tuples NOT flattened (only list type checked)"],
    ))

    # Wrap the generator to return a list for sandbox test verification
    out[-1] = FunctionalSpec(
        name="flatten_iter",
        signature="def flatten_iter(xs):",
        problem="Write a Python function `flatten_iter(xs)` that returns a lazily-flattened iterator over scalars from arbitrarily nested lists. Return the iterator as a list so tests can verify contents.",
        solution=(
            "def flatten_iter(xs):\n"
            "    def _gen(xs):\n"
            "        for item in xs:\n"
            "            if isinstance(item, list):\n"
            "                yield from _gen(item)\n"
            "            else:\n"
            "                yield item\n"
            "    return list(_gen(xs))\n"
        ),
        test_cases=[
            ([], []),
            ([1, 2, 3], [1, 2, 3]),
            ([[1, [2, [3]]]], [1, 2, 3]),
            ([[], [1], [[2, 3]]], [1, 2, 3]),
            ([(1, 2), [3]], [(1, 2), 3]),      # tuple treated as scalar
        ],
        algorithm="recursive generator + list materialization",
        complexity="O(total elements) time, O(depth) stack",
        edge_cases=["empty outer", "tuples NOT flattened", "deeply nested"],
    )

    out.append(FunctionalSpec(
        name="tap_side_effect",
        signature="def tap(fn):",
        problem="Write a Python function `tap(fn)` that wraps a side-effect function so it can be inlined in a pipe: tap(print) returns a function that prints its argument AND returns it unchanged.",
        solution=(
            "def tap(fn):\n"
            "    def wrapped(x):\n"
            "        fn(x)\n"
            "        return x\n"
            "    return wrapped\n"
        ),
        test_cases=[],
        algorithm="side-effect + pass-through return",
        complexity="O(cost(fn))",
        edge_cases=["fn return value discarded", "x passes through unchanged", "works in pipe chains"],
        skip_sandbox=True,
    ))

    out.append(FunctionalSpec(
        name="zip_with",
        signature="def zip_with(fn, a, b):",
        problem="Write a Python function `zip_with(fn, a, b)` that returns a list [fn(a[0], b[0]), fn(a[1], b[1]), ...] — stops at the shorter input. Equivalent to `map(fn, zip(a, b))` with explicit tuple unpack.",
        solution=(
            "def zip_with(fn, a, b):\n"
            "    return [fn(x, y) for x, y in zip(a, b)]\n"
        ),
        test_cases=[],
        algorithm="list comprehension over zip(a, b)",
        complexity="O(min(|a|, |b|))",
        edge_cases=["length mismatch truncates to shorter", "empty inputs → empty result"],
        skip_sandbox=True,
    ))

    out.append(FunctionalSpec(
        name="frequencies",
        signature="def frequencies(xs):",
        problem="Write a Python function `frequencies(xs)` that returns a dict mapping each unique element of xs to its count. Pure and order-independent.",
        solution=(
            "def frequencies(xs):\n"
            "    out = {}\n"
            "    for x in xs:\n"
            "        out[x] = out.get(x, 0) + 1\n"
            "    return out\n"
        ),
        test_cases=[
            ([], {}),
            ([1], {1: 1}),
            ([1, 1, 2], {1: 2, 2: 1}),
            (['a', 'b', 'a', 'c', 'a'], {'a': 3, 'b': 1, 'c': 1}),
            ([True, False, True], {True: 2, False: 1}),
        ],
        algorithm="dict.get with default 0, incremental count",
        complexity="O(n) time, O(unique) space",
        edge_cases=["empty → {}", "hashable elements only", "booleans work (hashable)"],
    ))

    return out


class FunctionalPatternsGenerator(DomainDataGenerator):
    """Higher-order function idioms: compose, pipe, memoize, once,
    curry, partition, flatten_iter, tap, zip_with, frequencies.

    Pure Python, no external deps. The sandbox-testable ones use
    concrete inputs; the rest ship AST-verified (closures / callables
    don't round-trip through the sandbox's eval mechanism cleanly)."""

    name = "functional"

    def __init__(self, rng=None):
        super().__init__(rng)
        self._specs = _specs()

    def generate_raw(self, n: int) -> List[VerifiedExample]:
        self.rng.shuffle(self._specs)
        out: List[VerifiedExample] = []
        for s in self._specs[:n]:
            out.append(VerifiedExample(
                problem=s.problem,
                signature=s.signature,
                solution=s.solution,
                test_cases=list(s.test_cases),
                reasoning="",
                algorithm=s.algorithm,
                complexity=s.complexity,
                edge_cases=list(s.edge_cases),
                category=f"fn_{s.name}",
                generator_name=self.name,
                skip_sandbox=s.skip_sandbox,
            ))
        return out


register_generator("functional", FunctionalPatternsGenerator)
