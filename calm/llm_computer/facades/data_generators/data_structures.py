"""DataStructuresGenerator — classic DS problems with verified solutions.

Linked lists, stacks, queues, BSTs, heaps, tries — the patterns
Gemma frequently sees and sometimes gets subtly wrong (null-pointer
handling, iterative vs recursive, stability, etc.).

All examples are pure Python stdlib + heapq (which the sandbox allows).
Solutions are minimal and idiomatic.
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
class DSSpec:
    name: str
    signature: str
    problem: str
    solution: str
    test_cases: List[Tuple]
    algorithm: str
    complexity: str
    edge_cases: List[str]


def _specs() -> List[DSSpec]:
    out: List[DSSpec] = []

    out.append(DSSpec(
        name="stack_push_pop",
        signature="def run_stack_ops(ops):",
        problem="Write a Python function `run_stack_ops(ops)` that processes a sequence of stack operations on a LIFO stack. Each op is a tuple: ('push', x) appends x, ('pop',) removes and returns the top. Return the list of all values returned by pop operations (in order). Raise IndexError on pop from empty.",
        solution=(
            "def run_stack_ops(ops):\n"
            "    stack = []\n"
            "    results = []\n"
            "    for op in ops:\n"
            "        if op[0] == 'push':\n"
            "            stack.append(op[1])\n"
            "        elif op[0] == 'pop':\n"
            "            if not stack:\n"
            "                raise IndexError('pop from empty stack')\n"
            "            results.append(stack.pop())\n"
            "    return results\n"
        ),
        test_cases=[
            ([], []),
            ([('push', 1), ('push', 2), ('pop',)], [2]),
            ([('push', 1), ('push', 2), ('pop',), ('pop',)], [2, 1]),
            ([('push', 'a'), ('push', 'b'), ('push', 'c'), ('pop',)], ['c']),
        ],
        algorithm="list as stack (append/pop from end)",
        complexity="O(1) amortized per op",
        edge_cases=["empty ops list", "pop from empty raises", "LIFO ordering"],
    ))

    out.append(DSSpec(
        name="queue_fifo",
        signature="def run_queue_ops(ops):",
        problem="Write a Python function `run_queue_ops(ops)` that processes FIFO queue operations using collections.deque. ('enq', x) adds to back, ('deq',) removes and returns from front. Return list of values returned by deq. Raise IndexError on deq from empty.",
        solution=(
            "def run_queue_ops(ops):\n"
            "    from collections import deque\n"
            "    q = deque()\n"
            "    results = []\n"
            "    for op in ops:\n"
            "        if op[0] == 'enq':\n"
            "            q.append(op[1])\n"
            "        elif op[0] == 'deq':\n"
            "            if not q:\n"
            "                raise IndexError('deq from empty queue')\n"
            "            results.append(q.popleft())\n"
            "    return results\n"
        ),
        test_cases=[
            ([], []),
            ([('enq', 1), ('enq', 2), ('deq',)], [1]),
            ([('enq', 1), ('enq', 2), ('deq',), ('deq',)], [1, 2]),
            ([('enq', 'a'), ('enq', 'b'), ('enq', 'c'), ('deq',)], ['a']),
        ],
        algorithm="deque for O(1) both-ends",
        complexity="O(1) per op",
        edge_cases=["FIFO not LIFO", "deq from empty raises", "empty ops"],
    ))

    out.append(DSSpec(
        name="reverse_linked_list",
        signature="def reverse_list(xs):",
        problem="Write a Python function `reverse_list(xs)` that reverses a list IN PLACE (modifies the input) without using list.reverse() or slicing. Returns the same list reference.",
        solution=(
            "def reverse_list(xs):\n"
            "    i, j = 0, len(xs) - 1\n"
            "    while i < j:\n"
            "        xs[i], xs[j] = xs[j], xs[i]\n"
            "        i += 1\n"
            "        j -= 1\n"
            "    return xs\n"
        ),
        test_cases=[
            ([], []),
            ([1], [1]),
            ([1, 2], [2, 1]),
            ([1, 2, 3], [3, 2, 1]),
            ([1, 2, 3, 4], [4, 3, 2, 1]),
            (['a', 'b', 'c'], ['c', 'b', 'a']),
        ],
        algorithm="two-pointer in-place swap",
        complexity="O(n) time, O(1) space",
        edge_cases=["empty", "single element", "odd vs even length"],
    ))

    out.append(DSSpec(
        name="bst_insert_inorder",
        signature="def bst_inorder(values):",
        problem="Write a Python function `bst_inorder(values)` that builds a binary search tree from the list of values (inserting one at a time), then returns an in-order traversal as a list. Duplicates go to the right subtree. Represent nodes as dicts `{'val', 'l', 'r'}`.",
        solution=(
            "def bst_inorder(values):\n"
            "    root = None\n"
            "    def insert(node, v):\n"
            "        if node is None:\n"
            "            return {'val': v, 'l': None, 'r': None}\n"
            "        if v < node['val']:\n"
            "            node['l'] = insert(node['l'], v)\n"
            "        else:\n"
            "            node['r'] = insert(node['r'], v)\n"
            "        return node\n"
            "    for v in values:\n"
            "        root = insert(root, v)\n"
            "    out = []\n"
            "    stack = []\n"
            "    cur = root\n"
            "    while stack or cur is not None:\n"
            "        while cur is not None:\n"
            "            stack.append(cur)\n"
            "            cur = cur['l']\n"
            "        cur = stack.pop()\n"
            "        out.append(cur['val'])\n"
            "        cur = cur['r']\n"
            "    return out\n"
        ),
        test_cases=[
            ([], []),
            ([5], [5]),
            ([5, 3, 7], [3, 5, 7]),
            ([5, 3, 7, 1, 4, 6, 8], [1, 3, 4, 5, 6, 7, 8]),
            ([3, 3, 3], [3, 3, 3]),
            ([5, 3, 5, 7], [3, 5, 5, 7]),   # dup goes right
        ],
        algorithm="BST insert + iterative in-order traversal (stack-based)",
        complexity="O(n log n) avg insert, O(n) traversal",
        edge_cases=["empty → empty", "single node", "duplicates right-biased", "iterative (no recursion depth issue)"],
    ))

    out.append(DSSpec(
        name="min_heap_top_k",
        signature="def smallest_k(xs, k):",
        problem="Write a Python function `smallest_k(xs, k)` that returns the k smallest elements of xs in SORTED order using a min-heap (heapq). Raise ValueError if k > len(xs) or k < 0.",
        solution=(
            "def smallest_k(xs, k):\n"
            "    import heapq\n"
            "    if k < 0 or k > len(xs):\n"
            "        raise ValueError('k out of range')\n"
            "    if k == 0:\n"
            "        return []\n"
            "    return heapq.nsmallest(k, xs)\n"
        ),
        test_cases=[
            ([1, 2, 3, 4, 5], 3, [1, 2, 3]),
            ([5, 3, 1, 4, 2], 2, [1, 2]),
            ([], 0, []),
            ([7], 1, [7]),
            ([3, 3, 3], 2, [3, 3]),
        ],
        algorithm="heapq.nsmallest (min-heap)",
        complexity="O(n log k)",
        edge_cases=["k = 0 returns empty", "k > len raises", "duplicates preserved"],
    ))

    out.append(DSSpec(
        name="trie_word_exists",
        signature="def trie_lookup(words, queries):",
        problem="Write a Python function `trie_lookup(words, queries)` that builds a prefix trie from `words` and returns a list of booleans — one per query — True if the query is an EXACT word in the trie. Represent the trie as nested dicts with a sentinel key '$' marking word ends.",
        solution=(
            "def trie_lookup(words, queries):\n"
            "    trie = {}\n"
            "    for w in words:\n"
            "        node = trie\n"
            "        for c in w:\n"
            "            node = node.setdefault(c, {})\n"
            "        node['$'] = True\n"
            "    results = []\n"
            "    for q in queries:\n"
            "        node = trie\n"
            "        for c in q:\n"
            "            if c not in node:\n"
            "                node = None\n"
            "                break\n"
            "            node = node[c]\n"
            "        results.append(node is not None and '$' in node)\n"
            "    return results\n"
        ),
        test_cases=[
            ([], [""], [False]),
            (["cat", "car", "care"], ["cat", "ca", "care", "dog"],
             [True, False, True, False]),
            (["a"], ["a", "aa"], [True, False]),
            (["hello"], ["hello", "hell"], [True, False]),
        ],
        algorithm="nested-dict trie with '$' end-marker",
        complexity="O(Σ|w|) build, O(|q|) per query",
        edge_cases=["prefix of a word is NOT a word", "empty words list", "duplicate insert ok"],
    ))

    out.append(DSSpec(
        name="union_find",
        signature="def count_components(n, edges):",
        problem="Write a Python function `count_components(n, edges)` that counts connected components in an undirected graph with n nodes (0..n-1) and a list of (u, v) edges using Union-Find (DSU) with path compression.",
        solution=(
            "def count_components(n, edges):\n"
            "    parent = list(range(n))\n"
            "    def find(x):\n"
            "        while parent[x] != x:\n"
            "            parent[x] = parent[parent[x]]   # path compression\n"
            "            x = parent[x]\n"
            "        return x\n"
            "    def union(a, b):\n"
            "        ra, rb = find(a), find(b)\n"
            "        if ra != rb:\n"
            "            parent[ra] = rb\n"
            "            return True\n"
            "        return False\n"
            "    for u, v in edges:\n"
            "        union(u, v)\n"
            "    return len({find(i) for i in range(n)})\n"
        ),
        test_cases=[
            (0, [], 0),
            (1, [], 1),
            (3, [], 3),
            (3, [(0, 1), (1, 2)], 1),
            (4, [(0, 1), (2, 3)], 2),
            (5, [(0, 1), (1, 2), (3, 4)], 2),
            (4, [(0, 0), (1, 1)], 4),     # self-loops don't merge
        ],
        algorithm="Union-Find with path compression",
        complexity="O((n + |E|) α(n)) ≈ O(n + |E|)",
        edge_cases=["n=0 returns 0", "disjoint nodes", "self-loops", "multi-edges"],
    ))

    out.append(DSSpec(
        name="lru_cache_class",
        signature="def run_lru(capacity, ops):",
        problem="Write a Python function `run_lru(capacity, ops)` that simulates an LRU cache using collections.OrderedDict. Ops are tuples: ('put', k, v) or ('get', k). Return list of values returned by 'get' (None if not present). On 'put' beyond capacity, evict the least recently used entry.",
        solution=(
            "def run_lru(capacity, ops):\n"
            "    from collections import OrderedDict\n"
            "    cache = OrderedDict()\n"
            "    results = []\n"
            "    for op in ops:\n"
            "        if op[0] == 'put':\n"
            "            _, k, v = op\n"
            "            if k in cache:\n"
            "                cache.move_to_end(k)\n"
            "            cache[k] = v\n"
            "            if len(cache) > capacity:\n"
            "                cache.popitem(last=False)\n"
            "        elif op[0] == 'get':\n"
            "            _, k = op\n"
            "            if k in cache:\n"
            "                cache.move_to_end(k)\n"
            "                results.append(cache[k])\n"
            "            else:\n"
            "                results.append(None)\n"
            "    return results\n"
        ),
        test_cases=[
            (2, [], []),
            (2, [('put', 1, 'a'), ('get', 1)], ['a']),
            (2, [('put', 1, 'a'), ('put', 2, 'b'), ('put', 3, 'c'), ('get', 1)], [None]),
            (2, [('put', 1, 'a'), ('get', 1), ('put', 2, 'b'), ('put', 3, 'c'), ('get', 1)], ['a', 'a']),
            (1, [('put', 1, 'x'), ('put', 2, 'y'), ('get', 1), ('get', 2)], [None, 'y']),
        ],
        algorithm="OrderedDict.move_to_end + popitem(last=False) for LRU",
        complexity="O(1) per op",
        edge_cases=["overflow evicts LRU", "get refreshes recency", "capacity = 1"],
    ))

    out.append(DSSpec(
        name="count_islands",
        signature="def count_islands(grid):",
        problem="Write a Python function `count_islands(grid)` that counts connected components of '1' cells in a 2D grid of 0s and 1s. Cells connect via 4-directional (up/down/left/right) adjacency. Use BFS with a visited set.",
        solution=(
            "def count_islands(grid):\n"
            "    if not grid or not grid[0]:\n"
            "        return 0\n"
            "    from collections import deque\n"
            "    rows, cols = len(grid), len(grid[0])\n"
            "    visited = set()\n"
            "    count = 0\n"
            "    for r in range(rows):\n"
            "        for c in range(cols):\n"
            "            if grid[r][c] != 1 or (r, c) in visited:\n"
            "                continue\n"
            "            count += 1\n"
            "            q = deque([(r, c)])\n"
            "            visited.add((r, c))\n"
            "            while q:\n"
            "                cr, cc = q.popleft()\n"
            "                for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):\n"
            "                    nr, nc = cr+dr, cc+dc\n"
            "                    if 0 <= nr < rows and 0 <= nc < cols \\\n"
            "                        and grid[nr][nc] == 1 and (nr, nc) not in visited:\n"
            "                        visited.add((nr, nc))\n"
            "                        q.append((nr, nc))\n"
            "    return count\n"
        ),
        test_cases=[
            ([], 0),
            ([[]], 0),
            ([[1]], 1),
            ([[0]], 0),
            ([[1, 0], [0, 1]], 2),
            ([[1, 1], [1, 1]], 1),
            ([[1, 0, 1], [0, 0, 0], [1, 0, 1]], 4),
            ([[1, 1, 0], [0, 1, 0], [0, 0, 1]], 2),
        ],
        algorithm="BFS per unvisited land cell; 4-directional neighbors",
        complexity="O(rows × cols)",
        edge_cases=["empty grid", "all water", "all land (one island)", "checkerboard"],
    ))

    out.append(DSSpec(
        name="detect_cycle_dg",
        signature="def has_cycle(n, edges):",
        problem="Write a Python function `has_cycle(n, edges)` that returns True if the directed graph with n nodes and `edges` (list of (u, v)) contains a cycle. Use DFS with a 3-color (white/gray/black) coloring.",
        solution=(
            "def has_cycle(n, edges):\n"
            "    adj = [[] for _ in range(n)]\n"
            "    for u, v in edges:\n"
            "        adj[u].append(v)\n"
            "    WHITE, GRAY, BLACK = 0, 1, 2\n"
            "    color = [WHITE] * n\n"
            "    def dfs(u):\n"
            "        color[u] = GRAY\n"
            "        for v in adj[u]:\n"
            "            if color[v] == GRAY:\n"
            "                return True\n"
            "            if color[v] == WHITE and dfs(v):\n"
            "                return True\n"
            "        color[u] = BLACK\n"
            "        return False\n"
            "    for i in range(n):\n"
            "        if color[i] == WHITE and dfs(i):\n"
            "            return True\n"
            "    return False\n"
        ),
        test_cases=[
            (0, [], False),
            (1, [], False),
            (1, [(0, 0)], True),                # self-loop
            (3, [(0, 1), (1, 2)], False),
            (3, [(0, 1), (1, 2), (2, 0)], True),
            (4, [(0, 1), (1, 2), (2, 3)], False),
            (4, [(0, 1), (1, 2), (2, 1)], True),
        ],
        algorithm="DFS with white/gray/black coloring (back-edge detection)",
        complexity="O(n + |E|)",
        edge_cases=["self-loop counts as cycle", "disconnected components", "multi-edge graphs"],
    ))

    return out


def _build_example(s: DSSpec) -> VerifiedExample:
    return VerifiedExample(
        problem=s.problem,
        signature=s.signature,
        solution=s.solution,
        test_cases=list(s.test_cases),
        reasoning="",
        algorithm=s.algorithm,
        complexity=s.complexity,
        edge_cases=list(s.edge_cases),
        category=f"ds_{s.name}",
        generator_name="data_structures",
    )


class DataStructuresGenerator(DomainDataGenerator):
    """Classic data-structure problems: stack, queue, linked-list,
    BST, heap, trie, union-find, LRU cache, graph components, cycle
    detection. All sandbox-verified Python."""

    name = "data_structures"

    def __init__(self, rng=None):
        super().__init__(rng)
        self._specs = _specs()

    def generate_raw(self, n: int) -> List[VerifiedExample]:
        self.rng.shuffle(self._specs)
        return [_build_example(s) for s in self._specs[:n]]


register_generator("data_structures", DataStructuresGenerator)
