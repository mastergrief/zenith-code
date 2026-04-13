"""
CALM algorithm tracing backend — verified algorithm execution.

The model writes "sorting [3,1,2] gives [1,2,3]" and Auto-CALM
verifies by running the actual algorithm on CPU.

Functions: sorting, searching, graph algorithms, combinatorics.
"""

from __future__ import annotations

import math
from itertools import combinations, permutations
from typing import List, Optional


# ---------------------------------------------------------------------------
# Sorting — returns both result and steps for explanation
# ---------------------------------------------------------------------------

def sort_list(data: list, reverse: bool = False) -> list:
    """Sort a list."""
    return sorted(data, reverse=bool(reverse))


def unique(data: list) -> list:
    """Remove duplicates, preserving order."""
    seen = set()
    result = []
    for x in data:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result


def binary_search(data: list, target) -> int:
    """Binary search on sorted list. Returns index or -1."""
    lo, hi = 0, len(data) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if data[mid] == target:
            return mid
        elif data[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


# ---------------------------------------------------------------------------
# Combinatorics
# ---------------------------------------------------------------------------

def nCr(n: int, r: int) -> int:
    """Combinations: n choose r."""
    n, r = int(n), int(r)
    if r < 0 or r > n:
        return 0
    return math.comb(n, r)


def nPr(n: int, r: int) -> int:
    """Permutations: n pick r."""
    n, r = int(n), int(r)
    if r < 0 or r > n:
        return 0
    return math.perm(n, r)


def list_combinations(data: list, r: int) -> list:
    """All combinations of r elements from data."""
    return [list(c) for c in combinations(data, int(r))]


def list_permutations(data: list, r: int = None) -> list:
    """All permutations of r elements from data."""
    r = int(r) if r is not None else len(data)
    return [list(p) for p in permutations(data, r)]


# ---------------------------------------------------------------------------
# Graph algorithms (adjacency list as dict)
# ---------------------------------------------------------------------------

def shortest_path(graph: dict, start, end) -> Optional[list]:
    """BFS shortest path in an unweighted graph.
    graph: {node: [neighbors]}
    """
    from collections import deque
    visited = {start}
    queue = deque([(start, [start])])
    while queue:
        node, path = queue.popleft()
        if node == end:
            return path
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None


def is_connected(graph: dict) -> bool:
    """Check if an undirected graph is connected."""
    if not graph:
        return True
    start = next(iter(graph))
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(graph.get(node, []))
    return len(visited) == len(graph)


def topological_sort(graph: dict) -> Optional[list]:
    """Topological sort of a DAG. Returns None if cycle detected.
    graph: {node: [dependencies]}
    """
    in_degree = {n: 0 for n in graph}
    for node in graph:
        for dep in graph[node]:
            in_degree[dep] = in_degree.get(dep, 0) + 1

    from collections import deque
    queue = deque(n for n, d in in_degree.items() if d == 0)
    result = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for dep in graph.get(node, []):
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)

    return result if len(result) == len(in_degree) else None


# ---------------------------------------------------------------------------
# Sequence operations
# ---------------------------------------------------------------------------

def cumsum(data: list) -> list:
    """Cumulative sum."""
    result = []
    s = 0
    for x in data:
        s += x
        result.append(s)
    return result


def running_max(data: list) -> list:
    """Running maximum."""
    result = []
    mx = float('-inf')
    for x in data:
        mx = max(mx, x)
        result.append(mx)
    return result


def longest_increasing_subsequence(data: list) -> int:
    """Length of the longest increasing subsequence."""
    if not data:
        return 0
    dp = []
    from bisect import bisect_left
    for x in data:
        pos = bisect_left(dp, x)
        if pos == len(dp):
            dp.append(x)
        else:
            dp[pos] = x
    return len(dp)


ALGO_FUNCTIONS = {
    "sort_list": sort_list,
    "unique": unique,
    "binary_search": binary_search,
    "nCr": nCr,
    "nPr": nPr,
    "list_combinations": list_combinations,
    "list_permutations": list_permutations,
    "shortest_path": shortest_path,
    "is_connected": is_connected,
    "topological_sort": topological_sort,
    "cumsum": cumsum,
    "running_max": running_max,
    "longest_increasing_subsequence": longest_increasing_subsequence,
}

ALGO_NL_PATTERNS = [
    (r'(\d+)\s+choose\s+(\d+)', 'nCr({0}, {1})'),
    (r'[Cc]\((\d+)\s*,\s*(\d+)\)', 'nCr({0}, {1})'),
    (r'(\d+)\s+permute\s+(\d+)', 'nPr({0}, {1})'),
    (r'[Pp]\((\d+)\s*,\s*(\d+)\)', 'nPr({0}, {1})'),
    (r'sort\s+\[([-\d.,\s]+)\]', 'sort_list([{0}])'),
    (r'unique\s+(?:values?\s+)?(?:in|of)\s+\[([-\d.,\s]+)\]', 'unique([{0}])'),
    (r'(?:longest increasing|LIS)\s+(?:subsequence\s+)?(?:of|in)\s+\[([-\d.,\s]+)\]', 'longest_increasing_subsequence([{0}])'),
]
