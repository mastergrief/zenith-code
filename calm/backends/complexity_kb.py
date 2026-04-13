"""
CALM Algorithm Complexity knowledge backend.

The single most hallucinated CS topic. Models confidently say quicksort
is O(n log n) worst case, hash table lookup is O(1) always, etc.
"""

from __future__ import annotations

# (best, average, worst, space, stable, notes)
_SORT_ALGORITHMS = {
    "quicksort": ("O(n log n)", "O(n log n)", "O(n²)", "O(log n)", False, "Worst case on already-sorted with bad pivot"),
    "mergesort": ("O(n log n)", "O(n log n)", "O(n log n)", "O(n)", True, "Guaranteed O(n log n) but not in-place"),
    "merge sort": ("O(n log n)", "O(n log n)", "O(n log n)", "O(n)", True, "Guaranteed O(n log n) but not in-place"),
    "heapsort": ("O(n log n)", "O(n log n)", "O(n log n)", "O(1)", False, "In-place, not stable"),
    "heap sort": ("O(n log n)", "O(n log n)", "O(n log n)", "O(1)", False, "In-place, not stable"),
    "timsort": ("O(n)", "O(n log n)", "O(n log n)", "O(n)", True, "Python/Java default, hybrid merge+insertion"),
    "insertion sort": ("O(n)", "O(n²)", "O(n²)", "O(1)", True, "Best for small/nearly-sorted arrays"),
    "bubble sort": ("O(n)", "O(n²)", "O(n²)", "O(1)", True, "Simple but slow"),
    "selection sort": ("O(n²)", "O(n²)", "O(n²)", "O(1)", False, "Always O(n²), minimal swaps"),
    "counting sort": ("O(n + k)", "O(n + k)", "O(n + k)", "O(k)", True, "k = range of input values"),
    "radix sort": ("O(nk)", "O(nk)", "O(nk)", "O(n + k)", True, "k = number of digits"),
    "bucket sort": ("O(n + k)", "O(n + k)", "O(n²)", "O(n)", True, "Assumes uniform distribution"),
    "shell sort": ("O(n log n)", "O(n^(4/3))", "O(n^(3/2))", "O(1)", False, "Gap-dependent complexity"),
}

# (average, worst, notes)
_DS_OPERATIONS = {
    "array_access": ("O(1)", "O(1)", "Direct index access"),
    "array_search": ("O(n)", "O(n)", "Linear scan"),
    "array_insert": ("O(n)", "O(n)", "Shift elements right"),
    "array_delete": ("O(n)", "O(n)", "Shift elements left"),
    "array_append": ("O(1)*", "O(n)", "Amortized O(1), worst case resize"),
    "linked_list_access": ("O(n)", "O(n)", "Must traverse from head"),
    "linked_list_search": ("O(n)", "O(n)", "Must traverse"),
    "linked_list_insert": ("O(1)", "O(1)", "At known position (head/tail)"),
    "linked_list_delete": ("O(1)", "O(1)", "At known position"),
    "hash_table_search": ("O(1)*", "O(n)", "Amortized O(1), worst case all collide"),
    "hash_table_insert": ("O(1)*", "O(n)", "Amortized O(1)"),
    "hash_table_delete": ("O(1)*", "O(n)", "Amortized O(1)"),
    "bst_search": ("O(log n)", "O(n)", "Worst case: degenerate (linked list)"),
    "bst_insert": ("O(log n)", "O(n)", "Worst case: degenerate"),
    "bst_delete": ("O(log n)", "O(n)", "Worst case: degenerate"),
    "balanced_bst_search": ("O(log n)", "O(log n)", "AVL/Red-Black guaranteed"),
    "balanced_bst_insert": ("O(log n)", "O(log n)", "With rebalancing"),
    "balanced_bst_delete": ("O(log n)", "O(log n)", "With rebalancing"),
    "heap_insert": ("O(log n)", "O(log n)", "Sift up"),
    "heap_delete_min": ("O(log n)", "O(log n)", "Sift down"),
    "heap_find_min": ("O(1)", "O(1)", "Root element"),
    "binary_search": ("O(log n)", "O(log n)", "Sorted array required"),
    "stack_push": ("O(1)", "O(1)", ""),
    "stack_pop": ("O(1)", "O(1)", ""),
    "queue_enqueue": ("O(1)", "O(1)", ""),
    "queue_dequeue": ("O(1)", "O(1)", ""),
}

_GRAPH_ALGORITHMS = {
    "bfs": ("O(V + E)", "O(V)", "Breadth-first search"),
    "dfs": ("O(V + E)", "O(V)", "Depth-first search"),
    "dijkstra": ("O((V + E) log V)", "O(V)", "With binary heap; no negative edges"),
    "bellman_ford": ("O(VE)", "O(V)", "Handles negative edges"),
    "floyd_warshall": ("O(V³)", "O(V²)", "All-pairs shortest paths"),
    "kruskal": ("O(E log E)", "O(V + E)", "Minimum spanning tree"),
    "prim": ("O((V + E) log V)", "O(V)", "MST with binary heap"),
    "topological_sort": ("O(V + E)", "O(V)", "DAG only"),
    "a_star": ("O(E)", "O(V)", "Depends on heuristic quality"),
}


def sort_complexity(algorithm: str) -> str:
    """Time and space complexity for a sorting algorithm."""
    key = algorithm.strip().lower().replace("-", " ")
    data = _SORT_ALGORITHMS.get(key)
    if not data:
        return f"unknown sort algorithm: {algorithm}"
    best, avg, worst, space, stable, notes = data
    stability = "stable" if stable else "unstable"
    return (f"{algorithm}: best={best}, avg={avg}, worst={worst}, "
            f"space={space}, {stability}. {notes}")


def ds_complexity(operation: str) -> str:
    """Time complexity for a data structure operation."""
    key = operation.strip().lower().replace(" ", "_").replace("-", "_")
    data = _DS_OPERATIONS.get(key)
    if not data:
        return f"unknown operation: {operation}"
    avg, worst, notes = data
    result = f"{operation}: avg={avg}, worst={worst}"
    if notes:
        result += f". {notes}"
    return result


def graph_complexity(algorithm: str) -> str:
    """Time and space complexity for a graph algorithm."""
    key = algorithm.strip().lower().replace(" ", "_").replace("-", "_").replace("'", "")
    data = _GRAPH_ALGORITHMS.get(key)
    if not data:
        return f"unknown graph algorithm: {algorithm}"
    time_c, space, notes = data
    return f"{algorithm}: time={time_c}, space={space}. {notes}"


def is_stable_sort(algorithm: str) -> bool:
    """Whether a sorting algorithm is stable."""
    key = algorithm.strip().lower().replace("-", " ")
    data = _SORT_ALGORITHMS.get(key)
    return data[4] if data else False


def worst_case(algorithm: str) -> str:
    """Worst-case time complexity for any known algorithm."""
    key = algorithm.strip().lower().replace("-", " ")
    if key in _SORT_ALGORITHMS:
        return _SORT_ALGORITHMS[key][2]
    key2 = key.replace(" ", "_")
    if key2 in _DS_OPERATIONS:
        return _DS_OPERATIONS[key2][1]
    if key2 in _GRAPH_ALGORITHMS:
        return _GRAPH_ALGORITHMS[key2][0]
    return f"unknown algorithm: {algorithm}"


COMPLEXITY_FUNCTIONS = {
    "sort_complexity": sort_complexity,
    "ds_complexity": ds_complexity,
    "graph_complexity": graph_complexity,
    "is_stable_sort": is_stable_sort,
    "worst_case": worst_case,
}
