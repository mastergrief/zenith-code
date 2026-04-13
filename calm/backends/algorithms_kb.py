"""
CALM Algorithms knowledge backend — search, dynamic programming, greedy, graph algorithms.

Models confuse algorithm categories and complexities. Reference data.
"""

from __future__ import annotations

_DATA_VERSION = "2025-01"

_SEARCH_ALGORITHMS = {
    "linear search": {"time_best": "O(1)", "time_avg": "O(n)", "time_worst": "O(n)", "space": "O(1)", "requires": "none", "use": "small/unsorted data"},
    "binary search": {"time_best": "O(1)", "time_avg": "O(log n)", "time_worst": "O(log n)", "space": "O(1)", "requires": "sorted array", "use": "large sorted data"},
    "interpolation search": {"time_best": "O(1)", "time_avg": "O(log log n)", "time_worst": "O(n)", "space": "O(1)", "requires": "sorted, uniformly distributed", "use": "uniformly distributed sorted data"},
    "exponential search": {"time_best": "O(1)", "time_avg": "O(log n)", "time_worst": "O(log n)", "space": "O(1)", "requires": "sorted array", "use": "unbounded/infinite lists"},
    "jump search": {"time_best": "O(1)", "time_avg": "O(√n)", "time_worst": "O(√n)", "space": "O(1)", "requires": "sorted array", "use": "jumping is cheaper than comparison"},
    "ternary search": {"time_best": "O(1)", "time_avg": "O(log₃ n)", "time_worst": "O(log₃ n)", "space": "O(1)", "requires": "unimodal function", "use": "finding extrema of unimodal functions"},
}

_DP_PROBLEMS = {
    "fibonacci": {"complexity": "O(n)", "naive": "O(2^n)", "technique": "memoization or tabulation", "recurrence": "F(n) = F(n-1) + F(n-2)"},
    "knapsack 0/1": {"complexity": "O(nW)", "technique": "2D table", "pseudo_polynomial": True, "use": "resource allocation with weight limit"},
    "knapsack unbounded": {"complexity": "O(nW)", "technique": "1D table", "difference": "items can be used multiple times"},
    "longest common subsequence": {"complexity": "O(mn)", "technique": "2D table", "use": "diff algorithms, DNA alignment"},
    "edit distance": {"complexity": "O(mn)", "alias": "Levenshtein distance", "technique": "2D table", "operations": ["insert", "delete", "replace"]},
    "coin change": {"complexity": "O(nS)", "technique": "1D table", "use": "minimum coins to make amount S"},
    "matrix chain multiplication": {"complexity": "O(n³)", "technique": "interval DP", "use": "optimal parenthesization"},
    "longest increasing subsequence": {"complexity": "O(n log n)", "technique": "patience sorting / binary search", "naive": "O(n²)"},
    "maximum subarray": {"complexity": "O(n)", "algorithm": "Kadane's algorithm", "alias": "max sum subarray"},
    "rod cutting": {"complexity": "O(n²)", "technique": "1D table", "similar_to": "unbounded knapsack"},
}

_GREEDY_ALGORITHMS = {
    "activity selection": {"complexity": "O(n log n)", "technique": "sort by end time, pick non-overlapping", "optimal": True},
    "Huffman coding": {"complexity": "O(n log n)", "technique": "priority queue, build tree bottom-up", "use": "data compression", "optimal": True},
    "Dijkstra": {"complexity": "O((V+E) log V)", "technique": "priority queue, relax edges", "use": "shortest path (non-negative weights)", "limitation": "no negative edges"},
    "Kruskal": {"complexity": "O(E log E)", "technique": "sort edges, union-find", "use": "minimum spanning tree"},
    "Prim": {"complexity": "O((V+E) log V)", "technique": "priority queue, grow tree", "use": "minimum spanning tree"},
    "fractional knapsack": {"complexity": "O(n log n)", "technique": "sort by value/weight ratio", "optimal": True, "vs_0_1": "can take fractions of items"},
}

_GRAPH_ALGORITHMS = {
    "BFS": {"complexity": "O(V+E)", "use": "shortest path (unweighted), connected components, bipartite check"},
    "DFS": {"complexity": "O(V+E)", "use": "cycle detection, topological sort, strongly connected components"},
    "Dijkstra": {"complexity": "O((V+E) log V)", "use": "shortest path (non-negative weights)", "data_structure": "min-heap"},
    "Bellman-Ford": {"complexity": "O(VE)", "use": "shortest path (negative weights OK)", "detects": "negative cycles"},
    "Floyd-Warshall": {"complexity": "O(V³)", "use": "all-pairs shortest path", "technique": "3 nested loops, DP"},
    "A*": {"complexity": "O(E)", "use": "shortest path with heuristic (maps, games)", "optimal": "if heuristic is admissible"},
    "topological sort": {"complexity": "O(V+E)", "use": "dependency resolution, build systems, task scheduling", "requires": "DAG"},
    "Tarjan SCC": {"complexity": "O(V+E)", "use": "strongly connected components", "technique": "DFS with lowlink"},
    "Kruskal MST": {"complexity": "O(E log E)", "use": "minimum spanning tree", "technique": "sort edges + union-find"},
    "Prim MST": {"complexity": "O((V+E) log V)", "use": "minimum spanning tree", "technique": "priority queue"},
}

_NP_PROBLEMS = {
    "traveling salesman": {"class": "NP-hard", "best_exact": "O(n² × 2^n)", "approximation": "Christofides (1.5× optimal)", "use": "routing, circuit design"},
    "graph coloring": {"class": "NP-complete", "decision": "is k-colorable?", "use": "register allocation, scheduling"},
    "satisfiability": {"alias": "SAT", "class": "NP-complete", "first_NP_complete": True, "solvers": ["MiniSat", "Z3"]},
    "subset sum": {"class": "NP-complete", "pseudo_polynomial": "O(nS) DP", "use": "cryptography basis"},
    "vertex cover": {"class": "NP-complete", "approximation": "2-approximation (greedy)"},
    "clique": {"class": "NP-complete", "decision": "does graph have k-clique?"},
    "hamiltonian path": {"class": "NP-complete", "decision": "does path visiting all vertices exist?"},
}


def search_algorithm(name: str) -> dict:
    """Get details about a search algorithm."""
    key = str(name).lower().strip()
    for k, v in _SEARCH_ALGORITHMS.items():
        if key in k or k in key:
            return {"algorithm": k, **v}
    return {"error": f"Unknown: {name}", "valid": list(_SEARCH_ALGORITHMS.keys())}


def dp_problem(name: str) -> dict:
    """Get details about a dynamic programming problem."""
    key = str(name).lower().strip()
    for k, v in _DP_PROBLEMS.items():
        if key in k or k in key:
            return {"problem": k, **v}
    return {"error": f"Unknown: {name}", "valid": list(_DP_PROBLEMS.keys())}


def greedy_algorithm(name: str) -> dict:
    """Get details about a greedy algorithm."""
    key = str(name).lower().strip()
    for k, v in _GREEDY_ALGORITHMS.items():
        if key in k or k in key:
            return {"algorithm": k, **v}
    return {"error": f"Unknown: {name}", "valid": list(_GREEDY_ALGORITHMS.keys())}


def graph_algorithm(name: str) -> dict:
    """Get details about a graph algorithm."""
    key = str(name).lower().strip()
    for k, v in _GRAPH_ALGORITHMS.items():
        if key in k.lower() or k.lower() in key:
            return {"algorithm": k, **v}
    return {"error": f"Unknown: {name}", "valid": list(_GRAPH_ALGORITHMS.keys())}


def np_problem(name: str) -> dict:
    """Get details about an NP-hard/NP-complete problem."""
    key = str(name).lower().strip()
    for k, v in _NP_PROBLEMS.items():
        if key in k or k in key:
            return {"problem": k, **v}
    return {"error": f"Unknown: {name}", "valid": list(_NP_PROBLEMS.keys())}


def dijkstra_vs_bellman_ford() -> dict:
    """Compare Dijkstra and Bellman-Ford."""
    return {"Dijkstra": _GRAPH_ALGORITHMS["Dijkstra"], "Bellman-Ford": _GRAPH_ALGORITHMS["Bellman-Ford"],
            "summary": "Dijkstra: faster but no negative edges. Bellman-Ford: slower but handles negatives and detects negative cycles."}


def bfs_vs_dfs() -> dict:
    """Compare BFS and DFS."""
    return {"BFS": _GRAPH_ALGORITHMS["BFS"], "DFS": _GRAPH_ALGORITHMS["DFS"],
            "BFS_space": "O(V) — stores frontier", "DFS_space": "O(V) — stores stack",
            "BFS_finds": "shortest path (unweighted)", "DFS_finds": "any path, cycles, topological order"}


ALGORITHMS_FUNCTIONS = {
    "search_algorithm": search_algorithm,
    "dp_problem": dp_problem,
    "greedy_algorithm": greedy_algorithm,
    "graph_algorithm": graph_algorithm,
    "np_problem": np_problem,
    "dijkstra_vs_bellman_ford": dijkstra_vs_bellman_ford,
    "bfs_vs_dfs": bfs_vs_dfs,
}

ALGORITHMS_NL_PATTERNS = [
    (r'(?:what is|explain|complexity of)\s+(binary search|linear search|interpolation search|jump search|ternary search)', 'search_algorithm("{0}")'),
    (r'(?:what is|explain)\s+(?:the\s+)?(knapsack|fibonacci|LCS|edit distance|coin change|matrix chain|LIS|kadane|rod cutting)\s+(?:problem|algorithm|DP)', 'dp_problem("{0}")'),
    (r'(?:what is|explain)\s+(Dijkstra|Bellman.Ford|Floyd.Warshall|A\*|BFS|DFS|topological sort|Tarjan|Kruskal|Prim)', 'graph_algorithm("{0}")'),
    (r'(?:what is|explain)\s+(?:the\s+)?(traveling salesman|TSP|graph coloring|SAT|satisfiability|subset sum|vertex cover|clique|hamiltonian)', 'np_problem("{0}")'),
    (r'(?:compare|difference|vs)\s+Dijkstra\s+(?:and|vs)\s+Bellman.Ford', 'dijkstra_vs_bellman_ford()'),
    (r'(?:compare|difference|vs)\s+BFS\s+(?:and|vs)\s+DFS', 'bfs_vs_dfs()'),
]
