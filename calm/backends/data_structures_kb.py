"""
CALM Data structures knowledge backend — complexity, use cases, comparisons.

Models confuse BST vs heap, hallucinate complexities, give wrong use cases.
"""

from __future__ import annotations

_DATA_VERSION = "2025-01"

_DATA_STRUCTURES = {
    "array": {
        "access": "O(1)", "search": "O(n)", "insert": "O(n)", "delete": "O(n)",
        "space": "O(n)", "ordered": True,
        "use_when": "random access needed, size known in advance",
        "avoid_when": "frequent insertions/deletions in middle",
    },
    "linked list": {
        "access": "O(n)", "search": "O(n)", "insert_head": "O(1)", "insert_tail": "O(1) with tail pointer", "delete": "O(1) with pointer",
        "space": "O(n)", "ordered": False,
        "use_when": "frequent insertions/deletions, unknown size",
        "avoid_when": "random access needed",
        "variants": ["singly linked", "doubly linked", "circular"],
    },
    "stack": {
        "push": "O(1)", "pop": "O(1)", "peek": "O(1)", "search": "O(n)",
        "space": "O(n)", "order": "LIFO",
        "use_when": "undo/redo, expression parsing, DFS, recursion simulation",
        "implementations": ["array-backed", "linked list"],
    },
    "queue": {
        "enqueue": "O(1)", "dequeue": "O(1)", "peek": "O(1)", "search": "O(n)",
        "space": "O(n)", "order": "FIFO",
        "use_when": "BFS, task scheduling, buffering",
        "variants": ["circular queue", "priority queue", "deque"],
    },
    "hash table": {
        "access": "O(1) avg", "search": "O(1) avg", "insert": "O(1) avg", "delete": "O(1) avg",
        "worst_case": "O(n) all ops", "space": "O(n)",
        "use_when": "fast lookup by key, counting, caching",
        "avoid_when": "ordered traversal needed",
        "collision_handling": ["chaining", "open addressing (linear/quadratic probing)"],
    },
    "binary search tree": {
        "access": "O(log n) avg", "search": "O(log n) avg", "insert": "O(log n) avg", "delete": "O(log n) avg",
        "worst_case": "O(n) if unbalanced", "space": "O(n)",
        "use_when": "sorted data, range queries, in-order traversal",
        "balanced_variants": ["AVL tree", "red-black tree", "B-tree"],
    },
    "heap": {
        "insert": "O(log n)", "extract_min_max": "O(log n)", "peek": "O(1)",
        "search": "O(n)", "space": "O(n)",
        "use_when": "priority queue, top-K, median finding",
        "types": ["min-heap", "max-heap", "binary heap", "fibonacci heap"],
        "NOT_for": "searching — use BST instead",
    },
    "trie": {
        "insert": "O(m)", "search": "O(m)", "delete": "O(m)",
        "space": "O(ALPHABET_SIZE × m × n)",
        "use_when": "prefix search, autocomplete, spell check, IP routing",
        "m_is": "length of the key/word",
    },
    "graph": {
        "add_vertex": "O(1)", "add_edge": "O(1)", "remove_edge": "O(E)",
        "search_bfs": "O(V+E)", "search_dfs": "O(V+E)",
        "space_adj_matrix": "O(V²)", "space_adj_list": "O(V+E)",
        "representations": ["adjacency matrix", "adjacency list", "edge list"],
        "use_when": "relationships, networks, routing, dependencies",
    },
    "red-black tree": {
        "access": "O(log n)", "search": "O(log n)", "insert": "O(log n)", "delete": "O(log n)",
        "space": "O(n)",
        "guarantees": "height ≤ 2×log₂(n+1)",
        "use_when": "balanced BST needed (Java TreeMap, C++ std::map)",
        "vs_avl": "fewer rotations on insert/delete, slightly less balanced than AVL",
    },
    "b-tree": {
        "search": "O(log n)", "insert": "O(log n)", "delete": "O(log n)",
        "space": "O(n)",
        "use_when": "database indexes, file systems — optimized for disk I/O",
        "branching_factor": "high (100s-1000s), minimizes disk seeks",
        "variants": ["B+tree (data only in leaves)", "B*tree"],
    },
    "bloom filter": {
        "insert": "O(k)", "query": "O(k)", "delete": "not supported",
        "space": "O(m bits)", "false_positives": True, "false_negatives": False,
        "use_when": "approximate set membership (cache, spell check, network)",
        "k_is": "number of hash functions",
    },
    "skip list": {
        "search": "O(log n) avg", "insert": "O(log n) avg", "delete": "O(log n) avg",
        "space": "O(n)",
        "use_when": "sorted data with simpler implementation than balanced BST",
        "used_by": ["Redis sorted sets", "LevelDB"],
    },
    "disjoint set": {
        "find": "O(α(n)) ≈ O(1)", "union": "O(α(n)) ≈ O(1)",
        "space": "O(n)",
        "use_when": "connected components, Kruskal's MST, cycle detection",
        "optimizations": ["path compression", "union by rank"],
        "alias": "union-find",
    },
}


def ds_info(name: str) -> dict:
    """Get complexity and details for a data structure."""
    key = str(name).lower().strip()
    entry = _DATA_STRUCTURES.get(key)
    if not entry:
        # Fuzzy match
        for k, v in _DATA_STRUCTURES.items():
            if key in k or k in key:
                return {"name": k, **v}
            if "alias" in v and key == v["alias"]:
                return {"name": k, **v}
        return {"error": f"Unknown: {name}", "valid": sorted(_DATA_STRUCTURES.keys())}
    return {"name": key, **entry}


def ds_compare(ds1: str, ds2: str) -> dict:
    """Compare two data structures."""
    d1 = ds_info(ds1)
    d2 = ds_info(ds2)
    if "error" in d1 or "error" in d2:
        return {"error": "Unknown data structure", "d1": d1, "d2": d2}
    return {"ds1": d1, "ds2": d2}


def best_for(operation: str) -> list[dict]:
    """Find the best data structures for a given operation."""
    op = str(operation).lower().strip()
    results = []
    for name, ds in _DATA_STRUCTURES.items():
        for key, val in ds.items():
            if isinstance(val, str) and op in key.lower():
                results.append({"name": name, "operation": key, "complexity": val})
    results.sort(key=lambda x: x["complexity"])
    return results[:5]


def list_data_structures() -> list[str]:
    """List all known data structures."""
    return sorted(_DATA_STRUCTURES.keys())


DATA_STRUCTURES_FUNCTIONS = {
    "ds_info": ds_info,
    "ds_compare": ds_compare,
    "best_for": best_for,
    "list_data_structures": list_data_structures,
}

DATA_STRUCTURES_NL_PATTERNS = [
    (r'(?:what is|explain|complexity of)\s+(?:a\s+)?(array|linked list|stack|queue|hash table|binary search tree|BST|heap|trie|graph|red.black tree|b.tree|bloom filter|skip list|disjoint set|union.find)', 'ds_info("{0}")'),
    (r'(?:compare|difference|vs)\s+(?:between\s+)?([\w\s]+?)\s+(?:and|vs)\s+([\w\s]+?)(?:\s+data structure)?$', 'ds_compare("{0}", "{1}")'),
    (r'(?:best|fastest)\s+(?:data structure|DS)\s+for\s+(\w+)', 'best_for("{0}")'),
]
