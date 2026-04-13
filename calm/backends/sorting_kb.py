"""
CALM Sorting algorithm knowledge backend — complexity, stability, use cases.

Models confuse stable vs unstable, give wrong average-case complexity.
"""

from __future__ import annotations

_DATA_VERSION = "2025-01"

_ALGORITHMS = {
    "bubble sort": {
        "best": "O(n)", "average": "O(n²)", "worst": "O(n²)",
        "space": "O(1)", "stable": True, "in_place": True,
        "use_when": "educational, nearly sorted small data",
        "avoid": "anything real — always slower than insertion sort",
    },
    "selection sort": {
        "best": "O(n²)", "average": "O(n²)", "worst": "O(n²)",
        "space": "O(1)", "stable": False, "in_place": True,
        "use_when": "minimizing swaps (expensive write operations)",
    },
    "insertion sort": {
        "best": "O(n)", "average": "O(n²)", "worst": "O(n²)",
        "space": "O(1)", "stable": True, "in_place": True,
        "use_when": "small arrays, nearly sorted data, online (streaming) data",
        "note": "Used by Timsort for small runs",
    },
    "merge sort": {
        "best": "O(n log n)", "average": "O(n log n)", "worst": "O(n log n)",
        "space": "O(n)", "stable": True, "in_place": False,
        "use_when": "guaranteed O(n log n), linked lists, external sort (disk)",
        "used_by": "Python (Timsort hybrid), Java (Arrays.sort for objects)",
    },
    "quick sort": {
        "best": "O(n log n)", "average": "O(n log n)", "worst": "O(n²)",
        "space": "O(log n)", "stable": False, "in_place": True,
        "use_when": "general purpose, cache-friendly, average fastest",
        "gotcha": "O(n²) on already-sorted with bad pivot — use median-of-3",
        "used_by": "C stdlib qsort, many languages' default",
    },
    "heap sort": {
        "best": "O(n log n)", "average": "O(n log n)", "worst": "O(n log n)",
        "space": "O(1)", "stable": False, "in_place": True,
        "use_when": "guaranteed O(n log n) with O(1) extra space",
        "avoid": "poor cache locality makes it slower than quicksort in practice",
    },
    "counting sort": {
        "best": "O(n+k)", "average": "O(n+k)", "worst": "O(n+k)",
        "space": "O(k)", "stable": True, "in_place": False,
        "use_when": "integers with known small range k",
        "limitation": "only works for integers, k must be manageable",
    },
    "radix sort": {
        "best": "O(nk)", "average": "O(nk)", "worst": "O(nk)",
        "space": "O(n+k)", "stable": True, "in_place": False,
        "use_when": "fixed-length integers or strings, k digits",
        "note": "k = number of digits. Faster than comparison sorts when k is small.",
    },
    "bucket sort": {
        "best": "O(n+k)", "average": "O(n+k)", "worst": "O(n²)",
        "space": "O(n)", "stable": True, "in_place": False,
        "use_when": "uniformly distributed floating-point numbers",
    },
    "timsort": {
        "best": "O(n)", "average": "O(n log n)", "worst": "O(n log n)",
        "space": "O(n)", "stable": True, "in_place": False,
        "use_when": "real-world data (often partially sorted)",
        "used_by": "Python, Java (for objects), Android, V8",
        "how": "Hybrid merge sort + insertion sort, detects natural runs",
    },
    "shell sort": {
        "best": "O(n log n)", "average": "O(n^1.25)", "worst": "O(n²)",
        "space": "O(1)", "stable": False, "in_place": True,
        "use_when": "medium arrays, embedded systems (no recursion, O(1) space)",
    },
    "introsort": {
        "best": "O(n log n)", "average": "O(n log n)", "worst": "O(n log n)",
        "space": "O(log n)", "stable": False, "in_place": True,
        "use_when": "general purpose with guaranteed worst case",
        "used_by": "C++ std::sort, .NET",
        "how": "Quicksort → heapsort fallback when depth exceeds 2×log₂(n)",
    },
}


def sort_info(name: str) -> dict:
    """Get complexity and details for a sorting algorithm."""
    key = str(name).lower().strip()
    entry = _ALGORITHMS.get(key)
    if not entry:
        for k, v in _ALGORITHMS.items():
            if key in k or k.replace(" ", "") == key.replace(" ", ""):
                return {"algorithm": k, **v}
        return {"error": f"Unknown: {name}", "valid": sorted(_ALGORITHMS.keys())}
    return {"algorithm": key, **entry}


def sort_compare(alg1: str, alg2: str) -> dict:
    """Compare two sorting algorithms."""
    return {"alg1": sort_info(alg1), "alg2": sort_info(alg2)}


def stable_sorts() -> list[str]:
    """List all stable sorting algorithms."""
    return sorted(k for k, v in _ALGORITHMS.items() if v.get("stable"))


def list_sorts() -> list[str]:
    """List all known sorting algorithms."""
    return sorted(_ALGORITHMS.keys())


SORTING_FUNCTIONS = {
    "sort_info": sort_info,
    "sort_compare": sort_compare,
    "stable_sorts": stable_sorts,
    "list_sorts": list_sorts,
}

SORTING_NL_PATTERNS = [
    (r'(?:what is|explain|complexity of)\s+(?:the\s+)?(bubble|selection|insertion|merge|quick|heap|counting|radix|bucket|tim|shell|intro)\s*sort', 'sort_info("{0} sort")'),
    (r'(?:compare|difference|vs)\s+(?:between\s+)?(\w+)\s*sort\s+(?:and|vs)\s+(\w+)\s*sort', 'sort_compare("{0} sort", "{1} sort")'),
    (r'(?:which|what|list)\s+(?:sorting?\s+)?(?:algorithms?\s+)?(?:are\s+)?stable', 'stable_sorts()'),
]
