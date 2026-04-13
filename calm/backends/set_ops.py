"""
CALM Set theory operations backend — set arithmetic, relations, properties.

Pure computation on sets represented as lists.
"""

from __future__ import annotations

import math
from itertools import combinations


def set_union(a: list, b: list) -> list:
    """A ∪ B."""
    return sorted(set(a) | set(b), key=str)


def set_intersection(a: list, b: list) -> list:
    """A ∩ B."""
    return sorted(set(a) & set(b), key=str)


def set_difference(a: list, b: list) -> list:
    """A \\ B (elements in A not in B)."""
    return sorted(set(a) - set(b), key=str)


def set_symmetric_difference(a: list, b: list) -> list:
    """A △ B (elements in exactly one of A, B)."""
    return sorted(set(a) ^ set(b), key=str)


def set_cartesian_product(a: list, b: list) -> list:
    """A × B (all ordered pairs)."""
    return [(x, y) for x in a for y in b]


def is_subset(a: list, b: list) -> bool:
    """A ⊆ B."""
    return set(a) <= set(b)


def is_proper_subset(a: list, b: list) -> bool:
    """A ⊂ B (subset but not equal)."""
    return set(a) < set(b)


def is_superset(a: list, b: list) -> bool:
    """A ⊇ B."""
    return set(a) >= set(b)


def is_disjoint(a: list, b: list) -> bool:
    """Whether A and B have no common elements."""
    return set(a).isdisjoint(set(b))


def power_set(s: list) -> list:
    """Power set 𝒫(S) = all subsets. |𝒫(S)| = 2^|S|."""
    items = list(s)
    n = len(items)
    if n > 20:
        return [["error: max 20 elements"]]
    result = []
    for r in range(n + 1):
        for combo in combinations(items, r):
            result.append(list(combo))
    return result


def power_set_size(n: int) -> int:
    """Size of power set: 2^n."""
    return 2 ** int(n)


def set_partition_count(n: int, k: int) -> int:
    """Number of ways to partition n-element set into k non-empty subsets (Stirling S(n,k))."""
    n, k = int(n), int(k)
    if k == 0:
        return 1 if n == 0 else 0
    if k == 1 or k == n:
        return 1
    if k > n:
        return 0
    result = 0
    for j in range(k + 1):
        result += ((-1) ** (k - j)) * math.comb(k, j) * (j ** n)
    return result // math.factorial(k)


def jaccard_index(a: list, b: list) -> float:
    """Jaccard index: |A ∩ B| / |A ∪ B|."""
    sa, sb = set(a), set(b)
    union = len(sa | sb)
    if union == 0:
        return 1.0
    return round(len(sa & sb) / union, 4)


def overlap_coefficient(a: list, b: list) -> float:
    """Overlap coefficient: |A ∩ B| / min(|A|, |B|)."""
    sa, sb = set(a), set(b)
    m = min(len(sa), len(sb))
    if m == 0:
        return 0.0
    return round(len(sa & sb) / m, 4)


def set_complement(s: list, universe: list) -> list:
    """Complement of S with respect to universe U: U \\ S."""
    return sorted(set(universe) - set(s), key=str)


SET_FUNCTIONS = {
    "set_union": set_union,
    "set_intersection": set_intersection,
    "set_difference": set_difference,
    "set_symmetric_difference": set_symmetric_difference,
    "set_cartesian_product": set_cartesian_product,
    "is_subset": is_subset,
    "is_proper_subset": is_proper_subset,
    "is_superset": is_superset,
    "is_disjoint": is_disjoint,
    "power_set": power_set,
    "power_set_size": power_set_size,
    "set_partition_count": set_partition_count,
    "jaccard_index": jaccard_index,
    "overlap_coefficient": overlap_coefficient,
    "set_complement": set_complement,
}

SET_NL_PATTERNS = [
    (r'(?:union)\s+(?:of\s+)?\[([^\]]+)\]\s+(?:and)\s+\[([^\]]+)\]', None),
    (r'(?:intersection)\s+(?:of\s+)?\[([^\]]+)\]\s+(?:and)\s+\[([^\]]+)\]', None),
    (r'(?:is)\s+\[([^\]]+)\]\s+(?:a\s+)?subset\s+(?:of)\s+\[([^\]]+)\]', None),
    (r'(?:are)\s+\[([^\]]+)\]\s+(?:and)\s+\[([^\]]+)\]\s+disjoint', None),
    (r'power\s+set\s+(?:of\s+)?\[([^\]]+)\]', None),
    (r'(?:size of\s+)?power\s+set\s+(?:of\s+)?(\d+)\s+elements', 'power_set_size({0})'),
    (r'jaccard\s+(?:index|similarity)\s+(?:of|between)', None),
]
