"""
CALM Combinatorics backend — permutations, derangements, Bell, Stirling, multinomial.

Models botch combinatorial formulas. Pure computation.
"""

from __future__ import annotations

import math


def derangements(n: int) -> int:
    """Number of derangements (permutations with no fixed points). D(n) = n! × Σ(-1)^k/k!."""
    n = int(n)
    if n == 0:
        return 1
    if n == 1:
        return 0
    result = 0
    for k in range(n + 1):
        result += ((-1) ** k) * math.factorial(n) // math.factorial(k)
    return result


def bell_number(n: int) -> int:
    """Nth Bell number: number of ways to partition a set of n elements."""
    n = int(n)
    if n == 0:
        return 1
    # Bell triangle
    tri = [[0] * (n + 1) for _ in range(n + 1)]
    tri[0][0] = 1
    for i in range(1, n + 1):
        tri[i][0] = tri[i - 1][i - 1]
        for j in range(1, i + 1):
            tri[i][j] = tri[i][j - 1] + tri[i - 1][j - 1]
    return tri[n][0]


def multinomial(n: int, groups: list) -> int:
    """Multinomial coefficient: n! / (k1! × k2! × ... × km!)."""
    n = int(n)
    result = math.factorial(n)
    for k in groups:
        result //= math.factorial(int(k))
    return result


def stars_and_bars(n: int, k: int) -> int:
    """Stars and bars: ways to put n identical items into k distinct bins. C(n+k-1, k-1)."""
    return math.comb(int(n) + int(k) - 1, int(k) - 1)


def circular_permutations(n: int) -> int:
    """Circular permutations of n objects: (n-1)!."""
    return math.factorial(int(n) - 1)


def permutations_with_repetition(n: int, items: list) -> int:
    """Permutations of n items where some repeat. n! / (n1! × n2! × ...)."""
    result = math.factorial(int(n))
    for count in items:
        result //= math.factorial(int(count))
    return result


def combinations_with_repetition(n: int, r: int) -> int:
    """Combinations with repetition: C(n+r-1, r)."""
    return math.comb(int(n) + int(r) - 1, int(r))


def subfactorial(n: int) -> int:
    """Subfactorial !n (same as derangements)."""
    return derangements(n)


def double_factorial(n: int) -> int:
    """Double factorial n!! = n × (n-2) × (n-4) × ... × (1 or 2)."""
    n = int(n)
    if n <= 0:
        return 1
    result = 1
    while n > 0:
        result *= n
        n -= 2
    return result


def falling_factorial(n: int, k: int) -> int:
    """Falling factorial n^(k) = n × (n-1) × ... × (n-k+1)."""
    n, k = int(n), int(k)
    result = 1
    for i in range(k):
        result *= (n - i)
    return result


def rising_factorial(n: int, k: int) -> int:
    """Rising factorial n^(k) = n × (n+1) × ... × (n+k-1). Pochhammer symbol."""
    n, k = int(n), int(k)
    result = 1
    for i in range(k):
        result *= (n + i)
    return result


def pigeonhole(items: int, containers: int) -> int:
    """Pigeonhole principle: at least ceil(items/containers) in one container."""
    return math.ceil(int(items) / max(int(containers), 1))


def inclusion_exclusion_2(a: int, b: int, a_and_b: int) -> int:
    """|A ∪ B| = |A| + |B| - |A ∩ B|."""
    return int(a) + int(b) - int(a_and_b)


def inclusion_exclusion_3(a: int, b: int, c: int, ab: int, ac: int, bc: int, abc: int) -> int:
    """|A ∪ B ∪ C| = |A|+|B|+|C| - |AB| - |AC| - |BC| + |ABC|."""
    return int(a) + int(b) + int(c) - int(ab) - int(ac) - int(bc) + int(abc)


MATH_COMBINATORICS_FUNCTIONS = {
    "derangements": derangements,
    "bell_number": bell_number,
    "multinomial": multinomial,
    "stars_and_bars": stars_and_bars,
    "circular_permutations": circular_permutations,
    "permutations_with_repetition": permutations_with_repetition,
    "combinations_with_repetition": combinations_with_repetition,
    "subfactorial": subfactorial,
    "double_factorial": double_factorial,
    "falling_factorial": falling_factorial,
    "rising_factorial": rising_factorial,
    "pigeonhole": pigeonhole,
    "inclusion_exclusion_2": inclusion_exclusion_2,
    "inclusion_exclusion_3": inclusion_exclusion_3,
}

MATH_COMBINATORICS_NL_PATTERNS = [
    (r'(?:number of\s+)?derangements?\s+(?:of|for)\s+(\d+)', 'derangements({0})'),
    (r'bell\s+number\s+(?:of|for|#)?\s*(\d+)', 'bell_number({0})'),
    (r'(?:circular|round table)\s+permutations?\s+(?:of|for)\s+(\d+)', 'circular_permutations({0})'),
    (r'stars?\s+and\s+bars?\s+(\d+)\s+(?:into|among)\s+(\d+)', 'stars_and_bars({0}, {1})'),
    (r'double\s+factorial\s+(?:of\s+)?(\d+)', 'double_factorial({0})'),
    (r'pigeonhole.*?(\d+)\s+(?:items?|objects?|pigeons?).*?(\d+)\s+(?:containers?|holes?|boxes?)', 'pigeonhole({0}, {1})'),
    (r'combinations?\s+with\s+repetition.*?(\d+).*?choose\s+(\d+)', 'combinations_with_repetition({0}, {1})'),
]
