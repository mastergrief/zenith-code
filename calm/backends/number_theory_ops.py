"""
CALM Number theory backend — Euler's totient, Legendre, Mobius, Catalan, partition.

Models approximate or hallucinate number theory results. Pure computation.
"""

from __future__ import annotations

import math


def euler_totient(n: int) -> int:
    """Euler's totient function φ(n): count of integers 1..n coprime to n."""
    n = int(n)
    if n <= 0:
        return 0
    result = n
    p = 2
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result


def mobius(n: int) -> int:
    """Möbius function μ(n): 0 if n has squared prime factor, (-1)^k if k distinct primes."""
    n = int(n)
    if n <= 0:
        return 0
    if n == 1:
        return 1
    count = 0
    p = 2
    while p * p <= n:
        if n % p == 0:
            n //= p
            count += 1
            if n % p == 0:
                return 0  # squared factor
        p += 1
    if n > 1:
        count += 1
    return 1 if count % 2 == 0 else -1


def catalan(n: int) -> int:
    """Nth Catalan number: C(2n,n)/(n+1)."""
    n = int(n)
    if n < 0:
        return 0
    return math.comb(2 * n, n) // (n + 1)


def stirling_second(n: int, k: int) -> int:
    """Stirling number of the second kind S(n,k): ways to partition n elements into k non-empty subsets."""
    n, k = int(n), int(k)
    if k == 0:
        return 1 if n == 0 else 0
    if k == 1 or k == n:
        return 1
    if k > n or k < 0:
        return 0
    result = 0
    for j in range(k + 1):
        sign = (-1) ** (k - j)
        result += sign * math.comb(k, j) * (j ** n)
    return result // math.factorial(k)


def partition_count(n: int) -> int:
    """Number of integer partitions of n (order doesn't matter)."""
    n = int(n)
    if n < 0:
        return 0
    dp = [0] * (n + 1)
    dp[0] = 1
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            dp[j] += dp[j - i]
    return dp[n]


def digital_root(n: int) -> int:
    """Digital root: repeatedly sum digits until single digit."""
    n = abs(int(n))
    if n == 0:
        return 0
    return 1 + (n - 1) % 9


def sum_of_divisors(n: int) -> int:
    """Sum of all positive divisors of n (σ(n)), including 1 and n."""
    n = int(n)
    if n <= 0:
        return 0
    total = 0
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            total += i
            if i != n // i:
                total += n // i
    return total


def count_divisors(n: int) -> int:
    """Count of positive divisors of n, including 1 and n."""
    n = int(n)
    if n <= 0:
        return 0
    count = 0
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            count += 1
            if i != n // i:
                count += 1
    return count


def is_perfect(n: int) -> bool:
    """Whether n is a perfect number (sum of proper divisors = n). E.g. 6, 28, 496."""
    n = int(n)
    return n > 0 and sum_of_divisors(n) - n == n


def is_abundant(n: int) -> bool:
    """Whether n is an abundant number (sum of proper divisors > n)."""
    n = int(n)
    return n > 0 and sum_of_divisors(n) - n > n


def is_deficient(n: int) -> bool:
    """Whether n is a deficient number (sum of proper divisors < n)."""
    n = int(n)
    return n > 0 and sum_of_divisors(n) - n < n


def legendre_symbol(a: int, p: int) -> int:
    """Legendre symbol (a/p): 0 if p|a, 1 if a is QR mod p, -1 if NQR."""
    a, p = int(a), int(p)
    if p < 2:
        return 0
    a = a % p
    if a == 0:
        return 0
    result = pow(a, (p - 1) // 2, p)
    return result if result <= 1 else -1


def is_carmichael(n: int) -> bool:
    """Whether n is a Carmichael number (composite but passes Fermat test for all coprime bases)."""
    n = int(n)
    if n < 2 or _is_prime_simple(n):
        return False
    for a in range(2, min(n, 1000)):
        if math.gcd(a, n) == 1:
            if pow(a, n - 1, n) != 1:
                return False
    return True


def _is_prime_simple(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def is_fibonacci(n: int) -> bool:
    """Whether n is a Fibonacci number. Uses the property that n is Fibonacci iff
    5n²+4 or 5n²-4 is a perfect square."""
    n = int(n)
    if n < 0:
        return False
    def is_square(x):
        if x < 0:
            return False
        s = int(math.isqrt(x))
        return s * s == x
    return is_square(5 * n * n + 4) or is_square(5 * n * n - 4)


NUMBER_THEORY_FUNCTIONS = {
    "euler_totient": euler_totient,
    "mobius": mobius,
    "catalan": catalan,
    "stirling_second": stirling_second,
    "partition_count": partition_count,
    "digital_root": digital_root,
    "sum_of_divisors": sum_of_divisors,
    "count_divisors": count_divisors,
    "is_perfect": is_perfect,
    "is_abundant": is_abundant,
    "is_deficient": is_deficient,
    "legendre_symbol": legendre_symbol,
    "is_carmichael": is_carmichael,
    "is_fibonacci": is_fibonacci,
}

NUMBER_THEORY_NL_PATTERNS = [
    (r'(?:euler\'?s?\s+)?totient\s+(?:of|for|φ\()?\s*(\d+)', 'euler_totient({0})'),
    (r'catalan\s+(?:number\s+)?(?:of|for|#)?\s*(\d+)', 'catalan({0})'),
    (r'(?:number of\s+)?(?:integer\s+)?partitions?\s+(?:of|for)\s+(\d+)', 'partition_count({0})'),
    (r'digital\s+root\s+(?:of|for)\s+(\d+)', 'digital_root({0})'),
    (r'sum\s+(?:of\s+)?divisors?\s+(?:of|for)\s+(\d+)', 'sum_of_divisors({0})'),
    (r'(?:count|number|how many)\s+divisors?\s+(?:of|does|for)\s+(\d+)', 'count_divisors({0})'),
    (r'(?:is)\s+(\d+)\s+(?:a\s+)?perfect\s+number', 'is_perfect({0})'),
    (r'(?:is)\s+(\d+)\s+(?:a\s+)?(?:abundant|excessive)', 'is_abundant({0})'),
    (r'mobius\s+(?:function\s+)?(?:of|for|μ\()?\s*(\d+)', 'mobius({0})'),
    (r'(?:is)\s+(\d+)\s+(?:a\s+)?fibonacci\s+number', 'is_fibonacci({0})'),
]
