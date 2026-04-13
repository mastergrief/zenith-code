"""
CALM Probability backend — dice, cards, binomial, Bayesian, expected value.

Models hand-wave probability. Pure math.comb.
"""

from __future__ import annotations

import math


def dice_probability(target: int, sides: int = 6, num_dice: int = 1) -> str:
    """Probability of rolling exactly target with num_dice dice of given sides."""
    target, sides, num_dice = int(target), int(sides), int(num_dice)
    if num_dice == 1:
        if 1 <= target <= sides:
            return f"1/{sides} = {1/sides:.6f}"
        return "0"
    # Brute force for small dice counts, inclusion-exclusion for larger
    if num_dice > 10:
        return "too many dice (max 10)"
    count = 0
    total = sides ** num_dice

    def _count(dice_left, remaining):
        nonlocal count
        if dice_left == 0:
            if remaining == 0:
                count += 1
            return
        for face in range(1, sides + 1):
            if remaining - face >= 0:
                _count(dice_left - 1, remaining - face)

    _count(num_dice, target)
    if count == 0:
        return "0"
    g = math.gcd(count, total)
    return f"{count // g}/{total // g} = {count / total:.6f}"


def coin_probability(heads: int, flips: int) -> str:
    """Probability of exactly `heads` heads in `flips` fair coin flips."""
    h, n = int(heads), int(flips)
    if h < 0 or h > n:
        return "0"
    ways = math.comb(n, h)
    total = 2 ** n
    g = math.gcd(ways, total)
    return f"{ways // g}/{total // g} = {ways / total:.6f}"


def card_probability(desired: int, total: int = 52, draw: int = 1) -> str:
    """Probability of drawing at least one desired card in `draw` draws from `total`."""
    desired, total, draw = int(desired), int(total), int(draw)
    if desired <= 0 or total <= 0 or draw <= 0:
        return "0"
    # P(at least one) = 1 - P(none)
    if draw > total:
        draw = total
    none = math.comb(total - desired, draw) / math.comb(total, draw)
    prob = 1 - none
    return f"{prob:.6f}"


def binomial_pmf(n: int, k: int, p: float) -> float:
    """Binomial probability: P(X=k) given n trials, probability p."""
    n, k, p = int(n), int(k), float(p)
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def binomial_cdf(n: int, k: int, p: float) -> float:
    """Cumulative binomial: P(X <= k) given n trials, probability p."""
    n, k, p = int(n), int(k), float(p)
    return sum(binomial_pmf(n, i, p) for i in range(k + 1))


def expected_value(values: list, probabilities: list) -> float:
    """Expected value: sum of value_i * probability_i."""
    return sum(float(v) * float(p) for v, p in zip(values, probabilities))


def bayes(prior: float, likelihood: float, evidence: float) -> float:
    """Bayes' theorem: P(A|B) = P(B|A) * P(A) / P(B)."""
    prior, likelihood, evidence = float(prior), float(likelihood), float(evidence)
    if evidence == 0:
        return -1.0
    return (likelihood * prior) / evidence


def permutations(n: int, r: int) -> int:
    """Number of permutations: n! / (n-r)!"""
    n, r = int(n), int(r)
    return math.perm(n, r)


def combinations(n: int, r: int) -> int:
    """Number of combinations: n! / (r! * (n-r)!)"""
    return math.comb(int(n), int(r))


def odds_to_probability(odds_for: int, odds_against: int) -> float:
    """Convert odds (e.g. 3:1 against) to probability."""
    f, a = int(odds_for), int(odds_against)
    return f / (f + a) if (f + a) > 0 else 0.0


def probability_to_odds(p: float) -> str:
    """Convert probability to odds ratio."""
    p = float(p)
    if p <= 0:
        return "0:1"
    if p >= 1:
        return "1:0"
    ratio = p / (1 - p)
    if ratio >= 1:
        # Simplify
        for denom in range(1, 100):
            numer = ratio * denom
            if abs(numer - round(numer)) < 0.01:
                return f"{round(numer)}:{denom}"
        return f"{ratio:.2f}:1"
    inv = (1 - p) / p
    for denom in range(1, 100):
        numer = inv * denom
        if abs(numer - round(numer)) < 0.01:
            return f"1:{round(numer)}"
    return f"1:{inv:.2f}"


PROBABILITY_FUNCTIONS = {
    "dice_probability": dice_probability,
    "coin_probability": coin_probability,
    "card_probability": card_probability,
    "binomial_pmf": binomial_pmf,
    "binomial_cdf": binomial_cdf,
    "expected_value": expected_value,
    "bayes": bayes,
    "permutations": permutations,
    "combinations": combinations,
    "odds_to_probability": odds_to_probability,
    "probability_to_odds": probability_to_odds,
}
