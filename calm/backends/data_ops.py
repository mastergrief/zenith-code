"""
CALM data/stats backend — verified statistical computations.

The model writes "the average is 42.5" and Auto-CALM verifies on CPU.

Functions: mean, median, mode, stdev, variance, percentile, correlation,
linear_regression, histogram, normalize, zscore.
"""

from __future__ import annotations

import math
from typing import List, Union


def mean(data: list) -> float:
    """Arithmetic mean."""
    if not data:
        raise ValueError("mean: empty data")
    return sum(data) / len(data)


def median(data: list) -> float:
    """Median value."""
    if not data:
        raise ValueError("median: empty data")
    s = sorted(data)
    n = len(s)
    if n % 2 == 1:
        return float(s[n // 2])
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def mode(data: list):
    """Most frequent value. Returns the smallest if tied."""
    if not data:
        raise ValueError("mode: empty data")
    counts = {}
    for x in data:
        counts[x] = counts.get(x, 0) + 1
    max_count = max(counts.values())
    modes = sorted(k for k, v in counts.items() if v == max_count)
    return modes[0]


def variance(data: list, population: bool = True) -> float:
    """Variance (population by default)."""
    if len(data) < 2:
        raise ValueError("variance: need at least 2 values")
    m = mean(data)
    ss = sum((x - m) ** 2 for x in data)
    return ss / len(data) if population else ss / (len(data) - 1)


def stdev(data: list, population: bool = True) -> float:
    """Standard deviation."""
    return math.sqrt(variance(data, population))


def percentile(data: list, p: float) -> float:
    """P-th percentile (0-100)."""
    if not data:
        raise ValueError("percentile: empty data")
    s = sorted(data)
    k = (len(s) - 1) * (float(p) / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(s[int(k)])
    return s[int(f)] * (c - k) + s[int(c)] * (k - f)


def correlation(x: list, y: list) -> float:
    """Pearson correlation coefficient."""
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("correlation: need equal-length lists with 2+ items")
    mx, my = mean(x), mean(y)
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def linear_regression(x: list, y: list) -> dict:
    """Simple linear regression. Returns {slope, intercept, r_squared}."""
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("linear_regression: need equal-length lists with 2+ items")
    mx, my = mean(x), mean(y)
    ss_xx = sum((xi - mx) ** 2 for xi in x)
    ss_xy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    if ss_xx == 0:
        raise ValueError("linear_regression: all x values are the same")
    slope = ss_xy / ss_xx
    intercept = my - slope * mx
    r = correlation(x, y)
    return {"slope": round(slope, 6), "intercept": round(intercept, 6), "r_squared": round(r ** 2, 6)}


def normalize(data: list) -> list:
    """Min-max normalize to [0, 1]."""
    lo, hi = min(data), max(data)
    if lo == hi:
        return [0.5] * len(data)
    return [(x - lo) / (hi - lo) for x in data]


def zscore(data: list) -> list:
    """Z-score normalize."""
    m = mean(data)
    s = stdev(data)
    if s == 0:
        return [0.0] * len(data)
    return [round((x - m) / s, 6) for x in data]


def histogram(data: list, bins: int = 10) -> list:
    """Bin data into a histogram. Returns list of {lo, hi, count}."""
    if not data:
        return []
    lo, hi = min(data), max(data)
    if lo == hi:
        return [{"lo": lo, "hi": hi, "count": len(data)}]
    width = (hi - lo) / bins
    result = []
    for i in range(bins):
        blo = lo + i * width
        bhi = lo + (i + 1) * width
        count = sum(1 for x in data if blo <= x < bhi) if i < bins - 1 \
            else sum(1 for x in data if blo <= x <= bhi)
        result.append({"lo": round(blo, 4), "hi": round(bhi, 4), "count": count})
    return result


DATA_FUNCTIONS = {
    "mean": mean,
    "median": median,
    "mode": mode,
    "variance": variance,
    "stdev": stdev,
    "percentile": percentile,
    "correlation": correlation,
    "linear_regression": linear_regression,
    "normalize": normalize,
    "zscore": zscore,
    "histogram": histogram,
}

DATA_NL_PATTERNS = [
    (r'(?:average|mean)\s+of\s+\[([-\d.,\s]+)\]', 'mean([{0}])'),
    (r'median\s+of\s+\[([-\d.,\s]+)\]', 'median([{0}])'),
    (r'mode\s+of\s+\[([-\d.,\s]+)\]', 'mode([{0}])'),
    (r'(?:standard deviation|stdev|std dev)\s+of\s+\[([-\d.,\s]+)\]', 'stdev([{0}])'),
    (r'(\d+)(?:th|st|nd|rd)\s+percentile\s+of\s+\[([-\d.,\s]+)\]', 'percentile([{1}], {0})'),
]
