"""
CALM Statistics backend — z-scores, percentiles, distributions, hypothesis tests.

Models approximate z-tables, mess up standard error, confuse one/two-tail.
Extends data_ops (mean/median/stdev) with inferential statistics.
"""

from __future__ import annotations

import math


def z_score(value: float, mean: float, stdev: float) -> float:
    """Z-score: (value - mean) / stdev."""
    sd = float(stdev)
    if sd == 0:
        return 0.0
    return round((float(value) - float(mean)) / sd, 4)


def _erf(x: float) -> float:
    """Approximation of the error function (Abramowitz & Stegun)."""
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * math.exp(-x * x)
    return sign * y


def normal_cdf(x: float, mean: float = 0, stdev: float = 1) -> float:
    """Cumulative distribution function for normal distribution."""
    z = (float(x) - float(mean)) / float(stdev)
    return round(0.5 * (1 + _erf(z / math.sqrt(2))), 6)


def normal_pdf(x: float, mean: float = 0, stdev: float = 1) -> float:
    """Probability density function for normal distribution."""
    m, s = float(mean), float(stdev)
    return round(math.exp(-0.5 * ((float(x) - m) / s) ** 2) / (s * math.sqrt(2 * math.pi)), 6)


def percentile_rank(value: float, data: list) -> float:
    """Percentile rank of a value in a dataset (0-100)."""
    vals = [float(x) for x in data]
    v = float(value)
    below = sum(1 for x in vals if x < v)
    equal = sum(1 for x in vals if x == v)
    return round((below + 0.5 * equal) / len(vals) * 100, 2)


def percentile(data: list, p: float) -> float:
    """Value at the p-th percentile (0-100) using linear interpolation."""
    vals = sorted(float(x) for x in data)
    k = (float(p) / 100) * (len(vals) - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    return round(vals[int(f)] * (c - k) + vals[int(c)] * (k - f), 4)


def iqr(data: list) -> float:
    """Interquartile range (Q3 - Q1)."""
    return round(percentile(data, 75) - percentile(data, 25), 4)


def standard_error(stdev: float, n: int) -> float:
    """Standard error of the mean: stdev / sqrt(n)."""
    return round(float(stdev) / math.sqrt(int(n)), 4)


def confidence_interval(mean: float, stdev: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Confidence interval for a mean. Default z=1.96 (95%)."""
    se = float(stdev) / math.sqrt(int(n))
    m = float(mean)
    margin = float(z) * se
    return (round(m - margin, 4), round(m + margin, 4))


def z_test(sample_mean: float, pop_mean: float, stdev: float, n: int) -> float:
    """One-sample z-test statistic."""
    se = float(stdev) / math.sqrt(int(n))
    return round((float(sample_mean) - float(pop_mean)) / se, 4)


def chi_squared(observed: list, expected: list) -> float:
    """Chi-squared test statistic from observed and expected frequencies."""
    obs = [float(x) for x in observed]
    exp = [float(x) for x in expected]
    if len(obs) != len(exp):
        return -1.0
    return round(sum((o - e) ** 2 / e for o, e in zip(obs, exp) if e != 0), 4)


def covariance(x: list, y: list) -> float:
    """Sample covariance of two datasets."""
    xv = [float(v) for v in x]
    yv = [float(v) for v in y]
    n = len(xv)
    if n < 2 or n != len(yv):
        return 0.0
    mx = sum(xv) / n
    my = sum(yv) / n
    return round(sum((xi - mx) * (yi - my) for xi, yi in zip(xv, yv)) / (n - 1), 4)


def correlation(x: list, y: list) -> float:
    """Pearson correlation coefficient."""
    xv = [float(v) for v in x]
    yv = [float(v) for v in y]
    n = len(xv)
    if n < 2 or n != len(yv):
        return 0.0
    mx = sum(xv) / n
    my = sum(yv) / n
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in xv) / (n - 1))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in yv) / (n - 1))
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(xv, yv)) / (n - 1)
    return round(cov / (sx * sy), 4)


def variance(data: list) -> float:
    """Sample variance."""
    vals = [float(x) for x in data]
    n = len(vals)
    if n < 2:
        return 0.0
    m = sum(vals) / n
    return round(sum((x - m) ** 2 for x in vals) / (n - 1), 4)


def coefficient_of_variation(data: list) -> float:
    """Coefficient of variation (CV) as percentage: (stdev/mean)*100."""
    vals = [float(x) for x in data]
    m = sum(vals) / len(vals)
    if m == 0:
        return 0.0
    v = variance(vals)
    return round(math.sqrt(v) / abs(m) * 100, 2)


def skewness(data: list) -> float:
    """Sample skewness (Fisher-Pearson)."""
    vals = [float(x) for x in data]
    n = len(vals)
    if n < 3:
        return 0.0
    m = sum(vals) / n
    s = math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1))
    if s == 0:
        return 0.0
    return round((n / ((n - 1) * (n - 2))) * sum(((x - m) / s) ** 3 for x in vals), 4)


def kurtosis(data: list) -> float:
    """Excess kurtosis (Fisher definition, normal = 0)."""
    vals = [float(x) for x in data]
    n = len(vals)
    if n < 4:
        return 0.0
    m = sum(vals) / n
    s = math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1))
    if s == 0:
        return 0.0
    m4 = sum(((x - m) / s) ** 4 for x in vals) / n
    return round(m4 - 3, 4)


STATISTICS_FUNCTIONS = {
    "z_score": z_score,
    "normal_cdf": normal_cdf,
    "normal_pdf": normal_pdf,
    "percentile_rank": percentile_rank,
    "percentile": percentile,
    "iqr": iqr,
    "standard_error": standard_error,
    "confidence_interval": confidence_interval,
    "z_test": z_test,
    "chi_squared": chi_squared,
    "covariance": covariance,
    "correlation": correlation,
    "variance": variance,
    "coefficient_of_variation": coefficient_of_variation,
    "skewness": skewness,
    "kurtosis": kurtosis,
}

STATISTICS_NL_PATTERNS = [
    (r'z.?score\s+(?:of|for)\s+([-\d.]+).*?mean\s+(?:of\s+)?([-\d.]+).*?(?:std|stdev|standard deviation)\s+(?:of\s+)?([\d.]+)', 'z_score({0}, {1}, {2})'),
    (r'standard error.*?(?:std|stdev|standard deviation)\s+(?:of\s+)?([\d.]+).*?n\s*=\s*(\d+)', 'standard_error({0}, {1})'),
    (r'(?:95%|99%)\s+confidence interval.*?mean\s+(?:of\s+)?([-\d.]+).*?(?:std|stdev)\s+(?:of\s+)?([\d.]+).*?n\s*=\s*(\d+)', 'confidence_interval({0}, {1}, {2})'),
    (r'correlation\s+(?:between|of).*?\[([-\d.,\s]+)\].*?\[([-\d.,\s]+)\]', 'correlation([{0}], [{1}])'),
    (r'variance\s+of\s+\[([-\d.,\s]+)\]', 'variance([{0}])'),
    (r'(?:IQR|interquartile range)\s+of\s+\[([-\d.,\s]+)\]', 'iqr([{0}])'),
    (r'skewness\s+of\s+\[([-\d.,\s]+)\]', 'skewness([{0}])'),
]
