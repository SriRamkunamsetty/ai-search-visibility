"""Small, dependency-free statistics helpers."""
from math import sqrt
from statistics import median


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion. Honest uncertainty for small samples."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def robust_z(values, floor=0.05):
    """MAD-based z-scores; the scale is floored so smooth series don't make z explode."""
    med = median(values)
    mad = max(median(abs(v - med) for v in values), floor)
    return [0.6745 * (v - med) / mad for v in values]


def pct_rank(values, x):
    """Percentile rank (0-100) of x within values (ties count half)."""
    if not values:
        return None
    below = sum(v < x for v in values)
    ties = sum(v == x for v in values)
    return 100.0 * (below + 0.5 * ties) / len(values)
