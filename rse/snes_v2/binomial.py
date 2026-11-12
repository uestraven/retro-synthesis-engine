"""Exact (Clopper-Pearson) binomial confidence intervals.

Endpoint A is a proportion over 36 seeds. A normal approximation is wrong at
that size and at proportions near 1 -- 36/36 would get an interval of zero
width -- so the exact interval is used. `scipy` is not installed here, so
the inverse beta is obtained by bisection on the regularised incomplete beta
function, which is computed by its continued fraction.

Verified against published Clopper-Pearson values in the tests, not against
another implementation of the same formula.
"""

from __future__ import annotations

import math


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Lentz's method)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return h


def regularised_incomplete_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b). 0 <= x <= 1."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + b * math.log1p(-x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def _inverse_beta(a: float, b: float, p: float) -> float:
    """Smallest x with I_x(a, b) >= p, by bisection. Monotone, so safe."""
    low, high = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if regularised_incomplete_beta(a, b, mid) < p:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def clopper_pearson(successes: int, trials: int,
                    confidence: float = 0.95) -> tuple[float, float]:
    """Exact two-sided interval for a binomial proportion.

    The boundary cases are the reason this exists: 0 successes gives
    (0, upper) and `trials` successes gives (lower, 1), rather than the
    zero-width interval a normal approximation produces.
    """
    if not 0 <= successes <= trials:
        raise ValueError(f"{successes} successes in {trials} trials")
    if trials == 0:
        return (0.0, 1.0)
    alpha = 1.0 - confidence
    low = 0.0 if successes == 0 else _inverse_beta(
        successes, trials - successes + 1, alpha / 2.0)
    high = 1.0 if successes == trials else _inverse_beta(
        successes + 1, trials - successes, 1.0 - alpha / 2.0)
    return (low, high)
