"""Clopper-Pearson against PUBLISHED values, not another copy of the formula.

Endpoint A is a proportion over 36 seeds, where a normal approximation is
wrong -- 36/36 would get a zero-width interval. These reference values are
the standard textbook/NIST Clopper-Pearson examples.
"""
import pytest

from rse.snes_v2.binomial import clopper_pearson, regularised_incomplete_beta


def test_incomplete_beta_matches_known_values():
    # I_0.5(1,1) = 0.5; I_x(1,1) = x. Also the symmetry I_x(a,b)=1-I_{1-x}(b,a).
    assert regularised_incomplete_beta(1, 1, 0.3) == pytest.approx(0.3)
    assert regularised_incomplete_beta(2, 3, 0.5) == pytest.approx(0.6875)
    assert regularised_incomplete_beta(5, 2, 0.5) == pytest.approx(0.109375)
    for a, b, x in ((2.5, 3.5, 0.4), (7, 3, 0.8), (1, 9, 0.05)):
        assert regularised_incomplete_beta(a, b, x) == pytest.approx(
            1.0 - regularised_incomplete_beta(b, a, 1.0 - x), abs=1e-12)


@pytest.mark.parametrize("successes,trials,low,high", [
    # Standard published Clopper-Pearson 95% intervals.
    (0, 10, 0.0000, 0.3085),
    (1, 10, 0.0025, 0.4450),
    (5, 10, 0.1871, 0.8129),
    (10, 10, 0.6915, 1.0000),
    (2, 20, 0.0123, 0.3170),
    (19, 20, 0.7513, 0.9987),
    (3, 36, 0.017526, 0.224690),
])
def test_published_intervals(successes, trials, low, high):
    got_low, got_high = clopper_pearson(successes, trials)
    assert got_low == pytest.approx(low, abs=5e-4)
    assert got_high == pytest.approx(high, abs=5e-4)


def test_intervals_satisfy_the_clopper_pearson_definition():
    """The strongest check: verify via the BINOMIAL CDF, not via the beta
    function the implementation uses, so the test cannot agree with a
    defective implementation of its own formula.

    p_high solves P(X <= x | p_high) = 0.025; p_low solves
    P(X >= x | p_low) = 0.025.
    """
    import math

    def cdf(k, n, p):
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
                   for i in range(k + 1))

    def sf_inclusive(k, n, p):
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
                   for i in range(k, n + 1))

    for successes, trials in ((3, 36), (19, 36), (5, 10), (2, 20),
                              (36, 36), (0, 36), (1, 5)):
        low, high = clopper_pearson(successes, trials)
        if successes < trials:
            assert cdf(successes, trials, high) == pytest.approx(0.025,
                                                                 abs=1e-9)
        if successes > 0:
            assert sf_inclusive(successes, trials, low) == pytest.approx(
                0.025, abs=1e-9)


def test_the_boundary_cases_are_not_zero_width():
    """The reason the exact interval is used at all."""
    low, high = clopper_pearson(36, 36)
    assert low < 1.0 and high == 1.0
    assert high - low > 0.07
    low, high = clopper_pearson(0, 36)
    assert low == 0.0 and high > 0.0


def test_the_interval_contains_the_point_estimate():
    for successes in range(0, 37):
        low, high = clopper_pearson(successes, 36)
        assert low <= successes / 36 <= high
        assert low <= high


def test_it_is_symmetric_under_relabelling_success_and_failure():
    for successes in (0, 1, 7, 19, 35, 36):
        low, high = clopper_pearson(successes, 36)
        flipped_low, flipped_high = clopper_pearson(36 - successes, 36)
        assert low == pytest.approx(1.0 - flipped_high, abs=1e-9)
        assert high == pytest.approx(1.0 - flipped_low, abs=1e-9)


def test_zero_trials_and_bad_input():
    assert clopper_pearson(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        clopper_pearson(5, 3)
