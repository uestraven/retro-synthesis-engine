"""The sealed test evaluator.

Tested before it is run, because it runs ONCE and a defect found afterwards
cannot be fixed without re-opening the test set.
"""

from __future__ import annotations

import math

import pytest

from rse.snes_lm.evaluate_test import (
    TrackScore,
    aggregate,
    level1_verdict,
    paired_bootstrap,
)
from rse.snes_lm.representation import STRIDE


def track(track_id, game, bits, tokens, slots=None):
    slots = slots or [bits / STRIDE] * STRIDE
    return TrackScore(
        track_id=track_id, game=game, bits=bits, n_tokens=tokens,
        slot_bits=list(slots), slot_counts=[tokens // STRIDE] * STRIDE,
    )


# --- aggregation ----------------------------------------------------------


def test_overall_loss_is_token_weighted_not_track_averaged() -> None:
    """A 57,510-token track and a 300-token jingle must not count equally.
    Averaging per-track bits/token would weight them the same."""
    scores = [track("a", "G", 1000.0, 1000), track("b", "G", 100.0, 10)]
    result = aggregate(scores)
    assert result["bits_per_token"] == pytest.approx(1100.0 / 1010.0)
    naive = (1000.0 / 1000 + 100.0 / 10) / 2
    assert result["bits_per_token"] != pytest.approx(naive)


def test_per_game_figures_cover_every_game() -> None:
    scores = [
        track("a", "Actraiser", 100.0, 100),
        track("b", "F-Zero", 200.0, 100),
        track("c", "Actraiser", 50.0, 100),
    ]
    result = aggregate(scores)
    assert set(result["per_game_bits"]) == {"Actraiser", "F-Zero"}
    assert result["per_game_bits"]["Actraiser"] == pytest.approx(150.0 / 200)


# --- the bootstrap --------------------------------------------------------


def test_the_bootstrap_is_paired_by_track() -> None:
    """Resampling the two models independently would discard the pairing and
    inflate the interval. A mismatch must raise, not be silently averaged."""
    model = [track("a", "G", 50.0, 100)]
    baseline = [track("DIFFERENT", "G", 100.0, 100)]
    with pytest.raises(AssertionError):
        paired_bootstrap(model, baseline, n_resamples=10)


def test_a_clear_improvement_gives_an_interval_above_five_percent() -> None:
    model = [track(f"t{i}", "G", 70.0, 100) for i in range(40)]
    baseline = [track(f"t{i}", "G", 100.0, 100) for i in range(40)]
    out = paired_bootstrap(model, baseline, n_resamples=2000, seed=0)
    assert out["observed_relative_improvement"] == pytest.approx(0.30)
    assert out["ci_lower_exceeds_5pct"]
    assert out["ci_excludes_zero"]


def test_no_improvement_gives_an_interval_containing_zero() -> None:
    model = [track(f"t{i}", "G", 100.0, 100) for i in range(40)]
    baseline = [track(f"t{i}", "G", 100.0, 100) for i in range(40)]
    out = paired_bootstrap(model, baseline, n_resamples=2000, seed=0)
    assert out["observed_relative_improvement"] == pytest.approx(0.0)
    assert not out["ci_excludes_zero"]
    assert not out["ci_lower_exceeds_5pct"]


def test_a_marginal_improvement_does_not_claim_the_threshold() -> None:
    """3% observed must not report the 5% criterion as supported, however
    tight the interval."""
    model = [track(f"t{i}", "G", 97.0, 100) for i in range(60)]
    baseline = [track(f"t{i}", "G", 100.0, 100) for i in range(60)]
    out = paired_bootstrap(model, baseline, n_resamples=2000, seed=0)
    assert out["observed_relative_improvement"] == pytest.approx(0.03)
    assert out["ci_excludes_zero"]
    assert not out["ci_lower_exceeds_5pct"]


def test_variable_tracks_widen_the_interval() -> None:
    """With heterogeneous per-track improvement the interval must be wider
    than in the uniform case, or it is not measuring track-level uncertainty."""
    import random as _r

    rng = _r.Random(0)
    model, baseline = [], []
    for i in range(40):
        gain = rng.choice([0.05, 0.5])
        baseline.append(track(f"t{i}", "G", 100.0, 100))
        model.append(track(f"t{i}", "G", 100.0 * (1 - gain), 100))
    varied = paired_bootstrap(model, baseline, n_resamples=2000, seed=0)
    uniform = paired_bootstrap(
        [track(f"t{i}", "G", 72.5, 100) for i in range(40)],
        [track(f"t{i}", "G", 100.0, 100) for i in range(40)],
        n_resamples=2000, seed=0,
    )
    width = lambda o: o["ci_upper"] - o["ci_lower"]  # noqa: E731
    assert width(varied) > width(uniform)


def test_the_bootstrap_is_deterministic_for_a_seed() -> None:
    args = (
        [track(f"t{i}", "G", 80.0, 100) for i in range(20)],
        [track(f"t{i}", "G", 100.0, 100) for i in range(20)],
    )
    a = paired_bootstrap(*args, n_resamples=500, seed=7)
    b = paired_bootstrap(*args, n_resamples=500, seed=7)
    assert a == b


# --- the registered verdict ----------------------------------------------


def test_the_verdict_uses_the_test_measured_baseline() -> None:
    """The threshold is 0.95 x the baseline measured ON TEST, not the
    validation reference of 3.2147. The splits are non-interchangeable, so a
    validation-derived bar would be the wrong one."""
    boot = {"ci_lower": 0.10, "ci_lower_exceeds_5pct": True}
    verdict = level1_verdict(3.00, 3.50, boot)
    assert verdict["threshold_bits"] == pytest.approx(3.325)
    assert verdict["meets_5pct_threshold"]
    assert verdict["relative_improvement"] == pytest.approx(0.5 / 3.5)


def test_a_sub_threshold_improvement_fails_the_criterion() -> None:
    """An improvement that is real but under 5% must read as NOT MET, which is
    the registered 'learning demonstrated, threshold not met' outcome."""
    boot = {"ci_lower": 0.02, "ci_lower_exceeds_5pct": False}
    verdict = level1_verdict(3.45, 3.50, boot)
    assert not verdict["meets_5pct_threshold"]
    assert "NOT MET" in verdict["verdict"]
    assert verdict["relative_improvement"] > 0


def test_the_verdict_disclaims_level_2() -> None:
    """A likelihood result must not be reported as a musical one."""
    boot = {"ci_lower": 0.2, "ci_lower_exceeds_5pct": True}
    note = level1_verdict(2.5, 3.4, boot)["note"]
    assert "Level 2" in note and "listener" in note
    assert "reproducibility" in note
