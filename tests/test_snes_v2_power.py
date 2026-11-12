"""The rotation must actually rotate, and must not flatter the design.

A power simulation that silently crosses every listener with every pair
overstates precision -- 20 listeners x 60 pairs is 1,200 judgements, each two
20-second clips, which no listener set will ever supply. The registered
Milestone 4 figures depend on the rotation being real.
"""
import importlib.util
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def power():
    spec = importlib.util.spec_from_file_location(
        "power_check", ROOT / "scripts/snes_v2_power_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fully_crossed_is_the_default_and_unchanged(power):
    """Milestone 3's reported figures used it; the default must not drift."""
    js = power.simulate(10, 0.5, seed=1, n_listeners=20)
    assert len(js) == 200
    assert len({j.pair for j in js}) == 10
    assert len({j.listener for j in js}) == 20


def test_rotation_limits_each_listener_to_its_assignment(power):
    js = power.simulate(30, 0.5, seed=1, n_listeners=20,
                        pairs_per_listener=10)
    assert len(js) == 200, "20 listeners x 10 items"
    per_listener = Counter(j.listener for j in js)
    assert set(per_listener.values()) == {10}
    assert len({j.pair for j in js}) == 30, "every pair must still be covered"


def test_rotation_spreads_judgements_evenly_across_pairs(power):
    js = power.simulate(24, 0.5, seed=7, n_listeners=100,
                        pairs_per_listener=10)
    per_pair = Counter(j.pair for j in js)
    assert len(per_pair) == 24
    assert max(per_pair.values()) - min(per_pair.values()) <= 1, \
        f"uneven coverage would bias the pair cluster: {per_pair}"
    assert sum(per_pair.values()) == 1000


def test_a_listener_never_judges_the_same_pair_twice(power):
    js = power.simulate(24, 0.5, seed=3, n_listeners=40,
                        pairs_per_listener=10)
    seen = Counter((j.listener, j.pair) for j in js)
    assert max(seen.values()) == 1


def test_an_assignment_larger_than_the_pair_set_is_clamped(power):
    js = power.simulate(6, 0.5, seed=1, n_listeners=5, pairs_per_listener=10)
    per_listener = Counter(j.listener for j in js)
    assert set(per_listener.values()) == {6}, "cannot assign more than exist"


def test_rotation_power_is_not_above_fully_crossed(power):
    """The rotation must cost precision, not gain it -- otherwise the
    registered listener requirement is built on a simulation bug."""
    from rse.snes_v2 import preference as px
    full = px.analyse(power.simulate(24, 0.65, seed=11, n_listeners=40),
                      n_replicates=800, seed=1)
    rot = px.analyse(power.simulate(24, 0.65, seed=11, n_listeners=40,
                                    pairs_per_listener=10),
                     n_replicates=800, seed=1)
    assert (rot.ci_high - rot.ci_low) > (full.ci_high - full.ci_low)
