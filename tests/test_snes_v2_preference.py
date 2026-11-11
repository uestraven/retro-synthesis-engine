"""The seven registered synthetic cases for the clustered bootstrap.

Fixtures are preserved with the implementation. The estimator is tested on
synthetic responses BEFORE any real listener response exists, so the CI method
cannot be tuned to the data it will judge.
"""
import random
import pytest
from rse.snes_v2 import preference as pf

#: The registered administration plan: 42 pairs, 20 listeners, 14 pairs each,
#: so each pair is judged by roughly 6-7 listeners under a fixed rotation.
N_LISTENERS, N_PAIRS, PER_LISTENER = 20, 42, 14


def plan():
    """The incomplete assignment structure, deterministic."""
    out = []
    for i in range(N_LISTENERS):
        for k in range(PER_LISTENER):
            out.append((f"L{i:02d}", f"P{(i * PER_LISTENER + k) % N_PAIRS:02d}"))
    return out


def build(choice_of, *, missing=()):
    return [pf.Judgement(listener, pair,
                         None if (listener, pair) in missing
                         else choice_of(listener, pair))
            for listener, pair in plan()]


# --- case 1: all ties ----------------------------------------------------

def test_all_ties_gives_50_percent_and_no_superiority():
    r = pf.analyse(build(lambda l, p: "tie"), n_replicates=400)
    assert r.adjusted_preference == pytest.approx(0.50)
    assert not r.superiority_established
    assert not r.threshold_met
    assert r.verdict == "threshold not met"
    assert r.n_ties == r.n_valid          # ties are data, not missing


# --- case 2: balanced preferences ---------------------------------------

def test_balanced_preferences_show_no_systematic_superiority():
    rng = random.Random(5)
    r = pf.analyse(build(lambda l, p: rng.choice(["v2", "baseline"])),
                   n_replicates=400)
    assert 0.40 < r.adjusted_preference < 0.60
    assert not r.superiority_established


# --- case 3: 12/20, the case that must NOT auto-pass ---------------------

def test_twelve_of_twenty_is_inconclusive_not_a_pass():
    """60% observed on 20 independent judgements is consistent with chance."""
    judgements = [pf.Judgement(f"L{i:02d}", "P00",
                               "v2" if i < 12 else "baseline")
                  for i in range(20)]
    r = pf.analyse(judgements, n_replicates=2000)
    assert r.adjusted_preference == pytest.approx(0.60)
    assert r.threshold_met                      # practical threshold met
    assert not r.superiority_established, (r.ci_low, r.ci_high)
    assert r.verdict == "threshold met, superiority not established"


# --- case 4: a clear effect is recovered ---------------------------------

def test_clear_advantage_recovers_direction_and_establishes_superiority():
    rng = random.Random(7)
    r = pf.analyse(build(lambda l, p: "v2" if rng.random() < 0.90
                         else "baseline"), n_replicates=600)
    assert r.adjusted_preference > 0.80
    assert r.superiority_established and r.threshold_met
    assert r.verdict == "demonstrated preference"


# --- case 5: missing responses ------------------------------------------

def test_missing_responses_follow_the_registered_rule():
    all_pairs = plan()
    # L00 misses 10 of 14 (>50%) and must be excluded entirely;
    # L01 misses 2 of 14 and must be kept with those two dropped.
    missing = {(l, p) for l, p in all_pairs if l == "L00"}
    missing = set(sorted(missing)[:10]) | set(
        sorted((l, p) for l, p in all_pairs if l == "L01")[:2])
    r = pf.analyse(build(lambda l, p: "v2", missing=missing),
                   n_replicates=300)
    assert "L00" in r.excluded_listeners
    assert "L01" not in r.excluded_listeners
    assert r.n_missing == 2                  # L00's are gone with the listener
    assert r.n_listeners == N_LISTENERS - 1
    assert r.n_pairs == N_PAIRS              # pairs stay in the study


# --- case 6: unequal judgements per pair ---------------------------------

def test_unequal_judgements_per_pair_are_not_silently_equalised():
    judgements = ([pf.Judgement(f"L{i:02d}", "P00", "v2") for i in range(9)]
                  + [pf.Judgement("L00", "P01", "baseline")])
    r = pf.analyse(judgements, n_replicates=300)
    assert r.judgements_per_pair == {"P00": 9, "P01": 1}
    # Judgement-weighted, so the 9-judgement pair dominates the point estimate;
    # an equal-pair assumption would have given 0.5.
    assert r.adjusted_preference == pytest.approx(0.9)


# --- case 7: label symmetry ---------------------------------------------

def test_result_is_symmetric_under_ab_label_reversal():
    rng = random.Random(11)
    judgements = build(lambda l, p: rng.choice(["v2", "baseline", "tie"]))
    a = pf.analyse(judgements, n_replicates=400)
    b = pf.analyse(pf.reverse_labels(judgements), n_replicates=400)
    assert a.adjusted_preference == pytest.approx(1.0 - b.adjusted_preference)
    assert a.ci_low == pytest.approx(1.0 - b.ci_high)
    assert a.ci_high == pytest.approx(1.0 - b.ci_low)


# --- the plan itself, and clustering ------------------------------------

def test_administration_plan_matches_the_registered_structure():
    assignments = plan()
    assert len(assignments) == N_LISTENERS * PER_LISTENER == 280
    per_pair = {}
    for _, p in assignments:
        per_pair[p] = per_pair.get(p, 0) + 1
    assert len(per_pair) == N_PAIRS
    assert min(per_pair.values()) >= 6 and max(per_pair.values()) <= 7


def test_two_way_clustering_widens_the_interval_versus_naive_resampling():
    """A listener with a consistent bias must widen the interval.

    If judgements were treated as independent, a block of correlated responses
    would shrink the interval instead -- the Level 1 error.
    """
    rng = random.Random(3)
    # Half the listeners always prefer v2, half are random: strong
    # within-listener correlation.
    def choice(l, p):
        return "v2" if int(l[1:]) % 2 == 0 else rng.choice(["v2", "baseline"])
    r = pf.analyse(build(choice), n_replicates=800)
    width = r.ci_high - r.ci_low
    assert width > 0.05, width          # clustering keeps it honestly wide
    assert r.ci_low < r.adjusted_preference < r.ci_high


def test_registered_constants_are_fixed():
    assert pf.PRACTICAL_THRESHOLD == 0.60
    assert pf.CONFIDENCE == 0.95
    assert pf.N_REPLICATES == 10_000
    assert pf.CI_METHOD == "percentile"
    assert pf.MAX_MISSING_SHARE == 0.50
