"""Amendment A1: eligibility, denominators, and declared selection failure."""
import pytest

from rse.snes_v2 import selection as sel
from rse.snes_v2 import statistics as sx

REFERENCE = {"attacks_per_sec": 17.8806, "mean_simultaneity": 2.3014,
             "interval_histogram": {"0": 0.3, "2": 0.4, "-2": 0.3}}


def stats(attacks, simultaneity, histogram=None):
    """A per-excerpt statistics dict, bypassing note construction."""
    return {"n_notes": 100, "span_sec": 50.0,
            "attacks_per_sec": attacks, "mean_simultaneity": simultaneity,
            "interval_counts": histogram or {0: 30, 2: 40, -2: 30}}


def candidate(temperature, top_p, *, per_seed):
    """per_seed: list of 4 (outcome, statistics-or-None) in seed order."""
    return sel.Candidate(temperature=temperature, top_p=top_p, outcomes=[
        sel.Outcome(seed=seed, outcome=outcome, statistics=statistics)
        for seed, (outcome, statistics) in zip(sel.DEVELOPMENT_SEEDS, per_seed)])


def good(attacks=17.8806, simultaneity=2.3014):
    return [("complete", stats(attacks, simultaneity))] * 4


def test_grid_and_denominator_constants_match_the_frozen_procedure():
    assert sel.TEMPERATURES == (0.8, 0.9, 1.0, 1.1, 1.2)
    assert sel.TOP_P_VALUES == (1.0, 0.95)
    assert sel.DEVELOPMENT_SEEDS == (1000, 1001, 1002, 1003)
    assert sel.EXPECTED_ATTEMPTED == 40


# --- the required test: an ineligible candidate cannot win -------------------

def test_ineligible_candidate_cannot_win_however_good_its_survivors():
    """A perfect single seed must not outrank a merely-good complete set."""
    perfect = stats(REFERENCE["attacks_per_sec"], REFERENCE["mean_simultaneity"])
    one_perfect_three_dead = candidate(0.8, 1.0, per_seed=[
        ("complete", perfect),
        ("invalid", None), ("invalid", None), ("complete", None)])
    mediocre_but_complete = candidate(1.2, 0.95, per_seed=good(14.0, 2.9))

    result = sel.select([one_perfect_three_dead, mediocre_but_complete],
                        REFERENCE)
    assert result.successful
    assert result.selected["temperature"] == 1.2
    assert result.n_eligible_candidates == 1

    # And it is not merely outranked: it carries no objective at all, so no
    # arithmetic over its survivors exists to compare.
    row = next(r for r in result.rows if r["temperature"] == 0.8)
    assert row["eligible"] is False
    assert row["objective"] is None
    assert row["aggregate"] is None
    assert len(row["ineligibility_reasons"]) == 3


def test_ineligible_candidate_cannot_win_when_it_is_the_only_scorable_one():
    """The degenerate form: nothing else in the grid at all."""
    result = sel.select([candidate(0.9, 1.0, per_seed=[
        ("complete", stats(17.8806, 2.3014)),
        ("invalid", None), ("invalid", None), ("invalid", None)])], REFERENCE)
    assert result.successful is False
    assert result.selected is None


def test_a_zero_note_complete_excerpt_is_ineligible_not_merely_odd():
    """Zero notes is the case Amendment A1 was written for: the outcome
    category is `complete`, yet the metrics are undefined."""
    empty = sx.excerpt_statistics([])          # None, by construction
    result = sel.select([candidate(1.0, 1.0, per_seed=[
        ("complete", stats(17.0, 2.3)), ("complete", stats(18.0, 2.3)),
        ("complete", stats(18.0, 2.3)), ("complete", empty)])], REFERENCE)
    assert result.successful is False
    assert "no defined metrics" in result.rows[0]["ineligibility_reasons"][0]


# --- the required test: all four outcomes accounted for ---------------------

def test_every_attempted_outcome_appears_in_exactly_one_category():
    grid = [candidate(t, p, per_seed=good())
            for t in sel.TEMPERATURES for p in sel.TOP_P_VALUES]
    result = sel.select(grid, REFERENCE)
    assert result.n_attempted_outcomes == sel.EXPECTED_ATTEMPTED == 40
    assert sum(result.outcome_tally.values()) == 40
    assert set(result.outcome_tally) == set(sel.OUTCOME_CATEGORIES)


def test_the_tally_reports_all_four_categories_including_the_empty_ones():
    """A category reported only when nonzero is how a denominator disappears."""
    result = sel.select([candidate(1.0, 1.0, per_seed=good())], REFERENCE)
    assert result.outcome_tally == {"complete": 4, "short": 0,
                                   "budget_exhausted": 0, "invalid": 0}


def test_mixed_categories_are_tallied_and_do_not_affect_eligibility():
    """short and budget_exhausted stay DECODABLE outcomes under A1."""
    mixed = candidate(1.0, 1.0, per_seed=[
        ("complete", stats(17.0, 2.3)), ("short", stats(18.0, 2.3)),
        ("budget_exhausted", stats(18.0, 2.3)), ("complete", stats(17.5, 2.3))])
    result = sel.select([mixed], REFERENCE)
    assert result.rows[0]["eligible"] is True
    assert result.successful
    assert result.outcome_tally == {"complete": 2, "short": 1,
                                   "budget_exhausted": 1, "invalid": 0}


def test_a_candidate_missing_a_seed_is_ineligible():
    short_grid = sel.Candidate(temperature=1.0, top_p=1.0, outcomes=[
        sel.Outcome(seed=s, outcome="complete", statistics=stats(17.9, 2.3))
        for s in (1000, 1001, 1002)])
    result = sel.select([short_grid], REFERENCE)
    assert result.successful is False
    assert result.n_attempted_outcomes == 3        # reported, not patched up
    assert "expected seeds" in result.rows[0]["ineligibility_reasons"][0]


def test_an_unregistered_outcome_category_is_rejected_at_construction():
    with pytest.raises(ValueError, match="unregistered outcome"):
        sel.Outcome(seed=1000, outcome="regenerated", statistics=None)


# --- the required test: an entirely ineligible grid fails ------------------

def test_an_entirely_ineligible_grid_records_a_selection_failure():
    grid = [candidate(t, p, per_seed=[("invalid", None)] * 4)
            for t in sel.TEMPERATURES for p in sel.TOP_P_VALUES]
    result = sel.select(grid, REFERENCE)
    assert result.successful is False
    assert result.selected is None
    assert result.n_eligible_candidates == 0
    assert result.n_attempted_outcomes == 40       # denominator survives
    assert result.outcome_tally["invalid"] == 40
    assert "declared unsuccessful" in result.failure_reason
    assert "no fallback" in result.failure_reason


def test_failure_is_declared_rather_than_filled_with_the_best_survivor():
    """Nine dead candidates and one candidate with three dead seeds: the
    tempting move is to take the survivor. The procedure forbids it."""
    grid = [candidate(t, 1.0, per_seed=[("invalid", None)] * 4)
            for t in sel.TEMPERATURES]
    grid.append(candidate(0.8, 0.95, per_seed=[
        ("complete", stats(17.8806, 2.3014)),
        ("invalid", None), ("invalid", None), ("invalid", None)]))
    result = sel.select(grid, REFERENCE)
    assert result.successful is False
    assert all(r["objective"] is None for r in result.rows)


# --- denominators are separate ---------------------------------------------

def test_eligible_and_attempted_denominators_are_independent():
    grid = [candidate(0.8, 1.0, per_seed=good()),
            candidate(0.9, 1.0, per_seed=good()),
            candidate(1.0, 1.0, per_seed=[("complete", stats(17.9, 2.3))] * 3
                      + [("invalid", None)])]
    result = sel.select(grid, REFERENCE)
    assert result.n_candidates == 3
    assert result.n_eligible_candidates == 2
    assert result.n_attempted_outcomes == 12
    # The eligible count is not a share of outcomes, and vice versa.
    assert result.n_eligible_candidates != result.n_attempted_outcomes


def test_every_outcome_is_retained_in_the_report_including_the_failures():
    grid = [candidate(1.1, 0.95, per_seed=[
        ("complete", stats(17.9, 2.3)), ("invalid", None),
        ("short", stats(18.2, 2.2)), ("budget_exhausted", None)])]
    result = sel.select(grid, REFERENCE)
    assert len(result.rows[0]["outcomes"]) == 4
    assert [o["seed"] for o in result.rows[0]["outcomes"]] == [1000, 1001,
                                                               1002, 1003]
    assert [o["scorable"] for o in result.rows[0]["outcomes"]] == [
        True, False, True, False]


# --- ranking and tie-break --------------------------------------------------

def test_the_closest_eligible_candidate_wins():
    grid = [candidate(0.8, 1.0, per_seed=good(5.0, 1.0)),
            candidate(0.9, 1.0, per_seed=good(17.8806, 2.3014)),
            candidate(1.0, 1.0, per_seed=good(40.0, 6.0))]
    result = sel.select(grid, REFERENCE)
    assert result.selected["temperature"] == 0.9
    assert result.selected["objective"] == pytest.approx(0.0, abs=1e-9)


def test_tie_break_prefers_the_lower_temperature_then_the_higher_top_p():
    identical = good(17.8806, 2.3014)
    grid = [candidate(1.1, 1.0, per_seed=identical),
            candidate(0.9, 0.95, per_seed=identical),
            candidate(0.9, 1.0, per_seed=identical)]
    result = sel.select(grid, REFERENCE)
    assert (result.selected["temperature"], result.selected["top_p"]) == (0.9, 1.0)


# --- durations are reported, never scored ----------------------------------

def test_duration_does_not_enter_the_objective():
    """The registered decision was to leave section 4 frozen. So a candidate
    whose excerpts barely play must score exactly as one that plays in
    target -- and the report must make that visible."""
    def with_durations(temperature, durations):
        return sel.Candidate(temperature=temperature, top_p=1.0, outcomes=[
            sel.Outcome(seed=seed, outcome="short", statistics=stats(17.9, 2.3),
                        duration_sec=duration)
            for seed, duration in zip(sel.DEVELOPMENT_SEEDS, durations)])

    barely_plays = with_durations(0.8, [0.05, 0.3, 0.5, 1.0])
    in_target = with_durations(0.9, [47.0, 48.0, 52.0, 59.0])
    result = sel.select([barely_plays, in_target], REFERENCE)
    rows = {r["temperature"]: r for r in result.rows}
    assert rows[0.8]["objective"] == rows[0.9]["objective"], \
        "duration must not change the frozen objective"
    # ...and the tie-break's lowest temperature still wins, barely-playing.
    assert result.selected["temperature"] == 0.8
    # The visibility that replaces the amendment:
    assert rows[0.8]["duration"]["n_in_target"] == 0
    assert rows[0.8]["duration"]["n_under_5_sec"] == 4
    assert rows[0.9]["duration"]["n_in_target"] == 4
    assert rows[0.8]["duration"]["scored"] is False
    assert result.selected["duration"]["n_in_target"] == 0


def test_the_overshoot_flag_is_carried_into_the_candidate_report():
    over = sel.Candidate(temperature=1.0, top_p=1.0, outcomes=[
        sel.Outcome(seed=seed, outcome="complete", statistics=stats(17.9, 2.3),
                    duration_sec=duration, exceeded_hard_cap=duration > 60.0)
        for seed, duration in zip(sel.DEVELOPMENT_SEEDS,
                                  [47.0, 85.23, 104.63, 50.0])])
    result = sel.select([over], REFERENCE)
    assert result.rows[0]["duration"]["n_exceeded_hard_cap"] == 2
    assert result.rows[0]["duration"]["n_in_target"] == 2
    assert result.rows[0]["duration"]["max_sec"] == 104.63
