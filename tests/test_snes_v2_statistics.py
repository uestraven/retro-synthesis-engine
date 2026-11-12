"""The frozen objective, and Amendment A1's eligibility rule."""
import pytest
from rse.snes_v2 import statistics as sx


def note(slot, start, midi=60.0):
    return {"slot": slot, "start_tick": start, "midi": midi,
            "status": "estimated", "end_tick": start + 100, "changes": []}


def test_zero_note_excerpt_has_no_statistics():
    """This is what makes Amendment A1 enforceable rather than aspirational."""
    assert sx.excerpt_statistics([]) is None


def test_single_instant_excerpt_has_no_span_and_no_statistics():
    assert sx.excerpt_statistics([note(0, 0), note(1, 0)]) is None


def test_attacks_per_second_uses_the_excerpt_span():
    notes = [note(0, 0), note(0, 4000), note(0, 8000)]      # 2 s span
    s = sx.excerpt_statistics(notes, grid_hz=4000.0)
    assert s["span_sec"] == pytest.approx(2.0)
    assert s["attacks_per_sec"] == pytest.approx(1.5)


def test_simultaneity_counts_notes_per_distinct_onset():
    notes = [note(0, 0), note(1, 0), note(2, 0), note(0, 4000)]
    s = sx.excerpt_statistics(notes, grid_hz=4000.0)
    assert s["mean_simultaneity"] == pytest.approx(2.0)      # (3 + 1) / 2


def test_median_not_pooled_aggregation():
    """Pooling decoded excerpts reported 296 attacks/s in Level 2 against a
    true corpus range of 13-26, because each item restarts its clock."""
    a = sx.excerpt_statistics([note(0, 0), note(0, 4000)], grid_hz=4000.0)
    b = sx.excerpt_statistics([note(0, 0), note(0, 40000)], grid_hz=4000.0)
    agg = sx.aggregate([a, b])
    assert agg["attacks_per_sec"] == pytest.approx(
        (a["attacks_per_sec"] + b["attacks_per_sec"]) / 2)


def test_objective_is_zero_against_itself():
    notes = [note(0, 0, 60.0), note(0, 4000, 62.0), note(1, 4000, 67.0)]
    agg = sx.aggregate([sx.excerpt_statistics(notes, grid_hz=4000.0)])
    assert sx.objective(agg, agg) == pytest.approx(0.0)


def test_objective_grows_with_density_mismatch():
    close = sx.aggregate([sx.excerpt_statistics(
        [note(0, 0), note(0, 4000), note(0, 8000)], grid_hz=4000.0)])
    far = sx.aggregate([sx.excerpt_statistics(
        [note(0, t * 100) for t in range(60)], grid_hz=4000.0)])
    assert sx.objective(far, close) > sx.objective(close, close)


def test_tie_break_is_lowest_temperature_then_highest_top_p():
    rows = [{"objective": 1.0000001, "temperature": 1.1, "top_p": 1.0},
            {"objective": 1.0000002, "temperature": 0.9, "top_p": 0.95},
            {"objective": 1.0000003, "temperature": 0.9, "top_p": 1.0}]
    best = min(rows, key=sx.tie_break_key)
    # All three round to the same 6dp objective, so temperature decides, then
    # the HIGHEST top_p among the equal temperatures.
    assert best["temperature"] == 0.9
    assert best["top_p"] == 1.0


def test_intervals_are_clipped_and_per_voice():
    notes = [note(0, 0, 60.0), note(0, 4000, 90.0), note(1, 0, 60.0)]
    s = sx.excerpt_statistics(notes, grid_hz=4000.0)
    assert set(s["interval_counts"]) == {sx.CLIP}      # +30 clipped to +12
