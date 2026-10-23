from __future__ import annotations

import numpy as np
import pytest

from rse.generation.loops import (
    LoopReport,
    max_adjacent_step,
    seam_loudness_db,
    seam_step,
    trailing_silence_sec,
    validate_loop,
)
from rse.generation.plan import make_plan
from rse.generation.procedural import generate
from rse.renderers.base import RenderSettings, notes_from_record
from rse.renderers.nes_apu import NesApuRenderer


def record_with(notes, *, n_bars=2, markers=True, tonic=0):
    return {
        "midi": {"ticks_per_beat": 480},
        "time_signatures": [{"tick": 0, "numerator": 4, "denominator": 4}],
        "loop_markers": [{"tick": 0, "text": "loop_start"}] if markers else [],
        "measurements": {"n_bars": n_bars},
        "events": notes,
        "generation": {
            "plan": {
                "n_bars": n_bars,
                "beats_per_bar": 4,
                "ticks_per_beat": 480,
                "key": {"tonic": tonic, "scale": "major"},
            }
        },
    }


def note(tick, pitch, dur=240):
    return {"ch": "P1", "type": "note", "tick": tick, "dur_ticks": dur, "pitch": pitch}


# -- measurements -------------------------------------------------------------


def test_seam_step_measures_the_wrap():
    # Wrapping from -1 back to +1 is a step of 2, against a peak of 1.
    assert seam_step(np.array([1.0, 0.0, -1.0])) == pytest.approx(2.0)
    assert seam_step(np.array([0.5, 0.2, 0.5])) == pytest.approx(0.0)
    assert seam_step(np.array([])) == 0.0
    assert seam_step(np.zeros(10)) == 0.0


def test_max_adjacent_step_finds_the_sharpest_internal_transition():
    square = np.array([1.0, 1.0, -1.0, -1.0, 1.0])
    assert max_adjacent_step(square) == pytest.approx(2.0)
    assert max_adjacent_step(np.zeros(5)) == 0.0


def test_trailing_silence_is_measured_in_seconds():
    samples = np.concatenate([np.ones(100), np.zeros(100)])
    assert trailing_silence_sec(samples, 100) == pytest.approx(0.99, abs=0.02)
    assert trailing_silence_sec(np.ones(100), 100) == 0.0
    assert trailing_silence_sec(np.zeros(50), 100) == pytest.approx(0.5)


def test_seam_loudness_compares_start_against_end():
    rate = 1000
    loud_then_quiet = np.concatenate([np.ones(rate) * 0.8, np.ones(rate) * 0.08])
    assert seam_loudness_db(loud_then_quiet, rate) == pytest.approx(20.0, abs=0.5)
    steady = np.ones(2 * rate) * 0.5
    assert seam_loudness_db(steady, rate) == pytest.approx(0.0, abs=0.1)


# -- musical validation -------------------------------------------------------


def test_an_empty_composition_cannot_loop():
    report = validate_loop(record_with([]))
    assert not report.ok
    assert "no notes" in report.issues[0]


def test_a_clean_score_passes():
    notes = [note(0, 60), note(480, 64), note(960, 67), note(1440, 60)]
    report = validate_loop(record_with(notes, n_bars=1))
    assert report.ok, report.issues
    assert report.resolves_to_tonic


def test_notes_past_the_boundary_are_caught():
    # Final note runs 480 ticks past the end of a one-bar loop.
    notes = [note(0, 60), note(1440, 60, dur=960)]
    report = validate_loop(record_with(notes, n_bars=1))
    assert not report.ok
    assert report.notes_crossing_boundary == 1
    assert "past the loop boundary" in report.issues[0]


def test_missing_loop_markers_are_flagged():
    report = validate_loop(record_with([note(0, 60)], n_bars=1, markers=False))
    assert not report.has_loop_markers
    assert any("no loop markers" in issue for issue in report.issues)


def test_an_unresolved_ending_is_flagged():
    # Ends on the fifth, never touching the tonic in the final bar.
    report = validate_loop(record_with([note(0, 67), note(1440, 67)], n_bars=1, tonic=0))
    assert not report.resolves_to_tonic
    assert any("does not feel resolved" in i or "not touch the tonic" in i for i in report.issues)


def test_noise_only_endings_are_not_judged_for_resolution():
    """The noise channel has no pitch, so it cannot resolve to anything."""
    notes = [
        {"ch": "NO", "type": "noise", "tick": 0, "dur_ticks": 60, "value": 7},
        {"ch": "NO", "type": "noise", "tick": 1440, "dur_ticks": 60, "value": 7},
    ]
    report = validate_loop(record_with(notes, n_bars=1))
    assert not any("tonic" in issue for issue in report.issues)


# -- acoustic validation ------------------------------------------------------


def test_trailing_silence_is_reported_as_a_gap():
    rate = 44_100
    samples = np.concatenate([np.ones(rate) * 0.5, np.zeros(rate // 2)])
    report = validate_loop(
        record_with([note(0, 60), note(1440, 60)], n_bars=1),
        samples=samples,
        sample_rate=rate,
    )
    assert any("silence at the end" in issue for issue in report.issues)


def test_a_loop_whose_end_fades_out_slams_on_repeat():
    # Loud start, quiet end: wrapping jumps from near-silence to full level.
    rate = 44_100
    samples = np.concatenate([np.ones(rate) * 0.9, np.ones(rate) * 0.01])
    report = validate_loop(
        record_with([note(0, 60), note(1440, 60)], n_bars=1),
        samples=samples,
        sample_rate=rate,
    )
    assert any("slams in" in issue for issue in report.issues), report.issues


def test_a_loop_that_dies_away_into_its_start_is_flagged():
    # Quiet start, loud end: the repeat drops in level instead.
    rate = 44_100
    samples = np.concatenate([np.ones(rate) * 0.01, np.ones(rate) * 0.9])
    report = validate_loop(
        record_with([note(0, 60), note(1440, 60)], n_bars=1),
        samples=samples,
        sample_rate=rate,
    )
    assert any("dies away" in issue for issue in report.issues), report.issues


def test_square_wave_transitions_are_not_mistaken_for_clicks():
    """A pulse wave steps full-scale constantly; that is not a seam defect."""
    rate = 44_100
    cycle = np.concatenate([np.ones(50), -np.ones(50)])
    samples = np.tile(cycle, rate // 100)
    report = validate_loop(
        record_with([note(0, 60), note(1440, 60)], n_bars=1),
        samples=samples,
        sample_rate=rate,
    )
    assert not any("click" in issue for issue in report.issues), report.issues


def test_report_serializes():
    payload = LoopReport(n_bars=8, seam_step=0.1, issues=["x"]).to_dict()
    assert payload["ok"] is False
    assert payload["n_bars"] == 8
    assert payload["seam_step"] == 0.1


# -- end to end ---------------------------------------------------------------


def test_generated_loops_validate_cleanly(nes_profile):
    """Every mood and energy must produce a loop that passes both checks."""
    renderer = NesApuRenderer(nes_profile)
    for mood in ("heroic", "dark", "mysterious", "melancholic"):
        for energy in ("low", "medium", "high"):
            composition = generate(
                nes_profile, make_plan(seed=7, mood=mood, energy=energy, n_bars=8)
            )
            result = renderer.render(
                notes_from_record(composition.record), RenderSettings(tail_sec=0.0)
            )
            report = validate_loop(
                composition.record,
                samples=result.samples,
                sample_rate=result.sample_rate,
            )
            assert report.ok, f"{mood}/{energy}: {report.issues}"


def test_a_render_with_a_tail_is_flagged_as_gapped(nes_profile):
    """Loops must be rendered with no trailing silence, and the check proves it."""
    composition = generate(nes_profile, make_plan(seed=3, n_bars=4))
    renderer = NesApuRenderer(nes_profile)
    result = renderer.render(notes_from_record(composition.record), RenderSettings(tail_sec=0.5))
    report = validate_loop(
        composition.record, samples=result.samples, sample_rate=result.sample_rate
    )
    assert any("silence at the end" in issue for issue in report.issues)
