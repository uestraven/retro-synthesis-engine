from __future__ import annotations

import random

import pytest

from rse.events.bar import (
    MIN_SIMILARITY,
    BarEstimate,
    detect_bar,
    step_contents,
)
from rse.events.pulse import PulseEstimate
from rse.events.schema import Composition, Noise, Note

TPB = 480
STEP = 120  # one grid step, which is one 24 Hz frame at 480 ticks per beat
SEC_PER_TICK = 1.0 / 960

#: A pulse whose grid is one step of 120 ticks, phase zero. Passed explicitly so
#: these tests exercise bar detection rather than re-testing the pulse fit.
PULSE = PulseEstimate(float(STEP), 0.0, 0.99, 200.0, 64, SEC_PER_TICK)


def composition_from_steps(events_by_step, *, phase=0) -> Composition:
    """Build a composition from {step: [(channel, value)]}."""
    events = []
    for step, entries in sorted(events_by_step.items()):
        for channel, value in entries:
            tick = (step * STEP) + phase
            if channel == "NO":
                events.append(
                    Noise(channel_id="NO", tick=tick, value=value, duration=STEP, velocity=100)
                )
            else:
                events.append(
                    Note(channel_id=channel, tick=tick, pitch=value, duration=STEP, velocity=100)
                )
    return Composition(hardware_profile="nes", ticks_per_beat=TPB, events=events)


def repeating(pattern, *, length, n_repeats, phase=0) -> Composition:
    """`pattern` keyed by step within a bar, repeated `n_repeats` times."""
    steps = {}
    for repeat in range(n_repeats):
        for step, entries in pattern.items():
            steps[repeat * length + step] = entries
    return composition_from_steps(steps, phase=phase)


#: Beat one is accented: all four channels start together there and fewer do
#: elsewhere. Without an accent the downbeat is not recoverable at all — a drum
#: part hitting every four steps has the same density at four offsets.
BEAT = {
    0: [("NO", 8), ("P1", 60), ("P2", 48), ("TR", 36)],
    4: [("NO", 3)],
    8: [("NO", 8), ("P1", 64)],
    12: [("NO", 3), ("P1", 67)],
}


# -- step contents ------------------------------------------------------------


def test_contents_are_keyed_by_grid_step():
    comp = composition_from_steps({0: [("P1", 60)], 3: [("NO", 8)]})
    assert step_contents(comp, PULSE) == {0: frozenset({("P1", 60)}), 3: frozenset({("NO", 8)})}


def test_contents_need_a_pulse():
    comp = composition_from_steps({0: [("P1", 60)]})
    assert step_contents(comp, PulseEstimate(0.0, 0.0, 0.0, 0.0, 0)) == {}


def test_a_steps_pitches_are_kept_not_just_its_channels():
    """Comparing only channel occupancy measurably picks worse bar lengths."""
    comp = composition_from_steps({0: [("P1", 60), ("P2", 67)]})
    assert step_contents(comp, PULSE)[0] == frozenset({("P1", 60), ("P2", 67)})


# -- detection ----------------------------------------------------------------


def test_a_sixteen_step_pattern_gives_a_sixteen_step_bar():
    estimate = detect_bar(repeating(BEAT, length=16, n_repeats=8), PULSE)
    assert estimate.found
    assert estimate.length_steps == 16
    assert estimate.is_musical


def test_a_thirty_two_step_pattern_gives_a_thirty_two_step_bar():
    """The corpus median. A fixed 16 straddles the bar line here, which is the
    whole reason this module exists."""
    wide = dict(BEAT)
    wide[20] = [("P1", 72)]  # content that only recurs every 32 steps
    estimate = detect_bar(repeating(wide, length=32, n_repeats=6), PULSE)
    assert estimate.found
    assert estimate.length_steps == 32


def test_the_fundamental_is_reported_not_a_multiple():
    """A 16-step pattern also matches at 32 and 48; the bar is the smallest."""
    estimate = detect_bar(repeating(BEAT, length=16, n_repeats=8), PULSE)
    assert estimate.repeat_steps == 16


def test_a_short_repeat_is_doubled_into_a_bar_sized_window():
    """Drums repeating every 4 steps have a real 4-step period, but counting
    positions against 4 would fire four Bar tokens per notated bar."""
    tight = {0: [("NO", 8)], 2: [("NO", 3)]}
    estimate = detect_bar(repeating(tight, length=4, n_repeats=40), PULSE)
    assert estimate.found
    assert estimate.repeat_steps == 4
    assert estimate.length_steps == 16


def test_the_downbeat_offset_is_recovered():
    """Shifting the whole piece by 5 steps must move the detected bar start."""
    shifted = repeating(BEAT, length=16, n_repeats=8, phase=5 * STEP)
    estimate = detect_bar(shifted, PULSE)
    assert estimate.found
    assert estimate.offset_steps == 5


def test_a_piece_with_no_repeat_is_refused():
    rng = random.Random(0)
    steps = {step: [("P1", rng.randrange(40, 90))] for step in sorted(rng.sample(range(400), 120))}
    estimate = detect_bar(composition_from_steps(steps), PULSE)
    assert not estimate.found
    assert estimate.confidence < MIN_SIMILARITY


def test_a_piece_too_short_to_confirm_a_repeat_is_refused():
    estimate = detect_bar(repeating(BEAT, length=16, n_repeats=2), PULSE)
    assert not estimate.found
    assert estimate.n_steps < 64


def test_an_empty_composition_is_refused():
    empty = Composition(hardware_profile="nes", ticks_per_beat=TPB)
    estimate = detect_bar(empty, PULSE)
    assert not estimate.found
    assert estimate.n_steps == 0


def test_detection_needs_a_pulse():
    comp = repeating(BEAT, length=16, n_repeats=8)
    assert not detect_bar(comp, PulseEstimate(0.0, 0.0, 0.0, 0.0, 0)).found


# -- derived values -----------------------------------------------------------


def test_is_musical_marks_powers_of_two():
    assert BarEstimate(16, 0, 0.5, 16, 128).is_musical
    assert BarEstimate(32, 0, 0.5, 32, 128).is_musical
    # 37% of detected lengths are not powers of two, and a caller that cannot
    # tolerate a wrong bar should treat those as a refusal.
    assert not BarEstimate(24, 0, 0.5, 24, 128).is_musical
    assert not BarEstimate(51, 0, 0.5, 51, 128).is_musical


def test_position_splits_a_step_into_bar_and_offset():
    estimate = BarEstimate(16, 0, 0.5, 16, 128)
    assert estimate.position(0) == (0, 0)
    assert estimate.position(5) == (0, 5)
    assert estimate.position(16) == (1, 0)
    assert estimate.position(37) == (2, 5)


def test_position_counts_from_the_downbeat_offset():
    estimate = BarEstimate(16, 5, 0.5, 16, 128)
    assert estimate.position(5) == (0, 0)
    assert estimate.position(21) == (1, 0)


def test_position_is_a_no_op_without_an_estimate():
    refused = BarEstimate(0, 0, 0.0, 0, 0)
    assert refused.position(37) == (0, 37)


def test_to_dict_carries_the_honest_flags():
    payload = BarEstimate(24, 3, 0.41, 24, 128).to_dict()
    assert payload["length_steps"] == 24
    assert payload["is_musical"] is False
    assert payload["found"] is True
    assert payload["confidence"] == pytest.approx(0.41)


def test_an_unaccented_piece_has_no_recoverable_downbeat():
    """A uniform pattern gives the same count at every offset, so the lowest is
    returned. A wrong offset rotates positions by a constant, which is far less
    damaging than a wrong length."""
    flat = {step: [("NO", 8)] for step in range(0, 256, 4)}
    estimate = detect_bar(composition_from_steps(flat), PULSE)
    assert estimate.found
    assert estimate.offset_steps == 0
