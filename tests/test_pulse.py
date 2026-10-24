from __future__ import annotations

import random

import pytest

from rse.events.convert import composition_from_intake
from rse.events.pulse import (
    MIN_Z,
    Composition,
    PulseEstimate,
    detect_pulse,
    grid_index,
    on_grid_share,
    onset_ticks,
    quantize,
)
from rse.events.schema import Noise, Note
from rse.generation.plan import make_plan
from rse.generation.procedural import generate

#: 480 ticks per beat at the default 120 BPM is 960 ticks per second, so a tick
#: is just over a millisecond and the periods below are easy to read in ticks.
TPB = 480
SEC_PER_TICK = 1.0 / 960


def composition_at(ticks, *, noise=True, duration=48, ticks_per_beat=TPB) -> Composition:
    """A composition whose onsets are exactly where asked, nothing else.

    Durations are short so that every onset is an attack: a note that runs into
    the next one is deliberately not counted as marking a position.
    """
    events = []
    for index, tick in enumerate(ticks):
        if noise:
            events.append(
                Noise(channel_id="NO", tick=int(tick), value=13, duration=duration, velocity=100)
            )
        else:
            events.append(
                Note(
                    channel_id="P1",
                    tick=int(tick),
                    pitch=60 + index % 5,
                    duration=duration,
                    velocity=100,
                )
            )
    return Composition(hardware_profile="nes", ticks_per_beat=ticks_per_beat, events=events)


# -- onsets -------------------------------------------------------------------


def test_onsets_are_ticks():
    assert onset_ticks(composition_at([0, 240, 480])) == [0, 240, 480]


def test_percussion_can_be_isolated():
    mixed = composition_at([0, 480])
    mixed.events.append(Note(channel_id="P1", tick=240, pitch=60, duration=48, velocity=100))
    assert onset_ticks(mixed, percussion_only=True) == [0, 480]
    assert onset_ticks(mixed) == [0, 240, 480]


def test_attacks_exclude_notes_that_follow_on_from_the_one_before():
    """A legato run marks its first note's position and nothing after it."""
    legato = composition_at([0, 240, 480, 720], duration=240)
    assert onset_ticks(legato) == [0, 240, 480, 720]
    assert onset_ticks(legato, attacks_only=True) == [0]


def test_a_gap_starts_a_new_attack():
    detached = composition_at([0, 240, 480, 960], duration=240)
    assert onset_ticks(detached, attacks_only=True) == [0, 960]


# -- detection ----------------------------------------------------------------


def test_a_regular_pattern_gives_its_own_period():
    estimate = detect_pulse(composition_at([i * 240 for i in range(32)]))
    assert estimate.found
    assert estimate.period_ticks == pytest.approx(240, rel=0.01)
    # Not exactly 1.0: the refinement searches a finite grid of periods.
    assert estimate.strength == pytest.approx(1.0, abs=1e-3)


def test_the_coarsest_grid_is_reported_not_a_subdivision():
    """Onsets spaced d apart also lie on d/2 and d/3, so the strength curve
    peaks at all of them. The grid is the coarsest, and an earlier revision that
    took the peak alone returned an arbitrary subdivision instead."""
    estimate = detect_pulse(composition_at([i * 240 for i in range(32)]))
    assert estimate.period_ticks == pytest.approx(240, rel=0.01)


def test_a_subdivided_rhythm_gives_the_subdivision():
    """Eighths against quarters are explained only by the eighth-note grid."""
    beats = [i * 240 for i in range(24)]
    offbeats = [i * 240 + 120 for i in range(0, 24, 2)]
    estimate = detect_pulse(composition_at(sorted(beats + offbeats)))
    assert estimate.period_ticks == pytest.approx(120, rel=0.02)


def test_phase_is_recovered():
    estimate = detect_pulse(composition_at([37 + i * 240 for i in range(32)]))
    assert estimate.period_ticks == pytest.approx(240, rel=0.01)
    assert estimate.phase_ticks == pytest.approx(37, abs=2)


def test_a_fractional_period_is_not_rounded_to_a_tick():
    """The real grid is a video frame, which is not a whole number of ticks in
    any sensible tick rate. Rounding it reintroduces the drift this measures."""
    period = 240.7
    estimate = detect_pulse(composition_at([round(i * period) for i in range(40)]))
    assert estimate.period_ticks == pytest.approx(period, rel=0.005)
    assert estimate.period_ticks != round(estimate.period_ticks)


def test_jitter_does_not_move_the_estimate():
    rng = random.Random(7)
    ticks = [i * 240 + rng.randint(-8, 8) for i in range(40)]
    estimate = detect_pulse(composition_at(ticks))
    assert estimate.period_ticks == pytest.approx(240, rel=0.02)
    assert estimate.found


def test_too_few_onsets_is_an_honest_refusal():
    estimate = detect_pulse(composition_at([0, 240, 480]))
    assert not estimate.found
    assert estimate.period_ticks == 0
    assert estimate.n_onsets == 3


def test_scattered_onsets_are_refused():
    rng = random.Random(0)
    scattered = sorted({rng.randrange(0, 40_000) for _ in range(40)})
    assert not detect_pulse(composition_at(scattered)).found


def test_an_empty_composition_is_refused():
    estimate = detect_pulse(Composition(hardware_profile="nes", ticks_per_beat=TPB))
    assert not estimate.found
    assert estimate.n_onsets == 0


def test_confidence_separates_structure_from_noise():
    rng = random.Random(0)
    regular = detect_pulse(composition_at([i * 240 for i in range(32)]))
    scattered = detect_pulse(composition_at(sorted({rng.randrange(0, 40_000) for _ in range(32)})))
    assert regular.confidence > scattered.confidence
    assert regular.confidence >= MIN_Z


# -- derived values -----------------------------------------------------------


def test_the_grid_is_reported_in_video_frames():
    """NES music is written in whole video frames, so a correct estimate of a
    six-frame grid must read as six frames and not as 6.2."""
    ticks_per_sec = 44_100  # the MIDI release's scale: 22050 tpb at 120 BPM
    frame = ticks_per_sec / 60.0988
    estimate = detect_pulse(
        composition_at(
            [round(i * 6 * frame) for i in range(40)],
            duration=400,
            ticks_per_beat=ticks_per_sec // 2,
        )
    )
    assert estimate.found
    assert estimate.period_frames == pytest.approx(6.0, abs=0.05)
    assert estimate.is_whole_frames


def test_an_off_grid_period_is_not_called_whole_frames():
    estimate = PulseEstimate(100.0, 0.0, 0.9, 50.0, 40, SEC_PER_TICK)
    assert estimate.period_frames == pytest.approx(6.26, abs=0.01)
    assert not estimate.is_whole_frames


def test_tempo_is_derived_from_the_period():
    estimate = PulseEstimate(480.0, 0.0, 0.9, 50.0, 40, SEC_PER_TICK)
    assert estimate.period_sec == pytest.approx(0.5)
    assert estimate.bpm == pytest.approx(120.0)


def test_a_refused_estimate_has_no_tempo():
    assert PulseEstimate(0.0, 0.0, 0.0, 0.0, 0).bpm == 0.0


def test_grid_index_counts_from_the_detected_phase():
    estimate = PulseEstimate(240.0, 30.0, 0.9, 50.0, 40, SEC_PER_TICK)
    assert grid_index(30, estimate) == 0
    assert grid_index(270, estimate) == 1
    assert grid_index(275, estimate) == 1  # nearest line, not the one before
    assert quantize(275, estimate) == pytest.approx(270.0)


def test_using_an_estimate_that_was_refused_changes_nothing():
    refused = PulseEstimate(0.0, 0.0, 0.0, 0.0, 0)
    assert grid_index(275, refused) == 275
    assert quantize(275, refused) == 275.0
    assert on_grid_share([0, 240], refused) == 0.0


def test_on_grid_share_is_one_when_every_onset_is_on_a_line():
    ticks = [i * 240 for i in range(32)]
    estimate = detect_pulse(composition_at(ticks))
    assert on_grid_share(ticks, estimate) == pytest.approx(1.0)


def test_on_grid_share_falls_when_onsets_are_off_the_line():
    estimate = PulseEstimate(240.0, 0.0, 0.9, 50.0, 40, SEC_PER_TICK)
    assert on_grid_share([0, 120, 240, 360], estimate) == pytest.approx(0.5)


# -- against the generator ----------------------------------------------------


def test_the_rule_based_generators_grid_is_recovered(nes_profile):
    """The control that matters: low energy puts kick on steps 0 and 8 and
    snare on 4 and 12 of a 16-step bar, so drum onsets fall on a regular grid
    with every onset on a line."""
    for seed in range(4):
        composition = composition_from_intake(
            generate(nes_profile, make_plan(seed=seed, n_bars=32, energy="low")).record
        )
        estimate = detect_pulse(composition)
        assert estimate.found, seed
        assert estimate.strength > 0.95, (seed, estimate.strength)
        drums = onset_ticks(composition, percussion_only=True)
        assert on_grid_share(drums, estimate) == pytest.approx(1.0), seed
