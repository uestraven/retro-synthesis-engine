from __future__ import annotations

import pytest

from rse.events.convert import composition_from_intake
from rse.events.pulse import (
    MIN_CONFIDENCE,
    Composition,
    PulseEstimate,
    align_to_pulse,
    concentration,
    detect_pulse,
    onset_frames,
)
from rse.events.schema import Noise, Note
from rse.generation.plan import make_plan
from rse.generation.procedural import generate

FRAME = 120  # ticks per frame at 480 ticks per beat


def composition_with(onsets, *, noise=True) -> Composition:
    events = []
    for index, frame in enumerate(onsets):
        if noise:
            events.append(
                Noise(channel_id="NO", tick=frame * FRAME, value=13, duration=FRAME, velocity=100)
            )
        else:
            events.append(
                Note(
                    channel_id="P1",
                    tick=frame * FRAME,
                    pitch=60 + index % 5,
                    duration=FRAME,
                    velocity=100,
                )
            )
    return Composition(hardware_profile="nes", ticks_per_beat=480, events=events)


# -- onsets -------------------------------------------------------------------


def test_onsets_are_frame_indexes():
    assert onset_frames(composition_with([0, 4, 8])) == [0, 4, 8]


def test_percussion_can_be_isolated():
    mixed = composition_with([0, 4], noise=True)
    mixed.events.append(
        Note(channel_id="P1", tick=2 * FRAME, pitch=60, duration=FRAME, velocity=100)
    )
    assert onset_frames(mixed, percussion_only=True) == [0, 4]
    assert onset_frames(mixed) == [0, 2, 4]


# -- detection ----------------------------------------------------------------


def test_a_regular_pattern_gives_its_own_period():
    estimate = detect_pulse(composition_with([f * 8 for f in range(24)]))
    assert estimate.found
    assert estimate.period_frames == 8
    assert estimate.phase_frames == 0


def test_the_fundamental_is_reported_not_a_multiple():
    """Autocorrelation at 4, 8, 12 ... is equal for a period-4 signal. An
    earlier version preferred the slowest lag and always returned the search
    ceiling, failing a control whose drums were every 4 frames."""
    estimate = detect_pulse(composition_with([f * 4 for f in range(40)]))
    assert estimate.period_frames == 4


def test_phase_is_recovered():
    offset = 3
    estimate = detect_pulse(composition_with([offset + f * 8 for f in range(24)]))
    assert estimate.period_frames == 8
    assert estimate.phase_frames == offset % 8


def test_too_few_onsets_is_an_honest_refusal():
    estimate = detect_pulse(composition_with([0, 8, 16]))
    assert not estimate.found
    assert estimate.period_frames == 0
    assert estimate.n_onsets == 3


def test_an_onset_on_every_frame_has_no_detectable_pulse():
    """A signal with no variation cannot have a period, and the detector must
    decline rather than invent one."""
    estimate = detect_pulse(composition_with(list(range(64))))
    assert not estimate.found


def test_an_empty_composition_is_refused():
    estimate = detect_pulse(Composition(hardware_profile="nes", ticks_per_beat=480))
    assert not estimate.found
    assert estimate.n_onsets == 0


def test_confidence_separates_structure_from_noise():
    import random

    rng = random.Random(0)
    regular = detect_pulse(composition_with([f * 8 for f in range(32)]))
    scattered = detect_pulse(composition_with(sorted({rng.randrange(0, 256) for _ in range(32)})))
    assert regular.confidence > scattered.confidence
    assert regular.confidence >= MIN_CONFIDENCE


# -- derived values -----------------------------------------------------------


def test_tempo_is_derived_from_the_period():
    estimate = PulseEstimate(period_frames=12, phase_frames=0, confidence=2.0, n_onsets=40)
    assert estimate.period_sec == pytest.approx(0.5)
    assert estimate.bpm == pytest.approx(120.0)


def test_a_refused_estimate_has_no_tempo():
    assert PulseEstimate(0, 0, 0.0, 0).bpm == 0.0


def test_alignment_counts_from_the_detected_phase():
    estimate = PulseEstimate(period_frames=8, phase_frames=3, confidence=2.0, n_onsets=40)
    assert align_to_pulse(3, estimate) == 0
    assert align_to_pulse(7, estimate) == 4
    assert align_to_pulse(11, estimate) == 0


def test_alignment_is_a_no_op_without_an_estimate():
    assert align_to_pulse(7, PulseEstimate(0, 0, 0.0, 0)) == 7


def test_concentration_is_one_for_evenly_spread_onsets():
    estimate = PulseEstimate(period_frames=4, phase_frames=0, confidence=2.0, n_onsets=40)
    assert concentration([0, 1, 2, 3, 4, 5, 6, 7], estimate) == pytest.approx(1.0)


def test_concentration_is_maximal_when_every_onset_is_on_the_beat():
    estimate = PulseEstimate(period_frames=4, phase_frames=0, confidence=2.0, n_onsets=40)
    assert concentration([0, 4, 8, 12], estimate) == pytest.approx(4.0)


def test_concentration_needs_an_estimate():
    assert concentration([0, 4], PulseEstimate(0, 0, 0.0, 0)) == 0.0


# -- against the generator ----------------------------------------------------


def test_the_rule_based_generators_grid_is_recovered(nes_profile):
    """The control that matters: low energy puts kick on steps 0 and 8 and
    snare on 4 and 12, so drum onsets fall every 4 frames with phase 0."""
    for seed in range(4):
        composition = composition_from_intake(
            generate(nes_profile, make_plan(seed=seed, n_bars=32, energy="low")).record
        )
        estimate = detect_pulse(composition)
        assert estimate.found, seed
        assert estimate.period_frames == 4, seed
        assert estimate.phase_frames == 0, seed
        # Every onset lands on one position of four.
        assert concentration(
            onset_frames(composition, percussion_only=True), estimate
        ) == pytest.approx(4.0)
