from __future__ import annotations

import pytest

from rse.events.convert import composition_from_intake
from rse.events.schema import Composition, Noise, Note, SectionSpan
from rse.events.structure import (
    RETURN_FLOOR,
    SECTION_LETTERS,
    analyze_structure,
    derive_sections,
    form_string,
    sections_from_analysis,
    segment_fingerprint,
    similarity,
)
from rse.generation.plan import make_plan
from rse.generation.procedural import generate

TICKS_PER_BAR = 480 * 4


def composition_of(*events) -> Composition:
    return Composition(hardware_profile="nes", ticks_per_beat=480, events=list(events))


def phrase(start_tick, pitches, channel="P1", step=240):
    return [
        Note(
            channel_id=channel,
            tick=start_tick + i * step,
            pitch=pitch,
            duration=step,
            velocity=100,
        )
        for i, pitch in enumerate(pitches)
    ]


MELODY = [60, 62, 64, 67, 65, 64, 62, 60]


# -- fingerprint --------------------------------------------------------------


def test_an_empty_segment_has_no_fingerprint():
    assert segment_fingerprint(composition_of(), 0, 8) == frozenset()


def test_a_fingerprint_is_transposition_invariant():
    """A section returning a fifth higher is still that section."""
    original = composition_of(*phrase(0, MELODY))
    transposed = composition_of(*phrase(0, [p + 7 for p in MELODY]))
    a = segment_fingerprint(original, 0, 8)
    b = segment_fingerprint(transposed, 0, 8)
    assert a and a == b


def test_a_fingerprint_tolerates_timing_drift():
    """The reason exact positions were abandoned: one frame of drift broke them."""
    tight = composition_of(*phrase(0, MELODY, step=240))
    loose = composition_of(*phrase(37, MELODY, step=241))
    assert similarity(
        segment_fingerprint(tight, 0, 8), segment_fingerprint(loose, 0, 8)
    ) == pytest.approx(1.0)


def test_a_fingerprint_is_order_aware():
    forward = composition_of(*phrase(0, MELODY))
    backward = composition_of(*phrase(0, list(reversed(MELODY))))
    assert similarity(segment_fingerprint(forward, 0, 8), segment_fingerprint(backward, 0, 8)) < 1.0


def test_repeated_values_collapse():
    """A held or retriggered note is one gesture, not several."""
    once = composition_of(*phrase(0, [60, 64, 67]))
    held = composition_of(*phrase(0, [60, 60, 60, 64, 64, 67]))
    assert segment_fingerprint(once, 0, 8) == segment_fingerprint(held, 0, 8)


def test_the_noise_channel_contributes():
    """A change of drum pattern is a clear marker of a new section."""
    without = composition_of(*phrase(0, MELODY))
    with_drums = composition_of(
        *phrase(0, MELODY),
        *[
            Noise(channel_id="NO", tick=i * 240, value=v, duration=60, velocity=100)
            for i, v in enumerate([13, 7, 13, 7, 13, 7, 13, 2])
        ],
    )
    assert segment_fingerprint(with_drums, 0, 8) > segment_fingerprint(without, 0, 8)


def test_segments_are_bounded_by_their_bars():
    composition = composition_of(*phrase(0, MELODY), *phrase(TICKS_PER_BAR * 8, MELODY))
    first = segment_fingerprint(composition, 0, 8)
    second = segment_fingerprint(composition, 8, 8)
    assert first and second and first == second


def test_similarity_edges():
    assert similarity(frozenset(), frozenset({1})) == 0.0
    assert similarity(frozenset({1, 2}), frozenset({3})) == 0.0
    assert similarity(frozenset({1, 2}), frozenset({1, 2})) == 1.0


# -- labelling ----------------------------------------------------------------


def test_nothing_to_analyze():
    assert analyze_structure(composition_of()) == []


def test_a_nonpositive_segment_length_is_refused():
    with pytest.raises(ValueError, match="segment_bars must be positive"):
        analyze_structure(composition_of(*phrase(0, MELODY)), segment_bars=0)


def test_a_repeated_segment_takes_the_earlier_label():
    other = [48, 50, 53, 55, 57, 55, 53, 50]
    composition = composition_of(
        *phrase(0, MELODY),
        *phrase(TICKS_PER_BAR * 8, MELODY),
        *phrase(TICKS_PER_BAR * 16, other),
        *phrase(TICKS_PER_BAR * 24, MELODY),
    )
    analyses = analyze_structure(composition, n_bars=32)
    assert form_string(analyses) == "AABA"
    assert analyses[1].is_return and analyses[3].is_return
    assert not analyses[2].is_return


def test_a_two_segment_window_can_still_detect_a_return():
    """With one pairwise value the median is not a baseline; the floor is."""
    composition = composition_of(*phrase(0, MELODY), *phrase(TICKS_PER_BAR * 8, MELODY))
    analyses = analyze_structure(composition, n_bars=16)
    assert form_string(analyses) == "AA"
    assert analyses[1].matched_similarity >= RETURN_FLOOR


def test_a_silent_segment_continues_the_current_section():
    composition = composition_of(*phrase(0, MELODY), *phrase(TICKS_PER_BAR * 16, MELODY))
    analyses = analyze_structure(composition, n_bars=24)
    # The middle segment is empty and must not invent a section of its own.
    assert analyses[1].n_grams == 0
    assert analyses[1].label == analyses[0].label


def test_letters_run_out_gracefully():
    events = []
    for index in range(10):
        events += phrase(TICKS_PER_BAR * 8 * index, [60 + index * 3, 62, 65, 69 + index])
    analyses = analyze_structure(composition_of(*events), n_bars=80)
    assert len({a.label for a in analyses}) <= len(SECTION_LETTERS)
    assert all(a.label in SECTION_LETTERS for a in analyses)


def test_the_generators_own_form_is_recovered(nes_profile):
    """The strongest available check: the planner declares a form, and the
    analysis has to find it without being told."""
    for form in ("AABA", "ABAB", "AABB"):
        plan = make_plan(seed=5, form=form, n_bars=32, energy="medium")
        composition = composition_from_intake(generate(nes_profile, plan).record)
        analyses = analyze_structure(composition, n_bars=32)
        assert form_string(analyses) == form, (form, form_string(analyses))


# -- spans --------------------------------------------------------------------


def test_spans_mark_where_the_form_changes():
    other = [48, 50, 53, 55]
    composition = composition_of(
        *phrase(0, MELODY),
        *phrase(TICKS_PER_BAR * 8, MELODY),
        *phrase(TICKS_PER_BAR * 16, other),
        *phrase(TICKS_PER_BAR * 24, MELODY),
    )
    spans = derive_sections(composition, n_bars=32)
    # A A B A is three spans, not four: consecutive same-label segments merge.
    assert [(s.start_bar, s.label) for s in spans] == [(0, "A"), (16, "B"), (24, "A")]


def test_spans_from_nothing():
    assert sections_from_analysis([]) == []


def test_derive_sections_on_real_generated_music(nes_profile):
    composition = composition_from_intake(
        generate(nes_profile, make_plan(seed=9, n_bars=32)).record
    )
    spans = derive_sections(composition, n_bars=32)
    assert spans
    assert spans[0].start_bar == 0
    assert all(isinstance(s, SectionSpan) for s in spans)
    assert [s.start_bar for s in spans] == sorted({s.start_bar for s in spans})
