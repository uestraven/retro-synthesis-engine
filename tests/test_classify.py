"""The event classifier, which every codec deletion metric depends on."""

from __future__ import annotations

import pytest

from rse.events.classify import (
    ARPEGGIO,
    ATTACK,
    PASSAGE,
    RETRIGGER,
    VIBRATO,
    classify_events,
)
from rse.events.schema import Composition, Noise, Note

ROW = 100.0


def _composition(events):
    return Composition(hardware_profile="nes", ticks_per_beat=480, events=list(events))


def _notes(channel, pairs):
    return [
        Note(channel_id=channel, tick=tick, pitch=pitch, duration=50, velocity=96)
        for tick, pitch in pairs
    ]


def test_isolated_notes_are_all_attacks():
    comp = _composition(_notes("P1", [(0, 60), (100, 62), (200, 64)]))
    assert classify_events(comp, row_ticks=ROW).labels == [ATTACK, ATTACK, ATTACK]


def test_wide_rapid_cycle_is_an_arpeggio():
    """A chord on a monophonic channel. Only the first event is a real note."""
    comp = _composition(_notes("P1", [(0, 60), (10, 64), (20, 67), (30, 60)]))
    result = classify_events(comp, row_ticks=ROW)
    assert result.labels == [ATTACK, ARPEGGIO, ARPEGGIO, ARPEGGIO]
    assert result.counts()[ARPEGGIO] == 3


def test_narrow_rapid_alternation_is_vibrato():
    comp = _composition(_notes("P1", [(0, 60), (10, 61), (20, 60), (30, 61)]))
    assert classify_events(comp, row_ticks=ROW).labels == [
        ATTACK,
        VIBRATO,
        VIBRATO,
        VIBRATO,
    ]


def test_same_pitch_repeated_is_a_retrigger():
    comp = _composition(_notes("P1", [(0, 60), (10, 60), (20, 60)]))
    assert classify_events(comp, row_ticks=ROW).labels == [ATTACK, RETRIGGER, RETRIGGER]


def test_rapid_scalar_run_is_a_passage_not_an_effect():
    """A fast scale is too wide to be vibrato and too narrow to be a chord.
    It might be real melody, so a codec must not take credit for discarding
    it -- the run stays unclassified rather than joining an effect class."""
    comp = _composition(_notes("P1", [(0, 60), (10, 62), (20, 64), (30, 67)]))
    labels = classify_events(comp, row_ticks=ROW).labels
    assert labels[0] == ATTACK
    assert set(labels[1:]) == {PASSAGE}


def test_noise_runs_are_never_called_arpeggio_or_vibrato():
    """A noise period is not a pitch, so a difference between two of them is
    not an interval."""
    comp = _composition(
        [
            Noise(channel_id="NO", tick=tick, value=value, duration=20, velocity=96)
            for tick, value in [(0, 1), (10, 9), (20, 4)]
        ]
    )
    labels = classify_events(comp, row_ticks=ROW).labels
    assert labels == [ATTACK, PASSAGE, PASSAGE]


def test_identical_noise_values_are_a_retrigger():
    comp = _composition(
        [
            Noise(channel_id="NO", tick=tick, value=5, duration=20, velocity=96)
            for tick in (0, 10, 20)
        ]
    )
    assert classify_events(comp, row_ticks=ROW).labels == [ATTACK, RETRIGGER, RETRIGGER]


def test_groups_do_not_span_channels():
    """Two channels hitting together are two attacks, not a chord effect."""
    comp = _composition(_notes("P1", [(0, 60)]) + _notes("P2", [(0, 67)]))
    result = classify_events(comp, row_ticks=ROW)
    assert result.labels == [ATTACK, ATTACK]
    assert result.group_of[0] != result.group_of[1]


def test_a_gap_wider_than_half_a_row_starts_a_new_group():
    """The threshold is structural: beyond half a row the events could be
    separate tracker rows, so they are separate notes."""
    comp = _composition(_notes("P1", [(0, 60), (49, 64), (200, 67), (249, 71)]))
    result = classify_events(comp, row_ticks=ROW)
    assert result.labels == [ATTACK, ARPEGGIO, ATTACK, ARPEGGIO]
    assert result.group_of[0] == result.group_of[1]
    assert result.group_of[2] == result.group_of[3]
    assert result.group_of[0] != result.group_of[2]


def test_row_ticks_must_be_positive():
    comp = _composition(_notes("P1", [(0, 60)]))
    with pytest.raises(ValueError, match="row_ticks"):
        classify_events(comp, row_ticks=0)


def test_labels_align_with_the_supplied_event_list_not_sorted_order():
    """A caller indexes these against the list it passed in, so grouping by
    sorted order must not reorder the result -- that would silently attach
    each label to the wrong event."""
    events = _notes("P2", [(0, 60), (10, 64)]) + _notes("P1", [(0, 72)])
    scrambled = [events[2], events[0], events[1]]
    result = classify_events(_composition(events), row_ticks=ROW, events=scrambled)
    # scrambled[0] is the lone P1 note; the two P2 notes are one arpeggio.
    assert result.labels == [ATTACK, ATTACK, ARPEGGIO]
    assert result.group_of[1] == result.group_of[2] != result.group_of[0]
