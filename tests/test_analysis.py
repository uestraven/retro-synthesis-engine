from __future__ import annotations

from rse.data import analysis
from rse.data.midi_io import Note


def note(tick: int, dur: int, pitch: int = 60) -> Note:
    return Note(
        channel_id="P1",
        tick=tick,
        dur_ticks=dur,
        pitch=pitch,
        velocity=100,
        track_index=1,
        midi_channel=0,
    )


def test_no_notes_means_no_polyphony():
    assert analysis.max_simultaneous([]) == 0
    assert analysis.overlaps([], 1) == []


def test_back_to_back_notes_are_monophonic():
    # A note ending exactly where the next starts is legal on a one-voice channel.
    notes = [note(0, 240), note(240, 240)]
    assert analysis.max_simultaneous(notes) == 1
    assert analysis.overlaps(notes, 1) == []


def test_overlapping_notes_are_detected_with_their_position():
    notes = [note(0, 480), note(240, 480, pitch=64)]
    assert analysis.max_simultaneous(notes) == 2
    found = analysis.overlaps(notes, 1)
    assert [(o.tick, o.simultaneous) for o in found] == [(240, 2)]


def test_three_way_overlap_counts_each_offending_onset():
    notes = [note(0, 960), note(100, 960, pitch=62), note(200, 960, pitch=64)]
    assert analysis.max_simultaneous(notes) == 3
    assert [o.simultaneous for o in analysis.overlaps(notes, 1)] == [2, 3]


def test_a_two_voice_limit_tolerates_two_notes():
    notes = [note(0, 480), note(240, 480, pitch=64)]
    assert analysis.overlaps(notes, 2) == []


def test_out_of_range_values_are_unique_and_sorted(nes_profile):
    spec = nes_profile.channel("P1")  # pitch 33..108
    notes = [note(0, 100, 20), note(100, 100, 60), note(200, 100, 120), note(300, 100, 20)]
    assert analysis.out_of_range(notes, spec) == [20, 120]


def test_noise_channel_range_is_measured_against_period_values(nes_profile):
    spec = nes_profile.channel("NO")  # value 1..16
    notes = [note(0, 100, 8), note(100, 100, 60)]
    assert analysis.out_of_range(notes, spec) == [60]


def test_duration_buckets_cover_common_note_values():
    assert analysis.duration_bucket(0.25) == "1/16"
    assert analysis.duration_bucket(1.0) == "1/4"
    assert analysis.duration_bucket(4.0) == "1"
    assert analysis.duration_bucket(8.0) == ">1"
