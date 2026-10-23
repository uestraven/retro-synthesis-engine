from __future__ import annotations

import pytest

from rse.generation.theory import (
    MOOD_SCALES,
    PROGRESSIONS,
    SCALES,
    Key,
    chord_pitches,
    is_consonant,
    transpose_into_range,
    triad,
)

C_MAJOR = Key(tonic=0, scale="major")
A_MINOR = Key(tonic=9, scale="minor")


def test_every_mood_has_a_scale_and_progressions():
    assert set(MOOD_SCALES) == set(PROGRESSIONS)
    for mood, scale in MOOD_SCALES.items():
        assert scale in SCALES
        assert PROGRESSIONS[mood], f"{mood} has no progressions"


def test_progressions_stay_inside_a_seven_note_scale():
    for mood, options in PROGRESSIONS.items():
        for progression in options:
            assert all(0 <= degree <= 6 for degree in progression), mood


def test_scales_start_on_the_tonic_and_stay_inside_an_octave():
    for name, intervals in SCALES.items():
        assert intervals[0] == 0, name
        assert all(0 <= i <= 11 for i in intervals), name
        assert list(intervals) == sorted(intervals), name


def test_key_rejects_nonsense():
    with pytest.raises(ValueError, match="unknown scale"):
        Key(tonic=0, scale="lydian_dominant")
    with pytest.raises(ValueError, match="pitch class"):
        Key(tonic=15, scale="major")


def test_degree_to_pitch_walks_the_scale():
    # C4 major: C D E F G A B
    assert [C_MAJOR.degree_to_pitch(d, 4) for d in range(7)] == [60, 62, 64, 65, 67, 69, 71]


def test_degrees_past_the_octave_wrap_upward():
    assert C_MAJOR.degree_to_pitch(7, 4) == C_MAJOR.degree_to_pitch(0, 4) + 12
    assert C_MAJOR.degree_to_pitch(-1, 4) == 59  # the B below middle C


def test_pentatonic_scales_have_five_degrees_per_octave():
    pentatonic = Key(tonic=0, scale="minor_pentatonic")
    assert pentatonic.size == 5
    assert pentatonic.degree_to_pitch(5, 4) == pentatonic.degree_to_pitch(0, 4) + 12


def test_triad_stacks_in_scale_steps():
    assert triad(C_MAJOR, 0) == (0, 2, 4)
    assert triad(C_MAJOR, 4) == (4, 6, 8)


def test_chord_quality_follows_the_scale():
    """A triad on the tonic is major in a major key and minor in a minor key."""
    major = chord_pitches(C_MAJOR, 0, 4)
    assert [p - major[0] for p in major] == [0, 4, 7]
    minor = chord_pitches(A_MINOR, 0, 4)
    assert [p - minor[0] for p in minor] == [0, 3, 7]


def test_key_membership():
    assert C_MAJOR.contains(64)  # E
    assert not C_MAJOR.contains(61)  # C#


def test_nearest_degree_snaps_a_pitch_into_the_scale():
    assert C_MAJOR.nearest_degree(60) == 0
    assert C_MAJOR.nearest_degree(64) == 2
    # C# is between the tonic and the second; either is acceptable.
    assert C_MAJOR.nearest_degree(61) in (0, 1)


def test_transpose_into_range_moves_by_octaves():
    assert transpose_into_range(36, 60, 84) == 60
    assert transpose_into_range(100, 60, 84) == 76
    assert transpose_into_range(72, 60, 84) == 72  # already inside


def test_transpose_into_range_clamps_when_no_octave_fits():
    # A range narrower than an octave may not hold this pitch class at all.
    assert 60 <= transpose_into_range(61, 60, 64) <= 64


def test_transpose_into_range_refuses_an_empty_range():
    with pytest.raises(ValueError, match="empty range"):
        transpose_into_range(60, 84, 60)


def test_consonance():
    assert is_consonant(0) and is_consonant(7) and is_consonant(12)
    assert not is_consonant(1) and not is_consonant(6)
