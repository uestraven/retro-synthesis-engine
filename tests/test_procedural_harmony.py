"""All seven triads in all six modes, named from their pitches.

A hard-coded roman-numeral table drifted from the music: it called A harmonic
minor's seventh `VII` and its fifth `v`, when the generator actually produces
G#-B-D diminished and E-G#-B major. The chords were correct and the report was
wrong, which is the dangerous direction -- it invites "fixing" music that
works.

These tests pin every label against the pitches, so the description cannot
disagree with the behaviour again. They assert nothing about composition, only
about naming.
"""

from __future__ import annotations

import pytest

from rse.procedural.analysis import analyse_progression, spell, triad
from rse.procedural.compose import SCALES, STYLES

A, C, G = 57, 60, 55


def test_harmonic_minor_seventh_is_diminished_not_borrowed_major() -> None:
    """The case that exposed the reporting defect."""
    seventh = triad(A, "harmonic_minor", 6)
    assert seventh["notes"] == ["G#", "B", "D"]
    assert seventh["quality"] == "diminished"
    assert seventh["numeral"] == "vii°"


def test_harmonic_minor_dominant_is_major() -> None:
    """The raised seventh makes V major, which is why it pulls back to i."""
    fifth = triad(A, "harmonic_minor", 4)
    assert fifth["notes"] == ["E", "G#", "B"]
    assert fifth["quality"] == "major"
    assert fifth["numeral"] == "V"


def test_harmonic_minor_mediant_is_augmented() -> None:
    """bIII+ is characteristic of harmonic minor and the old table could not
    express it at all."""
    third = triad(A, "harmonic_minor", 2)
    assert third["quality"] == "augmented"
    assert third["numeral"] == "bIII+"
    assert third["notes"] == ["C", "E", "G#"]


def test_raised_seventh_is_spelled_sharp_not_flat() -> None:
    """G# is the leading tone of A harmonic minor; Ab is not.

    Enharmonically identical, musically different, and a report that says Ab
    is unreadable to anyone checking the harmony.
    """
    assert spell(A, "harmonic_minor", 6) == "G#"
    assert "Ab" not in triad(A, "harmonic_minor", 6)["notes"]


def test_phrygian_second_is_flat_two_major() -> None:
    """Not ii°. The i/bII relationship is what makes Phrygian sound Phrygian."""
    second = triad(A, "phrygian", 1)
    assert second["notes"] == ["Bb", "D", "F"]
    assert second["quality"] == "major"
    assert second["numeral"] == "bII"


def test_phrygian_seventh_is_minor() -> None:
    seventh = triad(A, "phrygian", 6)
    assert seventh["quality"] == "minor"
    assert seventh["numeral"] == "bvii"


@pytest.mark.parametrize("mode", sorted(SCALES))
def test_every_degree_of_every_mode_is_nameable(mode: str) -> None:
    """No triad falls through to an `other(...)` quality."""
    for degree in range(7):
        result = triad(A, mode, degree)
        assert result["quality"] in (
            "major",
            "minor",
            "diminished",
            "augmented",
        ), f"{mode} degree {degree} -> {result['quality']}"
        assert result["numeral"]
        assert len(result["notes"]) == 3


@pytest.mark.parametrize("mode", sorted(SCALES))
def test_each_degree_gets_its_own_letter(mode: str) -> None:
    """Letter arithmetic: a seven-note mode uses all seven letters once.

    This is what forces the correct enharmonic spelling rather than a
    pitch-class lookup that would pick Ab over G#.
    """
    letters = [spell(A, mode, degree)[0] for degree in range(7)]
    assert sorted(letters) == sorted("ABCDEFG")


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("major", ["I", "ii", "iii", "IV", "V", "vi", "vii°"]),
        ("minor", ["i", "ii°", "bIII", "iv", "v", "bVI", "bVII"]),
        ("dorian", ["i", "ii", "bIII", "IV", "v", "vi°", "bVII"]),
        ("phrygian", ["i", "bII", "bIII", "iv", "v°", "bVI", "bvii"]),
        ("mixolydian", ["I", "ii", "iii°", "IV", "v", "vi", "bVII"]),
        ("harmonic_minor", ["i", "ii°", "bIII+", "iv", "V", "bVI", "vii°"]),
    ],
)
def test_full_numeral_sequence_per_mode(mode: str, expected: list[str]) -> None:
    """The complete diatonic set, which is the reference the report must match."""
    assert [triad(A, mode, d)["numeral"] for d in range(7)] == expected


def test_brief_examples_render_as_specified() -> None:
    """The two worked examples from the review."""
    boss = analyse_progression(A, "harmonic_minor", (0, 3, 4, 0))
    assert boss.chord_names == "Am - Dm - E - Am"
    assert boss.numerals == "i - iv - V - i"

    tension = analyse_progression(G, "phrygian", (0, 1, 0, 1))
    assert tension.chord_names == "Gm - Ab - Gm - Ab"
    assert tension.numerals == "i - bII - i - bII"


def test_every_shipped_progression_is_nameable() -> None:
    """Nothing in the frozen v1 ruleset produces an unnameable chord."""
    for name, style in STYLES.items():
        for progression in style.progressions:
            analysis = analyse_progression(style.tonic, style.mode, progression)
            for chord in analysis.chords:
                assert chord["quality"] in (
                    "major",
                    "minor",
                    "diminished",
                    "augmented",
                ), f"{name}: {chord}"
            assert analysis.numerals
            assert analysis.chord_names


def test_analysis_does_not_change_the_music() -> None:
    """Naming is read-only: composing twice either side of it is identical."""
    from rse.procedural.compose import compose

    before = compose(style_name="boss", seed=5)
    analyse_progression(A, "harmonic_minor", (0, 3, 4, 0))
    after = compose(style_name="boss", seed=5)
    signature = lambda r: [  # noqa: E731 - local, single use
        (e.channel_id, e.tick, getattr(e, "pitch", getattr(e, "value", 0)))
        for e in r.composition.sounding_events()
    ]
    assert signature(before) == signature(after)
