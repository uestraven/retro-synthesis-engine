"""Describe what the composer actually generated, derived from the pitches.

This exists because a hard-coded roman-numeral table drifted from the music.
The generator stacks scale degrees, so every triad is diatonic to its mode --
A harmonic minor's seventh-degree triad really is G#-B-D diminished and its
fifth really is E-G#-B major -- but the report called them `VII` and `v` from
a natural-minor lookup. The chords were right and the description was wrong,
which is the more dangerous way round: it invites "fixing" correct music.

So nothing here informs composition. Every label is computed from the actual
pitch classes, and the tests cover all seven triads in all six modes, so the
description cannot disagree with the behaviour again.

Spelling is by letter arithmetic, not a pitch-class table. In A harmonic minor
the raised seventh is **G#**, not Ab: enharmonically identical, but G# is the
leading tone and Ab is not, and a report that says Ab is unreadable to anyone
checking the harmony. The letter comes from the degree index and the
accidental from the distance to that letter's natural pitch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rse.procedural.compose import SCALES, chord_degrees, scale_pitch

#: Natural pitch class of each letter name.
LETTER_PITCH = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
LETTERS = ("C", "D", "E", "F", "G", "A", "B")

#: The major scale, which roman-numeral accidentals are measured against: a
#: third degree three semitones up is bIII because major's is four.
MAJOR = (0, 2, 4, 5, 7, 9, 11)

#: Triad qualities by the two stacked intervals.
QUALITIES = {
    (4, 3): "major",
    (3, 4): "minor",
    (3, 3): "diminished",
    (4, 4): "augmented",
}

#: Preferred letter for each pitch class when naming a tonic. Sharps for the
#: black keys, which is the commoner convention for minor-key game music.
TONIC_LETTER = {
    0: ("C", 0),
    1: ("C", 1),
    2: ("D", 0),
    3: ("E", -1),
    4: ("E", 0),
    5: ("F", 0),
    6: ("F", 1),
    7: ("G", 0),
    8: ("A", -1),
    9: ("A", 0),
    10: ("B", -1),
    11: ("B", 0),
}


def accidental(offset: int) -> str:
    """Render a semitone offset from a letter's natural pitch."""
    if offset == 0:
        return ""
    if offset > 0:
        return "#" * offset
    return "b" * -offset


def spell(tonic: int, mode: str, degree: int) -> str:
    """Letter name for a scale degree, spelled by letter arithmetic.

    The letter advances one step per scale degree, so each degree of a
    seven-note mode gets its own letter and the accidental follows. That is
    what makes A harmonic minor's seventh come out G# rather than Ab.
    """
    scale = SCALES[mode]
    letter_name, letter_offset = TONIC_LETTER[tonic % 12]
    start = LETTERS.index(letter_name)
    octaves, index = divmod(degree, len(scale))
    letter = LETTERS[(start + degree) % 7]
    actual = (tonic + 12 * octaves + scale[index]) % 12
    natural = LETTER_PITCH[letter]
    offset = (actual - natural + 6) % 12 - 6
    del letter_offset
    return f"{letter}{accidental(offset)}"


def triad(tonic: int, mode: str, degree: int) -> dict[str, Any]:
    """Everything derivable about the triad on a scale degree."""
    pitches = [scale_pitch(tonic, mode, g) for g in chord_degrees(degree)]
    intervals = (pitches[1] - pitches[0], pitches[2] - pitches[1])
    quality = QUALITIES.get(intervals, f"other{intervals}")
    index = degree % len(SCALES[mode])
    scale = SCALES[mode]

    # Accidental on the numeral: how this degree sits against the major scale.
    prefix = accidental((scale[index] - MAJOR[index] + 6) % 12 - 6)
    base = ("I", "II", "III", "IV", "V", "VI", "VII")[index]
    if quality == "minor":
        numeral = f"{prefix}{base.lower()}"
    elif quality == "diminished":
        numeral = f"{prefix}{base.lower()}°"
    elif quality == "augmented":
        numeral = f"{prefix}{base}+"
    else:
        numeral = f"{prefix}{base}"

    letters = [spell(tonic, mode, g) for g in chord_degrees(degree)]
    chord_name = letters[0] + {
        "major": "",
        "minor": "m",
        "diminished": "dim",
        "augmented": "aug",
    }.get(quality, "?")
    return {
        "degree": degree,
        "numeral": numeral,
        "chord": chord_name,
        "notes": letters,
        "pitch_classes": [p % 12 for p in pitches],
        "intervals": list(intervals),
        "quality": quality,
        "diatonic": True,
    }


@dataclass
class ProgressionAnalysis:
    """A progression described from its pitches."""

    mode: str
    tonic: str
    degrees: list[int]
    chords: list[dict[str, Any]]

    @property
    def numerals(self) -> str:
        return " - ".join(c["numeral"] for c in self.chords)

    @property
    def chord_names(self) -> str:
        return " - ".join(c["chord"] for c in self.chords)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "tonic": self.tonic,
            "degrees": list(self.degrees),
            "numerals": self.numerals,
            "chords": self.chord_names,
            "detail": self.chords,
        }

    def describe(self) -> str:
        return (
            f"Key: {self.tonic} {self.mode.replace('_', ' ')}\n"
            f"Degrees: {' -> '.join(str(d + 1) for d in self.degrees)}\n"
            f"Chords:  {self.chord_names}\n"
            f"Analysis: {self.numerals}"
        )


def analyse_progression(
    tonic: int, mode: str, degrees: tuple[int, ...] | list[int]
) -> ProgressionAnalysis:
    return ProgressionAnalysis(
        mode=mode,
        tonic=spell(tonic, mode, 0),
        degrees=list(degrees),
        chords=[triad(tonic, mode, d) for d in degrees],
    )


def analyse_style(style: Any) -> list[ProgressionAnalysis]:
    return [
        analyse_progression(style.tonic, style.mode, prog)
        for prog in style.progressions
    ]
