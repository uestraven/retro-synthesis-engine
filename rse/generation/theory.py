"""The small amount of music theory the procedural generator needs.

Kept separate from the generator because Phase 4's evaluation needs the same
vocabulary to judge harmonic consistency, and because theory is far easier to
test in isolation than a composition.

Degrees are zero-indexed scale positions: degree 0 is the tonic, degree 4 is
the fifth. Pitches are MIDI note numbers throughout.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Scale intervals in semitones from the tonic.
SCALES: dict[str, tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "major_pentatonic": (0, 2, 4, 7, 9),
    "minor_pentatonic": (0, 3, 5, 7, 10),
}

#: Chord progressions as scale degrees, grouped by the mood they tend to carry.
#: These are the progressions game soundtracks lean on, not a general survey.
PROGRESSIONS: dict[str, tuple[tuple[int, ...], ...]] = {
    "heroic": (
        (0, 3, 4, 0),  # I-IV-V-I
        (0, 4, 5, 3),  # I-V-vi-IV
        (0, 5, 3, 4),  # I-vi-IV-V
    ),
    "dark": (
        (0, 5, 2, 6),  # i-VI-III-VII, the staple minor loop
        (0, 3, 4, 4),  # i-iv-v-v
        (0, 6, 5, 6),  # i-VII-VI-VII
    ),
    "mysterious": (
        (0, 1, 0, 6),
        (0, 2, 5, 4),
        (0, 6, 2, 5),
    ),
    "melancholic": (
        (5, 2, 6, 3),
        (0, 4, 5, 2),
        (0, 2, 3, 4),
    ),
}

#: Which scale suits which mood. The generator may override these.
MOOD_SCALES: dict[str, str] = {
    "heroic": "major",
    "dark": "minor",
    "mysterious": "harmonic_minor",
    "melancholic": "minor",
}


@dataclass(frozen=True)
class Key:
    """A tonic and a scale."""

    tonic: int  # MIDI pitch class of the tonic, 0-11
    scale: str

    def __post_init__(self) -> None:
        if self.scale not in SCALES:
            raise ValueError(f"unknown scale {self.scale!r}; known: {sorted(SCALES)}")
        if not 0 <= self.tonic <= 11:
            raise ValueError(f"tonic must be a pitch class 0-11, got {self.tonic}")

    @property
    def intervals(self) -> tuple[int, ...]:
        return SCALES[self.scale]

    @property
    def size(self) -> int:
        """Notes per octave in this scale — five for pentatonic, seven otherwise."""
        return len(self.intervals)

    def degree_to_pitch(self, degree: int, octave: int = 4) -> int:
        """MIDI pitch for a scale degree, wrapping octaves as the degree runs on.

        Degree 7 of a seven-note scale is the tonic an octave up, and degree -1
        is the note below the tonic; callers can walk a melody past the octave
        without special-casing anything.
        """
        octave_shift, index = divmod(degree, self.size)
        return 12 * (octave + 1 + octave_shift) + self.tonic + self.intervals[index]

    def contains(self, pitch: int) -> bool:
        return (pitch - self.tonic) % 12 in self.intervals

    def nearest_degree(self, pitch: int) -> int:
        """The scale degree closest to `pitch`, ignoring octave."""
        pitch_class = (pitch - self.tonic) % 12
        best = min(range(self.size), key=lambda d: abs(self.intervals[d] - pitch_class))
        return best


def triad(key: Key, degree: int) -> tuple[int, int, int]:
    """The three scale degrees of the triad built on `degree`.

    Stacked in scale steps rather than fixed intervals, so the chord quality
    follows from the scale — degree 0 of a minor scale gives a minor triad
    without the caller having to say so.
    """
    return (degree, degree + 2, degree + 4)


def chord_pitches(key: Key, degree: int, octave: int = 4) -> tuple[int, ...]:
    """MIDI pitches of the triad on `degree`."""
    return tuple(key.degree_to_pitch(d, octave) for d in triad(key, degree))


def transpose_into_range(pitch: int, low: int, high: int) -> int:
    """Shift `pitch` by octaves until it sits inside `[low, high]`."""
    if low > high:
        raise ValueError(f"empty range {low}-{high}")
    while pitch < low:
        pitch += 12
    while pitch > high:
        pitch -= 12
    # A range narrower than an octave may not contain this pitch class at all;
    # clamping is the honest fallback.
    return max(low, min(high, pitch))


def is_consonant(interval: int) -> bool:
    """Whether a semitone interval is consonant, octave-reduced."""
    return interval % 12 in (0, 3, 4, 5, 7, 8, 9)
