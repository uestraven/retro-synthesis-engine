"""Chords that can be altered, and melodies that know about the alteration.

The v1 composer builds triads by stacking scale degrees, which is correct and
cannot express a borrowed chord: E major in A *natural* minor needs G#, and
the natural minor scale has G. So a progression like `i-iv-V-i` in natural
minor is unreachable, and that dominant is the single most useful missing
chord for boss and action writing.

Two rules shape this module.

**Mode-derived chords keep the old path exactly.** A `Chord` with no quality
override and no root shift resolves through the same degree arithmetic v1
used, so the 24 pieces in the frozen baseline stay byte-identical by
construction rather than by testing luck. Only altered chords take the new
pitch-space path.

**An altered chord suppresses the scale tone it replaced.** This is the
integration requirement that matters: if E major sounds in A natural minor and
the melody is still free to play any scale tone, it will put G against G# and
do it repeatedly. So resolving a chord also reports which pitch classes to
avoid while it sounds -- here, G natural. The melody is not restricted to
chord tones and keeps its passing tones; it loses exactly the one note the
alteration contradicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rse.procedural.compose import SCALES, chord_degrees, scale_pitch

#: Semitone intervals above the root, by triad quality.
QUALITY_INTERVALS: dict[str, tuple[int, int, int]] = {
    "major": (0, 4, 7),
    "minor": (0, 3, 7),
    "diminished": (0, 3, 6),
    "augmented": (0, 4, 8),
}

#: Shorthand accepted in progression definitions.
QUALITY_ALIASES = {
    "maj": "major",
    "M": "major",
    "min": "minor",
    "m": "minor",
    "dim": "diminished",
    "°": "diminished",
    "aug": "augmented",
    "+": "augmented",
}


@dataclass(frozen=True)
class Chord:
    """One chord in a progression.

    `degree` is a scale-degree index as before. `quality` overrides the triad
    the mode would give, and `root_shift` moves the root chromatically for a
    chord whose root is not in the scale at all. Both default to "whatever the
    mode says", which is the v1 behaviour.
    """

    degree: int
    quality: str | None = None
    root_shift: int = 0

    @property
    def altered(self) -> bool:
        return self.quality is not None or self.root_shift != 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "degree": self.degree,
            "quality": self.quality,
            "root_shift": self.root_shift,
            "altered": self.altered,
        }


@dataclass(frozen=True)
class Voicing:
    """A resolved chord: what actually sounds, and what to avoid.

    `degrees` is populated only for mode-derived chords, because the v1
    renderers work in degree space and must keep doing so exactly. Altered
    chords supply `semitones` instead.
    """

    chord: Chord
    #: Semitones above the tonic, one per chord tone.
    semitones: tuple[int, ...]
    #: Scale degrees, for the unaltered path only.
    degrees: tuple[int, ...] | None
    quality: str
    diatonic: bool
    #: Pitch classes the melody must not use while this chord sounds: the
    #: scale tones an alteration contradicts.
    avoid: frozenset[int]

    def pitch_classes(self, tonic: int) -> frozenset[int]:
        return frozenset((tonic + s) % 12 for s in self.semitones)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chord": self.chord.to_dict(),
            "semitones": list(self.semitones),
            "quality": self.quality,
            "diatonic": self.diatonic,
            "avoid": sorted(self.avoid),
        }


def as_chord(spec: Any) -> Chord:
    """Accept an int, a Chord, or a (degree, quality) pair.

    Integers are what v1 progressions contain, so they must keep working
    unchanged and must resolve to the mode-derived triad.
    """
    if isinstance(spec, Chord):
        return spec
    if isinstance(spec, int):
        return Chord(degree=spec)
    if isinstance(spec, (tuple, list)):
        degree = int(spec[0])
        quality = spec[1] if len(spec) > 1 else None
        shift = int(spec[2]) if len(spec) > 2 else 0
        if quality is not None:
            quality = QUALITY_ALIASES.get(str(quality), str(quality))
            if quality not in QUALITY_INTERVALS:
                raise ValueError(
                    f"unknown chord quality {spec[1]!r}; "
                    f"have {sorted(QUALITY_INTERVALS)} or "
                    f"{sorted(QUALITY_ALIASES)}"
                )
        return Chord(degree=degree, quality=quality, root_shift=shift)
    raise TypeError(f"cannot read {spec!r} as a chord")


def quality_of(semitones: tuple[int, ...]) -> str:
    """Triad quality from its intervals, derived rather than declared."""
    if len(semitones) != 3:
        return "other"
    intervals = (semitones[1] - semitones[0], semitones[2] - semitones[1])
    return {
        (4, 3): "major",
        (3, 4): "minor",
        (3, 3): "diminished",
        (4, 4): "augmented",
    }.get(intervals, "other")


def resolve(spec: Any, *, tonic: int, mode: str) -> Voicing:
    """Turn a chord spec into the pitches that sound and the notes to avoid."""
    chord = as_chord(spec)
    scale = SCALES[mode]
    scale_classes = frozenset(s % 12 for s in scale)

    if not chord.altered:
        # The v1 path, untouched. Degrees are kept so the renderers can go on
        # working in degree space and produce identical voicings.
        degrees = chord_degrees(chord.degree)
        semitones = tuple(
            scale_pitch(0, mode, d) for d in degrees
        )
        return Voicing(
            chord=chord,
            semitones=semitones,
            degrees=degrees,
            # Quality from the RAW stacked semitones. Reducing them mod 12
            # first destroys the ordering -- E-G-B comes out (7, 10, 14), and
            # mod 12 that is (7, 10, 2), whose intervals read (3, -8) and
            # resolve to "other" instead of minor.
            quality=quality_of(semitones),
            diatonic=True,
            avoid=frozenset(),
        )

    index = chord.degree % len(scale)
    root = scale[index] + chord.root_shift
    quality = chord.quality or quality_of(
        tuple(scale_pitch(0, mode, d) for d in chord_degrees(chord.degree))
    )
    semitones = tuple(root + i for i in QUALITY_INTERVALS[quality])

    # Which scale tones does this alteration contradict? For each chord pitch
    # that is not in the scale, the scale pitch a semitone either side is the
    # one it replaced -- G# replaces G, not A -- so prefer the nearer one
    # below, which is what a raised leading tone always is.
    avoid: set[int] = set()
    for pitch in semitones:
        pitch_class = pitch % 12
        if pitch_class in scale_classes:
            continue
        for offset in (-1, 1):
            neighbour = (pitch_class + offset) % 12
            if neighbour in scale_classes and neighbour not in {
                s % 12 for s in semitones
            }:
                avoid.add(neighbour)
                break

    return Voicing(
        chord=chord,
        semitones=semitones,
        degrees=None,
        quality=quality,
        diatonic=False,
        avoid=frozenset(avoid),
    )


def melody_candidates(
    voicing: Voicing, *, mode: str, octave_span: int = 2
) -> tuple[list[int], list[int]]:
    """(chord tones, scale tones) as semitones above the tonic.

    Both lists exclude anything in `voicing.avoid`, which is how an altered
    chord removes the note it contradicts without restricting the melody to
    chord tones or changing the mode globally.
    """
    scale = SCALES[mode]
    chord_tones = [
        s
        for octave in range(octave_span)
        for s in voicing.semitones
        if (s + 12 * octave) % 12 not in voicing.avoid
        for s in (s + 12 * octave,)
    ]
    scale_tones = [
        s + 12 * octave
        for octave in range(octave_span)
        for s in scale
        if (s % 12) not in voicing.avoid
    ]
    return sorted(set(chord_tones)), sorted(set(scale_tones))
