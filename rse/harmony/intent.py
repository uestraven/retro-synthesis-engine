"""What a chord means, as distinct from the notes that play it.

The NES chord model is `Chord(degree, quality, root_shift)` resolving to three
stacked scale degrees. That is honest for four monophonic channels: a triad is
already one note more than the two pulse voices can hold, so extensions were
never expressible and never needed.

SNES harmony needs sevenths, ninths, elevenths, suspensions and inversions, and
it needs them without consuming five voices per chord. Cmaj9 is five pitch
classes; a convincing SNES voicing of it is four notes, because the fifth
carries less of the chord's identity than anything else in it.

So intent and realisation become separate objects:

    ChordIntent   root, quality, extensions, inversion, function, alterations
        |
        v         the voicing planner, which may omit
    ChordVoicing  the pitches that actually sound, and what was dropped

Nothing here touches `rse/procedural/chords.py`. The NES path keeps its triad
generation exactly, so the twenty-four frozen baselines cannot move. This model
is additive and the NES composer never constructs one.

Pitch spelling is preserved: degrees and alterations are kept symbolically and
only resolved to pitch classes on request, so a raised ninth stays a raised
ninth rather than becoming an enharmonic flat tenth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Quality(StrEnum):
    """The triad at the core of the chord."""

    MAJOR = "major"
    MINOR = "minor"
    DIMINISHED = "diminished"
    AUGMENTED = "augmented"
    SUSPENDED_2 = "sus2"
    SUSPENDED_4 = "sus4"


class Seventh(StrEnum):
    """Which seventh, if any. Separate from the triad because the two vary
    independently: a minor triad takes a minor seventh in `m7` and a major one
    in `mMaj7`, and the pairing is the chord's identity."""

    NONE = "none"
    MINOR = "minor"  # 10 semitones -- dominant and minor sevenths
    MAJOR = "major"  # 11 semitones
    DIMINISHED = "diminished"  # 9 semitones -- fully diminished


class Function(StrEnum):
    """What the chord is doing. Carried so the voicing planner and the
    cadence logic can reason about intent rather than re-deriving it."""

    TONIC = "tonic"
    SUBDOMINANT = "subdominant"
    DOMINANT = "dominant"
    PREDOMINANT = "predominant"
    CHROMATIC = "chromatic"
    PASSING = "passing"


#: Semitones above the root for each triad quality.
TRIAD_INTERVALS: dict[Quality, tuple[int, ...]] = {
    Quality.MAJOR: (0, 4, 7),
    Quality.MINOR: (0, 3, 7),
    Quality.DIMINISHED: (0, 3, 6),
    Quality.AUGMENTED: (0, 4, 8),
    Quality.SUSPENDED_2: (0, 2, 7),
    Quality.SUSPENDED_4: (0, 5, 7),
}

#: Semitones above the root for each seventh.
SEVENTH_INTERVALS: dict[Seventh, int] = {
    Seventh.MINOR: 10,
    Seventh.MAJOR: 11,
    Seventh.DIMINISHED: 9,
}

#: Semitones above the root for the upper extensions, unaltered.
EXTENSION_INTERVALS: dict[int, int] = {9: 14, 11: 17, 13: 21}


class Tone(StrEnum):
    """A chord member, named by its function rather than its interval.

    Named because the omission priority is about function: the third and the
    seventh define the chord's quality, so they are the last to go, while the
    fifth is nearly always redundant once a bass states the root.
    """

    ROOT = "root"
    THIRD = "third"
    FIFTH = "fifth"
    SEVENTH = "seventh"
    NINTH = "ninth"
    ELEVENTH = "eleventh"
    THIRTEENTH = "thirteenth"


#: Which tones to shed first when voices run out, least important first.
#:
#: The fifth goes before anything else: it is implied by the root and
#: contributes no information about quality. The root can go next *when a bass
#: part is stating it* -- that is what makes a rootless voicing work and it is
#: why the planner has to know what the bass is doing. The third and seventh
#: are last, because dropping either stops the chord being the chord it is:
#: Cmaj7 without its B is just C major, and without its E it is ambiguous.
OMISSION_ORDER: tuple[Tone, ...] = (
    Tone.FIFTH,
    Tone.ELEVENTH,
    Tone.THIRTEENTH,
    Tone.ROOT,
    Tone.NINTH,
    Tone.SEVENTH,
    Tone.THIRD,
)


@dataclass(frozen=True)
class ChordIntent:
    """The harmony a composer means, before anyone decides what can be played.

    `root` is a pitch class, 0-11, with 0 = C. `degree` and `mode` are carried
    alongside it so a chord keeps its scale-degree identity for analysis and
    for roman-numeral reporting -- a chord that knows only its pitch class has
    lost the information that makes it a `IVmaj7` rather than an `Fmaj7`.
    """

    root: int
    quality: Quality = Quality.MAJOR
    seventh: Seventh = Seventh.NONE
    #: Upper extensions present, as scale numbers: 9, 11, 13.
    extensions: tuple[int, ...] = ()
    #: Which chord member is in the bass. 0 = root position, 1 = first
    #: inversion (third in the bass), 2 = second, 3 = third inversion.
    inversion: int = 0
    #: Semitone adjustments by scale number, e.g. {5: -1} for a flat fifth,
    #: {9: 1} for a sharp ninth. Kept symbolic so spelling survives.
    alterations: dict[int, int] = field(default_factory=dict)
    function: Function = Function.TONIC
    #: Set when the chord is borrowed from a parallel mode, naming the source
    #: so the report can say "minor iv borrowed from the parallel minor"
    #: rather than presenting a chromatic chord with no explanation.
    borrowed_from: str | None = None
    #: Scale degree in the home key, when known. 0-indexed.
    degree: int | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.root <= 11:
            raise ValueError(f"root must be a pitch class 0-11, got {self.root}")
        for extension in self.extensions:
            if extension not in EXTENSION_INTERVALS:
                raise ValueError(
                    f"unknown extension {extension}; have "
                    f"{sorted(EXTENSION_INTERVALS)}"
                )
        if self.extensions and self.seventh is Seventh.NONE:
            # An added ninth over a triad is legitimate (Cadd9), but a 9th
            # chord implies a seventh. Caught here rather than producing a
            # voicing nobody asked for.
            if 11 in self.extensions or 13 in self.extensions:
                raise ValueError(
                    "an eleventh or thirteenth implies a seventh; set one or "
                    "use extensions=(9,) for an added-ninth triad"
                )
        if not 0 <= self.inversion <= 3:
            raise ValueError(f"inversion must be 0-3, got {self.inversion}")
        # Force the lazy checks now. Validation lived only in `tones()`, so
        # an alteration naming a member the chord does not have -- a flat
        # seventh on a triad -- constructed happily and failed somewhere
        # downstream instead of at the point of the mistake.
        self.tones()
        self.bass_tone()

    # -- the tones this chord contains ------------------------------------

    def tones(self) -> dict[Tone, int]:
        """Chord members mapped to semitones above the root.

        Alterations are applied here, so a flat fifth is a fifth that has been
        flattened rather than a separate tone -- which keeps the omission
        priority meaningful.
        """
        triad = TRIAD_INTERVALS[self.quality]
        out: dict[Tone, int] = {Tone.ROOT: triad[0], Tone.THIRD: triad[1]}
        out[Tone.FIFTH] = triad[2]
        if self.seventh is not Seventh.NONE:
            out[Tone.SEVENTH] = SEVENTH_INTERVALS[self.seventh]
        by_number = {9: Tone.NINTH, 11: Tone.ELEVENTH, 13: Tone.THIRTEENTH}
        for extension in self.extensions:
            out[by_number[extension]] = EXTENSION_INTERVALS[extension]
        # Alterations, by the scale number they refer to.
        numbers = {
            3: Tone.THIRD, 5: Tone.FIFTH, 7: Tone.SEVENTH,
            9: Tone.NINTH, 11: Tone.ELEVENTH, 13: Tone.THIRTEENTH,
        }
        for number, shift in self.alterations.items():
            tone = numbers.get(number)
            if tone is None:
                raise ValueError(f"cannot alter scale number {number}")
            if tone not in out:
                raise ValueError(
                    f"cannot alter the {tone.value}: this chord has no {number}"
                )
            out[tone] = out[tone] + shift
        return out

    def pitch_classes(self) -> frozenset[int]:
        """Every pitch class the full chord contains."""
        return frozenset((self.root + s) % 12 for s in self.tones().values())

    def bass_tone(self) -> Tone:
        """Which member the inversion puts lowest."""
        order = [Tone.ROOT, Tone.THIRD, Tone.FIFTH, Tone.SEVENTH]
        available = [t for t in order if t in self.tones()]
        if self.inversion >= len(available):
            raise ValueError(
                f"inversion {self.inversion} needs a "
                f"{order[self.inversion].value}, which this chord has not got"
            )
        return available[self.inversion]

    # -- reporting ---------------------------------------------------------

    @property
    def symbol(self) -> str:
        """Chord symbol, e.g. `Cmaj9`, `Dm11`, `G7`, `Fm6`, `C/E`."""
        names = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
        text = names[self.root % 12]
        if self.quality is Quality.MINOR:
            text += "m"
        elif self.quality is Quality.DIMINISHED:
            text += "dim"
        elif self.quality is Quality.AUGMENTED:
            text += "aug"
        elif self.quality in (Quality.SUSPENDED_2, Quality.SUSPENDED_4):
            text += self.quality.value
        top = max(self.extensions) if self.extensions else None
        if self.seventh is Seventh.MAJOR:
            text += f"maj{top or 7}"
        elif self.seventh is Seventh.MINOR:
            text += str(top or 7)
        elif self.seventh is Seventh.DIMINISHED:
            text += "dim7"
        elif top:
            text += f"add{top}"
        for number, shift in sorted(self.alterations.items()):
            text += ("#" if shift > 0 else "b") + str(number)
        if self.inversion:
            bass = (self.root + self.tones()[self.bass_tone()]) % 12
            text += f"/{names[bass]}"
        return text

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "root": self.root,
            "quality": self.quality.value,
            "seventh": self.seventh.value,
            "extensions": list(self.extensions),
            "inversion": self.inversion,
            "alterations": dict(self.alterations),
            "function": self.function.value,
            "borrowed_from": self.borrowed_from,
            "degree": self.degree,
            "pitch_classes": sorted(self.pitch_classes()),
            "tones": {t.value: s for t, s in self.tones().items()},
        }


@dataclass(frozen=True)
class ChordVoicing:
    """What actually sounds, and what the planner gave up to fit it.

    `omitted` is the honest part. A voicing that silently drops the seventh of
    a maj7 has changed the harmony, and the arranger has to be able to see
    that it happened.
    """

    intent: ChordIntent
    #: Absolute MIDI pitches, lowest first.
    pitches: tuple[int, ...]
    #: Which members were left out, in the order the planner shed them.
    omitted: tuple[Tone, ...] = ()
    #: Pitches the bass part is covering, excluded from `pitches`.
    bass_pitches: tuple[int, ...] = ()
    #: Tones retained, parallel to `pitches`.
    tones: tuple[Tone, ...] = ()

    @property
    def voice_count(self) -> int:
        return len(self.pitches)

    @property
    def keeps_identity(self) -> bool:
        """Whether the voicing still says what the chord is.

        True when the third and the seventh survive -- or when there is no
        seventh to lose. Those two carry the quality; everything else is
        colour or reinforcement.
        """
        tones = set(self.tones) | set(self.bass_tones())
        if Tone.THIRD not in tones and self.intent.quality not in (
            Quality.SUSPENDED_2,
            Quality.SUSPENDED_4,
        ):
            return False
        if self.intent.seventh is not Seventh.NONE and Tone.SEVENTH not in tones:
            return False
        return True

    def bass_tones(self) -> tuple[Tone, ...]:
        """Which members the bass is covering."""
        by_class = {
            (self.intent.root + s) % 12: t for t, s in self.intent.tones().items()
        }
        return tuple(
            by_class[p % 12] for p in self.bass_pitches if p % 12 in by_class
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.intent.symbol,
            "pitches": list(self.pitches),
            "tones": [t.value for t in self.tones],
            "omitted": [t.value for t in self.omitted],
            "bass_pitches": list(self.bass_pitches),
            "voice_count": self.voice_count,
            "keeps_identity": self.keeps_identity,
        }
