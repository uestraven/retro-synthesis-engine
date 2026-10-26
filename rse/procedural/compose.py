"""A procedural NES composer: rules first, then controlled randomness.

The opposite design to Fermata. Instead of sampling plausible notes and
hoping harmony emerges, this fixes a harmonic plan and derives every voice
from it, so the four parts agree by construction rather than by luck.

That addresses a specific measured failure. A piece the listener called
"harmonies make no sense" had been arpeggiated 0%, truncated 0%, lost nothing
and merged nothing -- the arranger never touched it, so the bad harmony was
the composer's. A statistical model can produce four individually plausible
lines that disagree with each other. A shared progression cannot.

It also composes *into* the NES channels rather than onto them. Fermata writes
unrestricted multitrack MIDI that RSE then folds onto four monophonic voices;
this writes one note per channel per step from the start, so there is no
polyphony to reduce, no role to infer, and no chord to arpeggiate.

What is rule-based and what is random:

- **Rules**: key, scale, the chord progression, which pitches each voice may
  use against the current chord, the section form, and the drum pattern
  family. None of these are sampled freely.
- **Seeded randomness**: which progression from the style's set, the motif's
  contour and rhythm, which passing tones are taken, and the drum pattern's
  fills. One seed reproduces a piece exactly.

The motif is generated once and *re-harmonised* for each chord rather than
transposed blindly, so a return is recognisable without being a literal
replay -- the thing the Fermata pipeline could only achieve by pasting four
identical bars.

No claim is made that this sounds better than Fermata. It is a baseline to
test a design philosophy against the 60% NES-game-music pass rate.
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass, field
from typing import Any

from rse.events.schema import (
    Composition,
    Noise,
    Note,
    TempoChange,
    TimeSignatureChange,
)

#: Scale degrees in semitones, by mode name.
SCALES: dict[str, tuple[int, ...]] = {
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "major": (0, 2, 4, 5, 7, 9, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
}

#: D4 grid: 16 rows per bar, 4 rows per beat.
ROWS_PER_BAR = 16
ROWS_PER_BEAT = 4
TICKS_PER_BEAT = 480
TICKS_PER_ROW = TICKS_PER_BEAT // ROWS_PER_BEAT
TICKS_PER_BAR = TICKS_PER_ROW * ROWS_PER_BAR

#: Tempo is a frame count on this hardware, not a free-running BPM.
#: See rse/hardware/tempo.py for why the reachable tempos are a ladder.
from rse.hardware.tempo import NTSC_FRAME_RATE  # noqa: E402
from rse.hardware.tempo import snap as snap_tempo  # noqa: E402
from rse.procedural.phrase import BarRole, plan_phrases  # noqa: E402

#: NES pitch range the hardware profile accepts.
PITCH_MIN, PITCH_MAX = 33, 108

#: Noise periods used as drum voices. Low period reads as a snare-ish crack,
#: high as a hat; the renderer decides the timbre.
KICK, SNARE, HAT = 1, 6, 12


@dataclass(frozen=True)
class HarmonicRhythm:
    """How long each chord lasts, and whether the bass follows it.

    Durations are in rows, so a bar is 16 and a half-bar is 8 -- integers
    rather than fractional bars, because the D4 grid is rows and a float bar
    count would have to be rounded somewhere.

    The list cycles over the progression's chords, so a 4-chord progression
    with durations (8, 8) gives two chords per bar and completes in two bars
    instead of four. That changes the harmonic pacing without touching the
    progression itself, which is the point: the same chords at a different
    rate are a different musical experience.

    `pedal` holds the bass on the cycle's first root while the chords above it
    move. A pedal is not a chord duration, which is why it is a separate flag
    rather than another entry in the durations list.
    """

    name: str
    durations: tuple[int, ...]
    pedal: bool = False

    def plan(self, n_chords: int, n_bars: int) -> list[tuple[int, int, int]]:
        """(chord index, start row, length in rows) covering `n_bars` bars."""
        total = n_bars * ROWS_PER_BAR
        out: list[tuple[int, int, int]] = []
        row = 0
        step = 0
        while row < total:
            length = self.durations[step % len(self.durations)]
            length = min(length, total - row)
            out.append((step % n_chords, row, length))
            row += length
            step += 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "durations": list(self.durations),
            "pedal": self.pedal,
        }


#: One chord per bar: the only rhythm v1 used, and the default, so a caller
#: that does not ask for harmonic-rhythm variety gets byte-identical output.
ONE_PER_BAR = HarmonicRhythm("one_per_bar", (ROWS_PER_BAR,))

#: The stage-1 vocabulary. Each changes pacing only; no new chords.
RHYTHMS: dict[str, HarmonicRhythm] = {
    "one_per_bar": ONE_PER_BAR,
    "two_per_bar": HarmonicRhythm("two_per_bar", (8,)),
    "two_bar_hold": HarmonicRhythm("two_bar_hold", (ROWS_PER_BAR * 2,)),
    # Long-short: a chord held a bar then two at half a bar, which is the
    # commonest way NES writing accelerates into a cadence.
    "accelerating": HarmonicRhythm("accelerating", (ROWS_PER_BAR, 8, 8)),
    # A bar of tonic under moving harmony above it.
    "tonic_pedal": HarmonicRhythm("tonic_pedal", (8,), pedal=True),
}


@dataclass
class Style:
    """A preset: what the rules are allowed to choose from.

    `tempo_bpm` is snapped to the NES frame grid in `__post_init__`, so a
    preset cannot hold a tempo the console could not play and the value the
    report shows is always the value the music uses. Four of the six presets
    were off the grid -- `action` asked for 168 BPM, which is 5.37 frames per
    row -- and because every layer downstream agreed with that float, nothing
    noticed.

    Snapping here rather than at render time is deliberate: `dataclasses.replace`
    runs `__post_init__`, so a tempo override from a prompt or a slider lands on
    the grid by the same path as a preset.
    """

    name: str
    mode: str
    tempo_bpm: float
    #: Candidate progressions as scale-degree indices, one chord per bar.
    progressions: tuple[tuple[int, ...], ...]
    #: Rows per melody note, sampled per phrase. Smaller is busier.
    melody_rhythms: tuple[tuple[int, ...], ...]
    #: Which rows the bass attacks within a bar.
    bass_rows: tuple[int, ...]
    #: Accompaniment behaviour: "arpeggio", "offbeat", "pad".
    harmony_style: str
    #: Drum density: rows per hat, and whether a backbeat snare is used.
    hat_every: int
    backbeat: bool
    #: Root pitch, as a MIDI note for the tonic in the melody's octave.
    tonic: int = 60
    #: Probability a melody note is a passing tone rather than a chord tone.
    passing_chance: float = 0.3
    #: Harmonic rhythms this style may use when variety is requested. The
    #: default stays one-per-bar so v1 output is unchanged.
    rhythms: tuple[str, ...] = ("one_per_bar",)
    #: Frames per tracker row, the hardware's own tempo unit. Derived from
    #: `tempo_bpm` by snapping, and the authority for what actually plays.
    frames_per_row: int = 0

    def __post_init__(self) -> None:
        # Full precision kept internally and rounded only for display. Storing
        # the rounded BPM and then deriving the tempo map from it put the
        # playback 7 microseconds a beat off the frame count it claimed --
        # inaudible, but it meant `frames_per_row` was not quite what played,
        # which defeats the point of making it the source of truth.
        tempo = snap_tempo(self.tempo_bpm)
        object.__setattr__(self, "frames_per_row", tempo.frames_per_row)
        object.__setattr__(self, "tempo_bpm", tempo.bpm)

    @property
    def row_sec(self) -> float:
        """One tracker row, in seconds. What the driver would wait."""
        return self.frames_per_row / NTSC_FRAME_RATE

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "tempo_bpm": round(self.tempo_bpm, 2),
            "frames_per_row": self.frames_per_row,
            "row_ms": round(self.row_sec * 1000, 3),
            "harmony_style": self.harmony_style,
            "hat_every": self.hat_every,
            "backbeat": self.backbeat,
            "tonic": self.tonic,
            "passing_chance": self.passing_chance,
        }


#: Style presets. Progressions are written as scale-degree indices so they
#: transpose to any key and stay inside the mode.
STYLES: dict[str, Style] = {
    "boss": Style(
        name="boss",
        mode="harmonic_minor",
        tempo_bpm=150,
        progressions=(
            (0, 0, 5, 6),   # i i VI VII -- driving, unresolved
            (0, 6, 5, 4),   # i VII VI V -- descending
            (0, 3, 4, 0),   # i iv v i
        ),
        melody_rhythms=((2, 2, 2, 2, 4, 4), (4, 2, 2, 4, 4), (2, 2, 4, 2, 2, 4)),
        bass_rows=(0, 4, 8, 12),
        harmony_style="offbeat",
        hat_every=2,
        backbeat=True,
        tonic=57,
        passing_chance=0.35,
        rhythms=("one_per_bar", "two_per_bar", "accelerating", "tonic_pedal"),
    ),
    "action": Style(
        name="action",
        mode="minor",
        tempo_bpm=168,
        progressions=((0, 5, 2, 6), (0, 3, 6, 4), (0, 6, 3, 5)),
        melody_rhythms=((2, 2, 4, 2, 2, 4), (2, 2, 2, 2, 8), (4, 2, 2, 4, 4)),
        bass_rows=(0, 3, 6, 8, 11, 14),
        harmony_style="arpeggio",
        hat_every=2,
        backbeat=True,
        tonic=60,
        passing_chance=0.3,
        rhythms=("one_per_bar", "two_per_bar", "accelerating"),
    ),
    "adventure": Style(
        name="adventure",
        mode="major",
        tempo_bpm=137,
        progressions=((0, 4, 5, 3), (0, 3, 4, 0), (0, 5, 3, 4)),
        melody_rhythms=((4, 4, 4, 4), (4, 2, 2, 4, 4), (8, 4, 4)),
        bass_rows=(0, 8),
        harmony_style="arpeggio",
        hat_every=4,
        backbeat=True,
        tonic=60,
        passing_chance=0.25,
        rhythms=("one_per_bar", "two_bar_hold", "accelerating"),
    ),
    "town": Style(
        name="town",
        mode="major",
        tempo_bpm=128,
        progressions=((0, 3, 4, 0), (0, 5, 3, 4), (0, 4, 0, 4)),
        melody_rhythms=((4, 4, 4, 4), (4, 4, 2, 2, 4), (8, 8)),
        bass_rows=(0, 8),
        harmony_style="pad",
        hat_every=4,
        backbeat=False,
        tonic=62,
        passing_chance=0.2,
        rhythms=("one_per_bar", "two_bar_hold"),
    ),
    "tension": Style(
        name="tension",
        mode="phrygian",
        tempo_bpm=104,
        progressions=((0, 1, 0, 1), (0, 6, 0, 1), (0, 0, 1, 1)),
        melody_rhythms=((8, 8), (8, 4, 4), (16,)),
        bass_rows=(0, 8),
        harmony_style="pad",
        hat_every=8,
        backbeat=False,
        tonic=55,
        passing_chance=0.15,
        rhythms=("two_bar_hold", "tonic_pedal", "one_per_bar"),
    ),
    "victory": Style(
        name="victory",
        mode="major",
        tempo_bpm=144,
        progressions=((0, 4, 0, 4), (0, 3, 4, 0)),
        melody_rhythms=((2, 2, 4, 4, 4), (4, 4, 4, 4)),
        bass_rows=(0, 4, 8, 12),
        harmony_style="arpeggio",
        hat_every=2,
        backbeat=True,
        tonic=64,
        passing_chance=0.2,
        rhythms=("one_per_bar", "two_per_bar", "accelerating"),
    ),
}


#: Opt-in progression families, one list per style.
#:
#: Each is here for a stated musical reason, not to raise a diversity count.
#: Entries are plain degree integers where the mode already gives the wanted
#: chord, and (degree, quality) pairs only where it does not -- the new
#: mechanism is used exactly where the old one cannot reach.
EXTRA_PROGRESSIONS: dict[str, tuple[tuple[Any, ...], ...]] = {
    # harmonic minor already supplies V major, so these need no override.
    "boss": (
        (0, 5, 3, 4),            # i-bVI-iv-V: dramatic, strong resolution
        (0, 3, 1, 4),            # i-iv-ii°-V: stronger dominant preparation
    ),
    # natural minor does NOT supply V major -- degree 4 is minor there -- so
    # the cadence progressions borrow it explicitly. This is the single most
    # useful chord the old system could not express.
    "action": (
        (0, 5, 2, 6),            # i-bVI-bIII-bVII: driving, all diatonic
        (0, 3, (4, "major"), 0),  # i-iv-V-i: borrowed dominant, clear cadence
    ),
    "adventure": (
        (0, 3, 1, 4),            # I-IV-ii-V: forward motion, strong cadence
    ),
    "town": (
        (0, 2, 5, 3),            # I-iii-vi-IV: gentler movement
    ),
    "tension": (
        (0, 1, 0, 3),            # i-bII-i-iv: modal tension, all diatonic
    ),
    "victory": (
        (3, 4, 0, 0),            # IV-V-I-I: triumphant resolution
    ),
}


@dataclass
class Motif:
    """A short melodic idea, stored as scale-degree steps rather than pitches.

    Degrees, not semitones, so the same shape can be re-harmonised against a
    different chord and still fit the mode. That is what makes a return
    recognisable without being a transposed copy.
    """

    #: Offsets in scale degrees from the chord tone the phrase starts on.
    degrees: tuple[int, ...]
    #: Row lengths, one per note.
    rhythm: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"degrees": list(self.degrees), "rhythm": list(self.rhythm)}


@dataclass
class SectionPlan:
    """One section: its chords and how it treats the motif."""

    label: str
    bars: int
    progression: tuple[int, ...]
    #: "state" plays the motif, "vary" alters it, "contrast" uses a new one.
    treatment: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "bars": self.bars,
            "progression": list(self.progression),
            "treatment": self.treatment,
        }


def scale_pitch(tonic: int, mode: str, degree: int) -> int:
    """Pitch for a scale degree, wrapping octaves as the degree runs past 7."""
    scale = SCALES[mode]
    octave, index = divmod(degree, len(scale))
    return tonic + 12 * octave + scale[index]


def chord_degrees(root_degree: int) -> tuple[int, int, int]:
    """A triad on a scale degree: root, third, fifth, in degree space."""
    return (root_degree, root_degree + 2, root_degree + 4)


def clamp(pitch: int) -> int:
    """Fold into the NES range by octaves, which preserves pitch class."""
    while pitch < PITCH_MIN:
        pitch += 12
    while pitch > PITCH_MAX:
        pitch -= 12
    return pitch


def make_motif(rng: random.Random, style: Style) -> Motif:
    """A 2-bar idea: a contour in scale degrees with a chosen rhythm.

    Built from a small set of shapes rather than free random walk, because a
    random walk is what produces the "random notes" verdict. Each shape rises
    or falls and returns, so the phrase has a direction and an arrival.
    """
    shapes = (
        (0, 1, 2, 1, 0),
        (0, 2, 1, 3, 2),
        (0, -1, 0, 2, 1),
        (0, 2, 4, 2, 0),
        (0, 1, 3, 2, 0),
    )
    shape = list(rng.choice(shapes))
    rhythm = list(rng.choice(style.melody_rhythms))
    # Match lengths: trim or extend the contour to the rhythm's note count.
    while len(shape) < len(rhythm):
        shape.append(shape[-1] + rng.choice((-1, 1)))
    shape = shape[: len(rhythm)]
    return Motif(degrees=tuple(shape), rhythm=tuple(rhythm))


def vary_motif(rng: random.Random, motif: Motif) -> Motif:
    """Keep the contour recognisable, change one thing about it.

    Exactly one alteration, chosen at random: invert the last gesture, add a
    passing step, or augment the rhythm. Changing several at once makes a new
    idea rather than a variation of this one.
    """
    degrees = list(motif.degrees)
    rhythm = list(motif.rhythm)
    choice = rng.choice(("invert_tail", "ornament", "augment"))
    if choice == "invert_tail" and len(degrees) >= 3:
        pivot = degrees[-3]
        degrees[-2] = pivot - (degrees[-2] - pivot)
        degrees[-1] = pivot - (degrees[-1] - pivot)
    elif choice == "ornament" and len(degrees) >= 2:
        at = rng.randrange(len(degrees) - 1)
        degrees.insert(at + 1, degrees[at] + rng.choice((-1, 1)))
        half = max(1, rhythm[at] // 2)
        rhythm[at] = half
        rhythm.insert(at + 1, half)
    else:
        rhythm = [max(1, r * 2) for r in rhythm]
    return Motif(degrees=tuple(degrees), rhythm=tuple(rhythm))


def resolve_chord(spec: Any, style: Style) -> Any:
    """Resolve a progression entry, importing lazily to avoid a cycle."""
    from rse.procedural.chords import resolve

    return resolve(spec, tonic=style.tonic, mode=style.mode)


def chord_pitches(voicing: Any, style: Style, *, octave: int = 0) -> list[int]:
    """Absolute pitches for a resolved chord.

    Mode-derived chords go through the v1 degree arithmetic unchanged, which
    is what keeps the frozen baseline byte-identical: the degree path can
    place a tone above the octave and the pitch path cannot reproduce that
    voicing. Altered chords use their semitones directly, because a borrowed
    pitch has no scale degree.
    """
    if voicing.degrees is not None:
        return [
            scale_pitch(style.tonic, style.mode, d) + 12 * octave
            for d in voicing.degrees
        ]
    return [style.tonic + s + 12 * octave for s in voicing.semitones]


def plan_form(
    rng: random.Random,
    style: Style,
    *,
    target_seconds: float,
    section_bars: int = 8,
    progressions: tuple[Any, ...] | None = None,
) -> list[SectionPlan]:
    """A-A'-B-A'' sized to the duration, with B on its own progression.

    The contrast section gets a *different progression from the same mode*
    rather than a key change. A key change a fifth away was tried in the
    Fermata pipeline and the listener's verdict on the return was "makes no
    sense": B climbed to the loudest bar in the track and then the home key
    returned at half the level with no preparation.
    """
    seconds_per_bar = 4 * 60.0 / style.tempo_bpm
    count = max(2, round(target_seconds / (section_bars * seconds_per_bar)))
    pool = progressions or style.progressions
    main = rng.choice(pool)
    others = [p for p in pool if p != main] or [main]
    contrast = rng.choice(others)

    plans = [SectionPlan("A", section_bars, main, "state")]
    for index in range(1, count):
        if index == count - 2 and count >= 3:
            plans.append(SectionPlan("B", section_bars, contrast, "contrast"))
        elif index == count - 1:
            plans.append(SectionPlan("A''", section_bars, main, "state"))
        else:
            plans.append(SectionPlan(f"A'{index}", section_bars, main, "vary"))
    return plans


def cadence_pitch(
    style: Style, voicing: Any, cadence: str | None, fallback: int
) -> int:
    """The pitch a phrase arrives on, chosen by harmonic function.

    "closed" lands on the tonic, which is what makes a consequent sound
    finished. "open" lands on the nearest chord tone that is NOT the tonic, so
    the antecedent asks a question the next phrase can answer. Measured against
    the fallback so the arrival is a step away from where the line already was
    rather than a leap across the register.
    """
    tonic_pc = style.tonic % 12
    if cadence == "closed":
        # Nearest octave of the tonic to where the melody already is.
        candidates = [style.tonic + 12 * octave for octave in (-1, 0, 1, 2)]
        return min(candidates, key=lambda p: abs(p - fallback))

    # Open: a chord tone other than the tonic.
    if voicing.degrees is not None:
        pitches = [
            scale_pitch(style.tonic, style.mode, d) + 12 * octave
            for d in voicing.degrees
            for octave in (-1, 0, 1)
        ]
    else:
        pitches = [
            style.tonic + semitone + 12 * octave
            for semitone in voicing.semitones
            for octave in (-1, 0, 1)
        ]
    other = [p for p in pitches if p % 12 != tonic_pc]
    if not other:
        return fallback
    return min(other, key=lambda p: abs(p - fallback))


def plan_archetype(
    rng: random.Random,
    style: Style,
    *,
    form: str,
    section_bars: int,
    progressions: tuple[Any, ...] | None = None,
) -> list[SectionPlan]:
    """Sections from a form archetype, at the archetype's own length.

    The counterpart to `plan_form`, which derives the section COUNT from the
    duration and so had to stretch a three-section ternary into five -- at
    which point it was no longer a ternary. Here the archetype keeps its
    length and the section length absorbs the clock instead.

    Contrasting sections get their own progression; a `vary` section keeps the
    opening's, because A' is the opening re-voiced and changing its harmony
    would make it a new section rather than a variation.
    """
    from rse.procedural.variety import fit_template, natural_length

    treatments = fit_template(form, natural_length(form))
    pool = progressions or style.progressions
    main = rng.choice(pool)
    others = [p for p in pool if p != main] or [main]
    # Distinct progressions for distinct contrasts, so a linear form's B, C
    # and D are genuinely different places and not the same chords relabelled.
    contrast_progressions: dict[str, Any] = {}
    plans: list[SectionPlan] = []
    seen_state = 0
    for treatment in treatments:
        if treatment.startswith("contrast"):
            if treatment not in contrast_progressions:
                contrast_progressions[treatment] = rng.choice(others)
            progression = contrast_progressions[treatment]
        else:
            progression = main
        if treatment == "state":
            # A literal restatement IS the opening, so it carries the same
            # label. That is what distinguishes A-B-A (ternary) from A-B-A'
            # (rounded binary, where the return is re-voiced) -- labelling
            # both as A' hid the one difference between the two archetypes.
            label = "A"
        elif treatment == "vary":
            seen_state += 1
            label = "A" + "'" * seen_state
        else:
            label = {"contrast": "B", "contrast2": "C", "contrast3": "D"}.get(
                treatment, "B"
            )
        plans.append(
            SectionPlan(
                label=label,
                bars=section_bars,
                progression=progression,
                treatment=treatment,
            )
        )
    return plans


def render_melody(
    rng: random.Random,
    style: Style,
    motif: Motif,
    plan: SectionPlan,
    *,
    bar_offset: int,
    harmony_plan: list[tuple[int, int, int]],
    phrases: list[BarRole] | None = None,
    shape_velocity: Any = None,
) -> tuple[list[Note], dict[str, int]]:
    """Melody on P1, re-harmonised bar by bar against the progression.

    Each phrase starts on a tone of the bar's chord and follows the motif's
    degree contour. A step that lands outside the chord is kept only if it is
    in the scale, so a passing tone is a passing tone and never an accident.

    `phrases` is the opt-in phrase structure. With it `None` the melody behaves
    exactly as v1 did -- same draws, same output -- because v1 is the version
    the listener approved and a silent change to it is not something to risk on
    an untested idea. With a plan, each bar takes a role: the motif is stated,
    varied, climbed to a peak, and then resolved on a long cadence note
    followed by a rest. v1 had one distinct rhythm across eight bars and no
    rest at all, which is why it reads as a loop rather than a line.
    """
    notes: list[Note] = []
    # Measurement only. Truncating a melody note at a chord change is an
    # EXPERIMENTAL CONSTRAINT, not a music-theory requirement: a common tone
    # held from C major into A minor moves from third to fifth with no
    # conflict, and a suspension that resolves is a device rather than a
    # mistake. Counting the cost here so the listening test can say whether
    # the constraint is worth keeping.
    truncated = 0
    rows_removed = 0
    base = bar_offset * TICKS_PER_BAR
    for chord_index, start_row, span in harmony_plan:
        spec = plan.progression[chord_index % len(plan.progression)]
        voicing = resolve_chord(spec, style)
        altered = voicing.degrees is None
        if altered:
            # Pitch space. The motif's degree contour walks the scale tones
            # this chord allows, which preserves the shape while honouring
            # `avoid` -- so a raised leading tone is available and the natural
            # it replaced is not, without changing the mode globally.
            from rse.procedural.chords import melody_candidates

            chord_tones, scale_tones = melody_candidates(
                voicing, mode=style.mode, octave_span=3
            )
            tones = [style.tonic + t for t in chord_tones]
            ladder = [style.tonic + t for t in scale_tones]
            # Keep the generator aligned with the mode-derived path. That
            # path draws randrange(3) for its three stacked degrees; this one
            # would draw randrange(9) over three octaves of chord tones, so
            # every subsequent draw -- including the percussion, generated
            # per section after the melody -- would diverge. The arms of a
            # harmony experiment would then differ in drums too, confounding
            # the variable under test. Draw in the same range and fold.
            pick_from = tones
            tones = tones[: len(chord_degrees(0))] or tones
        else:
            tones = chord_degrees(spec if isinstance(spec, int) else spec.degree)
            ladder = None
        start = tones[rng.randrange(len(tones))]
        row = 0
        if altered:
            tones = pick_from

        # Which bar of the section this chord sits in, and what that bar is
        # for. `start_row` is in rows from the section start, so this works
        # whether the harmonic rhythm puts one chord in a bar or several.
        role = None
        if phrases:
            bar_index = start_row // ROWS_PER_BAR
            if bar_index < len(phrases):
                role = phrases[bar_index]

        degrees, rhythm = motif.degrees, motif.rhythm
        usable = span
        if role is not None:
            if role.role == "climb":
                # Lift the whole contour so the phrase has a registral peak
                # somewhere chosen rather than wherever the walk landed.
                degrees = tuple(d + role.lift for d in degrees)
            elif role.role == "vary":
                # Retrograde the tail: the repeat is recognisable but not
                # literal, and it costs no RNG draw.
                head, tail = degrees[: len(degrees) // 2], degrees[len(degrees) // 2 :]
                degrees = head + tuple(reversed(tail))
                # And vary the RHYTHM, which is what actually makes the bar
                # sound different. Altering only the pitches left bars 0-2 of
                # every phrase attacking identical rows -- the exact defect
                # this work exists to fix. Augmenting the opening pair into one
                # longer note removes an attack and shifts every later one,
                # without changing how many notes the bar holds enough to
                # disturb the contour.
                if len(rhythm) >= 3:
                    rhythm = (rhythm[0] + rhythm[1],) + tuple(rhythm[2:])
                    degrees = (degrees[0],) + tuple(degrees[2:])
                else:
                    # A one- or two-note bar has nothing to augment, so the
                    # sparse styles kept their single attack pattern. Displace
                    # the bar instead: the same notes entering late is
                    # syncopation, and it is the only rhythmic contrast
                    # available when there is just one note in the bar.
                    displace = min(2, max(1, span // 8))
                    rhythm = tuple(rhythm[:-1]) + (
                        max(1, rhythm[-1] - displace),
                    )
                    row = displace
            elif role.is_cadence:
                # Abandon the motif's rhythm. A cadence is an arrival: a short
                # approach and then one long note, with the rest after it.
                usable = max(1, span - role.rest_rows)
                approach = min(2, max(0, len(degrees) - 1))
                degrees = degrees[:approach] + (degrees[-1],)
                lead_rows = sum(rhythm[:approach]) if approach else 0
                hold = max(1, usable - lead_rows)
                rhythm = tuple(rhythm[:approach]) + (hold,)

        for step, length in zip(degrees, rhythm, strict=False):
            if row >= usable:
                break
            absolute = start_row + row
            if altered:
                # Walk the ladder by index so a degree step stays a step.
                base_index = min(
                    range(len(ladder)), key=lambda i: abs(ladder[i] - start)
                )
                pick = ladder[max(0, min(len(ladder) - 1, base_index + step))]
                if (
                    absolute % ROWS_PER_BEAT == 0
                    and rng.random() > style.passing_chance
                ):
                    pick = min(tones, key=lambda t: abs(t - pick))
                pitch = clamp(pick)
            else:
                degree = start + step
                # On strong rows prefer a chord tone; elsewhere the scale.
                if (
                    absolute % ROWS_PER_BEAT == 0
                    and rng.random() > style.passing_chance
                ):
                    degree = min(tones, key=lambda t: abs(t - degree))
                pitch = clamp(scale_pitch(style.tonic, style.mode, degree))

            # The arrival note is chosen for its function, not by the walk: a
            # consequent closes on the tonic, an antecedent leaves the question
            # open on a chord tone that is not the tonic. Without this, bar
            # endings hit the tonic once or twice in eight, by accident.
            if role is not None and role.is_cadence and step is degrees[-1]:
                pitch = clamp(cadence_pitch(style, voicing, role.cadence, pitch))
            kept = min(length, usable - row)
            if kept < length:
                truncated += 1
                rows_removed += length - kept
            velocity = 100
            if shape_velocity is not None:
                velocity = shape_velocity(
                    row_in_bar=absolute % ROWS_PER_BAR,
                    phrase_start=bool(role and role.position == 0 and row == 0),
                    cadence=bool(role and role.is_cadence and step is degrees[-1]),
                    section_start=absolute == 0,
                )
            notes.append(
                Note(
                    channel_id="P1",
                    tick=base + absolute * TICKS_PER_ROW,
                    pitch=pitch,
                    duration=kept * TICKS_PER_ROW,
                    velocity=velocity,
                )
            )
            row += length
    return notes, {"notes_truncated": truncated, "rows_removed": rows_removed}


def render_harmony(
    style: Style,
    plan: SectionPlan,
    *,
    bar_offset: int,
    harmony_plan: list[tuple[int, int, int]],
    figure: str | None = None,
    shape_velocity: Any = None,
) -> list[Note]:
    """Accompaniment on P2, built only from the bar's chord tones.

    One note at a time, because P2 is monophonic. An arpeggio is therefore
    the honest way to state a chord on this hardware, not a workaround.

    Fully determined by the progression and the figure, with no randomness
    inside a section: the accompaniment's job is to make the harmony audible,
    so varying it bar to bar would weaken the one voice whose predictability
    is the point. `figure` lets a SONG pick a different comp from the style's
    pool, and a `vary` section re-orchestrate -- predictable within a piece,
    different between pieces, which is how a soundtrack works.
    """
    notes: list[Note] = []
    base = bar_offset * TICKS_PER_BAR
    figure = figure or style.harmony_style
    for chord_index, start_row, span in harmony_plan:
        spec = plan.progression[chord_index % len(plan.progression)]
        voicing = resolve_chord(spec, style)
        pitches = chord_pitches(voicing, style, octave=-1)
        if figure == "arpeggio_down":
            pitches = list(reversed(pitches))
        if figure == "pad":
            pitch = clamp(pitches[0])
            notes.append(
                Note(
                    channel_id="P2",
                    tick=base + start_row * TICKS_PER_ROW,
                    pitch=pitch,
                    duration=span * TICKS_PER_ROW,
                    velocity=70,
                )
            )
            continue
        if figure == "stab":
            # One short hit per half bar, which leaves the most space of any
            # figure and is what a sparse arrangement wants.
            step, offset, hit = 8, 0, 2
        elif figure == "alternating":
            # Root and fifth traded on eighths.
            step, offset, hit = 2, 0, 2
            pitches = [pitches[0], pitches[-1]]
        elif figure in ("arpeggio", "arpeggio_down"):
            step, offset, hit = 2, 0, 2
        else:  # offbeat
            step, offset, hit = 4, 2, 2
        for index, row in enumerate(range(offset, span, step)):
            pitch = pitches[index % len(pitches)]
            if figure == "arpeggio" and index % len(pitches) == 0:
                # Alternate octave on each pass through the chord.
                pitch += 12 * (index // len(pitches) % 2)
            pitch = clamp(pitch)
            notes.append(
                Note(
                    channel_id="P2",
                    tick=base + (start_row + row) * TICKS_PER_ROW,
                    pitch=pitch,
                    duration=min(hit, span - row) * TICKS_PER_ROW,
                    velocity=(
                        shape_velocity(row_in_bar=(start_row + row) % ROWS_PER_BAR)
                        if shape_velocity is not None
                        else 72
                    ),
                )
            )
    return notes


def render_counter(
    melody: list[Note], style: Style, *, below: int = 2
) -> list[Note]:
    """P2 doubles the melody in parallel thirds instead of comping.

    One of the re-orchestration moves a `vary` section can make: the same
    sequence data, re-voiced. Parallel thirds because they are consonant
    against any diatonic line, and `below` is in SCALE DEGREES rather than
    semitones, so the interval follows the mode and stays inside the key -- a
    fixed four semitones would produce a major third over a minor chord.
    """
    scale = SCALES[style.mode]
    notes: list[Note] = []
    for note in melody:
        # Recover the melody note's scale degree, then step down the scale.
        offset = note.pitch - style.tonic
        octave, remainder = divmod(offset, 12)
        nearest = min(range(len(scale)), key=lambda i: abs(scale[i] - remainder))
        degree = octave * len(scale) + nearest - below
        notes.append(
            Note(
                channel_id="P2",
                tick=note.tick,
                pitch=clamp(scale_pitch(style.tonic, style.mode, degree)),
                duration=note.duration,
                velocity=68,
            )
        )
    return notes


def render_bass(
    style: Style,
    plan: SectionPlan,
    *,
    bar_offset: int,
    harmony_plan: list[tuple[int, int, int]],
    pedal: bool = False,
    rows: tuple[int, ...] | None = None,
) -> list[Note]:
    """Bass on the triangle: chord roots and fifths, no passing tones.

    Deterministic given the progression and the figure. The bass is what
    makes the harmony audible, so there is no randomness inside a section;
    `rows` lets a song choose which figure it keeps throughout.
    """
    notes: list[Note] = []
    base = bar_offset * TICKS_PER_BAR
    # A pedal holds the cycle's first root under everything above it, so the
    # bass is read from the plan's first chord rather than the current one.
    pedal_pitch = (
        chord_pitches(resolve_chord(plan.progression[0], style), style, octave=-2)[0]
        if pedal
        else None
    )
    bass_rows = rows or style.bass_rows
    for bar in range(plan.bars):
        # Which chord is sounding at this bar's downbeat.
        bar_start = bar * ROWS_PER_BAR
        chord_index = next(
            (
                ci
                for ci, start, span in harmony_plan
                if start <= bar_start < start + span
            ),
            0,
        )
        spec = plan.progression[chord_index % len(plan.progression)]
        low = chord_pitches(resolve_chord(spec, style), style, octave=-2)
        for index, row in enumerate(bass_rows):
            if pedal_pitch is not None:
                pitch = clamp(pedal_pitch)
            else:
                pitch = clamp(low[0] if index % 2 == 0 else low[2 % len(low)])
            length = (
                bass_rows[index + 1] - row
                if index + 1 < len(bass_rows)
                else ROWS_PER_BAR - row
            )
            notes.append(
                Note(
                    channel_id="TR",
                    tick=base + (bar_start + row) * TICKS_PER_ROW,
                    pitch=pitch,
                    duration=max(1, length) * TICKS_PER_ROW,
                    velocity=90,
                )
            )
    return notes


def render_drums(
    rng: random.Random, style: Style, plan: SectionPlan, *, bar_offset: int
) -> list[Noise]:
    """Percussion on the noise channel: kick, backbeat snare, hats.

    One hit at a time, since the channel is monophonic -- so a kick and a hat
    on the same row is impossible and the pattern is written to avoid it
    rather than relying on repair to drop one.
    """
    hits: list[Noise] = []
    for bar in range(plan.bars):
        base = (bar_offset + bar) * TICKS_PER_BAR
        taken: set[int] = set()

        # Explicit ordering instead of a closure: the noise channel is
        # monophonic, so the first hit claiming a row keeps it and later
        # candidates for that row are simply not written. Kick and snare
        # outrank hats, which is the usual priority on this hardware.
        wanted: list[tuple[int, int, int]] = [(0, KICK, 110), (8, KICK, 100)]
        if style.backbeat:
            wanted += [(4, SNARE, 105), (12, SNARE, 105)]
        if bar == plan.bars - 1:
            # A fill in a section's last bar, so the seam is announced.
            wanted += [
                (row, SNARE, 95) for row in (13, 14, 15) if rng.random() < 0.7
            ]
        wanted += [
            (row, HAT, 60) for row in range(0, ROWS_PER_BAR, style.hat_every)
        ]
        for row, value, velocity in wanted:
            if row in taken or row >= ROWS_PER_BAR:
                continue
            taken.add(row)
            hits.append(
                Noise(
                    channel_id="NO",
                    tick=base + row * TICKS_PER_ROW,
                    value=value,
                    duration=TICKS_PER_ROW,
                    velocity=velocity,
                )
            )
    return hits


@dataclass
class ProceduralResult:
    composition: Composition
    style: Style
    motif: Motif
    sections: list[SectionPlan]
    seed: int
    rhythm: HarmonicRhythm = ONE_PER_BAR
    #: How much melody the chord-boundary constraint removed.
    truncation: dict[str, int] = field(default_factory=dict)
    #: Which progression pool was used: None for v1.
    vocabulary: str | None = None
    variations: list[dict[str, Any]] = field(default_factory=list)
    #: Instrumentation, when sound design is on. None for the dry render.
    sound: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "style": self.style.to_dict(),
            "sound": self.sound.to_dict() if self.sound else None,
            "harmonic_rhythm": self.rhythm.to_dict(),
            "vocabulary": self.vocabulary,
            "truncation": dict(self.truncation),
            "motif": self.motif.to_dict(),
            "sections": [s.to_dict() for s in self.sections],
            "variations": self.variations,
            "n_events": len(self.composition.sounding_events()),
        }


def compose(
    *,
    style_name: str,
    seed: int,
    target_seconds: float = 60.0,
    section_bars: int = 8,
    harmonic_rhythm: str | None = None,
    vocabulary: str | None = None,
    tempo_bpm: int | None = None,
    tonic: int | None = None,
    phrasing: str | None = None,
    isolate_melody_rng: bool = False,
    variety: str | None = None,
    sound_design: str | None = "expressive",
    sound_profile: str | None = None,
    mood: str | None = None,
) -> ProceduralResult:
    """One complete NES piece from a style preset and a seed.

    `harmonic_rhythm` is `None` for the frozen v1 behaviour -- one chord per
    bar, and **no RNG draw**, so output is byte-identical to the 24 pieces in
    the listening baseline. Pass a name from `RHYTHMS` to force one, or
    `"auto"` to let the seed choose from the style's declared vocabulary.
    Opt-in rather than default, because the baseline is the thing that worked
    and a silent extra draw would have changed every piece.

    `vocabulary` is `None` for v1's progressions, `"extra"` for the Stage 2
    families only, or `"all"` for both. Also opt-in, and the v1 path draws
    from the same pool in the same order so the baseline is untouched.

    `phrasing` is `None` for v1's bar-by-bar melody and `"period"` for the
    phrase structure: four-bar phrases, an antecedent that leaves the question
    open and a consequent that closes on the tonic, a registral peak before
    each cadence, and a rest after it. Opt-in for the same reason the other
    expansions were -- v1 is what the listener approved, and v1 had one
    distinct rhythm across eight bars and no rest at all.

    `sound_design` defaults to `"expressive"`: per-song pulse timbres and
    position-shaped velocity. Promoted from opt-in after a 12-pair listening
    test accepted it -- the first non-null result after four composition
    experiments returned none, and the only one that changed how the music is
    PLAYED rather than what is played. Pass `None` for the dry v1 render, which
    is kept reproducible because the frozen baselines are expressed in it.

    It changes NO note:
    pitches, onsets, durations, channels and structure are identical, which is
    what makes it testable against the same score. Drawn from its own seeded
    stream so it cannot perturb a composition decision.

    `variety` is `None` for v1 and `"full"` to open the four axes measured as
    the causes of sameness: across 20 seeds of a style, the harmony and bass
    had 5-6 variants (victory: 2) and the form, key and tempo had exactly one
    each. With it on, a song draws its own key, tempo rung, section-form
    archetype, accompaniment figure and bass figure, and `vary` sections
    re-orchestrate rather than rewrite the motif. Opt-in, like the rest.

    `isolate_melody_rng` draws the melody's randomness from a generator seeded
    separately, so the NUMBER of melody notes cannot shift anything generated
    after it. It has to exist before phrasing can be tested: the melody draws
    once per note for its passing-tone decision, the phrased melody has fewer
    notes, and so the drums and every later section's motif came out different
    too -- a paired test would have been measuring phrasing plus a different
    arrangement. Off by default, because turning it on changes v1's output and
    v1 is the approved baseline; if phrasing is promoted this should be
    promoted with it, since melody density leaking into the drum pattern is a
    coupling defect either way.

    `tempo_bpm` and `tonic` override the preset so a prompt can ask for a
    tempo or a key without needing a new style. Both default to the preset's
    own values and neither consumes the generator, so the frozen baseline is
    unaffected: a request that names no tempo and no key composes exactly
    what v1 composed.
    """
    if style_name not in STYLES:
        raise KeyError(f"unknown style {style_name!r}; have {sorted(STYLES)}")
    style = STYLES[style_name]
    if tempo_bpm is not None or tonic is not None:
        # A copy, so the shared preset is never mutated by a request.
        style = dataclasses.replace(
            style,
            tempo_bpm=int(tempo_bpm) if tempo_bpm is not None else style.tempo_bpm,
            tonic=int(tonic) if tonic is not None else style.tonic,
        )
    rng = random.Random(seed)
    if harmonic_rhythm is None:
        rhythm = ONE_PER_BAR
    elif harmonic_rhythm == "auto":
        # Drawn from a generator seeded separately, so choosing a rhythm does
        # not shift the main stream and shift every melody with it.
        rhythm = RHYTHMS[random.Random(seed ^ 0x5F5F).choice(style.rhythms)]
    else:
        if harmonic_rhythm not in RHYTHMS:
            raise KeyError(
                f"unknown harmonic rhythm {harmonic_rhythm!r}; "
                f"have {sorted(RHYTHMS)}"
            )
        rhythm = RHYTHMS[harmonic_rhythm]
    extra = EXTRA_PROGRESSIONS.get(style_name, ())
    if vocabulary is None:
        pool = None
    elif vocabulary == "extra":
        pool = extra or style.progressions
    elif vocabulary == "all":
        pool = tuple(style.progressions) + tuple(extra)
    else:
        raise KeyError(
            f"unknown vocabulary {vocabulary!r}; use None, 'extra' or 'all'"
        )

    if phrasing not in (None, "period"):
        raise KeyError(
            f"unknown phrasing {phrasing!r}; use None or 'period'"
        )

    if variety not in (None, "full"):
        raise KeyError(f"unknown variety {variety!r}; use None or 'full'")
    if sound_design not in (None, "expressive"):
        raise KeyError(
            f"unknown sound_design {sound_design!r}; use None or 'expressive'"
        )

    voices = None
    lead_shaper = comp_shaper = None
    if sound_design == "expressive":
        from rse.procedural import sound as S

        voices = S.choose_voices(
            style_name, seed=seed, profile=sound_profile, mood=mood
        )
        lead_shaper = S.lead_velocity
        comp_shaper = S.comp_velocity

    arrangement = None
    if variety == "full":
        from rse.hardware.tempo import bpm_for_frames
        from rse.procedural import variety as V

        arrangement = V.choose(
            style_name,
            seed=seed,
            default_harmony=style.harmony_style,
            default_bass=style.bass_rows,
            default_tonic=style.tonic,
            default_frames=style.frames_per_row,
            target_seconds=target_seconds,
        )
        # Key and tempo land on the style before anything is composed, so the
        # whole piece -- melody, harmony, bass -- is written in them rather
        # than transposed afterwards.
        style = dataclasses.replace(
            style,
            tonic=arrangement.tonic,
            tempo_bpm=bpm_for_frames(arrangement.frames_per_row),
        )
        seconds_per_bar = 4 * 60.0 / style.tempo_bpm
        section_bars = V.section_bars_for(
            arrangement.form,
            target_seconds=target_seconds,
            seconds_per_bar=seconds_per_bar,
        )
        # The vocabulary expansion is part of the variety switch: the extra
        # progressions exist and were held back on the operator's instruction
        # not to promote them on variety grounds alone. That instruction was
        # lifted for this work.
        if vocabulary is None:
            pool = tuple(style.progressions) + tuple(extra)

    motif = make_motif(rng, style)
    if arrangement is not None:
        from rse.procedural import variety as V

        plans = plan_archetype(
            rng,
            style,
            form=arrangement.form,
            section_bars=section_bars,
            progressions=pool,
        )
    else:
        plans = plan_form(
            rng,
            style,
            target_seconds=target_seconds,
            section_bars=section_bars,
            progressions=pool,
        )

    events: list[Note | Noise] = []
    variations: list[dict[str, Any]] = []
    truncation = {"notes_truncated": 0, "rows_removed": 0}
    bar_offset = 0
    contrast_motifs: dict[str, Motif] = {}
    reorch_index = 0
    for plan in plans:
        reorch = None
        if plan.treatment == "vary":
            if arrangement is not None:
                # Re-orchestration, not rewriting. A' in the repertoire was the
                # same sequence data with the instruments reassigned, the track
                # transposed, or a counter-melody layer toggled -- new melodic
                # material cost memory that competed with the samples. Keeping
                # the motif is also what lets the theme get established.
                from rse.procedural.variety import REORCHESTRATIONS, Reorchestration

                reorch = Reorchestration(
                    **REORCHESTRATIONS[reorch_index % len(REORCHESTRATIONS)]
                )
                reorch_index += 1
                used = motif
            else:
                used = vary_motif(rng, motif)
        elif plan.treatment.startswith("contrast"):
            # contrast, contrast2, contrast3 are DISTINCT ideas, which is what
            # makes a linear A-B-C-D form linear rather than A-B-A-B.
            if plan.treatment not in contrast_motifs:
                contrast_motifs[plan.treatment] = make_motif(rng, style)
            used = contrast_motifs[plan.treatment]
        else:
            used = motif
        harmony_plan = rhythm.plan(len(plan.progression), plan.bars)
        # The phrase plan is deterministic from the bar count, so it costs no
        # RNG draw and cannot shift the percussion generated after it.
        phrases = plan_phrases(plan.bars) if phrasing == "period" else None
        # Seeded from the main seed and the bar offset, so it is deterministic
        # and distinct per section, and consumes nothing from `rng`.
        melody_rng = (
            random.Random(seed ^ 0xA3E1 ^ (bar_offset << 8))
            if isolate_melody_rng
            else rng
        )
        variations.append(
            {
                "label": plan.label,
                "motif": used.to_dict(),
                "n_chord_changes": len(harmony_plan),
                "phrases": [r.to_dict() for r in phrases] if phrases else None,
                "reorchestration": reorch.to_dict() if reorch else None,
            }
        )
        melody, melody_stats = render_melody(
            melody_rng,
            style,
            used,
            plan,
            bar_offset=bar_offset,
            harmony_plan=harmony_plan,
            phrases=phrases,
            shape_velocity=lead_shaper,
        )
        events += melody
        truncation["notes_truncated"] += melody_stats["notes_truncated"]
        truncation["rows_removed"] += melody_stats["rows_removed"]
        section_style = style
        if reorch is not None and reorch.transpose:
            section_style = dataclasses.replace(
                style, tonic=style.tonic + reorch.transpose
            )
        if reorch is not None and reorch.counter_melody:
            # P2 doubles the line instead of comping: the layer change.
            events += render_counter(melody, section_style)
        elif reorch is not None and reorch.thin:
            pass  # accompaniment dropped entirely for this section
        else:
            figure = arrangement.harmony_style if arrangement else None
            if reorch is not None and reorch.harmony_style:
                figure = reorch.harmony_style
            events += render_harmony(
                section_style,
                plan,
                bar_offset=bar_offset,
                harmony_plan=harmony_plan,
                figure=figure,
                shape_velocity=comp_shaper,
            )
        bass_rows = arrangement.bass_rows if arrangement else None
        if reorch is not None and reorch.bass_pattern:
            from rse.procedural.variety import BASS_PATTERNS

            bass_rows = BASS_PATTERNS[reorch.bass_pattern]
        events += render_bass(
            section_style,
            plan,
            bar_offset=bar_offset,
            harmony_plan=harmony_plan,
            pedal=rhythm.pedal,
            rows=bass_rows,
        )
        events += render_drums(rng, style, plan, bar_offset=bar_offset)
        bar_offset += plan.bars

    composition = Composition(
        hardware_profile="nes", ticks_per_beat=TICKS_PER_BEAT, events=events
    )
    # Microseconds per beat from the FRAME COUNT, not from the BPM. A beat is
    # `rows_per_beat * frames_per_row` video frames, and that is the quantity
    # the console actually counts.
    composition.tempo_map = [
        TempoChange(
            0,
            int(round(ROWS_PER_BEAT * style.frames_per_row / NTSC_FRAME_RATE * 1e6)),
        )
    ]
    composition.time_signatures = [TimeSignatureChange(0, 4, 4)]
    composition.loop_start_tick = 0
    composition.loop_end_tick = bar_offset * TICKS_PER_BAR
    composition.normalize()
    return ProceduralResult(
        composition=composition,
        style=style,
        sound=voices,
        motif=motif,
        sections=plans,
        seed=seed,
        rhythm=rhythm,
        truncation=truncation,
        vocabulary=vocabulary,
        variations=variations,
    )
