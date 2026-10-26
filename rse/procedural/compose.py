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

#: NES pitch range the hardware profile accepts.
PITCH_MIN, PITCH_MAX = 33, 108

#: Noise periods used as drum voices. Low period reads as a snare-ish crack,
#: high as a hat; the renderer decides the timbre.
KICK, SNARE, HAT = 1, 6, 12


@dataclass
class Style:
    """A preset: what the rules are allowed to choose from."""

    name: str
    mode: str
    tempo_bpm: int
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "tempo_bpm": self.tempo_bpm,
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


def plan_form(
    rng: random.Random, style: Style, *, target_seconds: float, section_bars: int = 8
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
    main = rng.choice(style.progressions)
    others = [p for p in style.progressions if p != main] or [main]
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


def render_melody(
    rng: random.Random,
    style: Style,
    motif: Motif,
    plan: SectionPlan,
    *,
    bar_offset: int,
) -> list[Note]:
    """Melody on P1, re-harmonised bar by bar against the progression.

    Each phrase starts on a tone of the bar's chord and follows the motif's
    degree contour. A step that lands outside the chord is kept only if it is
    in the scale, so a passing tone is a passing tone and never an accident.
    """
    notes: list[Note] = []
    for bar in range(plan.bars):
        chord_root = plan.progression[bar % len(plan.progression)]
        tones = chord_degrees(chord_root)
        start = tones[rng.randrange(len(tones))]
        row = 0
        for step, length in zip(motif.degrees, motif.rhythm, strict=False):
            if row >= ROWS_PER_BAR:
                break
            degree = start + step
            # On strong rows prefer a chord tone; elsewhere allow the scale.
            if row % ROWS_PER_BEAT == 0 and rng.random() > style.passing_chance:
                degree = min(tones, key=lambda t: abs(t - degree))
            pitch = clamp(scale_pitch(style.tonic, style.mode, degree))
            notes.append(
                Note(
                    channel_id="P1",
                    tick=(bar_offset + bar) * TICKS_PER_BAR + row * TICKS_PER_ROW,
                    pitch=pitch,
                    duration=length * TICKS_PER_ROW,
                    velocity=100,
                )
            )
            row += length
    return notes


def render_harmony(
    style: Style, plan: SectionPlan, *, bar_offset: int
) -> list[Note]:
    """Accompaniment on P2, built only from the bar's chord tones.

    One note at a time, because P2 is monophonic. An arpeggio is therefore
    the honest way to state a chord on this hardware, not a workaround.

    Fully determined by the progression and the style, with no randomness:
    the accompaniment's job is to make the harmony audible, so varying it
    per seed would weaken the one voice whose predictability is the point.
    """
    notes: list[Note] = []
    for bar in range(plan.bars):
        chord_root = plan.progression[bar % len(plan.progression)]
        tones = chord_degrees(chord_root)
        if style.harmony_style == "pad":
            pitch = clamp(scale_pitch(style.tonic, style.mode, tones[0]) - 12)
            notes.append(
                Note(
                    channel_id="P2",
                    tick=(bar_offset + bar) * TICKS_PER_BAR,
                    pitch=pitch,
                    duration=TICKS_PER_BAR,
                    velocity=70,
                )
            )
            continue
        rows = range(0, ROWS_PER_BAR, 2) if style.harmony_style == "arpeggio" else range(
            2, ROWS_PER_BAR, 4
        )
        for index, row in enumerate(rows):
            degree = tones[index % len(tones)]
            if style.harmony_style == "arpeggio" and index % len(tones) == 0:
                degree += 7 * (index // len(tones) % 2)
            pitch = clamp(scale_pitch(style.tonic, style.mode, degree) - 12)
            notes.append(
                Note(
                    channel_id="P2",
                    tick=(bar_offset + bar) * TICKS_PER_BAR + row * TICKS_PER_ROW,
                    pitch=pitch,
                    duration=2 * TICKS_PER_ROW,
                    velocity=72,
                )
            )
    return notes


def render_bass(style: Style, plan: SectionPlan, *, bar_offset: int) -> list[Note]:
    """Bass on the triangle: chord roots and fifths, no passing tones.

    Deterministic given the progression. The bass is what makes the harmony
    audible, so it is the one voice with no randomness in it at all.
    """
    notes: list[Note] = []
    for bar in range(plan.bars):
        chord_root = plan.progression[bar % len(plan.progression)]
        tones = chord_degrees(chord_root)
        for index, row in enumerate(style.bass_rows):
            degree = tones[0] if index % 2 == 0 else tones[2]
            pitch = clamp(scale_pitch(style.tonic, style.mode, degree) - 24)
            length = (
                style.bass_rows[index + 1] - row
                if index + 1 < len(style.bass_rows)
                else ROWS_PER_BAR - row
            )
            notes.append(
                Note(
                    channel_id="TR",
                    tick=(bar_offset + bar) * TICKS_PER_BAR + row * TICKS_PER_ROW,
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
    variations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "style": self.style.to_dict(),
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
) -> ProceduralResult:
    """One complete NES piece from a style preset and a seed."""
    if style_name not in STYLES:
        raise KeyError(f"unknown style {style_name!r}; have {sorted(STYLES)}")
    style = STYLES[style_name]
    rng = random.Random(seed)
    motif = make_motif(rng, style)
    plans = plan_form(rng, style, target_seconds=target_seconds, section_bars=section_bars)

    events: list[Note | Noise] = []
    variations: list[dict[str, Any]] = []
    bar_offset = 0
    contrast_motif: Motif | None = None
    for plan in plans:
        if plan.treatment == "vary":
            used = vary_motif(rng, motif)
        elif plan.treatment == "contrast":
            contrast_motif = contrast_motif or make_motif(rng, style)
            used = contrast_motif
        else:
            used = motif
        variations.append({"label": plan.label, "motif": used.to_dict()})
        events += render_melody(rng, style, used, plan, bar_offset=bar_offset)
        events += render_harmony(style, plan, bar_offset=bar_offset)
        events += render_bass(style, plan, bar_offset=bar_offset)
        events += render_drums(rng, style, plan, bar_offset=bar_offset)
        bar_offset += plan.bars

    composition = Composition(
        hardware_profile="nes", ticks_per_beat=TICKS_PER_BEAT, events=events
    )
    composition.tempo_map = [
        TempoChange(0, int(round(60_000_000 / style.tempo_bpm)))
    ]
    composition.time_signatures = [TimeSignatureChange(0, 4, 4)]
    composition.loop_start_tick = 0
    composition.loop_end_tick = bar_offset * TICKS_PER_BAR
    composition.normalize()
    return ProceduralResult(
        composition=composition,
        style=style,
        motif=motif,
        sections=plans,
        seed=seed,
        variations=variations,
    )
