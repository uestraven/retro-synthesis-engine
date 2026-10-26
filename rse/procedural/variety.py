"""What differs between two songs of the same style.

Measured across 20 seeds of each style, before this module existed:

    channel / property        unique across 20 seeds
    melody   (P1)                                 20
    drums    (NO)                                 20
    harmony  (P2)                               5-6   (victory: 2)
    bass     (TR)                               5-6   (victory: 2)
    section form                                   1
    key                                            1
    tempo                                          1

So a "boss song" was always A harmonic minor at 150.25 BPM, offbeat harmony,
bass on all four beats, in the form A-A'1-A'2-B-A''. Only the melody contour
and the drums moved. `render_harmony` and `render_bass` take no generator at
all, so the entire backing was a pure function of which of three progressions
got drawn.

That determinism was deliberate and its reasoning was sound -- the
accompaniment's job is to make the harmony audible, and a bass line that
wanders is a bass line that fails at it. So the fix is not to randomise those
voices; it is to choose a pattern **per song** and then hold it. Predictable
within a piece, different between pieces, which is how a real soundtrack works:
every track in a game has its own bass figure and keeps it.

Everything here is drawn from a generator seeded separately from the one the
melody and drums use, so adding variety cannot shift the voices that were
already varying.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

#: Accompaniment figures for P2. Each states the bar's chord a different way;
#: all are monophonic, because P2 is.
#:
#: `arpeggio`      -- run up the chord, alternating octave each pass
#: `arpeggio_down` -- run down it
#: `offbeat`       -- stabs between the beats, the classic chiptune comp
#: `pad`           -- hold the root for the whole chord
#: `alternating`   -- root and fifth traded on eighths
#: `stab`          -- one short hit per half bar, leaving space
HARMONY_STYLES = (
    "arpeggio",
    "arpeggio_down",
    "offbeat",
    "pad",
    "alternating",
    "stab",
)

#: Bass figures as the rows of a bar the triangle attacks.
BASS_PATTERNS: dict[str, tuple[int, ...]] = {
    "whole": (0,),
    "halves": (0, 8),
    "four": (0, 4, 8, 12),
    "eighths": (0, 2, 4, 6, 8, 10, 12, 14),
    "syncopated": (0, 3, 6, 8, 11, 14),
    "gallop": (0, 2, 4, 8, 10, 12),
    "backbeat": (0, 6, 8, 14),
    "dotted": (0, 6, 8, 12),
}

#: Which bass figures suit which style. A town theme does not get the gallop.
STYLE_BASS: dict[str, tuple[str, ...]] = {
    "boss": ("four", "eighths", "syncopated", "gallop", "dotted"),
    "action": ("eighths", "syncopated", "gallop", "four", "backbeat"),
    "adventure": ("halves", "four", "dotted", "backbeat"),
    "town": ("halves", "four", "whole", "dotted"),
    "tension": ("whole", "halves", "dotted"),
    "victory": ("four", "eighths", "halves", "gallop"),
}

#: Which accompaniment figures suit which style.
STYLE_HARMONY: dict[str, tuple[str, ...]] = {
    "boss": ("offbeat", "arpeggio", "stab", "alternating"),
    "action": ("arpeggio", "offbeat", "alternating", "arpeggio_down"),
    "adventure": ("arpeggio", "arpeggio_down", "offbeat", "pad"),
    "town": ("pad", "arpeggio", "alternating", "arpeggio_down"),
    "tension": ("pad", "stab", "arpeggio_down"),
    "victory": ("arpeggio", "alternating", "offbeat", "arpeggio_down"),
}

#: Tonics offered per style, as MIDI pitches. The mode is unchanged, so these
#: are transpositions rather than re-colourings: a boss theme in D harmonic
#: minor is still a boss theme. Kept inside one octave so the melody's own
#: octave handling does not have to move.
STYLE_TONICS: dict[str, tuple[int, ...]] = {
    "boss": (57, 55, 59, 62, 53),
    "action": (60, 57, 62, 55, 64),
    "adventure": (60, 62, 65, 57, 55),
    "town": (62, 60, 65, 67, 57),
    "tension": (55, 57, 53, 60, 50),
    "victory": (64, 60, 62, 67, 57),
}

#: Tempo options as frames per tracker row -- the hardware's own unit, so
#: every option is a tempo an NES can actually play. Adjacent rungs only: the
#: ladder is coarse, and two rungs apart is a different style, not a variation.
#:
#: 5 frames = 180.30 BPM, 6 = 150.25, 7 = 128.78, 8 = 112.69, 9 = 100.16,
#: 10 = 90.15.
STYLE_FRAMES: dict[str, tuple[int, ...]] = {
    "boss": (6, 5),
    "action": (5, 6),
    "adventure": (7, 6),
    "town": (7, 8),
    "tension": (9, 10),
    "victory": (6, 5),
}


@dataclass(frozen=True)
class Reorchestration:
    """How a section varies the theme WITHOUT new melodic material.

    This is what A'1 and A'2 actually were. Writing a genuinely new section
    cost sequence data, and sequence data competed with the samples and the
    engine inside 64KB, so composers varied a section with tracker commands
    instead: reassign the instruments, shift the track transpose, add or drop
    a counter-melody layer. The player hears an evolving composition; the
    cartridge holds one copy of the notes.

    RSE had been doing the expensive thing and rewriting the motif's contour
    and rhythm, which is a different musical gesture and a worse one -- it
    makes A'1 a new idea rather than the same idea in new clothes, so the
    theme never gets established.
    """

    #: Semitones to transpose the whole section. The tracker transpose.
    transpose: int = 0
    #: Accompaniment figure for this section, overriding the song's.
    harmony_style: str | None = None
    #: Bass figure for this section.
    bass_pattern: str | None = None
    #: Whether P2 doubles the melody a third/sixth below instead of comping.
    counter_melody: bool = False
    #: Drop the accompaniment entirely, leaving melody, bass and drums.
    thin: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "transpose": self.transpose,
            "harmony_style": self.harmony_style,
            "bass_pattern": self.bass_pattern,
            "counter_melody": self.counter_melody,
            "thin": self.thin,
        }


#: The re-orchestration moves a `vary` section may draw, cheapest first. Each
#: is a tracker-command-level change, not new melodic data.
REORCHESTRATIONS: tuple[dict[str, Any], ...] = (
    {"counter_melody": True},
    {"thin": True},
    {"transpose": 12},
    {"transpose": -12},
    {"harmony_style": "arpeggio"},
    {"harmony_style": "offbeat"},
    {"harmony_style": "pad"},
    {"bass_pattern": "eighths"},
    {"bass_pattern": "halves"},
    {"counter_melody": True, "transpose": 12},
)


@dataclass(frozen=True)
class Arrangement:
    """The choices that make one song of a style differ from the next."""

    harmony_style: str
    bass_pattern: str
    bass_rows: tuple[int, ...]
    tonic: int
    frames_per_row: int
    form: str
    #: Extra drum fill density, 0-2. Higher adds fills at phrase ends.
    fill: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "harmony_style": self.harmony_style,
            "bass_pattern": self.bass_pattern,
            "tonic": self.tonic,
            "frames_per_row": self.frames_per_row,
            "form": self.form,
            "fill": self.fill,
        }


#: Section-form archetypes, as treatment sequences.
#:
#: These are the shapes 8- and 16-bit console music actually used, not a set
#: invented to look varied. An earlier draft of this module had `abab` and
#: `arch`, which are not archetypes of the repertoire, and was missing the
#: linear form entirely -- the one boss battles are built on.
#:
#: The 64KB audio RAM of an SPC700 (and the far tighter NES budget) is what
#: shaped all of them: sequence data competed with samples and engine code, so
#: composers favoured structures that got maximum mileage from a short file.
#:
#: `ternary`     A-B-A. Town, overworld and character themes: a melody, a
#:               contrasting section to shift the mood, a clean loop home.
#: `ternary_var` A-A'-B-A. The same with the opening restated once.
#: `pop32`       A-A-B-A, the 32-bar pop form. Establishes a hook, repeats it,
#:               bridges away, delivers a conclusion before looping.
#: `linear`      A-B-C-D. Boss battles: constantly shifting statements that
#:               keep urgency up by never immediately repeating. The LOOP is
#:               the return, so this form has no closing restatement -- which
#:               is why `returns` exists below.
#: `rounded`     A-B-A'. Rounded binary. The return is a VARIATION of the
#:               opening rather than a literal restatement, which is the
#:               distinction from ternary and the shape Uematsu favoured for
#:               character and town themes alongside it -- establish a motif,
#:               take it somewhere darker or wider in B, resolve to a re-voiced
#:               A that loops cleanly. "Terra's Theme" works this way: the
#:               march-like motif, a sweeping bridge, and a transition back.
#: `variation`   A-A'1-A'2-B-A'. Theme-and-variation engineering, and the
#:               cheapest of all in memory: A' was not new sequence data but
#:               the same data re-orchestrated.
FORM_TEMPLATES: dict[str, tuple[str, ...]] = {
    "ternary": ("state", "contrast", "state"),
    "rounded": ("state", "contrast", "vary"),
    "ternary_var": ("state", "vary", "contrast", "state"),
    "pop32": ("state", "state", "contrast", "state"),
    "linear": ("state", "contrast", "contrast2", "contrast3"),
    "variation": ("state", "vary", "vary", "contrast", "state"),
}

#: Whether a form closes by restating the opening. `linear` does not: it
#: relies on the loop point, so forcing a trailing `state` would turn the one
#: non-repeating archetype into another repeating one.
FORM_RETURNS: dict[str, bool] = {
    "ternary": True,
    # A' is still a return: the same material, re-voiced.
    "rounded": True,
    "ternary_var": True,
    "pop32": True,
    "linear": False,
    "variation": True,
}

#: Which archetypes suit which style, following where the repertoire used
#: them: linear for the driving fight music, ternary and pop32 for the
#: settled themes.
STYLE_FORMS: dict[str, tuple[str, ...]] = {
    "boss": ("linear", "variation", "ternary_var", "pop32"),
    "action": ("linear", "variation", "pop32", "ternary_var"),
    "adventure": ("pop32", "ternary", "rounded", "ternary_var", "variation"),
    "town": ("ternary", "rounded", "pop32", "ternary_var"),
    "tension": ("ternary", "rounded", "linear", "ternary_var"),
    "victory": ("pop32", "ternary", "rounded", "variation"),
}


#: New statements a linear form may reach for, in order, as it grows.
LINEAR_STATEMENTS = ("contrast", "contrast2", "contrast3")


def fit_template(name: str, count: int) -> list[str]:
    """Stretch or trim an archetype to `count` sections.

    Sections are sized to the requested duration elsewhere, so a form has to
    adapt rather than dictate the length.

    A returning form keeps its opening and its closing restatement and grows
    or shrinks in the middle. `linear` is handled separately: it has no
    closing restatement to protect, and growing it means reaching for another
    NEW statement rather than repeating one -- which is the whole point of the
    archetype. Once it runs out of distinct statements it varies the last one
    instead of looping back, so urgency is maintained without the form
    quietly turning into A-B-A-B.
    """
    template = list(FORM_TEMPLATES[name])
    count = max(1, count)
    if count == 1:
        return ["state"]

    if not FORM_RETURNS.get(name, True):
        shape = ["state"]
        for index in range(count - 1):
            if index < len(LINEAR_STATEMENTS):
                shape.append(LINEAR_STATEMENTS[index])
            else:
                shape.append("vary")
        return shape

    if len(template) < count:
        # Grow the varying middle, never the contrast or the return, so the
        # song gets longer without acquiring a second ending.
        middle = [t for t in template[1:-1] if t in ("vary", "state")] or ["vary"]
        grown = [template[0]]
        index = 0
        while len(grown) < count - 2:
            grown.append(middle[index % len(middle)])
            index += 1
        grown.extend(template[-2:])
        return grown[:count]

    if len(template) > count:
        # Shrink, protecting what makes the archetype itself. Drop a `vary`
        # first, then a DUPLICATE `state` -- never a contrast. Dropping from
        # the end of the middle turned pop32 at three sections into
        # A-A-A: a form with no contrasting section at all, which is not a
        # shorter pop32, it is nothing.
        head, tail = template[0], template[-1]
        middle = list(template[1:-1])
        while len(middle) > count - 2:
            vary = [i for i, t in enumerate(middle) if t == "vary"]
            states = [i for i, t in enumerate(middle) if t == "state"]
            if vary:
                middle.pop(vary[-1])
            elif states:
                middle.pop(states[-1])
            else:
                # Only contrasts left; keep the first and stop.
                middle = middle[: max(0, count - 2)]
                break
        return [head, *middle, tail]

    return template


def natural_length(name: str) -> int:
    """How many sections the archetype wants, before any fitting."""
    return len(FORM_TEMPLATES[name])


#: Section lengths in bars, all powers-of-two multiples of a four-bar phrase.
SECTION_LENGTHS = (4, 8, 12, 16)


def section_bars_for(
    form: str,
    *,
    target_seconds: float,
    seconds_per_bar: float,
    lengths: tuple[int, ...] = SECTION_LENGTHS,
) -> int:
    """Section length that puts the archetype closest to the requested time.

    The form is the musical decision and the section length is the knob that
    fits the clock -- not the other way round. Deriving the section COUNT from
    the duration, as the first version did, meant stretching a three-section
    ternary to five and it stopped being a ternary: `ternary`, `ternary_var`
    and `variation` all collapsed to the same shape at every length above
    four. Letting the form keep its own length and moving the bars instead
    preserves all five, and varies section length as a side effect.
    """
    sections = max(1, natural_length(form))
    return min(
        lengths,
        key=lambda bars: abs(target_seconds - sections * bars * seconds_per_bar),
    )


def forms_that_fit(
    style_name: str,
    *,
    target_seconds: float,
    seconds_per_bar: float,
    tolerance: float = 10.0,
) -> tuple[str, ...]:
    """The style's archetypes that can land near the requested duration.

    An archetype keeps its own section count, so not every one can hit every
    length: a four-section linear form at `tension`'s slow tempo is 77 seconds
    at its best fit, which is nowhere near a 60-second request. Filtering here
    rather than stretching the form keeps the shape intact, and means a
    30-second request and a 120-second one genuinely draw from different sets.
    """
    options = STYLE_FORMS.get(style_name, ("variation",))
    fitting = []
    for form in options:
        bars = section_bars_for(
            form, target_seconds=target_seconds, seconds_per_bar=seconds_per_bar
        )
        length = natural_length(form) * bars * seconds_per_bar
        if abs(length - target_seconds) <= tolerance:
            fitting.append(form)
    if fitting:
        return tuple(fitting)
    # Nothing fits; return the single closest so a song is still produced.
    return (
        min(
            options,
            key=lambda form: abs(
                natural_length(form)
                * section_bars_for(
                    form,
                    target_seconds=target_seconds,
                    seconds_per_bar=seconds_per_bar,
                )
                * seconds_per_bar
                - target_seconds
            ),
        ),
    )


def choose(
    style_name: str,
    *,
    seed: int,
    default_harmony: str,
    default_bass: tuple[int, ...],
    default_tonic: int,
    default_frames: int,
    target_seconds: float = 60.0,
    seconds_per_bar: float | None = None,
) -> Arrangement:
    """Draw one song's arrangement.

    From a generator seeded off the song seed alone, so the melody and drum
    streams are untouched: adding variety must not change the voices that were
    already varying, or every A/B would be measuring two things at once.
    """
    rng = random.Random(seed ^ 0x5EED_1A77)
    harmony = STYLE_HARMONY.get(style_name, (default_harmony,))
    bass = STYLE_BASS.get(style_name)
    tonics = STYLE_TONICS.get(style_name, (default_tonic,))
    frames = STYLE_FRAMES.get(style_name, (default_frames,))

    bass_name = rng.choice(bass) if bass else "custom"
    bass_rows = BASS_PATTERNS[bass_name] if bass else default_bass
    # Tempo is drawn before the form, because the form's fit depends on it.
    chosen_frames = rng.choice(frames)
    if seconds_per_bar is None:
        from rse.hardware.tempo import bpm_for_frames

        seconds_per_bar = 4 * 60.0 / bpm_for_frames(chosen_frames)
    forms = forms_that_fit(
        style_name,
        target_seconds=target_seconds,
        seconds_per_bar=seconds_per_bar,
    )
    return Arrangement(
        harmony_style=rng.choice(harmony),
        bass_pattern=bass_name,
        bass_rows=bass_rows,
        tonic=rng.choice(tonics),
        frames_per_row=chosen_frames,
        form=rng.choice(forms),
        fill=rng.randrange(3),
    )


@dataclass
class Space:
    """How many distinct songs a style can produce. For reporting only."""

    style: str
    harmony: int
    bass: int
    tonics: int
    tempos: int
    forms: int
    progressions: int
    motif_shapes: int = 5
    motif_rhythms: int = 3

    @property
    def total(self) -> int:
        return (
            self.harmony
            * self.bass
            * self.tonics
            * self.tempos
            * self.forms
            * self.progressions
            * self.motif_shapes
            * self.motif_rhythms
        )

    def to_dict(self) -> dict[str, int | str]:
        return {
            "style": self.style,
            "harmony": self.harmony,
            "bass": self.bass,
            "tonics": self.tonics,
            "tempos": self.tempos,
            "forms": self.forms,
            "progressions": self.progressions,
            "total": self.total,
        }


def space(style_name: str, *, progressions: int) -> Space:
    return Space(
        style=style_name,
        harmony=len(STYLE_HARMONY.get(style_name, ("",))),
        bass=len(STYLE_BASS.get(style_name, ("",))),
        tonics=len(STYLE_TONICS.get(style_name, (0,))),
        tempos=len(STYLE_FRAMES.get(style_name, (0,))),
        forms=len(STYLE_FORMS.get(style_name, ("",))),
        progressions=progressions,
    )

