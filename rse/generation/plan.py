"""The composition plan.

Simple next-token prediction does not produce well-formed structure (proposal
6.5), so structure is decided *before* any note exists: form, key, harmony and
tempo first, notes second. Phase 4's model generates inside a plan like this
rather than replacing it.

A plan is the complete reproducibility record for a composition: same plan,
same output, forever.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from rse.generation.theory import MOOD_SCALES, PROGRESSIONS, SCALES, Key

#: Phrase forms. Repeated letters reuse material, which is what makes a tune
#: sound composed instead of merely continuous.
FORMS: tuple[str, ...] = ("AABA", "ABAC", "AABB", "ABAB")

#: Supported moods. Each selects a scale and a pool of progressions.
MOODS: tuple[str, ...] = tuple(PROGRESSIONS)

#: Energy levels, which drive note density, accompaniment style and drums.
ENERGIES: tuple[str, ...] = ("low", "medium", "high")

#: Bar counts the generator supports.
LENGTHS: tuple[int, ...] = (4, 8, 16, 32)


@dataclass(frozen=True)
class Section:
    """One phrase of the form, with the chord under each of its bars."""

    label: str
    start_bar: int
    n_bars: int
    chord_degrees: tuple[int, ...]

    def chord_at(self, bar_offset: int) -> int:
        return self.chord_degrees[bar_offset % len(self.chord_degrees)]


@dataclass(frozen=True)
class CompositionPlan:
    """Everything decided before note generation starts."""

    seed: int
    key: Key
    mood: str
    energy: str
    tempo_bpm: float
    n_bars: int
    form: str
    sections: tuple[Section, ...]
    beats_per_bar: int = 4
    ticks_per_beat: int = 480
    loop: bool = True

    @property
    def ticks_per_bar(self) -> int:
        return self.ticks_per_beat * self.beats_per_bar

    @property
    def total_ticks(self) -> int:
        return self.ticks_per_bar * self.n_bars

    def section_at(self, bar: int) -> Section:
        for section in self.sections:
            if section.start_bar <= bar < section.start_bar + section.n_bars:
                return section
        return self.sections[-1]

    def chord_at_bar(self, bar: int) -> int:
        section = self.section_at(bar)
        return section.chord_at(bar - section.start_bar)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "key": {"tonic": self.key.tonic, "scale": self.key.scale},
            "mood": self.mood,
            "energy": self.energy,
            "tempo_bpm": self.tempo_bpm,
            "n_bars": self.n_bars,
            "form": self.form,
            "beats_per_bar": self.beats_per_bar,
            "ticks_per_beat": self.ticks_per_beat,
            "loop": self.loop,
            "sections": [
                {
                    "label": s.label,
                    "start_bar": s.start_bar,
                    "n_bars": s.n_bars,
                    "chord_degrees": list(s.chord_degrees),
                }
                for s in self.sections
            ],
        }


def make_plan(
    *,
    seed: int | None = None,
    mood: str = "heroic",
    energy: str = "medium",
    tempo_bpm: float | None = None,
    n_bars: int = 16,
    form: str | None = None,
    tonic: int | None = None,
    scale: str | None = None,
    loop: bool = True,
) -> CompositionPlan:
    """Build a plan, filling anything unspecified from the seeded RNG."""
    if mood not in MOODS:
        raise ValueError(f"unknown mood {mood!r}; known: {list(MOODS)}")
    if energy not in ENERGIES:
        raise ValueError(f"unknown energy {energy!r}; known: {list(ENERGIES)}")
    if n_bars not in LENGTHS:
        raise ValueError(f"n_bars must be one of {list(LENGTHS)}, got {n_bars}")
    if scale is not None and scale not in SCALES:
        raise ValueError(f"unknown scale {scale!r}; known: {sorted(SCALES)}")

    if seed is None:
        seed = random.Random().randrange(2**31)
    rng = random.Random(seed)

    key = Key(
        tonic=rng.randrange(12) if tonic is None else tonic % 12,
        scale=scale or MOOD_SCALES[mood],
    )
    if tempo_bpm is None:
        tempo_bpm = float(rng.choice(_TEMPO_RANGE[energy]))
    if form is None:
        form = rng.choice(FORMS)

    sections = _build_sections(form, n_bars, mood, rng)
    return CompositionPlan(
        seed=seed,
        key=key,
        mood=mood,
        energy=energy,
        tempo_bpm=tempo_bpm,
        n_bars=n_bars,
        form=form,
        sections=sections,
        loop=loop,
    )


#: Tempo pools per energy level, in BPM. Chosen to sit in the range game music
#: actually occupies rather than the full 60-220 the proposal allows.
_TEMPO_RANGE: dict[str, tuple[int, ...]] = {
    "low": (72, 80, 88, 96),
    "medium": (104, 112, 120, 128),
    "high": (140, 150, 160, 170, 180),
}


def _build_sections(form: str, n_bars: int, mood: str, rng: random.Random) -> tuple[Section, ...]:
    """Split the bars across the form's phrases and give each a progression.

    Phrases sharing a letter share a progression, which is what makes an
    AABA form hear as a return rather than as four unrelated ideas.
    """
    letters = list(form)
    n_phrases = len(letters)
    base = n_bars // n_phrases
    remainder = n_bars % n_phrases
    pool = PROGRESSIONS[mood]

    progressions: dict[str, tuple[int, ...]] = {}
    for letter in letters:
        if letter not in progressions:
            progressions[letter] = rng.choice(pool)

    sections: list[Section] = []
    bar = 0
    for index, letter in enumerate(letters):
        # Spread any leftover bars over the earliest phrases so section lengths
        # differ by at most one bar.
        length = base + (1 if index < remainder else 0)
        if length == 0:
            continue
        sections.append(
            Section(
                label=letter,
                start_bar=bar,
                n_bars=length,
                chord_degrees=progressions[letter],
            )
        )
        bar += length
    return tuple(sections)


@dataclass
class PlanSummary:
    """Human-readable description of a plan, for CLI output."""

    plan: CompositionPlan
    note_names: tuple[str, ...] = field(
        default=("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"),
        repr=False,
    )

    @property
    def key_name(self) -> str:
        return f"{self.note_names[self.plan.key.tonic]} {self.plan.key.scale}"

    def lines(self) -> list[str]:
        plan = self.plan
        out = [
            f"key {self.key_name}, {plan.mood}, {plan.energy} energy, "
            f"{plan.tempo_bpm:g} BPM, {plan.n_bars} bars, form {plan.form}",
        ]
        for section in plan.sections:
            degrees = " ".join(f"{d + 1}" for d in section.chord_degrees)
            out.append(
                f"  {section.label}: bars {section.start_bar + 1}-"
                f"{section.start_bar + section.n_bars}, chords {degrees}"
            )
        return out
