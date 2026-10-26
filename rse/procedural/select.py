"""Choose a procedural style from an interpreted prompt.

The missing link. The interpreter already reads a request into style,
scenario, mood, energy, tempo, key and duration; the procedural composer
already turns a style preset into a complete NES piece. Nothing connected
them, so the product shipped the composer that scores 60% while the one the
listener called "fantastic" was reachable only by naming a preset in an
experiment script.

Selection is a lookup, not a heuristic stack. Scenario decides first because
it is the most specific thing a request states ("boss battle" is unambiguous);
mood breaks a tie when no scenario is named; energy is the last resort. Each
step records what it matched, so the choice is inspectable rather than
emergent.

Tempo and key pass through as overrides, which the composer accepts without
needing a new preset. Everything the composer cannot take is reported
unsupported rather than approximated into something it can.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rse.procedural.compose import STYLES

#: Scenario name to procedural style. The interpreter's scenario vocabulary is
#: larger than the composer's six presets, so several scenarios share one.
SCENARIO_STYLE: dict[str, str] = {
    "boss_battle": "boss",
    "battle": "action",
    "dungeon": "tension",
    "overworld": "adventure",
    "exploration": "adventure",
    "town": "town",
    "title": "victory",
    "credits": "town",
    "game_over": "tension",
}

#: Mood to style, used only when no scenario was named.
MOOD_STYLE: dict[str, str] = {
    "dark": "boss",
    "ominous": "boss",
    "sinister": "boss",
    "evil": "boss",
    "menacing": "boss",
    "tense": "tension",
    "haunting": "tension",
    "mysterious": "tension",
    "sad": "tension",
    "melancholy": "tension",
    "hopeful": "adventure",
    "bright": "adventure",
    "happy": "town",
    "cheerful": "town",
    "peaceful": "town",
    "dreamy": "town",
    "triumphant": "victory",
    "heroic": "victory",
}

#: Energy to style, the last resort when nothing else matched.
ENERGY_STYLE: dict[str, str] = {
    "very_high": "action",
    "high": "action",
    "medium_high": "adventure",
    "medium": "town",
    "low": "tension",
    "very_low": "tension",
}

DEFAULT_STYLE = "adventure"

#: Modes that share a tonal colour. A prompt asking for "dark" sets mode
#: `minor`, and the boss preset is `harmonic_minor` -- different strings,
#: the same musical request. Reporting that as unsupported was a false
#: negative that would make the report less trustworthy, not more.
MODE_FAMILY: dict[str, str] = {
    "minor": "minor",
    "harmonic_minor": "minor",
    "dorian": "minor",
    "phrygian": "minor",
    "major": "major",
    "mixolydian": "major",
    "lydian": "major",
}


@dataclass
class Selection:
    """Which preset was chosen, why, and what the request could not reach."""

    style: str = DEFAULT_STYLE
    reason: str = "default"
    matched_on: str = ""
    tempo_bpm: int | None = None
    tonic: int | None = None
    section_bars: int = 8
    #: Prompt requirements the composer has no input for.
    unsupported: list[dict[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "style": self.style,
            "reason": self.reason,
            "matched_on": self.matched_on,
            "tempo_bpm": self.tempo_bpm,
            "tonic": self.tonic,
            "section_bars": self.section_bars,
            "unsupported": list(self.unsupported),
            "notes": list(self.notes),
        }


#: Semitone offset per fifth, for turning the interpreter's key root into a
#: tonic pitch. The interpreter counts in fifths because that is what
#: Fermata's `<KEY_root:mode>` family uses; a fifth is seven semitones.
FIFTH = 7

#: Middle of the NES melodic range, so an override lands somewhere playable.
BASE_TONIC = 60


def tonic_for(key_root: int) -> int:
    """Tonic pitch for a key root measured in fifths from C."""
    return BASE_TONIC + (key_root * FIFTH) % 12


def select(plan: Any) -> Selection:
    """Map an interpreted prompt onto a procedural style and overrides."""
    chosen = Selection()

    scenario_style = SCENARIO_STYLE.get(plan.scenario or "")
    mood_style = MOOD_STYLE.get(plan.mood or "")
    energy_style = ENERGY_STYLE.get(plan.energy or "")

    if scenario_style:
        chosen.style = scenario_style
        chosen.reason = f"scenario {plan.scenario!r}"
        chosen.matched_on = "scenario"
        # A scenario fixes the style, but a mood that disagrees is worth
        # recording: "a cheerful boss theme" gets the boss preset, and the
        # listener should know the mood did not steer it.
        if mood_style and mood_style != scenario_style:
            chosen.notes.append(
                f"mood {plan.mood!r} suggests {mood_style!r} but scenario "
                f"{plan.scenario!r} takes precedence"
            )
    elif mood_style:
        chosen.style = mood_style
        chosen.reason = f"mood {plan.mood!r}"
        chosen.matched_on = "mood"
    elif energy_style:
        chosen.style = energy_style
        chosen.reason = f"energy {plan.energy!r}"
        chosen.matched_on = "energy"
    else:
        chosen.reason = "nothing in the prompt named a scenario, mood or energy"
        chosen.notes.append(f"fell back to the {DEFAULT_STYLE!r} preset")

    preset = STYLES[chosen.style]

    # Tempo: only override when the prompt actually implied one, so a request
    # that says nothing about speed keeps the preset's own tempo.
    if plan.tempo_bpm and plan.tempo_bpm != preset.tempo_bpm:
        if plan.energy or any(
            r.text.endswith("bpm") for r in plan.requirements
        ):
            chosen.tempo_bpm = int(plan.tempo_bpm)
        else:
            chosen.notes.append(
                f"kept the preset tempo {preset.tempo_bpm} BPM; the prompt "
                "named no tempo or energy"
            )

    if plan.key_root:
        chosen.tonic = tonic_for(plan.key_root)
        chosen.notes.append(
            f"key root {plan.key_root} (fifths from C) -> tonic {chosen.tonic}"
        )

    # The mode is the preset's, not the prompt's: a style's mode is part of
    # what makes it that style, and swapping harmonic minor for major would
    # not give a cheerful boss theme, it would give a different style.
    #
    # Only reported when the prompt ACTUALLY asked -- `plan.mood` is what
    # sets the mode, and the interpreter defaults to minor whether or not a
    # mood word appeared. Reporting an unstated default as an unmet request
    # would fill the report with requirements nobody made. And a request is
    # met when the modes share a family: "dark" asking for minor is satisfied
    # by harmonic minor.
    if (
        plan.mood
        and plan.key_mode
        and MODE_FAMILY.get(plan.key_mode) != MODE_FAMILY.get(preset.mode)
    ):
        chosen.unsupported.append(
            {
                "requirement": f"mode {plan.key_mode!r}",
                "reason": f"the {chosen.style!r} preset is {preset.mode!r}, a "
                f"{MODE_FAMILY.get(preset.mode, 'different')} mode, and "
                "its mode is part of its identity; choose a different style "
                "instead of re-moding this one",
            }
        )

    for requirement in plan.requirements:
        if requirement.status == "unsupported":
            chosen.unsupported.append(
                {"requirement": requirement.text, "reason": requirement.detail}
            )

    return chosen
