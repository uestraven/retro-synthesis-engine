"""Instrumentation and expression: which sounds play the score, and how.

The composer decides what is played. Nothing decided which sounds played it,
or how they behaved. Audited across every style and seed, every note of every
song RSE has produced carried:

    instrument_id   None
    effects         ()
    lead velocity   100        (every note, every song, forever)
    harmony velocity 70-72
    bass velocity   90
    duty cycle      the renderer's DEFAULT_DUTY: P1 50%, P2 25%

So forty melodies measuring forty distinct interval contours were all
performed by the same two square patches at one fixed volume with no
articulation. That is why four composition-level experiments in a row returned
no audible preference: the material was being rearranged, and the performance
never changed.

Two hardware facts shape everything here, both verified against this
repository's own renderer rather than taken from a reference.

**There are three pulse timbres, not four.** `DUTY_SEQUENCES[3]` is exactly
`1 - DUTY_SEQUENCES[1]`: the 25% waveform inverted. Measured over ten
harmonics the two are identical to 0.0000, so offering "75%" as a separate
instrument would be a choice that changes nothing. 12.5%, 25% and 50% are
genuinely distinct -- 50% has no even harmonics at all, which is what makes it
read as hollow and clarinet-like, while 12.5% keeps every partial and reads as
bright and nasal.

**Volume is a 4-bit register, so dynamics are quantised to fifteen steps.**
127/15 is 8.47, which means a velocity accent of five does nothing at all:
velocities 98 through 104 all land on register 12. Expression is therefore
authored in REGISTER space and converted to the velocity that lands on each
register exactly. Authoring in velocity and hoping is how you get a dynamic
plan that measures beautifully and is inaudible.

**The triangle has no volume register.** Its velocity is ignored by the
hardware and by this renderer, which is correct. Bass dynamics are not
available on the NES and this module does not pretend otherwise.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

#: Pulse duty indices that are genuinely distinct timbres.
#:
#: Index 3 is omitted deliberately: it is the 25% waveform inverted, with an
#: identical magnitude spectrum, so it is the same instrument. Phase inversion
#: is audible only against another correlated signal, which two independently
#: sequenced NES channels are not.
TIMBRES: dict[str, int] = {
    "bright": 0,  # 12.5% -- every partial present; thin, nasal, cuts through
    "classic": 1,  # 25%  -- the standard chiptune lead
    "hollow": 2,  # 50%  -- no even harmonics; rounded, clarinet-like
}

#: The duty index the renderer uses when nothing says otherwise. Kept so the
#: dry arm can be reproduced exactly.
DRY_DUTY: dict[str, int] = {"P1": 2, "P2": 1}

#: Which named profiles each style may draw.
#:
#: Declared by PROFILE NAME rather than by raw timbre pair, because the first
#: version was written as ad-hoc pairs and three styles ended up able to draw a
#: profile that contradicted them: `boss` could draw `warm` and could never
#: draw `aggressive`, `tension` could draw `aggressive`, and `victory` could
#: draw `mellow`. Nothing in a pair of duty indices says what it will sound
#: like; the name does, so the name is what the pool is written in.
#:
#: Character grouping: `aggressive` and `sharp` are assertive, `full`, `warm`
#: and `mellow` are rounded, `eerie` is thin and unsettled.
STYLE_PROFILES: dict[str, tuple[str, ...]] = {
    "boss": ("aggressive", "full", "sharp"),
    "action": ("aggressive", "sharp", "full"),
    "adventure": ("full", "sharp", "warm"),
    "town": ("warm", "mellow", "full"),
    # Thin and restrained, and never assertive: the reverse of boss.
    "tension": ("eerie", "mellow", "full"),
    "victory": ("sharp", "aggressive", "full"),
}

#: Named instrument profiles, as (lead, accompaniment) timbres.
#:
#: Named so an interface can offer them and a report can state which was used.
#: Every one is a real, reachable pair of duty cycles -- there is no profile
#: here that the engine cannot produce, which is the rule for anything an
#: interpreter is allowed to put in a validated configuration.
SOUND_PROFILES: dict[str, tuple[str, str]] = {
    # A thin, cutting lead over the standard comp. Reads as hard and urgent.
    "aggressive": ("bright", "classic"),
    # The classic pairing: rounded lead, standard comp. The old default.
    "full": ("hollow", "classic"),
    # Thin lead over a rounded bed. Unsettled, hollow, exposed.
    "eerie": ("bright", "hollow"),
    # Rounded lead over a thin comp: warm foreground, glassy background.
    "warm": ("hollow", "bright"),
    # Standard lead over a thin comp. Bright without being harsh.
    "sharp": ("classic", "bright"),
    # Standard lead over a rounded bed.
    "mellow": ("classic", "hollow"),
}

#: Mood words to a profile.
#:
#: This exists because `mood` was an inert control. The interpreter read it
#: into `plan.key_mode`, which `compose` never consults -- it uses the style's
#: own mode -- so "a dark theme" and "a bright theme" produced byte-identical
#: output once a style was chosen. Timbre is the dimension a mood can actually
#: move now that it is proven audible, so mood selects a profile.
MOOD_PROFILES: dict[str, str] = {
    "dark": "full",
    "ominous": "full",
    "sinister": "aggressive",
    "evil": "aggressive",
    "menacing": "aggressive",
    "tense": "eerie",
    "haunting": "eerie",
    "mysterious": "eerie",
    "sad": "mellow",
    "melancholy": "mellow",
    "hopeful": "sharp",
    "bright": "sharp",
    "happy": "sharp",
    "cheerful": "sharp",
    "peaceful": "warm",
    "dreamy": "warm",
    "triumphant": "aggressive",
    "heroic": "sharp",
}

#: 4-bit volume register, and the velocity that lands on each value exactly.
MAX_REGISTER = 15


def velocity_for_register(register: int) -> int:
    """Velocity that maps onto `register` exactly.

    The inverse of the renderer's `round(velocity * 15 / 127)`. Going through
    this rather than picking velocities directly is what makes an accent
    audible: the register is what the hardware hears.
    """
    register = max(0, min(MAX_REGISTER, int(register)))
    return max(1, round(register * 127 / MAX_REGISTER))


#: Lead dynamics, in register values. The span is deliberately narrow: this is
#: a 4-bit register and the lead has to stay above the comp at every point, so
#: three steps of range is the whole budget before the balance breaks.
LEAD_REGISTERS: dict[str, int] = {
    "accent": 14,  # phrase opening, cadential arrival, section start
    "strong": 12,  # on the beat
    "normal": 11,  # off the beat
    "weak": 10,  # between eighths
}

#: Accompaniment dynamics. Kept under the lead at every level, so the existing
#: balance -- lead register 12 against comp register 8 -- is preserved.
COMP_REGISTERS: dict[str, int] = {
    "strong": 9,
    "normal": 8,
    "weak": 7,
}

#: The triangle's velocity is ignored by the hardware. Held at the v1 value so
#: the channel is unchanged rather than given meaningless dynamics.
BASS_VELOCITY = 90


@dataclass(frozen=True)
class SoundDesign:
    """One song's instrumentation."""

    lead_timbre: str
    comp_timbre: str
    style: str
    #: Which named profile this is, when one was asked for or matched.
    profile: str | None = None
    #: Why it was chosen: "profile", "mood", or "style".
    reason: str = "style"

    @property
    def duty_by_channel(self) -> dict[str, int]:
        return {"P1": TIMBRES[self.lead_timbre], "P2": TIMBRES[self.comp_timbre]}

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "reason": self.reason,
            "lead_timbre": self.lead_timbre,
            "comp_timbre": self.comp_timbre,
            "duty_by_channel": self.duty_by_channel,
            "lead_duty_percent": _percent(TIMBRES[self.lead_timbre]),
            "comp_duty_percent": _percent(TIMBRES[self.comp_timbre]),
        }


#: Derived from `STYLE_PROFILES`, so the two cannot disagree.
VOICE_PRESETS: dict[str, tuple[tuple[str, str], ...]] = {
    style: tuple(SOUND_PROFILES[name] for name in names)
    for style, names in STYLE_PROFILES.items()
}


def _percent(duty: int) -> str:
    return {0: "12.5%", 1: "25%", 2: "50%", 3: "25% inverted"}[duty]


def profile_name_for(lead: str, comp: str) -> str | None:
    """The named profile matching a timbre pair, if there is one."""
    for name, pair in SOUND_PROFILES.items():
        if pair == (lead, comp):
            return name
    return None


def choose_voices(
    style_name: str,
    *,
    seed: int,
    profile: str | None = None,
    mood: str | None = None,
) -> SoundDesign:
    """Pick a song's lead and accompaniment timbres.

    Precedence: an explicitly named `profile` wins, then `mood`, then a draw
    from the style's own presets. Each records which applied, so a chosen
    profile and an inferred one stay distinguishable in the report.

    The draw is per song and from a separately seeded stream: the duty must not
    wander from note to note, and choosing it must not perturb any composition
    decision. The seed is mixed with a constant unique to this module so the
    sound design cannot correlate with the arrangement draw.
    """
    if profile is not None:
        if profile not in SOUND_PROFILES:
            raise KeyError(
                f"unknown sound profile {profile!r}; have {sorted(SOUND_PROFILES)}"
            )
        lead, comp = SOUND_PROFILES[profile]
        return SoundDesign(
            lead_timbre=lead, comp_timbre=comp, style=style_name,
            profile=profile, reason="profile",
        )

    if mood and mood in MOOD_PROFILES:
        name = MOOD_PROFILES[mood]
        lead, comp = SOUND_PROFILES[name]
        return SoundDesign(
            lead_timbre=lead, comp_timbre=comp, style=style_name,
            profile=name, reason="mood",
        )

    rng = random.Random(seed ^ 0x50_17D_51)
    presets = VOICE_PRESETS.get(style_name) or (("hollow", "classic"),)
    lead, comp = rng.choice(presets)
    return SoundDesign(
        lead_timbre=lead, comp_timbre=comp, style=style_name,
        profile=profile_name_for(lead, comp), reason="style",
    )


def lead_velocity(
    *,
    row_in_bar: int,
    rows_per_beat: int = 4,
    phrase_start: bool = False,
    cadence: bool = False,
    section_start: bool = False,
) -> int:
    """Velocity for one lead note, from its musical position.

    Modest and motivated, in this order of priority: a section opening or a
    cadential arrival is the loudest thing in the phrase, a phrase opening next,
    then the beat hierarchy. Everything resolves to a register first, so each
    distinction is one the hardware can actually represent.
    """
    if section_start or cadence or phrase_start:
        return velocity_for_register(LEAD_REGISTERS["accent"])
    if row_in_bar % rows_per_beat == 0:
        return velocity_for_register(LEAD_REGISTERS["strong"])
    if row_in_bar % (rows_per_beat // 2 or 1) == 0:
        return velocity_for_register(LEAD_REGISTERS["normal"])
    return velocity_for_register(LEAD_REGISTERS["weak"])


def comp_velocity(*, row_in_bar: int, rows_per_beat: int = 4) -> int:
    """Velocity for one accompaniment note. Follows the beat only.

    The comp does not take phrase or cadence accents: its job is to state the
    harmony steadily, and accenting it competes with the lead for the one thing
    a 4-bit register has little of, which is range.
    """
    if row_in_bar % rows_per_beat == 0:
        return velocity_for_register(COMP_REGISTERS["strong"])
    if row_in_bar % (rows_per_beat // 2 or 1) == 0:
        return velocity_for_register(COMP_REGISTERS["normal"])
    return velocity_for_register(COMP_REGISTERS["weak"])


def registers_are_distinct(velocities: list[int]) -> bool:
    """Whether a set of velocities actually lands on different registers.

    The check that catches a dynamic plan which reads well and is inaudible.
    """
    from rse.renderers.nes_apu import _velocity_to_volume

    return len({_velocity_to_volume(v) for v in velocities}) == len(set(velocities))
