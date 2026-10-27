"""The wire contract for the generator.

These mirror the controls in the interface concept -- console, prompt, style,
duration, energy, tempo, loop -- and nothing more. Each one maps onto something
the composer already takes, so the API adds no musical behaviour of its own:
`style` picks an existing preset, `energy` picks an existing energy bucket,
`tempo_bpm` uses the override `compose()` already accepts. A control that
implied a new musical capability would be a composer change pretending to be a
web change.

`console` is validated, not quietly ignored. SNES is not implemented, and the
API says so with a 501 rather than returning NES audio under an SNES label.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

#: Energy words the interpreter already understands, lowest to highest, so a
#: percentage slider maps onto them instead of inventing a new scale.
ENERGY_LADDER = (
    "very_low",
    "low",
    "medium",
    "medium_high",
    "high",
    "very_high",
)


def energy_from_percent(percent: float) -> str:
    """Map a 0-100 slider onto the interpreter's existing energy buckets.

    Equal bands rather than rounding to the nearest rung: rounding gives the
    two ends half-width bands, so the extremes of the slider are harder to
    reach than the middle ones.
    """
    clamped = min(100.0, max(0.0, float(percent)))
    band = int(clamped / 100.0 * len(ENERGY_LADDER))
    return ENERGY_LADDER[min(band, len(ENERGY_LADDER) - 1)]


class GenerateRequest(BaseModel):
    """One Generate Song click."""

    prompt: str = Field(
        default="",
        max_length=600,
        description="Free text describing the soundtrack.",
    )
    console: Literal["nes", "snes"] = "nes"
    style: str | None = Field(
        default=None,
        description="Composer preset. Omit to infer it from the prompt.",
    )
    seconds: float | None = Field(
        default=None, description="Requested duration. Capped by the server."
    )
    energy: float | None = Field(
        default=None, ge=0, le=100, description="0-100 slider, calm to intense."
    )
    tempo_bpm: int | None = Field(default=None, ge=20, le=380)
    loop: bool | None = Field(
        default=None,
        description="Omit to let the prompt decide; set to force it either way.",
    )
    mood: str | None = Field(
        default=None,
        description="Selects the instrument profile. Only moods the engine "
        "maps to a real profile are accepted.",
    )
    sound_profile: str | None = Field(
        default=None,
        description="Named instrument profile. Overrides mood. NES only.",
    )
    echo: str | None = Field(
        default=None,
        description="SNES only. Echo preset; costs audio RAM, so a long "
        "reverb is paid for in instrument memory.",
    )
    reserved_voices: int = Field(
        default=0, ge=0, le=6,
        description="SNES only. Voices held back for sound effects, which is "
        "the game-ready mode.",
    )

    @field_validator("echo")
    @classmethod
    def _known_echo(cls, value: str | None) -> str | None:
        if value is None:
            return None
        from rse.song.snes_generate import ECHO_PRESETS

        key = value.strip().lower()
        if key and key not in ECHO_PRESETS:
            raise ValueError(
                f"unknown echo preset {value!r}; have {sorted(ECHO_PRESETS)}"
            )
        return key or None

    @field_validator("mood")
    @classmethod
    def _known_mood(cls, value: str | None) -> str | None:
        if value is None:
            return None
        from rse.procedural.sound import MOOD_PROFILES

        key = value.strip().lower()
        if key and key not in MOOD_PROFILES:
            raise ValueError(
                f"mood {value!r} does not map to an instrument profile; "
                f"have {sorted(MOOD_PROFILES)}"
            )
        return key or None

    @field_validator("sound_profile")
    @classmethod
    def _known_profile(cls, value: str | None) -> str | None:
        if value is None:
            return None
        from rse.procedural.sound import SOUND_PROFILES

        key = value.strip().lower()
        if key and key not in SOUND_PROFILES:
            raise ValueError(
                f"unknown sound profile {value!r}; have {sorted(SOUND_PROFILES)}"
            )
        return key or None
    seed: int | None = Field(
        default=None,
        ge=0,
        le=2**31 - 1,
        description="Omit for a new variation; supply to reproduce a song.",
    )

    @field_validator("prompt")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    @field_validator("style")
    @classmethod
    def _known_style(cls, value: str | None) -> str | None:
        if value is None:
            return None
        from rse.procedural.compose import STYLES

        key = value.strip().lower().replace(" ", "_").replace("-", "_")
        # The interface labels a preset "Boss Battle"; the composer calls it
        # "boss". Accept the label without making the UI know the internal
        # name, and reject anything that is neither.
        aliases = {"boss_battle": "boss", "battle": "action", "overworld": "adventure"}
        key = aliases.get(key, key)
        if key not in STYLES:
            raise ValueError(
                f"unknown style {value!r}; have {sorted(STYLES)}"
            )
        return key

    def effective_prompt(self) -> str:
        """What the interpreter is given.

        An empty prompt is legitimate -- someone may only touch the controls --
        so the style becomes the request rather than failing. Phrased the way
        the interpreter reads prompts, since it is a prompt interpreter and not
        a form parser.
        """
        if self.prompt:
            return self.prompt
        if self.style:
            return f"a {self.style.replace('_', ' ')} theme"
        return "retro game music"


class SongLinks(BaseModel):
    wav: str
    midi: str | None = None
    report: str
    ogg: str | None = None


class GenerateResponse(BaseModel):
    """Everything the page needs to play, download and reproduce the song."""

    song_id: str
    seed: int
    console: str
    engine: str = "procedural"
    style: str
    style_reason: str
    duration_sec: float
    requested_seconds: float
    duration_within_tolerance: bool
    tempo_bpm: float
    #: The hardware's own tempo unit. Sent so the page can show that the
    #: number is a tempo a console could actually produce.
    frames_per_row: int
    mode: str
    #: What actually played it: the profile, its duty cycles, and why.
    sound: dict[str, Any] = Field(default_factory=dict)
    bars: int
    form: list[str]
    loopable: bool
    hardware_violations: int
    loop_timing_continuous: bool
    links: SongLinks
    #: The honesty buckets, unchanged from the CLI: what the request got, what
    #: was approximated, what the assembler imposed, and what nothing in the
    #: system can take as input.
    requirements: dict[str, Any]
    notes: list[str] = Field(default_factory=list)
    unsupported: list[dict[str, str]] = Field(default_factory=list)
    #: SNES only: the eight-voice allocation, the ARAM budget, and the
    #: preview's own caveats. Absent for NES.
    snes: dict[str, Any] | None = None


class StyleInfo(BaseModel):
    name: str
    label: str
    tempo_bpm: float
    #: The hardware's own tempo unit. Sent so the page can show that the
    #: number is a tempo a console could actually produce.
    frames_per_row: int
    mode: str


class EchoPreset(BaseModel):
    """One SNES echo setting and what it costs in audio RAM."""

    name: str
    delay_ms: int
    aram_bytes: int


class SoundProfileInfo(BaseModel):
    """One named instrument profile the engine can actually produce."""

    name: str
    label: str
    lead_timbre: str
    comp_timbre: str
    lead_duty_percent: str
    comp_duty_percent: str
    moods: list[str] = Field(default_factory=list)


class TempoStep(BaseModel):
    """One rung of the tempo ladder the NES can actually play.

    The console advances a tracker row every integer number of video frames,
    so tempo is a discrete ladder rather than a dial. Sent to the page so its
    slider snaps to the grid instead of offering tempos no console could
    produce -- `action` used to ask for 168 BPM, which is 5.37 frames a row.
    """

    frames_per_row: int
    bpm: float
    row_ms: float


class CapabilitiesResponse(BaseModel):
    """What the interface should offer, read from the engine rather than
    hard-coded in the frontend."""

    styles: list[StyleInfo]
    consoles: list[dict[str, Any]]
    min_seconds: float
    max_seconds: float
    default_seconds: float
    formats: list[str]
    engine: str = "procedural"
    #: Every playable tempo, so the page's slider snaps to the hardware grid
    #: instead of offering BPM values no console could output.
    tempo_ladder: list[TempoStep] = Field(default_factory=list)
    #: Instrument profiles, so the page offers only profiles that exist.
    sound_profiles: list[SoundProfileInfo] = Field(default_factory=list)
    #: Mood words that map to a profile. A mood outside this list is refused
    #: rather than silently ignored, which is what it used to be.
    moods: list[str] = Field(default_factory=list)
    #: SNES echo presets, each with its ARAM cost so the interface can show
    #: that a longer reverb is memory the instruments do not get.
    echo_presets: list[EchoPreset] = Field(default_factory=list)
    #: Maximum voices a caller may reserve for sound effects.
    max_reserved_voices: int = 6
