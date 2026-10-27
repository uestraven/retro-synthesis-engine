"""The wire contract for the generator.

These mirror the controls in the interface -- console, style, duration,
energy, tempo, loop -- and nothing more. Each one maps onto something the
composer already takes, so the API adds no musical behaviour of its own:
`style` picks an existing preset, `energy` picks an existing energy bucket,
`tempo_bpm` uses the override `compose()` already accepts. A control that
implied a new musical capability would be a composer change pretending to be
a web change.

There is no `prompt`. There was: a free-text field read by a rule-based
parser, behind a "Describe a Song" tab. Both are gone, because
natural-language input is not a direction this project is taking and a text
box promises an understanding of prose that only a language model has. The
settings ARE the request.

The parser itself survives as `rse.prompt.interpret`, and not as dead code:
it is still how a request becomes a `SongPlan`, which is the structure the
composer and the requirement report both read. `effective_prompt()` phrases
the settings as a short description for it. That indirection is now
vestigial in shape -- building a `SongPlan` from the settings directly would
be cleaner -- but it is well covered by tests and changing it is a refactor
of the engine rather than of the web surface.

`console` is validated, not quietly ignored, and a console this deployment
does not serve gets a 503 rather than audio from the other renderer.
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

    console: Literal["nes", "snes"] = "nes"
    style: str | None = Field(
        default=None,
        description="Composer preset. Omit to let the engine choose one.",
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
        description="Omit for the style's default; set to force it either way.",
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
        description="SNES only. BUDGET ONLY -- DOES NOT AFFECT THE AUDIO. "
        "Reserves the S-DSP echo buffer, so it changes the audio-RAM "
        "account and nothing you can hear: the preview renderer has no echo "
        "processor, and the per-voice echo sends in the voice plan are "
        "recorded but not applied. Useful for seeing how much instrument "
        "memory a given reverb would cost; useless for making a song "
        "sound wet.",
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
        """The settings, phrased as the short description the planner reads.

        Not a prompt from a user -- there is no longer any way for one to
        arrive. `rse.prompt.interpret` is the engine's settings-to-`SongPlan`
        step, and it happens to take its input as a sentence, so this renders
        the one setting it can infer something from.

        Named `effective_prompt` because that is what it feeds, and renaming
        it would churn the call sites for no behavioural gain. The shape is
        vestigial: a `SongPlan` built straight from the fields would be
        clearer. That is an engine refactor, deliberately not bundled with
        removing a web control.
        """
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
    """One SNES echo setting and what it costs in audio RAM.

    `affects_audio` is False for every preset and is sent to the client so a
    settings panel can label the control honestly. Release validation
    generated the same song at dry, room, hall and cavern and got four
    byte-identical WAVs: the setting reserves the DSP's echo buffer in the
    memory account, but the preview renderer has no echo processor, so
    nothing downstream reads it. Shipping it as a reverb knob would be
    exactly the "metadata exists, therefore the capability exists" mistake.
    """

    name: str
    delay_ms: int
    aram_bytes: int
    #: Always False today. If an echo processor is ever added, this flips and
    #: the tests that assert silence-of-effect will fail, which is the point.
    affects_audio: bool = False


class CapabilityClaim(BaseModel):
    """What RSE claims about one SNES capability, and on what evidence.

    Served rather than written into the page so the claims have exactly one
    source. A marketing-ish sentence in HTML and a boolean in Python drift;
    this cannot.
    """

    capability: str
    #: Deliberately a free-text claim and not an enum. The useful distinctions
    #: here -- "implemented, internally validated" against "verified on tested
    #: output" against "partial" -- are differences in the KIND of evidence,
    #: and flattening them to a status would lose the thing worth saying.
    claim: str
    detail: str


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
    #: Per-capability SNES claims, so a client can show what is implemented,
    #: what is merely measured, what is partial and what is unsupported,
    #: rather than inferring all of it from one `hardware_exact: false`.
    snes_claims: list[CapabilityClaim] = Field(default_factory=list)
