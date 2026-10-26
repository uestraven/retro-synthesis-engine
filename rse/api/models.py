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
    tempo_bpm: int
    mode: str
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


class StyleInfo(BaseModel):
    name: str
    label: str
    tempo_bpm: int
    mode: str


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
