"""The canonical RSE Event Format (RSE-EF).

Versioned separately from the Phase 1 intake record, which stays the lossless
landing zone for source data. RSE-EF is what the rest of the system agrees on:
the tokenizer reads it, the constraint engine rewrites it, the renderers
consume it, and the exporters write it out.

Two rules shape the design (proposal 5.3):

- A noise event carries a `value`, never a `pitch`, and the field that does not
  apply is absent rather than null. Anything that transposes or harmonizes by
  pitch therefore cannot touch the noise channel by accident.
- Source ticks are canonical. Beats, bars and seconds are derived on demand
  from the tempo map, never stored in place of the tick they came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rse.data.timing import MusicalTime, TempoChange, TimeSignatureChange

#: Bump the minor version for additive changes, the major for breaking ones.
RSE_EF_SCHEMA_VERSION = "rse-ef/1.1.0"

#: Major version this build can read.
SUPPORTED_SCHEMA_MAJOR = "rse-ef/1"

NOTE = "note"
NOISE = "noise"
CONTROLLER = "controller"
PROGRAM = "program"
EFFECT = "effect"
EVENT_TYPES = (NOTE, NOISE, CONTROLLER, PROGRAM, EFFECT)


class SchemaError(ValueError):
    """Raised when a serialized composition cannot be read."""


@dataclass(frozen=True)
class SectionSpan:
    """Where a section of the form begins, and which section it is.

    Derived from repetition rather than annotated — NES-MDB carries no form
    labels — so a span says "this material resembles the earlier A", not "the
    composer wrote an A here". See `rse.events.structure`.
    """

    start_bar: int
    label: str

    def to_dict(self) -> dict[str, Any]:
        return {"start_bar": self.start_bar, "label": self.label}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SectionSpan:
        return cls(start_bar=int(data["start_bar"]), label=str(data["label"]))


@dataclass(frozen=True)
class Event:
    """Base for everything on the timeline."""

    channel_id: str
    tick: int

    @property
    def event_type(self) -> str:  # pragma: no cover - overridden everywhere
        raise NotImplementedError

    @property
    def end_tick(self) -> int:
        return self.tick

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass(frozen=True)
class Note(Event):
    """A pitched note on a melodic channel."""

    pitch: int = 60
    duration: int = 0
    velocity: int = 96
    instrument_id: str | None = None
    effects: tuple[tuple[str, Any], ...] = ()

    @property
    def event_type(self) -> str:
        return NOTE

    @property
    def end_tick(self) -> int:
        return self.tick + self.duration

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "type": NOTE,
            "ch": self.channel_id,
            "tick": self.tick,
            "pitch": self.pitch,
            "duration": self.duration,
            "velocity": self.velocity,
        }
        if self.instrument_id is not None:
            payload["instrument_id"] = self.instrument_id
        if self.effects:
            payload["effects"] = dict(self.effects)
        return payload


@dataclass(frozen=True)
class Noise(Event):
    """A percussive event on an unpitched channel.

    `value` selects a hardware period or timbre. It is not a pitch and must
    never be transposed.
    """

    value: int = 1
    duration: int = 0
    velocity: int = 96
    instrument_id: str | None = None
    effects: tuple[tuple[str, Any], ...] = ()

    @property
    def event_type(self) -> str:
        return NOISE

    @property
    def end_tick(self) -> int:
        return self.tick + self.duration

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "type": NOISE,
            "ch": self.channel_id,
            "tick": self.tick,
            "value": self.value,
            "duration": self.duration,
            "velocity": self.velocity,
        }
        if self.instrument_id is not None:
            payload["instrument_id"] = self.instrument_id
        if self.effects:
            payload["effects"] = dict(self.effects)
        return payload


@dataclass(frozen=True)
class Controller(Event):
    """Continuous expression data, such as volume or timbre."""

    controller: int = 0
    value: int = 0

    @property
    def event_type(self) -> str:
        return CONTROLLER

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": CONTROLLER,
            "ch": self.channel_id,
            "tick": self.tick,
            "controller": self.controller,
            "value": self.value,
        }


@dataclass(frozen=True)
class Program(Event):
    """An instrument or patch selection."""

    program: int = 0

    @property
    def event_type(self) -> str:
        return PROGRAM

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": PROGRAM,
            "ch": self.channel_id,
            "tick": self.tick,
            "program": self.program,
        }


@dataclass(frozen=True)
class EffectEvent(Event):
    """A hardware-register effect: duty change, pitch slide, vibrato, sweep."""

    effect: str = ""
    params: tuple[tuple[str, Any], ...] = ()

    @property
    def event_type(self) -> str:
        return EFFECT

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": EFFECT,
            "ch": self.channel_id,
            "tick": self.tick,
            "effect": self.effect,
            "params": dict(self.params),
        }


def event_sort_key(event: Event) -> tuple[int, str, int, int]:
    """Deterministic timeline order.

    Note-offs are implied by durations here, so ordering only has to be stable:
    by tick, then channel, then event type, then the event's own value.
    """
    value = 0
    if isinstance(event, Note):
        value = event.pitch
    elif isinstance(event, Noise):
        value = event.value
    elif isinstance(event, Controller):
        value = event.controller
    elif isinstance(event, Program):
        value = event.program
    return (event.tick, event.channel_id, EVENT_TYPES.index(event.event_type), value)


@dataclass
class Composition:
    """A complete piece in canonical form."""

    hardware_profile: str
    ticks_per_beat: int
    events: list[Event] = field(default_factory=list)
    tempo_map: list[TempoChange] = field(default_factory=list)
    time_signatures: list[TimeSignatureChange] = field(default_factory=list)
    loop_start_tick: int | None = None
    loop_end_tick: int | None = None
    #: Sections of the form, in bar order. Empty when unknown; nothing requires
    #: them, so a document written before schema 1.1.0 still loads.
    sections: list[SectionSpan] = field(default_factory=list)
    source_metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = RSE_EF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.ticks_per_beat <= 0:
            raise SchemaError(f"ticks_per_beat must be positive, got {self.ticks_per_beat}")

    # -- queries ----------------------------------------------------------

    @property
    def time(self) -> MusicalTime:
        """Derived timing map. Beats and seconds come from here, not storage."""
        return MusicalTime(self.ticks_per_beat, list(self.tempo_map), list(self.time_signatures))

    @property
    def end_tick(self) -> int:
        return max((e.end_tick for e in self.events), default=0)

    @property
    def channel_ids(self) -> list[str]:
        return sorted({e.channel_id for e in self.events})

    def sorted_events(self) -> list[Event]:
        return sorted(self.events, key=event_sort_key)

    def events_on(self, channel_id: str) -> list[Event]:
        return [e for e in self.sorted_events() if e.channel_id == channel_id]

    def sounding_events(self, channel_id: str | None = None) -> list[Note | Noise]:
        """Note and noise events, which are the ones that occupy a channel."""
        return [
            e
            for e in self.sorted_events()
            if isinstance(e, Note | Noise) and (channel_id is None or e.channel_id == channel_id)
        ]

    def normalize(self) -> None:
        """Put events into canonical order, in place."""
        self.events = self.sorted_events()

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "hardware_profile": self.hardware_profile,
            "ticks_per_beat": self.ticks_per_beat,
            "tempo_map": [{"tick": t.tick, "us_per_beat": t.us_per_beat} for t in self.tempo_map],
            "time_signatures": [
                {"tick": s.tick, "numerator": s.numerator, "denominator": s.denominator}
                for s in self.time_signatures
            ],
            "loop_start_tick": self.loop_start_tick,
            "loop_end_tick": self.loop_end_tick,
            "sections": [s.to_dict() for s in self.sections],
            "source_metadata": self.source_metadata,
            "events": [e.to_dict() for e in self.sorted_events()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Composition:
        version = str(data.get("schema_version", ""))
        if not version.startswith(SUPPORTED_SCHEMA_MAJOR):
            raise SchemaError(
                f"unsupported RSE-EF schema {version!r}; "
                f"this build reads {SUPPORTED_SCHEMA_MAJOR}.x"
            )
        return cls(
            hardware_profile=data["hardware_profile"],
            ticks_per_beat=int(data["ticks_per_beat"]),
            events=[event_from_dict(e) for e in data.get("events", ())],
            tempo_map=[
                TempoChange(int(t["tick"]), int(t["us_per_beat"]))
                for t in data.get("tempo_map", ())
            ],
            time_signatures=[
                TimeSignatureChange(int(s["tick"]), int(s["numerator"]), int(s["denominator"]))
                for s in data.get("time_signatures", ())
            ],
            loop_start_tick=data.get("loop_start_tick"),
            loop_end_tick=data.get("loop_end_tick"),
            sections=[SectionSpan.from_dict(s) for s in data.get("sections", ())],
            source_metadata=data.get("source_metadata", {}),
            schema_version=version,
        )


def event_from_dict(data: dict[str, Any]) -> Event:
    """Rebuild one event from its serialized form."""
    event_type = data.get("type")
    channel_id = data["ch"]
    tick = int(data["tick"])
    effects = tuple(sorted((data.get("effects") or {}).items()))

    if event_type == NOTE:
        return Note(
            channel_id=channel_id,
            tick=tick,
            pitch=int(data["pitch"]),
            duration=int(data.get("duration", 0)),
            velocity=int(data.get("velocity", 96)),
            instrument_id=data.get("instrument_id"),
            effects=effects,
        )
    if event_type == NOISE:
        return Noise(
            channel_id=channel_id,
            tick=tick,
            value=int(data["value"]),
            duration=int(data.get("duration", 0)),
            velocity=int(data.get("velocity", 96)),
            instrument_id=data.get("instrument_id"),
            effects=effects,
        )
    if event_type == CONTROLLER:
        return Controller(
            channel_id=channel_id,
            tick=tick,
            controller=int(data["controller"]),
            value=int(data["value"]),
        )
    if event_type == PROGRAM:
        return Program(channel_id=channel_id, tick=tick, program=int(data["program"]))
    if event_type == EFFECT:
        return EffectEvent(
            channel_id=channel_id,
            tick=tick,
            effect=str(data.get("effect", "")),
            params=tuple(sorted((data.get("params") or {}).items())),
        )
    raise SchemaError(f"unknown event type {event_type!r}; known: {list(EVENT_TYPES)}")
