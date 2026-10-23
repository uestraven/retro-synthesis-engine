"""Renderer interface and the record-to-note-list adaptation.

A renderer turns validated musical events into audio. It must stay replaceable
without touching the composition model or the parsers (proposal 4.2), so the
only thing a backend sees is `RenderSettings` plus a list of `RenderNote`.

Phase 1 intake records are the input today. When Phase 2 introduces the
canonical RSE Event Format, `notes_from_record` is the single function that
needs to learn the new shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from rse.data.timing import MusicalTime, TempoChange, TimeSignatureChange

#: Default render sample rate. 44.1 kHz because that is what the NES-MDB
#: reference recordings use, which keeps future A/B comparison honest.
DEFAULT_SAMPLE_RATE = 44_100


@dataclass(frozen=True)
class RenderSettings:
    """Everything that affects the audio output, logged for reproducibility."""

    sample_rate: int = DEFAULT_SAMPLE_RATE
    #: Internal oversampling factor. The APU emits stepped waveforms at CPU
    #: rate; rendering above the output rate and decimating keeps the harmonics
    #: that fall above Nyquist from folding back as audible aliasing.
    oversample: int = 4
    #: Peak normalization target. None leaves the raw mixer output alone.
    peak: float | None = 0.89
    #: Apply the console's output filters (two high-pass, one low-pass).
    apply_output_filters: bool = True
    #: Shape unpitched notes that carry no envelope data with the measured
    #: percussive decay. Without it, generated drums render at one constant
    #: level and read as static rather than as hits.
    default_noise_decay: bool = True
    #: Seconds of silence appended so release tails are not clipped.
    tail_sec: float = 0.25

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.oversample < 1:
            raise ValueError(f"oversample must be at least 1, got {self.oversample}")
        if self.peak is not None and not 0 < self.peak <= 1:
            raise ValueError(f"peak must be in (0, 1], got {self.peak}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "oversample": self.oversample,
            "peak": self.peak,
            "apply_output_filters": self.apply_output_filters,
            "default_noise_decay": self.default_noise_decay,
            "tail_sec": self.tail_sec,
        }


#: Controller number carrying per-frame volume, matching what NES-MDB's own
#: MIDI release uses and what the exprsco reader writes.
VOLUME_CONTROLLER = 11


@dataclass(frozen=True)
class EnvelopePoint:
    """A volume register change: a step, not a ramp."""

    sec: float
    gain: float


@dataclass(frozen=True)
class RenderNote:
    """One note, resolved to absolute seconds and a hardware channel."""

    channel_id: str
    start_sec: float
    end_sec: float
    #: MIDI pitch for pitched channels; a hardware period index for noise.
    value: int
    velocity: int
    pitched: bool

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass
class RenderResult:
    """Rendered audio plus what it took to produce it."""

    samples: Any  # numpy float32 array, mono, nominally in [-1, 1]
    sample_rate: int
    settings: RenderSettings
    renderer_id: str
    #: Per-channel audio before mixing, for stem export and debugging.
    stems: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        return len(self.samples) / self.sample_rate

    def to_dict(self) -> dict[str, Any]:
        """Metadata only — never the audio itself."""
        return {
            "renderer_id": self.renderer_id,
            "sample_rate": self.sample_rate,
            "duration_sec": round(self.duration_sec, 6),
            "n_samples": int(len(self.samples)),
            "channels_rendered": sorted(self.stems),
            "settings": self.settings.to_dict(),
            "warnings": self.warnings,
        }


class Renderer(Protocol):
    """What every audio backend must provide."""

    renderer_id: str
    profile_id: str

    def render(
        self, notes: list[RenderNote], settings: RenderSettings | None = None
    ) -> RenderResult: ...


def musical_time_from_record(record: dict[str, Any]) -> MusicalTime:
    """Rebuild the timing map stored in an intake record."""
    return MusicalTime(
        record["midi"]["ticks_per_beat"],
        [TempoChange(t["tick"], t["us_per_beat"]) for t in record.get("tempo_map", ())],
        [
            TimeSignatureChange(s["tick"], s["numerator"], s["denominator"])
            for s in record.get("time_signatures", ())
        ],
    )


def envelopes_from_record(record: dict[str, Any]) -> dict[str, list[EnvelopePoint]]:
    """Per-channel volume envelopes, from the controller data.

    Real NES instruments decay: measured over 8098 noise hits in NES-MDB, mean
    volume falls 9.50 -> 5.98 -> 4.48 of 15 across the first three frames. That
    movement is in the source, the parsers preserve it, and a renderer that
    ignores it holds every note at its onset level — which is what makes a drum
    hit sound like static rather than percussion.

    The register steps rather than glides, so these are breakpoints to be held,
    never interpolated.
    """
    time = musical_time_from_record(record)
    envelopes: dict[str, list[EnvelopePoint]] = {}
    for event in record.get("events", ()):
        if event.get("type") != "control" or event.get("control") != VOLUME_CONTROLLER:
            continue
        channel_id = event.get("ch")
        if channel_id is None:
            continue
        gain = max(0.0, min(1.0, event.get("value", 0) / 127.0))
        envelopes.setdefault(channel_id, []).append(
            EnvelopePoint(sec=time.seconds(event["tick"]), gain=gain)
        )
    for points in envelopes.values():
        points.sort(key=lambda point: point.sec)
    return envelopes


def notes_from_record(record: dict[str, Any]) -> list[RenderNote]:
    """Extract renderable notes from an intake record.

    Note *start* times are already stored in seconds on each event; note *ends*
    are not, so they are derived from the record's own tempo map rather than
    assuming a constant tempo.
    """
    time = musical_time_from_record(record)
    notes: list[RenderNote] = []
    for event in record.get("events", ()):
        event_type = event.get("type")
        if event_type not in ("note", "noise"):
            continue
        # A noise event carries `value`, a pitched one carries `pitch`; the key
        # that is absent is the one that must not be read.
        pitched = event_type == "note"
        value = event["pitch"] if pitched else event["value"]
        start_tick = event["tick"]
        end_tick = start_tick + event.get("dur_ticks", 0)
        notes.append(
            RenderNote(
                channel_id=event["ch"],
                start_sec=time.seconds(start_tick),
                end_sec=time.seconds(end_tick),
                value=value,
                velocity=event.get("velocity", 127),
                pitched=pitched,
            )
        )
    notes.sort(key=lambda n: (n.start_sec, n.channel_id))
    return notes
