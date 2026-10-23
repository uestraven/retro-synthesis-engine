"""Conversion into and out of the canonical format, plus round-trip measurement.

MVP criterion 2 (proposal 11.3) asks that the canonical format round-trip
supported compositions "within defined timing and controller tolerances". The
tolerances are defined here, as a value you can print, so fidelity is a
measurement rather than a claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mido

from rse.data.midi_io import parse_midi
from rse.data.normalize import build_intake_record
from rse.data.provenance import USE_UNKNOWN, describe_file
from rse.data.timing import TempoChange, TimeSignatureChange
from rse.events.schema import (
    Composition,
    Controller,
    EffectEvent,
    Event,
    Noise,
    Note,
    Program,
    event_sort_key,
)
from rse.hardware.profile import HardwareProfile

#: Intake event types that become RSE-EF effect events. They are hardware
#: register changes, not notes, and the canonical format says so explicitly
#: rather than flattening them into controller data.
_EFFECT_FROM_INTAKE = {
    "pitchwheel": "pitch_bend",
    "aftertouch": "channel_pressure",
    "polytouch": "key_pressure",
}
_EFFECT_TO_INTAKE = {v: k for k, v in _EFFECT_FROM_INTAKE.items()}

#: Text that marks a loop boundary in a source file.
_LOOP_START_HINTS = ("loop_start", "loop start", "loopstart", "repeat start")
_LOOP_END_HINTS = ("loop_end", "loop end", "loopend", "repeat end")


# -- intake record <-> canonical ---------------------------------------------


def composition_from_intake(record: dict[str, Any]) -> Composition:
    """Build a canonical composition from a Phase 1 intake record."""
    events: list[Event] = []
    for event in record.get("events", ()):
        converted = _event_from_intake(event)
        if converted is not None:
            events.append(converted)

    loop_start, loop_end = _loop_ticks(record.get("loop_markers", ()))
    source = dict(record.get("source", {}))
    if "composition_id" in record:
        source["composition_id"] = record["composition_id"]
    if "name" in record:
        source["name"] = record["name"]
    if "generation" in record:
        source["generation"] = record["generation"]

    composition = Composition(
        hardware_profile=record.get("hardware_profile", "nes"),
        ticks_per_beat=record.get("midi", {}).get("ticks_per_beat", 480),
        events=events,
        tempo_map=[TempoChange(t["tick"], t["us_per_beat"]) for t in record.get("tempo_map", ())],
        time_signatures=[
            TimeSignatureChange(s["tick"], s["numerator"], s["denominator"])
            for s in record.get("time_signatures", ())
        ],
        loop_start_tick=loop_start,
        loop_end_tick=loop_end,
        source_metadata=source,
    )
    composition.normalize()
    return composition


def composition_to_intake(
    composition: Composition, *, profile: HardwareProfile | None = None
) -> dict[str, Any]:
    """Project a canonical composition back onto the intake record shape.

    Lossy by design: the intake record has no place for RSE-EF effect events
    other than the MIDI messages they came from, and derived measurements are
    not recomputed here. Its purpose is feeding the existing MIDI exporter.
    """
    channel_ids = list(profile.channel_ids) if profile else composition.channel_ids
    events = [_event_to_intake(e) for e in composition.sorted_events()]
    markers = []
    if composition.loop_start_tick is not None:
        markers.append({"tick": composition.loop_start_tick, "text": "loop_start"})
    if composition.loop_end_tick is not None:
        markers.append({"tick": composition.loop_end_tick, "text": "loop_end"})

    return {
        "hardware_profile": composition.hardware_profile,
        "midi": {"ticks_per_beat": composition.ticks_per_beat},
        "tempo_map": [{"tick": t.tick, "us_per_beat": t.us_per_beat} for t in composition.tempo_map]
        or [{"tick": 0, "us_per_beat": 500_000}],
        "time_signatures": [
            {"tick": s.tick, "numerator": s.numerator, "denominator": s.denominator}
            for s in composition.time_signatures
        ],
        "loop_markers": markers,
        "channels": [{"channel_id": cid} for cid in channel_ids],
        "events": [e for e in events if e is not None],
        "source": composition.source_metadata,
    }


def _event_from_intake(event: dict[str, Any]) -> Event | None:
    event_type = event.get("type")
    channel_id = event.get("ch")
    if channel_id is None:
        return None
    tick = event["tick"]

    if event_type == "note":
        return Note(
            channel_id=channel_id,
            tick=tick,
            pitch=event["pitch"],
            duration=event.get("dur_ticks", 0),
            velocity=event.get("velocity", 96),
        )
    if event_type == "noise":
        return Noise(
            channel_id=channel_id,
            tick=tick,
            value=event["value"],
            duration=event.get("dur_ticks", 0),
            velocity=event.get("velocity", 96),
        )
    if event_type == "control":
        return Controller(
            channel_id=channel_id,
            tick=tick,
            controller=event["control"],
            value=event["value"],
        )
    if event_type == "program":
        return Program(channel_id=channel_id, tick=tick, program=event["program"])
    if event_type in _EFFECT_FROM_INTAKE:
        params = {
            k: v
            for k, v in event.items()
            if k
            not in ("type", "ch", "tick", "beat", "bar", "bar_pos", "sec", "q_beat", "q_dur_beats")
        }
        return EffectEvent(
            channel_id=channel_id,
            tick=tick,
            effect=_EFFECT_FROM_INTAKE[event_type],
            params=tuple(sorted(params.items())),
        )
    return None


def _event_to_intake(event: Event) -> dict[str, Any] | None:
    if isinstance(event, Note):
        return {
            "ch": event.channel_id,
            "type": "note",
            "tick": event.tick,
            "dur_ticks": event.duration,
            "pitch": event.pitch,
            "velocity": event.velocity,
        }
    if isinstance(event, Noise):
        return {
            "ch": event.channel_id,
            "type": "noise",
            "tick": event.tick,
            "dur_ticks": event.duration,
            "value": event.value,
            "velocity": event.velocity,
        }
    if isinstance(event, Controller):
        return {
            "ch": event.channel_id,
            "type": "control",
            "tick": event.tick,
            "control": event.controller,
            "value": event.value,
        }
    if isinstance(event, Program):
        return {
            "ch": event.channel_id,
            "type": "program",
            "tick": event.tick,
            "program": event.program,
        }
    if isinstance(event, EffectEvent) and event.effect in _EFFECT_TO_INTAKE:
        return {
            "ch": event.channel_id,
            "type": _EFFECT_TO_INTAKE[event.effect],
            "tick": event.tick,
            **dict(event.params),
        }
    return None


def _loop_ticks(markers) -> tuple[int | None, int | None]:
    start = end = None
    for marker in markers:
        text = str(marker.get("text", "")).strip().lower()
        if start is None and any(hint in text for hint in _LOOP_START_HINTS):
            start = marker["tick"]
        elif end is None and any(hint in text for hint in _LOOP_END_HINTS):
            end = marker["tick"]
    return start, end


# -- MIDI <-> canonical -------------------------------------------------------


def composition_from_midi(path: Path, profile: HardwareProfile) -> Composition:
    """Parse a MIDI file straight into canonical form."""
    path = Path(path)
    parsed = parse_midi(path, profile)
    source = describe_file(
        path,
        root=path.parent,
        dataset_id="adhoc",
        dataset_version="adhoc/convert",
        use_class=USE_UNKNOWN,
        acquisition_method="convert",
    )
    return composition_from_intake(build_intake_record(parsed, source, profile))


def composition_to_midi(
    composition: Composition, *, profile: HardwareProfile | None = None
) -> mido.MidiFile:
    """Reconstruct a MIDI file from canonical form."""
    from rse.exporters.midi import record_to_midi

    return record_to_midi(composition_to_intake(composition, profile=profile))


def write_composition_midi(
    path: Path, composition: Composition, *, profile: HardwareProfile | None = None
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    composition_to_midi(composition, profile=profile).save(str(path))
    return path


# -- round-trip measurement ---------------------------------------------------


@dataclass(frozen=True)
class Tolerance:
    """How much drift a round trip may introduce before it counts as loss.

    Defaults are exact: every path inside RSE is lossless for notes and
    controllers, and a regression should fail rather than be absorbed. The
    knobs exist so a lossy external format can be measured against a stated
    budget instead of an opinion.
    """

    timing_ticks: int = 0
    duration_ticks: int = 0
    velocity: int = 0
    controller_value: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "timing_ticks": self.timing_ticks,
            "duration_ticks": self.duration_ticks,
            "velocity": self.velocity,
            "controller_value": self.controller_value,
        }


EXACT = Tolerance()


@dataclass
class RoundTripReport:
    """What survived a round trip, and what did not."""

    n_original: int = 0
    n_result: int = 0
    n_matched: int = 0
    missing: list[dict[str, Any]] = field(default_factory=list)
    added: list[dict[str, Any]] = field(default_factory=list)
    drifted: list[dict[str, Any]] = field(default_factory=list)
    tolerance: Tolerance = EXACT
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing and not self.added and not self.drifted and not self.notes

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "n_original": self.n_original,
            "n_result": self.n_result,
            "n_matched": self.n_matched,
            "n_missing": len(self.missing),
            "n_added": len(self.added),
            "n_drifted": len(self.drifted),
            "missing": self.missing[:20],
            "added": self.added[:20],
            "drifted": self.drifted[:20],
            "tolerance": self.tolerance.to_dict(),
            "notes": self.notes,
        }


def compare_compositions(
    original: Composition, result: Composition, *, tolerance: Tolerance = EXACT
) -> RoundTripReport:
    """Match `result`'s events against `original`'s within `tolerance`."""
    report = RoundTripReport(tolerance=tolerance)
    originals = original.sorted_events()
    results = list(result.sorted_events())
    report.n_original = len(originals)
    report.n_result = len(results)

    unmatched = list(results)
    for event in originals:
        match = _take_match(event, unmatched, tolerance)
        if match is None:
            report.missing.append(_describe(event))
            continue
        report.n_matched += 1
        drift = _drift(event, match)
        if drift:
            report.drifted.append({**_describe(event), "drift": drift})
    report.added = [_describe(e) for e in unmatched]

    if original.ticks_per_beat != result.ticks_per_beat:
        report.notes.append(
            f"resolution changed: {original.ticks_per_beat} -> {result.ticks_per_beat} "
            "ticks per beat"
        )
    if (original.loop_start_tick, original.loop_end_tick) != (
        result.loop_start_tick,
        result.loop_end_tick,
    ):
        report.notes.append("loop boundaries did not survive")
    if _tempo_key(original) != _tempo_key(result):
        report.notes.append("tempo map did not survive")
    return report


def _take_match(event: Event, candidates: list[Event], tolerance: Tolerance) -> Event | None:
    """Pop the closest acceptable counterpart to `event` from `candidates`."""
    best_index = None
    best_distance = None
    for index, candidate in enumerate(candidates):
        if type(candidate) is not type(event):
            continue
        if candidate.channel_id != event.channel_id:
            continue
        if not _identity_matches(event, candidate):
            continue
        distance = abs(candidate.tick - event.tick)
        if distance > tolerance.timing_ticks:
            continue
        if best_distance is None or distance < best_distance:
            best_index, best_distance = index, distance
    if best_index is None:
        return None
    return candidates.pop(best_index)


def _identity_matches(a: Event, b: Event) -> bool:
    """Whether two events refer to the same musical thing, ignoring drift."""
    if isinstance(a, Note) and isinstance(b, Note):
        return a.pitch == b.pitch
    if isinstance(a, Noise) and isinstance(b, Noise):
        return a.value == b.value
    if isinstance(a, Controller) and isinstance(b, Controller):
        return a.controller == b.controller
    if isinstance(a, Program) and isinstance(b, Program):
        return a.program == b.program
    if isinstance(a, EffectEvent) and isinstance(b, EffectEvent):
        return a.effect == b.effect
    return False


def _drift(original: Event, result: Event) -> dict[str, int]:
    """Differences that exceeded tolerance, per field."""
    drift: dict[str, int] = {}
    if result.tick != original.tick:
        drift["tick"] = result.tick - original.tick
    if isinstance(original, Note | Noise) and isinstance(result, Note | Noise):
        if result.duration != original.duration:
            drift["duration"] = result.duration - original.duration
        if result.velocity != original.velocity:
            drift["velocity"] = result.velocity - original.velocity
    if isinstance(original, Controller) and isinstance(result, Controller):
        if result.value != original.value:
            drift["value"] = result.value - original.value
    return drift


def _describe(event: Event) -> dict[str, Any]:
    return {
        "type": event.event_type,
        "ch": event.channel_id,
        "tick": event.tick,
        "sort_key": list(event_sort_key(event)),
    }


def _tempo_key(composition: Composition) -> list[tuple[int, int]]:
    return [(t.tick, t.us_per_beat) for t in composition.tempo_map]


def round_trip_midi(
    composition: Composition,
    profile: HardwareProfile,
    *,
    tolerance: Tolerance = EXACT,
    directory: Path | None = None,
) -> tuple[Composition, RoundTripReport]:
    """Write a composition to MIDI, read it back, and report the difference."""
    import tempfile

    with tempfile.TemporaryDirectory() as temporary:
        base = Path(directory) if directory else Path(temporary)
        path = base / "round_trip.mid"
        write_composition_midi(path, composition, profile=profile)
        restored = composition_from_midi(path, profile)
    # Provenance legitimately differs after a round trip; the comparison is of
    # musical content, so carry the original's metadata across.
    restored.source_metadata = composition.source_metadata
    return restored, compare_compositions(composition, restored, tolerance=tolerance)
