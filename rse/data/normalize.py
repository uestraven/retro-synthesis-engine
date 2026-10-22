"""Building intake records from parsed MIDI.

The intake record is the lossless landing zone: original ticks are always
present, derived musical positions sit beside them, and quantization — when
requested — is written to an extra field rather than replacing anything
(proposal 5.3). Noise-channel events keep noise semantics instead of being
dressed up as pitched notes.
"""

from __future__ import annotations

from typing import Any

from rse import INTAKE_SCHEMA_VERSION
from rse.data import analysis
from rse.data.midi_io import Control, IntakeComposition, Note
from rse.data.provenance import SourceRecord
from rse.hardware.profile import HardwareProfile

#: Text/marker events whose content suggests a loop boundary.
LOOP_HINTS = ("loop", "repeat start", "repeat end")


def quantize_beat(beat: float, grid: float) -> float:
    """Snap a beat position to the nearest multiple of `grid` quarter notes."""
    if grid <= 0:
        raise ValueError(f"grid must be positive, got {grid}")
    return round(beat / grid) * grid


def build_intake_record(
    comp: IntakeComposition,
    source: SourceRecord,
    profile: HardwareProfile,
    *,
    quantize_grid: float | None = None,
) -> dict[str, Any]:
    """Serialize one parsed composition into an intake record."""
    time = comp.time
    tpb = comp.ticks_per_beat

    def position(tick: int) -> dict[str, Any]:
        bar, bar_pos = time.bar_position(tick)
        pos: dict[str, Any] = {
            "tick": tick,
            "beat": round(time.beat(tick), 6),
            "bar": bar,
            "bar_pos": round(bar_pos, 6),
            "sec": round(time.seconds(tick), 6),
        }
        if quantize_grid is not None:
            pos["q_beat"] = round(quantize_beat(time.beat(tick), quantize_grid), 6)
        return pos

    events: list[dict[str, Any]] = []
    dropped_notes = 0
    dropped_controls = 0

    for note in comp.notes:
        if note.channel_id is None:
            dropped_notes += 1
            continue
        events.append(_note_event(note, profile, position, tpb, quantize_grid))

    for control in comp.controls:
        if control.channel_id is None:
            dropped_controls += 1
            continue
        events.append(_control_event(control, position))

    events.sort(key=lambda e: (e["tick"], e["ch"], e["type"]))

    channels = [_channel_summary(comp, profile, channel_id) for channel_id in profile.channel_ids]
    loop_markers = [
        {**position(evt["tick"]), "type": evt["type"], "text": evt["text"]}
        for evt in comp.text_events
        if any(hint in (evt.get("text") or "").lower() for hint in LOOP_HINTS)
    ]

    end_tick = comp.end_tick
    # The end tick is exclusive: a note finishing exactly on the downbeat of
    # bar 8 makes an eight-bar piece, not a nine-bar one.
    last_bar, _ = time.bar_position(max(0, end_tick - 1))
    measurements = {
        "n_events": len(events),
        "n_notes": sum(1 for e in events if e["type"] in ("note", "noise")),
        "n_controls": sum(1 for e in events if e["type"] not in ("note", "noise")),
        "dropped_notes": dropped_notes,
        "dropped_controls": dropped_controls,
        "end_tick": end_tick,
        "end_beat": round(time.beat(end_tick), 6),
        "end_sec": round(time.seconds(end_tick), 6),
        "n_bars": last_bar + 1 if end_tick else 0,
        "constant_tempo": time.is_constant_tempo,
        "constant_meter": time.is_constant_meter,
        "channels_used": sum(1 for ch in channels if ch["n_notes"] > 0),
        "hardware_compliant": all(
            ch["max_simultaneous"] <= profile.channel(ch["channel_id"]).max_simultaneous_notes
            and not ch["out_of_range_values"]
            for ch in channels
        ),
    }

    return {
        "schema_version": INTAKE_SCHEMA_VERSION,
        "composition_id": source.source_id,
        "name": source.relpath,
        "hardware_profile": profile.profile_id,
        "hardware_schema_version": profile.schema_version,
        "source": source.to_dict(),
        "midi": {
            "format": comp.midi_format,
            "ticks_per_beat": tpb,
            "n_tracks": comp.n_tracks,
        },
        "quantize_grid": quantize_grid,
        "tempo_map": [
            {**position(t.tick), "us_per_beat": t.us_per_beat, "bpm": round(t.bpm, 4)}
            for t in time.tempo_changes
        ],
        "time_signatures": [
            {**position(s.tick), "numerator": s.numerator, "denominator": s.denominator}
            for s in time.time_signatures
        ],
        "key_signatures": comp.key_signatures,
        "loop_markers": loop_markers,
        "text_events": comp.text_events,
        "channels": channels,
        "events": events,
        "measurements": measurements,
        "warnings": comp.warnings,
    }


def _note_event(
    note: Note,
    profile: HardwareProfile,
    position,
    ticks_per_beat: int,
    quantize_grid: float | None,
) -> dict[str, Any]:
    spec = profile.channel(note.channel_id)  # type: ignore[arg-type]
    event: dict[str, Any] = {
        "ch": note.channel_id,
        **position(note.tick),
        "dur_ticks": note.dur_ticks,
        "dur_beats": round(note.dur_ticks / ticks_per_beat, 6),
        "velocity": note.velocity,
    }
    if quantize_grid is not None:
        event["q_dur_beats"] = round(
            max(quantize_grid, quantize_beat(note.dur_ticks / ticks_per_beat, quantize_grid)),
            6,
        )
    if spec.pitched:
        event["type"] = "note"
        event["pitch"] = note.pitch
    else:
        # The NES noise channel's "note number" selects a period, not a pitch;
        # transposition and harmonic analysis must never touch it.
        event["type"] = "noise"
        event["value"] = note.pitch
    return event


def _control_event(control: Control, position) -> dict[str, Any]:
    type_map = {
        "control_change": "control",
        "program_change": "program",
        "pitchwheel": "pitchwheel",
        "aftertouch": "aftertouch",
        "polytouch": "polytouch",
    }
    return {
        "ch": control.channel_id,
        "type": type_map.get(control.type, control.type),
        **position(control.tick),
        **control.data,
    }


def _channel_summary(
    comp: IntakeComposition, profile: HardwareProfile, channel_id: str
) -> dict[str, Any]:
    spec = profile.channel(channel_id)
    notes = comp.notes_for(channel_id)
    voice = next(
        (v for v in comp.voices if comp.assignment.channel_id_for(v.key) == channel_id),
        None,
    )
    assignment = comp.assignment.assignments.get(voice.key) if voice else None
    values = [n.pitch for n in notes]
    return {
        "channel_id": channel_id,
        "role": spec.role,
        "type": spec.type,
        "pitched": spec.pitched,
        "midi_track": voice.track_index if voice else None,
        "midi_channel": voice.midi_channel if voice else None,
        "track_name": voice.track_name if voice else None,
        "assignment_method": assignment.method if assignment else "unassigned",
        "n_notes": len(notes),
        "value_min": min(values) if values else None,
        "value_max": max(values) if values else None,
        "max_simultaneous": analysis.max_simultaneous(notes),
        "overlap_count": len(analysis.overlaps(notes, spec.max_simultaneous_notes)),
        "out_of_range_values": analysis.out_of_range(notes, spec),
        "programs": sorted(
            {
                c.data["program"]
                for c in comp.controls
                if c.channel_id == channel_id and c.type == "program_change"
            }
        ),
        "controllers_used": sorted(
            {
                c.data["control"]
                for c in comp.controls
                if c.channel_id == channel_id and c.type == "control_change"
            }
        ),
    }
