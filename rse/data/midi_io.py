"""Parsing source MIDI into an in-memory intake composition.

Controller, program and pitch-bend data is carried through unchanged. NES-MDB
encodes expressive properties such as volume and timbre in controllers, so
stripping them would silently cost synthesis fidelity (proposal 5.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mido

from rse.data.roles import AssignmentResult, SourceVoice, assign_channels
from rse.data.timing import MusicalTime, TempoChange, TimeSignatureChange
from rse.hardware.profile import HardwareProfile

#: Message types kept as expressive control data.
CONTROL_TYPES = frozenset(
    {"control_change", "program_change", "pitchwheel", "aftertouch", "polytouch"}
)

#: Meta messages kept at file level.
TEXT_META_TYPES = frozenset({"marker", "text", "cue_marker", "lyrics"})


class MidiParseError(ValueError):
    """Raised when a file cannot be read as MIDI at all."""


@dataclass(frozen=True)
class Note:
    channel_id: str | None
    tick: int
    dur_ticks: int
    pitch: int
    velocity: int
    track_index: int
    midi_channel: int | None


@dataclass(frozen=True)
class Control:
    channel_id: str | None
    tick: int
    type: str
    data: dict[str, Any]
    track_index: int
    midi_channel: int | None


@dataclass
class IntakeComposition:
    """One parsed source file, before normalization."""

    source_path: Path
    midi_format: int
    ticks_per_beat: int
    n_tracks: int
    time: MusicalTime
    voices: list[SourceVoice]
    assignment: AssignmentResult
    notes: list[Note]
    controls: list[Control]
    text_events: list[dict[str, Any]] = field(default_factory=list)
    key_signatures: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def end_tick(self) -> int:
        note_end = max((n.tick + n.dur_ticks for n in self.notes), default=0)
        control_end = max((c.tick for c in self.controls), default=0)
        return max(note_end, control_end)

    def notes_for(self, channel_id: str) -> list[Note]:
        return [n for n in self.notes if n.channel_id == channel_id]


def parse_midi(path: Path, profile: HardwareProfile) -> IntakeComposition:
    """Read a MIDI file and assign its voices to `profile`'s channels."""
    try:
        # clip=True clamps out-of-range data bytes instead of raising, which
        # keeps a handful of otherwise-usable files in the dataset.
        midi = mido.MidiFile(str(path), clip=True)
    except Exception as exc:  # mido raises several unrelated exception types
        raise MidiParseError(f"{path.name}: {type(exc).__name__}: {exc}") from exc

    warnings: list[str] = []
    tempo_changes: list[TempoChange] = []
    time_signatures: list[TimeSignatureChange] = []
    text_events: list[dict[str, Any]] = []
    key_signatures: list[dict[str, Any]] = []

    track_names: dict[int, str | None] = {}
    instrument_names: dict[int, str | None] = {}
    raw_notes: list[dict[str, Any]] = []
    raw_controls: list[dict[str, Any]] = []
    voice_keys: list[tuple[int, int | None]] = []

    for track_index, track in enumerate(midi.tracks):
        tick = 0
        open_notes: dict[tuple[int | None, int], dict[str, Any]] = {}
        for msg in track:
            tick += msg.time
            if msg.is_meta:
                _collect_meta(
                    msg,
                    tick,
                    track_index,
                    tempo_changes,
                    time_signatures,
                    text_events,
                    key_signatures,
                    track_names,
                    instrument_names,
                )
                continue

            midi_channel = getattr(msg, "channel", None)
            key = (track_index, midi_channel)
            if key not in voice_keys:
                voice_keys.append(key)

            if msg.type == "note_on" and msg.velocity > 0:
                note_key = (midi_channel, msg.note)
                if note_key in open_notes:
                    # A repeated note_on without an intervening note_off: close
                    # the first one here rather than losing it.
                    _close_note(open_notes.pop(note_key), tick, raw_notes, warnings, path)
                open_notes[note_key] = {
                    "tick": tick,
                    "pitch": msg.note,
                    "velocity": msg.velocity,
                    "track_index": track_index,
                    "midi_channel": midi_channel,
                }
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                note_key = (midi_channel, msg.note)
                pending = open_notes.pop(note_key, None)
                if pending is None:
                    warnings.append(
                        f"note_off without note_on (track {track_index}, pitch {msg.note}, "
                        f"tick {tick})"
                    )
                else:
                    _close_note(pending, tick, raw_notes, warnings, path)
            elif msg.type in CONTROL_TYPES:
                raw_controls.append(
                    {
                        "tick": tick,
                        "type": msg.type,
                        "data": _control_data(msg),
                        "track_index": track_index,
                        "midi_channel": midi_channel,
                    }
                )

        for pending in open_notes.values():
            warnings.append(
                f"note left open at end of track {track_index} "
                f"(pitch {pending['pitch']}, tick {pending['tick']}); truncated to end of track"
            )
            _close_note(pending, tick, raw_notes, warnings, path)

    note_counts: dict[tuple[int, int | None], int] = {}
    for note in raw_notes:
        key = (note["track_index"], note["midi_channel"])
        note_counts[key] = note_counts.get(key, 0) + 1
        if key not in voice_keys:
            voice_keys.append(key)

    voices = [
        SourceVoice(
            key=key,
            track_index=key[0],
            midi_channel=key[1],
            track_name=track_names.get(key[0]),
            instrument_name=instrument_names.get(key[0]),
            n_notes=note_counts.get(key, 0),
        )
        for key in voice_keys
    ]
    assignment = assign_channels(voices, profile)
    warnings.extend(assignment.warnings)

    time = MusicalTime(midi.ticks_per_beat, tempo_changes, time_signatures)

    notes = [
        Note(
            channel_id=assignment.channel_id_for((n["track_index"], n["midi_channel"])),
            tick=n["tick"],
            dur_ticks=n["dur_ticks"],
            pitch=n["pitch"],
            velocity=n["velocity"],
            track_index=n["track_index"],
            midi_channel=n["midi_channel"],
        )
        for n in raw_notes
    ]
    notes.sort(key=lambda n: (n.tick, n.channel_id or "", n.pitch))

    controls = [
        Control(
            channel_id=assignment.channel_id_for((c["track_index"], c["midi_channel"])),
            tick=c["tick"],
            type=c["type"],
            data=c["data"],
            track_index=c["track_index"],
            midi_channel=c["midi_channel"],
        )
        for c in raw_controls
    ]
    controls.sort(key=lambda c: (c.tick, c.channel_id or "", c.type))

    return IntakeComposition(
        source_path=path,
        midi_format=midi.type,
        ticks_per_beat=midi.ticks_per_beat,
        n_tracks=len(midi.tracks),
        time=time,
        voices=voices,
        assignment=assignment,
        notes=notes,
        controls=controls,
        text_events=text_events,
        key_signatures=key_signatures,
        warnings=warnings,
    )


def _collect_meta(
    msg: mido.MetaMessage,
    tick: int,
    track_index: int,
    tempo_changes: list[TempoChange],
    time_signatures: list[TimeSignatureChange],
    text_events: list[dict[str, Any]],
    key_signatures: list[dict[str, Any]],
    track_names: dict[int, str | None],
    instrument_names: dict[int, str | None],
) -> None:
    if msg.type == "set_tempo":
        tempo_changes.append(TempoChange(tick, msg.tempo))
    elif msg.type == "time_signature":
        time_signatures.append(TimeSignatureChange(tick, msg.numerator, msg.denominator))
    elif msg.type == "track_name":
        track_names.setdefault(track_index, msg.name)
    elif msg.type == "instrument_name":
        instrument_names.setdefault(track_index, msg.name)
    elif msg.type == "key_signature":
        key_signatures.append({"tick": tick, "key": msg.key})
    elif msg.type in TEXT_META_TYPES:
        value = getattr(msg, "text", None) or getattr(msg, "name", "")
        text_events.append(
            {"tick": tick, "type": msg.type, "text": value, "track_index": track_index}
        )


def _close_note(
    pending: dict[str, Any],
    tick: int,
    out: list[dict[str, Any]],
    warnings: list[str],
    path: Path,
) -> None:
    duration = tick - pending["tick"]
    if duration <= 0:
        warnings.append(
            f"zero-length note dropped (pitch {pending['pitch']}, tick {pending['tick']})"
        )
        return
    out.append({**pending, "dur_ticks": duration})


def _control_data(msg: mido.Message) -> dict[str, Any]:
    if msg.type == "control_change":
        return {"control": msg.control, "value": msg.value}
    if msg.type == "program_change":
        return {"program": msg.program}
    if msg.type == "pitchwheel":
        return {"pitch": msg.pitch}
    if msg.type == "aftertouch":
        return {"value": msg.value}
    if msg.type == "polytouch":
        return {"note": msg.note, "value": msg.value}
    return {}  # pragma: no cover - CONTROL_TYPES is exhaustive
