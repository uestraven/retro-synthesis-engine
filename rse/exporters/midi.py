"""MIDI export.

MIDI is an interchange format and cannot carry every chip-specific instrument
or playback behaviour (proposal 7.4). What it does carry is editable note data,
which is what makes a generated composition usable in a tracker or DAW.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mido

DEFAULT_TICKS_PER_BEAT = 480


def record_to_midi(record: dict[str, Any]) -> mido.MidiFile:
    """Build a MIDI file from an intake record.

    One track per hardware channel, named so that re-parsing the result maps
    the tracks back to the same channels.
    """
    ticks_per_beat = record.get("midi", {}).get("ticks_per_beat") or DEFAULT_TICKS_PER_BEAT
    midi = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)

    # Everything is assembled with absolute ticks in `time` and converted to
    # delta times in one pass at the end; mixing the two is how MIDI writers
    # end up with events in the wrong place.
    meta: list[mido.MetaMessage] = []
    for tempo in record.get("tempo_map", ()) or [{"tick": 0, "us_per_beat": 500_000}]:
        meta.append(
            mido.MetaMessage("set_tempo", tempo=int(tempo["us_per_beat"]), time=tempo["tick"])
        )
    for signature in record.get("time_signatures", ()):
        meta.append(
            mido.MetaMessage(
                "time_signature",
                numerator=signature["numerator"],
                denominator=signature["denominator"],
                time=signature["tick"],
            )
        )
    for marker in record.get("loop_markers", ()):
        meta.append(
            mido.MetaMessage("marker", text=marker.get("text", "loop"), time=marker["tick"])
        )
    meta.sort(key=lambda m: m.time)

    meta_track = mido.MidiTrack()
    meta_track.append(mido.MetaMessage("track_name", name="meta", time=0))
    meta_track.extend(meta)
    _to_delta(meta_track)
    midi.tracks.append(meta_track)

    channel_ids = [channel["channel_id"] for channel in record.get("channels", ())]
    if not channel_ids:
        channel_ids = sorted({e["ch"] for e in record.get("events", ()) if "ch" in e})

    for index, channel_id in enumerate(channel_ids):
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=channel_id, time=0))
        absolute: list[mido.Message] = []
        for event in record.get("events", ()):
            if event.get("ch") != channel_id:
                continue
            absolute.extend(_event_messages(event, index))
        # Note-offs before note-ons at the same tick keeps a monophonic channel
        # looking monophonic when the file is re-parsed.
        absolute.sort(key=lambda m: (m.time, 0 if m.type == "note_off" else 1))
        track.extend(absolute)
        _to_delta(track)
        midi.tracks.append(track)
    return midi


def write_midi(path: Path, record: dict[str, Any]) -> Path:
    """Write an intake record out as a MIDI file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record_to_midi(record).save(str(path))
    return path


def _event_messages(event: dict[str, Any], midi_channel: int) -> list[mido.Message]:
    """Messages for one event, with absolute ticks in `time`."""
    tick = event["tick"]
    event_type = event.get("type")
    if event_type in ("note", "noise"):
        value = event.get("pitch", event.get("value"))
        velocity = max(1, min(127, event.get("velocity", 96)))
        duration = max(1, event.get("dur_ticks", 1))
        return [
            mido.Message("note_on", channel=midi_channel, note=value, velocity=velocity, time=tick),
            mido.Message(
                "note_off", channel=midi_channel, note=value, velocity=0, time=tick + duration
            ),
        ]
    if event_type == "control":
        return [
            mido.Message(
                "control_change",
                channel=midi_channel,
                control=event["control"],
                value=event["value"],
                time=tick,
            )
        ]
    if event_type == "program":
        return [
            mido.Message(
                "program_change", channel=midi_channel, program=event["program"], time=tick
            )
        ]
    if event_type == "pitchwheel":
        return [mido.Message("pitchwheel", channel=midi_channel, pitch=event["pitch"], time=tick)]
    return []


def _to_delta(track: mido.MidiTrack) -> None:
    """Rewrite absolute tick values in `time` as MIDI delta times, in place."""
    cursor = 0
    for message in track:
        absolute = message.time
        message.time = max(0, absolute - cursor)
        cursor = absolute
