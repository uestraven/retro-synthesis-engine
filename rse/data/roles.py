"""Mapping source MIDI tracks onto hardware channels.

A General MIDI program number says nothing about which console channel a part
belongs on (proposal 6.2), so the mapping is inferred from track naming first
and positional order second. Every assignment records *how* it was decided, and
anything that could not be assigned is reported rather than silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

from rse.hardware.profile import HardwareProfile

#: How a voice was matched to a channel, most trustworthy first.
BY_TRACK_NAME = "track_name"
BY_INSTRUMENT_NAME = "instrument_name"
BY_ORDINAL = "ordinal"
UNASSIGNED = "unassigned"


@dataclass(frozen=True)
class SourceVoice:
    """A distinct `(track, MIDI channel)` stream in a source file."""

    key: tuple[int, int | None]
    track_index: int
    midi_channel: int | None
    track_name: str | None
    instrument_name: str | None
    n_notes: int


@dataclass(frozen=True)
class ChannelAssignment:
    channel_id: str | None
    role: str | None
    method: str

    @property
    def assigned(self) -> bool:
        return self.channel_id is not None


@dataclass
class AssignmentResult:
    assignments: dict[tuple[int, int | None], ChannelAssignment]
    warnings: list[str]

    def channel_id_for(self, key: tuple[int, int | None]) -> str | None:
        assignment = self.assignments.get(key)
        return assignment.channel_id if assignment else None


def assign_channels(voices: list[SourceVoice], profile: HardwareProfile) -> AssignmentResult:
    """Assign each source voice to at most one hardware channel."""
    assignments: dict[tuple[int, int | None], ChannelAssignment] = {}
    warnings: list[str] = []
    claimed: set[str] = set()

    # Pass 1: explicit names. A name match is the only evidence strong enough to
    # reorder channels relative to track order, so it wins.
    for voice in voices:
        spec = profile.match_track_name(voice.track_name)
        method = BY_TRACK_NAME
        if spec is None:
            spec = profile.match_track_name(voice.instrument_name)
            method = BY_INSTRUMENT_NAME
        if spec is None:
            continue
        if spec.channel_id in claimed:
            warnings.append(
                f"track {voice.track_index} ({voice.track_name!r}) also names channel "
                f"{spec.channel_id}, which is already taken; falling back to ordinal order"
            )
            continue
        claimed.add(spec.channel_id)
        assignments[voice.key] = ChannelAssignment(spec.channel_id, spec.role, method)

    # Pass 2: everything still unassigned that actually plays notes, in track
    # order, onto whatever channels are left in profile order.
    free = [ch for ch in profile.channels if ch.channel_id not in claimed]
    for voice in voices:
        if voice.key in assignments:
            continue
        if voice.n_notes == 0:
            continue
        if not free:
            assignments[voice.key] = ChannelAssignment(None, None, UNASSIGNED)
            warnings.append(
                f"track {voice.track_index} (channel {voice.midi_channel}, "
                f"{voice.n_notes} notes) has no free {profile.profile_id} channel and was dropped"
            )
            continue
        spec = free.pop(0)
        claimed.add(spec.channel_id)
        assignments[voice.key] = ChannelAssignment(spec.channel_id, spec.role, BY_ORDINAL)

    # Pass 3: note-free voices still get a record so their controller data can be
    # attributed if they were named, and so the report can account for them.
    for voice in voices:
        assignments.setdefault(voice.key, ChannelAssignment(None, None, UNASSIGNED))

    unclaimed = [ch.channel_id for ch in profile.channels if ch.channel_id not in claimed]
    if unclaimed:
        warnings.append(f"no source voice mapped to channel(s): {', '.join(unclaimed)}")

    return AssignmentResult(assignments=assignments, warnings=warnings)
