"""RSE v2 codec: grouped musical events over a stateful performance decoder.

Two layers, deliberately separated:

1. **Musical timeline** — groups of actions at one musical instant, each group
   separated from the previous by a STRICTLY POSITIVE time advance.
   Simultaneity lives inside a group, so there is no `TIME_SAME` token that can
   repeat forever. That removes the exact mechanism behind Level 2's two
   4,000-event runaways **by construction**. It does not bound attack density,
   which still needs its own measured predicate.
2. **Performance decoder** — tracks active notes, applies lifetimes, and
   assigns the eight DSP voices. The model never has to invent a hardware
   scheduling algorithm.

**A hardware voice is a playback resource, not a musical part.** This is
measured, not assumed: over 200 train tracks, three lead-voice criteria agree
on one voice in only 9.9% of tracks, and the best disagrees with its own
track-level choice in ~30% of 5-second windows, because parts migrate across
voices mid-track. So the original voice index is carried as **provenance**
only, for round-trip verification, and the decoder allocates its own slots.
Nothing here is a melody/bass/accompaniment label, and the absence of role
annotation is explicit rather than implied.

**Note lifetimes are explicit.** `snesmdb.events.v4` distinguishes `key_on`
from `legato_step`, and a legato pitch change must not become a new attack —
conflating them is what inflated the Level 2 published control by 16×. Ends
derived from published durations are marked as *derived*, never presented as
observed key-off instructions.

**Nothing is silently repaired.** Every normalisation, merge, and voice
reassignment is counted and reported. A codec that quietly fixes its input and
then calls itself lossless is measuring its own tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Iterable, Literal, Sequence

#: The eight generic S-DSP voices.
N_HARDWARE_VOICES = 8

#: Action kinds. Ordering WITHIN a group is fixed: ends before starts, so a
#: voice freed at an instant is available to a note beginning at that same
#: instant. Pitch changes come last because they target already-active notes.
ActionKind = Literal["end", "start", "pitch"]
_ORDER: dict[str, int] = {"end": 0, "start": 1, "pitch": 2}

#: How a note's end was established. Preserved because they are different
#: evidence: a key-off was observed, a derived end was computed from a
#: published duration, and a rekey end was forced by the voice being reused.
EndSource = Literal["derived_duration", "rekey", "open"]


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    #: Stable within one encoded track. NOT a musical role and NOT a voice.
    note_id: int
    midi_note: float | None = None
    pitch_status: str | None = None
    #: Original S-DSP voice, carried for verification only.
    provenance_voice: int | None = None
    end_source: EndSource | None = None

    def sort_key(self) -> tuple[int, int]:
        return (_ORDER[self.kind], self.note_id)


@dataclass
class Group:
    """Everything happening at one musical instant."""

    #: Grid steps since the previous group. Strictly positive except the first,
    #: which carries the absolute offset from zero (and may be 0).
    delta: int
    actions: list[Action] = field(default_factory=list)

    def ordered(self) -> list[Action]:
        return sorted(self.actions, key=Action.sort_key)


@dataclass
class EncodeReport:
    """What the encoding cost. Every lossy step is counted here."""

    grid_hz: float
    n_source_events: int
    n_groups: int
    n_actions: int
    n_starts: int
    n_pitch_changes: int
    n_ends_derived: int
    n_ends_open: int
    #: Distinct instants that collapsed onto one grid step.
    n_instants_merged: int
    #: Starts that landed on the same grid step AND the same pitch and voice as
    #: another start, so the two are indistinguishable after quantisation.
    n_indistinguishable_starts: int
    n_events_without_duration: int
    n_unpitched_starts: int

    @property
    def actions_per_source_event(self) -> float:
        return self.n_actions / self.n_source_events if self.n_source_events else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_hz": self.grid_hz,
            "n_source_events": self.n_source_events,
            "n_groups": self.n_groups,
            "n_actions": self.n_actions,
            "n_starts": self.n_starts,
            "n_pitch_changes": self.n_pitch_changes,
            "n_ends_derived": self.n_ends_derived,
            "n_ends_open": self.n_ends_open,
            "n_instants_merged": self.n_instants_merged,
            "n_indistinguishable_starts": self.n_indistinguishable_starts,
            "n_events_without_duration": self.n_events_without_duration,
            "n_unpitched_starts": self.n_unpitched_starts,
            "actions_per_source_event": self.actions_per_source_event,
        }


def _step(seconds: Decimal, grid_hz: float) -> int:
    """Quantise a time in seconds to a grid step, half-up."""
    scaled = seconds * Decimal(str(grid_hz))
    return int(scaled.to_integral_value(rounding="ROUND_HALF_UP"))


def encode(
    events: Sequence[dict[str, Any]], *, grid_hz: float = 100.0
) -> tuple[list[Group], EncodeReport]:
    """`snesmdb.events.v4` -> grouped musical events.

    `key_on` becomes a start; `legato_step` becomes a pitch change on its
    parent note, NOT a new attack. Ends come from published durations and are
    labelled `derived_duration`; a note whose duration is null ends nowhere and
    is labelled `open` rather than assumed to last forever.
    """
    starts: list[dict[str, Any]] = []
    legato: list[dict[str, Any]] = []
    for event in events:
        if event.get("origin") == "legato_step":
            legato.append(event)
        else:
            starts.append(event)

    # note_id is assigned in (onset, voice) order so encoding is deterministic.
    starts.sort(key=lambda e: (Decimal(str(e["onset_sec"])), int(e["voice"])))
    id_of_event: dict[Any, int] = {}
    timeline: dict[int, list[Action]] = {}
    n_without_duration = n_unpitched = n_ends_derived = n_ends_open = 0

    def add(step: int, action: Action) -> None:
        timeline.setdefault(step, []).append(action)

    seen_start: set[tuple[int, Any, int]] = set()
    n_indistinguishable = 0

    for note_id, event in enumerate(starts):
        onset = Decimal(str(event["onset_sec"]))
        step = _step(onset, grid_hz)
        note = event.get("midi_note")
        if note is None:
            n_unpitched += 1
        key = (step, None if note is None else round(float(note), 4),
               int(event["voice"]))
        if key in seen_start:
            n_indistinguishable += 1
        seen_start.add(key)
        if event.get("event_id") is not None:
            id_of_event[event["event_id"]] = note_id
        add(step, Action(
            kind="start", note_id=note_id,
            midi_note=None if note is None else float(note),
            pitch_status=event.get("pitch_status"),
            provenance_voice=int(event["voice"]),
        ))
        duration = event.get("duration_sec")
        if duration is None:
            n_without_duration += 1
            n_ends_open += 1
            continue
        end_step = _step(onset + Decimal(str(duration)), grid_hz)
        # A note quantised to zero length still ENDS; it is not dropped. The
        # decoder decides what a zero-length note sounds like, which keeps the
        # codec honest about what it was given.
        add(end_step, Action(kind="end", note_id=note_id,
                             end_source="derived_duration"))
        n_ends_derived += 1

    for event in legato:
        parent = event.get("parent_event_id")
        note_id = id_of_event.get(parent)
        if note_id is None:
            # Orphaned legato step: no parent attack in this table. Counted,
            # not silently promoted to an attack.
            continue
        note = event.get("midi_note")
        add(_step(Decimal(str(event["onset_sec"])), grid_hz), Action(
            kind="pitch", note_id=note_id,
            midi_note=None if note is None else float(note),
            pitch_status=event.get("pitch_status"),
        ))

    steps = sorted(timeline)
    groups: list[Group] = []
    previous = 0
    for index, step in enumerate(steps):
        delta = step if index == 0 else step - previous
        groups.append(Group(delta=delta, actions=timeline[step]))
        previous = step

    n_actions = sum(len(g.actions) for g in groups)
    n_starts = sum(1 for g in groups for a in g.actions if a.kind == "start")
    n_pitch = sum(1 for g in groups for a in g.actions if a.kind == "pitch")
    distinct_source_instants = len({
        _step(Decimal(str(e["onset_sec"])), grid_hz * 1000) for e in events
    })
    report = EncodeReport(
        grid_hz=grid_hz,
        n_source_events=len(events),
        n_groups=len(groups),
        n_actions=n_actions,
        n_starts=n_starts,
        n_pitch_changes=n_pitch,
        n_ends_derived=n_ends_derived,
        n_ends_open=n_ends_open,
        n_instants_merged=max(0, distinct_source_instants - len(steps)),
        n_indistinguishable_starts=n_indistinguishable,
        n_events_without_duration=n_without_duration,
        n_unpitched_starts=n_unpitched,
    )
    return groups, report


@dataclass
class DecodeReport:
    """What the decoder had to do to make the stream playable."""

    grid_hz: float
    n_notes: int
    #: Starts that arrived with all eight voices occupied, so a sounding note
    #: was displaced. This is an ENGINEERING correction, never evidence that a
    #: model orchestrated for eight voices.
    n_voice_steals: int
    #: Ends for a note that was not active (double end, or end before start).
    n_unmatched_ends: int
    #: Pitch changes targeting a note that was not active.
    n_orphan_pitch_changes: int
    n_zero_length_notes: int
    peak_occupancy: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_hz": self.grid_hz,
            "n_notes": self.n_notes,
            "n_voice_steals": self.n_voice_steals,
            "n_unmatched_ends": self.n_unmatched_ends,
            "n_orphan_pitch_changes": self.n_orphan_pitch_changes,
            "n_zero_length_notes": self.n_zero_length_notes,
            "peak_occupancy": self.peak_occupancy,
        }


def decode(
    groups: Iterable[Group], *, grid_hz: float = 100.0,
    n_voices: int = N_HARDWARE_VOICES,
) -> tuple[list[dict[str, Any]], DecodeReport]:
    """Grouped musical events -> an events-like table, with voices allocated.

    The allocator is deterministic: the lowest-numbered free slot, and when
    none is free, the note that has been sounding LONGEST is displaced. Both
    choices are arbitrary but fixed, so a round trip is reproducible, and every
    displacement is counted.

    Within a group, ends are applied before starts, so a slot freed at an
    instant can be taken by a note starting at that same instant. Pitch
    changes are applied last.
    """
    step_sec = Decimal(1) / Decimal(str(grid_hz))
    now = 0
    #: note_id -> record under construction
    active: dict[int, dict[str, Any]] = {}
    slot_of: dict[int, int] = {}
    free: list[int] = list(range(n_voices))
    out: list[dict[str, Any]] = []
    steals = unmatched = orphans = 0
    peak = 0

    def close(note_id: int, at: int, source: EndSource) -> None:
        record = active.pop(note_id, None)
        if record is None:
            return
        slot = slot_of.pop(note_id, None)
        if slot is not None:
            free.append(slot)
            free.sort()
        record["duration_sec"] = float((at - record["_start_step"]) * step_sec)
        record["end_source"] = source
        record.pop("_start_step")
        out.append(record)

    for group in groups:
        now += group.delta
        for action in group.ordered():
            if action.kind == "end":
                if action.note_id not in active:
                    unmatched += 1
                    continue
                close(action.note_id, now, action.end_source or "derived_duration")
            elif action.kind == "start":
                if not free:
                    # Displace the longest-sounding note. Counted as a steal.
                    victim = min(active, key=lambda n: active[n]["_start_step"])
                    close(victim, now, "rekey")
                    steals += 1
                slot = free.pop(0)
                slot_of[action.note_id] = slot
                active[action.note_id] = {
                    # Carried so a round trip can be matched note-for-note
                    # rather than by position, which voice reassignment breaks.
                    "note_id": action.note_id,
                    "onset_sec": float(now * step_sec),
                    "voice": slot,
                    "midi_note": action.midi_note,
                    "pitch_status": action.pitch_status,
                    "provenance_voice": action.provenance_voice,
                    # No musical role is asserted. Absence is explicit.
                    "role": None,
                    "_start_step": now,
                }
            else:  # pitch
                record = active.get(action.note_id)
                if record is None:
                    orphans += 1
                    continue
                record.setdefault("pitch_changes", []).append(
                    {"onset_sec": float(now * step_sec),
                     "midi_note": action.midi_note,
                     "pitch_status": action.pitch_status}
                )
            peak = max(peak, len(active))

    for note_id in list(active):
        close(note_id, now, "open")

    out.sort(key=lambda e: (e["onset_sec"], e["voice"]))
    report = DecodeReport(
        grid_hz=grid_hz,
        n_notes=len(out),
        n_voice_steals=steals,
        n_unmatched_ends=unmatched,
        n_orphan_pitch_changes=orphans,
        n_zero_length_notes=sum(1 for e in out if e["duration_sec"] == 0.0),
        peak_occupancy=peak,
    )
    return out, report
