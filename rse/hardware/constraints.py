"""The hardware constraint engine.

Proposal 3.2 is specific that overlapping notes must not simply be discarded
after generation. So this module does two separate jobs, and keeps them
separate:

- **Prevention.** `ChannelAllocator` tracks occupancy so a generator can ask
  whether a note fits *before* committing to it.
- **Repair.** `ConstraintEngine.apply` rewrites a composition that already
  violates the profile, preferring to shorten or transpose rather than drop,
  and records every change it makes.

`validate` is read-only and answers "is this playable"; `apply` answers "make
this playable and tell me what it cost". Phase 1 measures violations, this
phase resolves them.
"""

from __future__ import annotations

from bisect import bisect_left, insort
from dataclasses import dataclass, field
from typing import Any

from rse.events.schema import Composition, Event, Noise, Note
from rse.hardware.profile import ChannelSpec, HardwareProfile

#: Modification kinds recorded in a validation report.
SHORTENED = "shortened"
TRANSPOSED = "transposed"
CLAMPED = "clamped"
REJECTED = "rejected"
RETIMED = "retimed"


@dataclass(frozen=True)
class Violation:
    """Something the target hardware could not play."""

    kind: str
    channel_id: str
    tick: int
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "channel_id": self.channel_id,
            "tick": self.tick,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class Modification:
    """A change the engine made to resolve a violation."""

    kind: str
    channel_id: str
    tick: int
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "channel_id": self.channel_id,
            "tick": self.tick,
            "detail": self.detail,
        }


@dataclass
class ValidationReport:
    """Everything the engine found and everything it changed."""

    profile_id: str = ""
    n_events_before: int = 0
    n_events_after: int = 0
    violations: list[Violation] = field(default_factory=list)
    modifications: list[Modification] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when nothing had to be changed."""
        return not self.modifications

    @property
    def n_rejected(self) -> int:
        return sum(1 for m in self.modifications if m.kind == REJECTED)

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for modification in self.modifications:
            counts[modification.kind] = counts.get(modification.kind, 0) + 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "profile_id": self.profile_id,
            "n_events_before": self.n_events_before,
            "n_events_after": self.n_events_after,
            "n_violations": len(self.violations),
            "n_modifications": len(self.modifications),
            "n_rejected": self.n_rejected,
            "counts": self.counts(),
            "violations": [v.to_dict() for v in self.violations[:50]],
            "modifications": [m.to_dict() for m in self.modifications[:50]],
        }


class ChannelAllocator:
    """Tracks what each channel is playing, so conflicts can be prevented.

    Intervals are half-open: a note ending exactly where the next begins is
    legal monophony on real hardware, not an overlap.
    """

    def __init__(self, profile: HardwareProfile) -> None:
        self.profile = profile
        self._spans: dict[str, list[tuple[int, int]]] = {
            channel_id: [] for channel_id in profile.channel_ids
        }

    def spans(self, channel_id: str) -> list[tuple[int, int]]:
        return list(self._spans[channel_id])

    def capacity(self, channel_id: str) -> int:
        return self.profile.channel(channel_id).max_simultaneous_notes

    def overlapping(self, channel_id: str, tick: int, duration: int) -> list[tuple[int, int]]:
        """Spans already occupying any part of `[tick, tick + duration)`."""
        if channel_id not in self._spans:
            raise KeyError(f"no channel {channel_id!r} in profile {self.profile.profile_id!r}")
        end = tick + max(0, duration)
        spans = self._spans[channel_id]
        index = max(0, bisect_left(spans, (tick, tick)) - 1)
        found = []
        for start, stop in spans[index:]:
            if start >= end:
                break
            if stop > tick:
                found.append((start, stop))
        return found

    def can_place(self, channel_id: str, tick: int, duration: int) -> bool:
        if duration <= 0:
            return False
        limit = self.capacity(channel_id)
        return len(self.overlapping(channel_id, tick, duration)) < limit

    def place(self, channel_id: str, tick: int, duration: int) -> bool:
        """Claim a span if it fits. Returns whether it was claimed."""
        if not self.can_place(channel_id, tick, duration):
            return False
        insort(self._spans[channel_id], (tick, tick + duration))
        return True

    def room_until(self, channel_id: str, tick: int) -> int | None:
        """How long a note starting at `tick` could last before it collides.

        `None` means nothing is in the way. This is what lets a caller shorten
        a note to fit instead of dropping it.
        """
        spans = self._spans[channel_id]
        nearest = None
        for start, _stop in spans:
            if start > tick and (nearest is None or start < nearest):
                nearest = start
        return None if nearest is None else nearest - tick


class ConstraintEngine:
    """Checks and repairs compositions against a hardware profile."""

    def __init__(self, profile: HardwareProfile, *, min_duration: int = 1) -> None:
        self.profile = profile
        self.min_duration = min_duration

    # -- read-only --------------------------------------------------------

    def validate(self, composition: Composition) -> list[Violation]:
        """Everything the profile cannot play, without changing anything."""
        violations: list[Violation] = []
        if composition.hardware_profile != self.profile.profile_id:
            violations.append(
                Violation(
                    "profile_mismatch",
                    "-",
                    0,
                    f"composition targets {composition.hardware_profile!r}, "
                    f"engine is {self.profile.profile_id!r}",
                )
            )

        allocator = ChannelAllocator(self.profile)
        for event in composition.sorted_events():
            if event.channel_id not in self.profile.channel_ids:
                violations.append(
                    Violation(
                        "unknown_channel",
                        event.channel_id,
                        event.tick,
                        f"no channel {event.channel_id!r} in this profile",
                    )
                )
                continue
            if not isinstance(event, Note | Noise):
                continue
            spec = self.profile.channel(event.channel_id)
            value = _value_of(event)

            if not spec.in_range(value):
                violations.append(
                    Violation(
                        "out_of_range",
                        event.channel_id,
                        event.tick,
                        f"{'pitch' if spec.pitched else 'value'} {value} outside "
                        f"{spec.value_min}-{spec.value_max}",
                    )
                )
            if event.duration <= 0:
                violations.append(
                    Violation(
                        "zero_duration", event.channel_id, event.tick, "duration is not positive"
                    )
                )
                continue
            if not allocator.can_place(event.channel_id, event.tick, event.duration):
                violations.append(
                    Violation(
                        "polyphony",
                        event.channel_id,
                        event.tick,
                        f"channel allows {spec.max_simultaneous_notes} simultaneous note(s)",
                    )
                )
            allocator.place(event.channel_id, event.tick, event.duration)

        if composition.loop_end_tick is not None:
            for event in composition.sounding_events():
                if event.end_tick > composition.loop_end_tick:
                    violations.append(
                        Violation(
                            "past_loop_end",
                            event.channel_id,
                            event.tick,
                            f"ends at {event.end_tick}, past the loop end at "
                            f"{composition.loop_end_tick}",
                        )
                    )
        return violations

    # -- repair -----------------------------------------------------------

    def apply(self, composition: Composition) -> tuple[Composition, ValidationReport]:
        """Return a playable copy of `composition`, plus what it cost.

        Order of preference, following proposal 3.2: keep the event by
        transposing or shortening it; steal the channel only when two events
        want the same instant; reject only what cannot be represented at all.
        """
        report = ValidationReport(profile_id=self.profile.profile_id)
        report.violations = self.validate(composition)
        report.n_events_before = len(composition.events)

        kept: list[Event] = []
        contending: dict[str, list[Note | Noise]] = {
            channel_id: [] for channel_id in self.profile.channel_ids
        }

        # Pass one: make each event individually representable.
        for event in composition.sorted_events():
            if event.channel_id not in self.profile.channel_ids:
                report.modifications.append(
                    Modification(
                        REJECTED,
                        event.channel_id,
                        event.tick,
                        f"no channel {event.channel_id!r} in this profile",
                    )
                )
                continue
            if not isinstance(event, Note | Noise):
                kept.append(event)  # controllers and effects never contend
                continue

            resolved = self._fit_value(event, report)
            if resolved is None:
                continue
            resolved = self._fit_loop(resolved, composition, report)
            if resolved is None:
                continue
            contending[resolved.channel_id].append(resolved)

        # Pass two: resolve contention within each channel, where the fix
        # depends on the note's neighbours rather than on the note alone.
        for channel_id, events in contending.items():
            kept.extend(self._resolve_channel(channel_id, events, report))

        result = Composition(
            hardware_profile=self.profile.profile_id,
            ticks_per_beat=composition.ticks_per_beat,
            events=kept,
            tempo_map=list(composition.tempo_map),
            time_signatures=list(composition.time_signatures),
            loop_start_tick=composition.loop_start_tick,
            loop_end_tick=composition.loop_end_tick,
            source_metadata=dict(composition.source_metadata),
        )
        result.normalize()
        report.n_events_after = len(result.events)
        return result, report

    def _fit_value(self, event: Note | Noise, report: ValidationReport) -> Note | Noise | None:
        """Bring a pitch or hardware value into the channel's range."""
        spec = self.profile.channel(event.channel_id)
        value = _value_of(event)
        if spec.in_range(value):
            return event

        if spec.pitched:
            # Octave transposition preserves the note's musical identity, so
            # try it before giving up on the note.
            shifted = value
            while spec.value_min is not None and shifted < spec.value_min:
                shifted += 12
            while spec.value_max is not None and shifted > spec.value_max:
                shifted -= 12
            if spec.in_range(shifted):
                report.modifications.append(
                    Modification(
                        TRANSPOSED,
                        event.channel_id,
                        event.tick,
                        f"pitch {value} -> {shifted} to fit {spec.value_min}-{spec.value_max}",
                    )
                )
                return _with_value(event, shifted)
            report.modifications.append(
                Modification(
                    REJECTED,
                    event.channel_id,
                    event.tick,
                    f"pitch {value} cannot be transposed into {spec.value_min}-{spec.value_max}",
                )
            )
            return None

        # An unpitched value selects a timbre, so the nearest valid one is a
        # reasonable substitute; there is no octave to move it by.
        clamped = max(spec.value_min or value, min(spec.value_max or value, value))
        report.modifications.append(
            Modification(
                CLAMPED,
                event.channel_id,
                event.tick,
                f"value {value} -> {clamped} to fit {spec.value_min}-{spec.value_max}",
            )
        )
        return _with_value(event, clamped)

    def _fit_loop(
        self, event: Note | Noise, composition: Composition, report: ValidationReport
    ) -> Note | Noise | None:
        """Stop notes bleeding over the loop boundary."""
        end = composition.loop_end_tick
        if end is None or event.end_tick <= end:
            return event
        if event.tick >= end:
            report.modifications.append(
                Modification(
                    REJECTED,
                    event.channel_id,
                    event.tick,
                    f"starts at or after the loop end at {end}",
                )
            )
            return None
        new_duration = end - event.tick
        report.modifications.append(
            Modification(
                SHORTENED,
                event.channel_id,
                event.tick,
                f"duration {event.duration} -> {new_duration} to end at the loop boundary",
            )
        )
        return _with_duration(event, new_duration)

    def _resolve_channel(
        self, channel_id: str, events: list[Note | Noise], report: ValidationReport
    ) -> list[Note | Noise]:
        """Fit a channel's notes within its simultaneous-note limit.

        For a monophonic channel the fix is voice stealing in the direction the
        hardware actually works: a new note retriggers the channel, so the note
        already sounding is truncated to the newcomer's onset. Shortening the
        incoming note instead would be backwards, and dropping it loses music
        that the console could have played.
        """
        capacity = self.profile.channel(channel_id).max_simultaneous_notes
        ordered = sorted(events, key=lambda e: (e.tick, -e.velocity, -_value_of(e)))
        kept: list[Note | Noise] = []

        for event in ordered:
            if event.duration < self.min_duration:
                report.modifications.append(
                    Modification(
                        REJECTED,
                        channel_id,
                        event.tick,
                        f"duration {event.duration} is below the minimum of {self.min_duration}",
                    )
                )
                continue

            sounding = [k for k in kept if k.end_tick > event.tick]
            if len(sounding) < capacity:
                kept.append(event)
                continue

            # The channel is full. Truncate whichever sounding note started
            # earliest; if none can yield, this note cannot be represented.
            stealable = [
                (index, k)
                for index, k in enumerate(kept)
                if k.end_tick > event.tick and k.tick < event.tick
            ]
            if not stealable:
                report.modifications.append(
                    Modification(
                        REJECTED,
                        channel_id,
                        event.tick,
                        f"another note already starts here and the channel allows "
                        f"{capacity} note(s) at once",
                    )
                )
                continue

            index, victim = min(stealable, key=lambda pair: pair[1].tick)
            new_duration = event.tick - victim.tick
            if new_duration < self.min_duration:
                report.modifications.append(
                    Modification(
                        REJECTED,
                        channel_id,
                        event.tick,
                        "truncating the sounding note would leave it shorter than "
                        f"{self.min_duration} tick(s)",
                    )
                )
                continue
            kept[index] = _with_duration(victim, new_duration)
            report.modifications.append(
                Modification(
                    SHORTENED,
                    channel_id,
                    victim.tick,
                    f"duration {victim.duration} -> {new_duration}; the note at "
                    f"{event.tick} retriggers the channel",
                )
            )
            kept.append(event)
        return kept


def _value_of(event: Note | Noise) -> int:
    return event.pitch if isinstance(event, Note) else event.value


def _with_value(event: Note | Noise, value: int) -> Note | Noise:
    from dataclasses import replace

    if isinstance(event, Note):
        return replace(event, pitch=value)
    return replace(event, value=value)


def _with_duration(event: Note | Noise, duration: int) -> Note | Noise:
    from dataclasses import replace

    return replace(event, duration=duration)


def channel_priority(spec: ChannelSpec) -> int:
    """Rank a channel by how much its material matters when stealing voices.

    Melody first, then bass, then harmony, then percussion — the order
    proposal 3.2 asks for. Used when two events contend for one instant and
    one of them has to go.
    """
    return {
        "pulse1": 100,
        "triangle": 90,
        "pulse2": 80,
        "noise": 70,
    }.get(spec.role, 50)
