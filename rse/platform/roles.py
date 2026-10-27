"""Musical parts, and the voices that happen to play them.

On NES these are the same thing. The composer writes `channel_id="P1"` in
`render_melody` and `channel_id="TR"` in `render_bass`, and NES channel
literals appear in seventeen modules. That is correct for a console with four
fixed, non-interchangeable channels -- the triangle *is* the bass voice,
because it cannot be anything else.

It is wrong for the SNES, whose eight S-DSP voices are interchangeable. A part
is a musical idea; a voice is a slot that plays one note at a time. The same
voice carries the lead in one bar and a percussion hit in the next, a part may
need two voices at once for a doubled line, and a note's release tail keeps
occupying its voice after the next note has started somewhere else.

Nothing here touches NES. `NES_VOICES` is a fixed one-to-one map that
reproduces the existing assignment exactly, so the NES path keeps writing to
its own channels and its baselines cannot move. The abstraction is additive:
platforms that need allocation get it, and the platform that does not, does
not pay for it.

Allocation is deterministic and every displacement is reported. A silently
dropped note is the defect this module exists to make impossible -- when eight
voices are not enough, the arranger has to know what it lost and why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Role(StrEnum):
    """What a part is for, independent of what plays it."""

    LEAD = "lead"
    COUNTERMELODY = "countermelody"
    HARMONY = "harmony"
    PAD = "pad"
    BASS = "bass"
    ARPEGGIO = "arpeggio"
    PERCUSSION = "percussion"
    ACCENT = "accent"

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.value


#: Default allocation priority, most important first.
#:
#: The melody and the harmonic foundation survive; decoration is what goes.
#: A style or a passage may override this -- a percussion-driven battle
#: section may well rank drums above harmony -- so it is a default and not a
#: rule.
DEFAULT_PRIORITY: tuple[Role, ...] = (
    Role.LEAD,
    Role.BASS,
    Role.PERCUSSION,
    Role.HARMONY,
    Role.PAD,
    Role.ARPEGGIO,
    Role.COUNTERMELODY,
    Role.ACCENT,
)

#: NES: one role per channel, fixed, because the hardware is fixed. This is
#: the compatibility path -- it reproduces the existing assignment exactly.
NES_VOICES: dict[Role, str] = {
    Role.LEAD: "P1",
    Role.HARMONY: "P2",
    Role.COUNTERMELODY: "P2",
    Role.PAD: "P2",
    Role.ARPEGGIO: "P2",
    Role.BASS: "TR",
    Role.PERCUSSION: "NO",
    Role.ACCENT: "NO",
}


@dataclass(frozen=True)
class PartNote:
    """A note belonging to a musical part, with no voice assigned yet."""

    role: Role
    tick: int
    duration: int
    #: Pitch for a pitched part; a sample or period index for percussion.
    value: int
    velocity: int = 100
    instrument_id: str | None = None
    #: Ticks the sound continues after `duration` -- an envelope release. It
    #: keeps occupying the voice, which is the thing a naive allocator misses.
    release: int = 0
    #: Overrides the role's default priority for this note alone.
    priority: int | None = None
    #: Whether `value` is a musical pitch. False for percussion, where it
    #: selects an instrument instead -- on NES it is a noise period index, and
    #: treating 1, 6 and 12 as MIDI notes pitched a kick down nearly three
    #: octaves and stretched a 70 ms hat into a 6-second drone.
    pitched: bool = True

    @property
    def end_tick(self) -> int:
        """When the note stops sounding, release included."""
        return self.tick + self.duration + self.release

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "tick": self.tick,
            "duration": self.duration,
            "value": self.value,
            "velocity": self.velocity,
            "instrument_id": self.instrument_id,
            "release": self.release,
            "pitched": self.pitched,
        }


@dataclass(frozen=True)
class VoiceAssignment:
    """One part note, placed on a physical voice."""

    note: PartNote
    voice: str

    def to_dict(self) -> dict[str, Any]:
        return {"voice": self.voice, **self.note.to_dict()}


@dataclass(frozen=True)
class Displacement:
    """A note that could not be placed, or one that was cut short for it.

    Reported rather than silently dropped. `stolen_from` is set when a
    lower-priority note was evicted to make room; `dropped` when the incoming
    note itself lost.
    """

    tick: int
    role: Role
    value: int
    reason: str
    #: The other party in the collision. For `stolen` it is the role that
    #: took the voice; for `dropped` it is the role that kept it.
    taken_by: Role | None = None
    #: Ticks of sound lost, for a note cut short rather than removed.
    truncated_ticks: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "role": self.role.value,
            "value": self.value,
            "reason": self.reason,
            "taken_by": self.taken_by.value if self.taken_by else None,
            "truncated_ticks": self.truncated_ticks,
        }


@dataclass
class AllocationReport:
    """What the allocator did, in full."""

    assignments: list[VoiceAssignment] = field(default_factory=list)
    displacements: list[Displacement] = field(default_factory=list)
    peak_voices: int = 0
    voice_count: int = 0

    @property
    def dropped(self) -> int:
        return sum(1 for d in self.displacements if d.reason == "dropped")

    @property
    def stolen(self) -> int:
        return sum(1 for d in self.displacements if d.reason == "stolen")

    def to_dict(self) -> dict[str, Any]:
        return {
            "voice_count": self.voice_count,
            "peak_voices": self.peak_voices,
            "assigned": len(self.assignments),
            "dropped": self.dropped,
            "stolen": self.stolen,
            "displacements": [d.to_dict() for d in self.displacements],
        }

    def summary(self) -> str:
        return (
            f"{len(self.assignments)} notes on {self.voice_count} voices, "
            f"peak {self.peak_voices} concurrent, "
            f"{self.stolen} stolen, {self.dropped} dropped"
        )


class VoiceAllocator:
    """Assigns part notes to a fixed pool of voices, deterministically.

    Greedy in tick order, which is what a hardware driver does: it cannot see
    the future. A free voice is preferred; failing that, the lowest-priority
    sounding note is evicted if the incoming note outranks it, and otherwise
    the incoming note is dropped. Either way the decision is recorded.

    `reserved` holds voices back for sound effects, which is the game-ready
    mode: music gets `voice_count - reserved`.
    """

    def __init__(
        self,
        voices: list[str],
        *,
        priority: tuple[Role, ...] = DEFAULT_PRIORITY,
        reserved: int = 0,
    ) -> None:
        if reserved >= len(voices):
            raise ValueError(
                f"reserving {reserved} of {len(voices)} voices leaves none for music"
            )
        self.all_voices = list(voices)
        self.voices = list(voices[: len(voices) - reserved]) if reserved else list(voices)
        self.reserved = reserved
        self.rank = {role: index for index, role in enumerate(priority)}

    def _rank(self, note: PartNote) -> int:
        if note.priority is not None:
            return note.priority
        return self.rank.get(note.role, len(self.rank))

    def allocate(self, notes: list[PartNote]) -> AllocationReport:
        report = AllocationReport(voice_count=len(self.voices))
        # voice -> (end_tick, rank, note). Empty means free.
        busy: dict[str, tuple[int, int, PartNote]] = {}
        # Sorted by tick, then by priority so the important note of a
        # simultaneous group claims a voice first, then by value for a stable
        # order regardless of input ordering.
        ordered = sorted(
            notes, key=lambda n: (n.tick, self._rank(n), n.value, n.role.value)
        )
        for note in ordered:
            for voice, (end, _rank, _held) in list(busy.items()):
                if end <= note.tick:
                    del busy[voice]

            free = [v for v in self.voices if v not in busy]
            if free:
                voice = free[0]
            else:
                # Steal from the lowest-priority sounding note, if we outrank it.
                victim_voice, (_end, victim_rank, victim) = max(
                    busy.items(), key=lambda kv: (kv[1][1], kv[1][0])
                )
                if victim_rank <= self._rank(note):
                    report.displacements.append(
                        Displacement(
                            tick=note.tick,
                            role=note.role,
                            value=note.value,
                            reason="dropped",
                            taken_by=victim.role,
                        )
                    )
                    continue
                report.displacements.append(
                    Displacement(
                        tick=note.tick,
                        role=victim.role,
                        value=victim.value,
                        reason="stolen",
                        taken_by=note.role,
                        truncated_ticks=max(0, victim.end_tick - note.tick),
                    )
                )
                voice = victim_voice
                del busy[victim_voice]

            busy[voice] = (note.end_tick, self._rank(note), note)
            report.assignments.append(VoiceAssignment(note=note, voice=voice))
            report.peak_voices = max(report.peak_voices, len(busy))
        return report


def nes_voice_for(role: Role) -> str:
    """The NES channel a role has always used.

    The compatibility path. NES channels are not interchangeable -- the
    triangle is the bass because it cannot be anything else -- so there is
    nothing to allocate and this is a lookup.
    """
    try:
        return NES_VOICES[role]
    except KeyError as error:  # pragma: no cover - every Role is mapped
        raise KeyError(f"no NES channel for role {role!r}") from error


def concurrency(notes: list[PartNote]) -> int:
    """Peak simultaneous notes, release tails included.

    Separate from allocation so an arranger can ask "will this fit?" before
    committing to a voicing.
    """
    events: list[tuple[int, int]] = []
    for note in notes:
        events.append((note.tick, 1))
        events.append((note.end_tick, -1))
    peak = current = 0
    for _tick, delta in sorted(events, key=lambda e: (e[0], e[1])):
        current += delta
        peak = max(peak, current)
    return peak
