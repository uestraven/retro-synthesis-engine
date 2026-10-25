"""An intermediate score, between an external model and RSE's NES pipeline.

Deliberately *not* a second codec. D4 describes what the APU does; this
describes what a general symbolic music model can say. The gap between them is
the thing Phase 2 measures, so the two representations have to stay separate
or the measurement collapses.

Three properties matter for the audit:

**Track identity is explicit and typed.** An NES channel is not a General MIDI
program, and the noise channel is not a drum kit. A conversion that forgets
which track was the triangle cannot put it back, so `Track.role` carries that
through rather than relying on a program number to imply it.

**Percussion is a separate event type.** A noise period is an unpitched timbre
selector with 16 values; a GM drum key is a sound choice out of ~47. Treating
them as the same integer is the error that would make the round trip look
lossless while silently changing which drum plays.

**Conversion warnings are carried, not raised.** A lossy conversion is the
expected case here, and the measurement needs the losses enumerated, not an
exception at the first one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Musical role of a track, which survives conversion even when the
#: instrument assignment does not.
MELODY = "melody"
BASS = "bass"
ACCOMPANIMENT = "accompaniment"
PERCUSSION = "percussion"
ROLES = (MELODY, BASS, ACCOMPANIMENT, PERCUSSION)

#: NES channel to default musical role. P2 is accompaniment by convention
#: rather than by measurement -- `rse.conditioning.lead_channel` shows the
#: lead swaps between P1 and P2 and sometimes sits on the triangle, so this is
#: a starting assignment the arranger may override.
NES_ROLES = {"P1": MELODY, "P2": ACCOMPANIMENT, "TR": BASS, "NO": PERCUSSION}


@dataclass(frozen=True)
class ScoreNote:
    """A pitched note, in ticks of the score's own resolution."""

    tick: int
    duration: int
    pitch: int
    velocity: int = 96


@dataclass(frozen=True)
class ScorePercussion:
    """An unpitched hit.

    `key` is in the score's own percussion namespace -- a GM drum number for a
    general score, a noise period for an NES one. `source_value` keeps the
    original so a round trip can be checked rather than assumed.
    """

    tick: int
    duration: int
    key: int
    velocity: int = 96
    source_value: int | None = None


@dataclass
class Track:
    """One voice. `channel_id` is set only when it came from, or is going to,
    a specific piece of hardware."""

    role: str
    notes: list[ScoreNote] = field(default_factory=list)
    percussion: list[ScorePercussion] = field(default_factory=list)
    #: General MIDI program, where the representation uses one.
    program: int | None = None
    channel_id: str | None = None
    #: Whether this track may sound more than one note at a time. An NES
    #: pulse channel may not; a piano track may.
    monophonic: bool = False

    @property
    def n_events(self) -> int:
        return len(self.notes) + len(self.percussion)


@dataclass
class Score:
    """A whole piece, at a stated tick resolution."""

    #: Ticks per quarter note. The single most consequential field: it is
    #: what decides whether RSE's beat/16 onsets survive.
    ticks_per_quarter: int
    tempo_bpm: float
    tracks: list[Track] = field(default_factory=list)
    beats_per_bar: int = 4
    #: What the conversion could not represent, in order.
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def n_events(self) -> int:
        return sum(t.n_events for t in self.tracks)

    def track_for(self, channel_id: str) -> Track | None:
        for track in self.tracks:
            if track.channel_id == channel_id:
                return track
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticks_per_quarter": self.ticks_per_quarter,
            "tempo_bpm": round(self.tempo_bpm, 3),
            "beats_per_bar": self.beats_per_bar,
            "n_events": self.n_events,
            "tracks": [
                {
                    "role": t.role,
                    "channel_id": t.channel_id,
                    "program": t.program,
                    "monophonic": t.monophonic,
                    "n_notes": len(t.notes),
                    "n_percussion": len(t.percussion),
                }
                for t in self.tracks
            ],
            "n_warnings": len(self.warnings),
            "provenance": dict(self.provenance),
        }
