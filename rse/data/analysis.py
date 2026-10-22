"""Per-composition measurements used by both normalization and statistics.

These are measurements, not enforcement: Phase 1 records how far source data is
from the hardware rules so the Phase 2 constraint engine can be built against
real numbers instead of assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass

from rse.data.midi_io import Note
from rse.hardware.profile import ChannelSpec


@dataclass(frozen=True)
class Overlap:
    """Two notes sounding at once on a channel that can only hold one."""

    tick: int
    simultaneous: int


def max_simultaneous(notes: list[Note]) -> int:
    """Greatest number of notes sounding at the same instant."""
    if not notes:
        return 0
    # Sweep note starts and ends in time order; a start that arrives while the
    # count is already at the limit is what the constraint engine will have to
    # resolve later.
    edges: list[tuple[int, int]] = []
    for note in notes:
        edges.append((note.tick, 1))
        edges.append((note.tick + note.dur_ticks, -1))
    # Ends sort before starts at the same tick: a note ending exactly where the
    # next begins is legal monophony, not an overlap.
    edges.sort(key=lambda e: (e[0], e[1]))
    current = peak = 0
    for _tick, delta in edges:
        current += delta
        peak = max(peak, current)
    return peak


def overlaps(notes: list[Note], limit: int) -> list[Overlap]:
    """Every instant where more than `limit` notes sound together."""
    if not notes or limit <= 0:
        return []
    edges: list[tuple[int, int]] = []
    for note in notes:
        edges.append((note.tick, 1))
        edges.append((note.tick + note.dur_ticks, -1))
    edges.sort(key=lambda e: (e[0], e[1]))
    found: list[Overlap] = []
    current = 0
    for tick, delta in edges:
        current += delta
        if delta == 1 and current > limit:
            found.append(Overlap(tick=tick, simultaneous=current))
    return found


def out_of_range(notes: list[Note], spec: ChannelSpec) -> list[int]:
    """Note values the channel cannot represent, as a sorted unique list."""
    return sorted({n.pitch for n in notes if not spec.in_range(n.pitch)})


#: Duration bucket upper edges in quarter notes, with their labels.
DURATION_EDGES: tuple[tuple[float, str], ...] = (
    (0.0625, "<=1/64"),
    (0.125, "1/32"),
    (0.25, "1/16"),
    (0.5, "1/8"),
    (1.0, "1/4"),
    (2.0, "1/2"),
    (4.0, "1"),
)


def duration_bucket(beats: float) -> str:
    """Label a duration in quarter notes with the nearest common note value."""
    for edge, label in DURATION_EDGES:
        if beats <= edge * 1.01:  # small tolerance for non-exact source timing
            return label
    return ">1"
