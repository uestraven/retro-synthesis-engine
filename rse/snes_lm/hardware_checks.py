"""Hardware-consistency diagnostics on generated and real event tables.

**Token grammar and hardware-consistent note lifetimes are different
measurements.** A sequence can satisfy the registered structural-validity
criterion -- the TIME/VOICE/PITCH/DUR cycle -- and still describe a performance
eight-voice hardware could not produce. The registered criterion is not
reinterpreted to cover this; this is a separate diagnostic.

The case that prompted it, from the first generated excerpt:

    voice 3, onset 0.415, duration 1.487  -> nominal end 1.902
    voice 3, onset 0.952, duration 0.460  -> starts 0.950 s early

On the S-DSP, a key-on replaces whatever that voice was playing. A conventional
MIDI renderer would instead sound both, which the hardware cannot do on one
voice.

Generated event tables are NEVER altered here. Overlap is measured and
reported; the voice-stealing rule belongs to the renderer and is applied
identically to both arms.

The control matters as much as the measurement. snesmdb's extractor ends a note
when its voice is re-keyed without a key-off (`ended_by_rekey`), so the real
corpus should show almost no same-voice overlap. If generated output shows
substantially more, the model is producing something the training data never
contained.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Sequence

#: Overlaps at or below this are the documented rounding artefact, not real.
#: The release warns that `onset_sec` and `duration_sec` are rounded
#: independently, so `onset + duration` can cross the next onset on the same
#: voice by exactly 1 microsecond: 3,695 consecutive pairs across 106 tracks,
#: none by more than 1 us.
ROUNDING_TOLERANCE_SEC = 1e-6


@dataclass
class OverlapReport:
    n_events: int
    n_pairs: int
    n_overlapping: int
    total_overlap_sec: float
    max_overlap_sec: float
    n_rounding_artefacts: int

    @property
    def fraction(self) -> float:
        return self.n_overlapping / self.n_pairs if self.n_pairs else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_events": self.n_events,
            "n_consecutive_same_voice_pairs": self.n_pairs,
            "n_overlapping": self.n_overlapping,
            "fraction_overlapping": self.fraction,
            "total_overlap_sec": self.total_overlap_sec,
            "max_overlap_sec": self.max_overlap_sec,
            "n_rounding_artefacts_excluded": self.n_rounding_artefacts,
        }


def same_voice_overlap(events: Sequence[dict[str, Any]]) -> OverlapReport:
    """How often a note is still nominally sounding when its voice re-keys.

    Counted over consecutive pairs on the same voice. A note with no duration
    (null, or still sounding at the window end) is skipped rather than assumed
    to last forever.
    """
    by_voice: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        by_voice.setdefault(int(event["voice"]), []).append(event)

    pairs = overlapping = artefacts = 0
    total = 0.0
    worst = 0.0
    for voice_events in by_voice.values():
        ordered = sorted(voice_events, key=lambda e: float(e["onset_sec"]))
        for first, second in zip(ordered, ordered[1:]):
            duration = first.get("duration_sec")
            if duration is None:
                continue
            pairs += 1
            end = float(first["onset_sec"]) + float(duration)
            overlap = end - float(second["onset_sec"])
            if overlap <= 0:
                continue
            if overlap <= ROUNDING_TOLERANCE_SEC:
                artefacts += 1
                continue
            overlapping += 1
            total += overlap
            worst = max(worst, overlap)
    return OverlapReport(
        n_events=len(events), n_pairs=pairs, n_overlapping=overlapping,
        total_overlap_sec=total, max_overlap_sec=worst,
        n_rounding_artefacts=artefacts,
    )


def apply_voice_stealing(
    events: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    """Clamp each note's end to the next key-on on its voice.

    The RENDERING policy, applied identically to both generation arms, so
    neither is advantaged by a renderer sounding notes the hardware could not.
    Returns new event dicts and the number clamped; the input is not mutated.

    This is deliberately separate from `same_voice_overlap`, which measures the
    generated tables as they are. Measuring and rendering are different jobs
    and conflating them would hide the diagnostic behind the fix.
    """
    by_voice: dict[int, list[int]] = {}
    for index, event in enumerate(events):
        by_voice.setdefault(int(event["voice"]), []).append(index)

    out = [dict(event) for event in events]
    clamped = 0
    for indices in by_voice.values():
        ordered = sorted(indices, key=lambda i: Decimal(str(events[i]["onset_sec"])))
        for current, following in zip(ordered, ordered[1:]):
            duration = out[current].get("duration_sec")
            if duration is None:
                continue
            start = Decimal(str(out[current]["onset_sec"]))
            limit = Decimal(str(events[following]["onset_sec"]))
            overlap = start + Decimal(str(duration)) - limit
            if overlap <= 0:
                continue
            # Clamp EXACTLY, in Decimal, with no tolerance. Clamping in float
            # against `limit + 1e-6` left `onset + duration` a hair past the
            # next key-on, so the voice read as still sounding when it
            # retriggered: eight voices each holding a sub-microsecond
            # residual produced the impossible 13- and 14-voice figures.
            out[current]["duration_sec"] = float(limit - start)
            # Authoritative end: the following onset's own value, so the two
            # spans are exactly adjacent rather than within an ULP.
            out[current]["end_sec"] = events[following]["onset_sec"]
            # The RATE keeps its musical meaning: a 1 us documented rounding
            # artefact is not a voice-stealing event, so it is clamped but not
            # counted. This keeps the reported rates comparable to the ones
            # already recorded.
            if overlap > Decimal(str(ROUNDING_TOLERANCE_SEC)):
                out[current]["clamped_by_voice_stealing"] = True
                clamped += 1
    return out, clamped


#: The eight generic S-DSP voices.
N_HARDWARE_VOICES = 8


@dataclass
class OccupancyReport:
    """Three DIFFERENT quantities that were previously conflated.

    `max_simultaneous_voices` was named for quantity 3 but computes quantity 1,
    and was then applied to a voice-stolen table, where the two should coincide
    and did not: it reported 13 and 14 concurrent voices on eight-voice
    hardware. Those figures measured nothing about the corpus or the model.
    """

    #: (1) Concurrent NOTE INSTANCES as generated. MAY legitimately exceed 8:
    #: that is the diagnostic -- it is how far the table is from playable.
    concurrent_notes: int
    #: (2) Concurrent note instances after the renderer's voice-stealing
    #: policy. Must be <= 8; more means the renderer is not actually capping.
    concurrent_after_stealing: int
    #: (3) Distinct OCCUPIED HARDWARE VOICES. Cannot exceed 8 by construction,
    #: because a key-on ends whatever its voice was playing. A value above 8 is
    #: a defect in this function, never a property of the data.
    occupied_voices: int
    n_distinct_voices: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "concurrent_notes": self.concurrent_notes,
            "concurrent_after_stealing": self.concurrent_after_stealing,
            "occupied_voices": self.occupied_voices,
            "n_distinct_voices": self.n_distinct_voices,
        }


def _span(event: dict[str, Any]) -> tuple[Decimal, Decimal] | None:
    """Exact [onset, onset+duration) from the 6-dp literals.

    `Decimal(str(x))` rather than `Decimal(x)`: the release's values are not
    float64-representable, so going through the shortest repr is what keeps
    `onset + duration` comparable to the next onset at all.
    """
    duration = event.get("duration_sec")
    if duration is None:
        return None
    start = Decimal(str(event["onset_sec"]))
    # An explicit end wins over `onset + duration`. Voice stealing sets it to
    # the FOLLOWING onset's own value, so adjacency is exact by construction.
    # Re-deriving the end by addition cannot achieve that here: decoded onsets
    # are accumulated floats, not the release's exact 6-dp literals, so the sum
    # lands an ULP past the next onset and the voice reads as still sounding --
    # which is what kept quantity 2 at 9 and 10 after the clamp was made exact.
    end = event.get("end_sec")
    if end is not None:
        return start, Decimal(str(end))
    return start, start + Decimal(str(duration))


def _peak(intervals: Sequence[tuple[Decimal, Decimal]]) -> int:
    """Peak overlap of HALF-OPEN intervals.

    Empty intervals are dropped: a zero-duration key-on sounds for no time, so
    it occupies nothing -- though it still ends the previous note on its voice,
    which is `_voice_intervals`' job, not this one. Closes are processed before
    opens at equal timestamps, which is what makes [onset, end) half-open: a
    note ending exactly when the next begins does not overlap it.
    """
    boundaries: list[tuple[Decimal, int]] = []
    for start, end in intervals:
        if end <= start:
            continue
        boundaries.append((end, -1))
        boundaries.append((start, +1))
    boundaries.sort(key=lambda b: (b[0], b[1]))
    current = peak = 0
    for _, delta in boundaries:
        current += delta
        peak = max(peak, current)
    return peak


def _voice_intervals(
    events: Sequence[dict[str, Any]]
) -> list[tuple[Decimal, Decimal]]:
    """Sounding spans per voice, each truncated by the next key-on on it.

    This is the hardware's own rule, applied exactly: a voice sounds one note
    at a time. Because every span on a voice is cut at the following key-on,
    the spans within a voice cannot overlap, so the peak over all voices is
    bounded by the number of distinct voices -- and so quantity 3 cannot
    exceed 8 unless the generator emitted a voice outside 0..7.
    """
    by_voice: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        by_voice.setdefault(int(event["voice"]), []).append(event)

    out: list[tuple[Decimal, Decimal]] = []
    for voice_events in by_voice.values():
        ordered = sorted(voice_events, key=lambda e: Decimal(str(e["onset_sec"])))
        for index, event in enumerate(ordered):
            span = _span(event)
            if span is None:
                continue
            start, end = span
            if index + 1 < len(ordered):
                # A repeated key-on at the SAME onset truncates to empty, which
                # is right: the voice is immediately retriggered.
                end = min(end, Decimal(str(ordered[index + 1]["onset_sec"])))
            out.append((start, end))
    return out


def occupancy(events: Sequence[dict[str, Any]]) -> OccupancyReport:
    """All three quantities, measured on one event table.

    Quantity 2 is computed from `apply_voice_stealing`'s actual output rather
    than re-deriving it, so the renderer's policy is checked as implemented --
    in float, as it runs -- instead of being assumed correct.
    """
    stolen, _ = apply_voice_stealing(events)
    spans = [s for s in (_span(e) for e in events) if s is not None]
    stolen_spans = [s for s in (_span(e) for e in stolen) if s is not None]
    voices = {int(e["voice"]) for e in events}
    return OccupancyReport(
        concurrent_notes=_peak(spans),
        concurrent_after_stealing=_peak(stolen_spans),
        occupied_voices=_peak(_voice_intervals(events)),
        n_distinct_voices=len(voices),
    )


def max_simultaneous_voices(events: Sequence[dict[str, Any]]) -> int:
    """Quantity 1: peak concurrent NOTE INSTANCES, ignoring voice.

    Kept under its original name because the Level 1 and corpus-scan records
    cite it, but it is no longer described as a voice count. It never was one:
    it does not group by voice, so on a table with same-voice overlap it
    exceeds 8 whether or not the hardware could play the table.
    """
    return _peak([s for s in (_span(e) for e in events) if s is not None])
