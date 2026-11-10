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
        ordered = sorted(indices, key=lambda i: float(events[i]["onset_sec"]))
        for current, following in zip(ordered, ordered[1:]):
            duration = out[current].get("duration_sec")
            if duration is None:
                continue
            start = float(out[current]["onset_sec"])
            limit = float(events[following]["onset_sec"])
            if start + float(duration) > limit + ROUNDING_TOLERANCE_SEC:
                out[current]["duration_sec"] = max(0.0, limit - start)
                out[current]["clamped_by_voice_stealing"] = True
                clamped += 1
    return out, clamped


def max_simultaneous_voices(events: Sequence[dict[str, Any]]) -> int:
    """Peak simultaneous sounding voices, with the documented traps handled.

    The release is explicit about both: parse the 6-dp literals as `Decimal`
    because they are not float64-representable, and process CLOSES BEFORE
    OPENS at equal timestamps. A naive sweep reports 9 to 13 voices on
    eight-voice hardware; closes-first alone takes nine games over the limit
    down to six.
    """
    boundaries: list[tuple[Decimal, int]] = []
    for event in events:
        duration = event.get("duration_sec")
        if duration is None:
            continue
        start = Decimal(str(event["onset_sec"]))
        end = start + Decimal(str(duration))
        boundaries.append((end, -1))   # closes sort before opens at equal time
        boundaries.append((start, +1))
    boundaries.sort(key=lambda b: (b[0], b[1]))
    current = peak = 0
    for _, delta in boundaries:
        current += delta
        peak = max(peak, current)
    return peak
