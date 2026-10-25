"""What a codec keeps, what it moves, and what it invents.

Three measurement rules shape this module, and each exists because the obvious
alternative gives a codec undeserved credit.

**Matching is one-to-one.** A reconstructed event may stand in for at most one
original event. Without that, a codec that collapses an arpeggio to a single
note could be scored as having recovered all four of its frames, and the
collapse would look free.

**Spurious events are reported, not netted off.** A codec that emits more
events than it was given has not preserved more information. Missing and extra
are separate columns, so no cancellation can hide either.

**Monophony is enforced before measuring durations.** The collapse key
guarantees one onset per channel per slot, which is not the same as guaranteeing
non-overlapping notes: a four-row note followed by another note two rows later
overlaps on a monophonic channel. Real hardware steals the voice, so a duration
measured without that truncation describes something the APU would never play.
This is applied as an explicit, counted step rather than inside `decode`,
because codec A's decoded output has to stay bit-identical to the
implementation the foundation was trained on.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from statistics import fmean, median, pstdev

from rse.events.schema import Composition, Noise, Note

#: Widest onset displacement that can still count as the same event, in rows.
#: Two rows is deliberately generous: the question is which original event a
#: reconstructed one corresponds to, and a tight window would score a badly
#: displaced note as *missing plus spurious* rather than as displaced, which
#: flatters whichever codec displaces least into looking like the one that
#: deletes least.
MATCH_TOLERANCE_ROWS = 2.0


def ioi_entropy(times: list[float]) -> float:
    """Shannon entropy of inter-onset intervals, in bits.

    Buckets at half-octave resolution on a log scale, which is the definition
    the earlier representation analysis used (2.358 bits on originals against
    0.686 after row collapse). Kept identical so Branch B's figures can be
    read against those, and shared from here so the two cannot drift apart.
    """
    ordered = sorted(times)
    iois = [b - a for a, b in zip(ordered, ordered[1:], strict=False) if b > a]
    if not iois:
        return 0.0
    buckets = Counter(round(math.log2(max(gap, 1e-3)) * 2) for gap in iois)
    total = sum(buckets.values())
    return -sum((c / total) * math.log2(c / total) for c in buckets.values() if c)


def coefficient_of_variation(values: list[float]) -> float:
    """Spread of note lengths, normalized by their mean.

    Scale-free on purpose: codec D measures durations in finer units, so a raw
    standard deviation would differ between codecs for reasons of unit rather
    than of articulation.
    """
    if len(values) < 2:
        return 0.0
    mean = fmean(values)
    return pstdev(values) / mean if mean else 0.0


def enforce_monophony(composition: Composition) -> tuple[Composition, int]:
    """Truncate each note where the next note on its channel begins.

    Returns the repaired composition and how many events had to be shortened.
    The count is itself a codec statistic: a codec that needs more truncation
    is one whose durations the hardware would silently override.
    """
    events = list(composition.events)
    by_channel: dict[str, list[int]] = {}
    for index, event in enumerate(events):
        if isinstance(event, Note | Noise):
            by_channel.setdefault(event.channel_id, []).append(index)

    truncated = 0
    for indices in by_channel.values():
        indices.sort(key=lambda i: events[i].tick)
        for current, following in zip(indices, indices[1:], strict=False):
            here, nxt = events[current], events[following]
            room = nxt.tick - here.tick
            if room < 1:
                # Same tick on a monophonic channel: the later write wins on
                # real hardware, so the earlier note never sounds. Kept at
                # length 1 rather than deleted, so event counts stay
                # comparable and the loss shows up as duration error.
                room = 1
            if here.duration > room:
                events[current] = type(here)(**{**vars(here), "duration": room})
                truncated += 1

    repaired = Composition(
        hardware_profile=composition.hardware_profile,
        ticks_per_beat=composition.ticks_per_beat,
        events=events,
        tempo_map=list(composition.tempo_map),
        time_signatures=list(composition.time_signatures),
        sections=list(composition.sections),
    )
    repaired.loop_start_tick = composition.loop_start_tick
    repaired.loop_end_tick = composition.loop_end_tick
    return repaired, truncated


def _key(event: Note | Noise) -> tuple[str, int]:
    """Channel and value. A note at a different pitch is a different note."""
    return (event.channel_id, event.pitch if isinstance(event, Note) else event.value)


@dataclass
class MatchResult:
    """A one-to-one correspondence between original and reconstructed events."""

    #: (original index, reconstructed index) pairs.
    pairs: list[tuple[int, int]] = field(default_factory=list)
    unmatched_original: list[int] = field(default_factory=list)
    spurious_reconstructed: list[int] = field(default_factory=list)

    def recall(self, n_original: int) -> float:
        return len(self.pairs) / n_original if n_original else 0.0


def match_events(
    original: list[Note | Noise],
    reconstructed: list[Note | Noise],
    *,
    tolerance: float,
    original_scale: float = 1.0,
    reconstructed_scale: float = 1.0,
) -> MatchResult:
    """Pair each original event with at most one reconstructed event.

    The two scale factors convert each stream's ticks into a shared unit, and
    `tolerance` is in that unit. They are not optional in practice: a source
    piece runs at 44100 ticks per second while `decode` rebuilds on a nominal
    480-ticks-per-beat grid, so matching raw tick differences compares a row of
    4410 ticks against a row of 120 and pairs almost nothing. Passing seconds
    for both is the intended use; the defaults exist for tick-space tests.

    Candidates must share a channel and a value. Pairs are taken in order of
    increasing displacement, so the closest correspondences are fixed first;
    ties break on index order, which makes the result deterministic.
    """
    buckets: dict[tuple[str, int], list[int]] = {}
    for index, event in enumerate(reconstructed):
        buckets.setdefault(_key(event), []).append(index)

    candidates: list[tuple[float, int, int]] = []
    for oi, event in enumerate(original):
        at = event.tick * original_scale
        for ri in buckets.get(_key(event), ()):
            gap = abs(reconstructed[ri].tick * reconstructed_scale - at)
            if gap <= tolerance:
                candidates.append((gap, oi, ri))
    candidates.sort()

    used_original: set[int] = set()
    used_reconstructed: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _gap, oi, ri in candidates:
        if oi in used_original or ri in used_reconstructed:
            continue
        used_original.add(oi)
        used_reconstructed.add(ri)
        pairs.append((oi, ri))

    pairs.sort()
    return MatchResult(
        pairs=pairs,
        unmatched_original=[i for i in range(len(original)) if i not in used_original],
        spurious_reconstructed=[
            i for i in range(len(reconstructed)) if i not in used_reconstructed
        ],
    )


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


@dataclass
class FidelityMetrics:
    """One codec measured against one piece's original event stream."""

    n_original: int = 0
    n_reconstructed: int = 0
    n_matched: int = 0
    n_missing: int = 0
    n_spurious: int = 0
    n_truncated_for_monophony: int = 0
    #: Retained / total, per event class from `rse.events.classify`.
    recall_by_class: dict[str, float] = field(default_factory=dict)
    n_by_class: dict[str, int] = field(default_factory=dict)
    onset_mae_sec: float = 0.0
    onset_p95_sec: float = 0.0
    duration_mae_sec: float = 0.0
    ioi_entropy_original: float = 0.0
    ioi_entropy_reconstructed: float = 0.0
    note_length_cv_original: float = 0.0
    note_length_cv_reconstructed: float = 0.0
    n_tokens: int = 0

    @property
    def ioi_entropy_retained(self) -> float:
        if self.ioi_entropy_original <= 0:
            return 0.0
        return self.ioi_entropy_reconstructed / self.ioi_entropy_original

    def to_dict(self) -> dict[str, object]:
        return {
            "n_original": self.n_original,
            "n_reconstructed": self.n_reconstructed,
            "n_matched": self.n_matched,
            "n_missing": self.n_missing,
            "n_spurious": self.n_spurious,
            "n_truncated_for_monophony": self.n_truncated_for_monophony,
            "event_recall": round(self.n_matched / self.n_original, 5) if self.n_original else 0.0,
            "recall_by_class": {k: round(v, 5) for k, v in self.recall_by_class.items()},
            "n_by_class": dict(self.n_by_class),
            "onset_mae_sec": round(self.onset_mae_sec, 6),
            "onset_p95_sec": round(self.onset_p95_sec, 6),
            "duration_mae_sec": round(self.duration_mae_sec, 6),
            "ioi_entropy_original": round(self.ioi_entropy_original, 4),
            "ioi_entropy_reconstructed": round(self.ioi_entropy_reconstructed, 4),
            "ioi_entropy_retained": round(self.ioi_entropy_retained, 4),
            "note_length_cv_original": round(self.note_length_cv_original, 4),
            "note_length_cv_reconstructed": round(self.note_length_cv_reconstructed, 4),
            "n_tokens": self.n_tokens,
        }


def measure_fidelity(
    original: list[Note | Noise],
    reconstructed: list[Note | Noise],
    *,
    classes: list[str],
    original_sec_per_tick: float,
    reconstructed_sec_per_tick: float,
    row_sec: float,
    n_truncated: int = 0,
    n_tokens: int = 0,
) -> FidelityMetrics:
    """Compare two event streams in seconds, not ticks.

    Seconds because the two streams do not share a tick scale: `decode`
    rebuilds time on a nominal grid and recovers the real row rate from the
    Tempo token. Comparing raw ticks would report a tempo-label difference as a
    timing error, which is how an earlier comparison was rigged by accident.
    """
    if len(classes) != len(original):
        raise ValueError(
            f"got {len(classes)} class labels for {len(original)} original events; "
            "they must correspond one-to-one"
        )
    result = match_events(
        original,
        reconstructed,
        tolerance=row_sec * MATCH_TOLERANCE_ROWS,
        original_scale=original_sec_per_tick,
        reconstructed_scale=reconstructed_sec_per_tick,
    )

    onset_errors: list[float] = []
    duration_errors: list[float] = []
    for oi, ri in result.pairs:
        o, r = original[oi], reconstructed[ri]
        onset_errors.append(
            abs(o.tick * original_sec_per_tick - r.tick * reconstructed_sec_per_tick)
        )
        duration_errors.append(
            abs(o.duration * original_sec_per_tick - r.duration * reconstructed_sec_per_tick)
        )

    matched_original = {oi for oi, _ in result.pairs}
    totals: Counter[str] = Counter(classes)
    kept: Counter[str] = Counter(classes[oi] for oi in matched_original)

    metrics = FidelityMetrics(
        n_original=len(original),
        n_reconstructed=len(reconstructed),
        n_matched=len(result.pairs),
        n_missing=len(result.unmatched_original),
        n_spurious=len(result.spurious_reconstructed),
        n_truncated_for_monophony=n_truncated,
        recall_by_class={name: kept[name] / count for name, count in totals.items() if count},
        n_by_class=dict(totals),
        onset_mae_sec=fmean(onset_errors) if onset_errors else 0.0,
        onset_p95_sec=percentile(onset_errors, 0.95),
        duration_mae_sec=fmean(duration_errors) if duration_errors else 0.0,
        ioi_entropy_original=ioi_entropy([e.tick * original_sec_per_tick for e in original]),
        ioi_entropy_reconstructed=ioi_entropy(
            [e.tick * reconstructed_sec_per_tick for e in reconstructed]
        ),
        note_length_cv_original=coefficient_of_variation(
            [e.duration * original_sec_per_tick for e in original]
        ),
        note_length_cv_reconstructed=coefficient_of_variation(
            [e.duration * reconstructed_sec_per_tick for e in reconstructed]
        ),
        n_tokens=n_tokens,
    )
    return metrics


def overlapping_events(composition: Composition) -> list[tuple[int, int]]:
    """Pairs of indices that sound at once on one monophonic channel.

    Used as an assertion rather than a metric: after `enforce_monophony` this
    must be empty for every codec at every resolution.
    """
    events = list(composition.events)
    by_channel: dict[str, list[int]] = {}
    for index, event in enumerate(events):
        if isinstance(event, Note | Noise):
            by_channel.setdefault(event.channel_id, []).append(index)

    clashes: list[tuple[int, int]] = []
    for indices in by_channel.values():
        indices.sort(key=lambda i: events[i].tick)
        for current, following in zip(indices, indices[1:], strict=False):
            if events[current].tick + events[current].duration > events[following].tick:
                clashes.append((current, following))
    return clashes


def summarize(values: list[float]) -> dict[str, float]:
    """Median and p95 of a per-piece statistic, for the ranking table."""
    if not values:
        return {"median": 0.0, "p95": 0.0, "n": 0}
    return {
        "median": round(median(values), 6),
        "p95": round(percentile(values, 0.95), 6),
        "n": len(values),
    }
