"""Deriving musical form from the notes.

The first model's one clear weakness is organisation above the bar: at
temperature 1 it almost never repeats a bar, at 0.7 it repeats until it is
stuck. Proposal 6.5 predicted this — next-token prediction does not produce
well-formed structure — and 6.2 reserved a phrase-boundary token family that
was never implemented. This module supplies what that family needs.

NES-MDB carries no form annotation: no section labels, no phrase marks, not
even loop points. So form is inferred from repetition, and *how* it is inferred
had to be measured rather than assumed.

What the measurements showed
----------------------------
Comparing exact `(frame, channel, pitch)` positions finds almost nothing: mean
self-similarity 0.03-0.07 across every segment length, which labelled 64% of
real pieces as having no repetition at all. That is not what game music is
like. exprsco is frame-exact performance data, so a repeated phrase is almost
never frame-identical — vibrato, arpeggio artifacts and envelope timing all
shift it, and one frame of drift destroys an exact match.

Comparing **pitch-interval n-grams per channel** instead gives 0.18-0.35, four
to five times higher, because it is order-aware, timing-tolerant and
transposition-invariant. That is the fingerprint used here.

Self-similarity also rises with segment length (0.18 at 2 bars, 0.29 at 12,
0.35 at 24), so a segment must be long enough to hold a phrase, and a *window*
must be long enough to hold several segments or there is no form to see.
"""

from __future__ import annotations

import zlib
from collections import defaultdict
from dataclasses import dataclass
from statistics import median

from rse.events.schema import Composition, Note, SectionSpan

#: Bars per segment. Eight of RSE's 0.667 s bars is 5.3 s, which is long enough
#: to hold a phrase at the tempos this corpus implies while still fitting four
#: segments into a 32-bar window.
SEGMENT_BARS = 8

#: Letters available. A piece needing more reuses the closest rather than
#: inventing a seventh, past which the label carries nothing the model can use.
SECTION_LETTERS = ("A", "B", "C", "D", "E", "F")

#: Interval n-gram length for the fingerprint.
NGRAM = 3

#: A segment is a return of an earlier one when its best similarity clears
#: both of these: an absolute floor, and a multiple of the piece's own median
#: pairwise similarity. The relative part matters because the corpus is
#: heterogeneous — a dense arpeggio piece and a sparse ballad have very
#: different baseline similarities, and one global constant cannot serve both.
RETURN_FLOOR = 0.15
RETURN_RELATIVE = 1.5


@dataclass(frozen=True)
class SegmentAnalysis:
    """One segment, its label, and how strongly it matched."""

    index: int
    start_bar: int
    n_bars: int
    label: str
    #: Similarity to the segment whose label this one took; 0.0 if new.
    matched_similarity: float
    n_grams: int

    @property
    def is_return(self) -> bool:
        return self.matched_similarity > 0.0


def segment_fingerprint(
    composition: Composition,
    start_bar: int,
    n_bars: int,
    *,
    beats_per_bar: int = 4,
    ngram: int = NGRAM,
) -> frozenset[int]:
    """Pitch-interval n-grams per channel for one segment.

    Order-aware so it distinguishes a phrase from its retrograde, tolerant of
    timing drift because it ignores exact positions, and transposition-
    invariant so a section returning a fifth up still matches. Consecutive
    repeats of a value collapse: a held or retriggered note is one gesture.

    The noise channel contributes too — its values are hardware periods rather
    than pitches, but a change of drum pattern is one of the clearest markers
    of a new section in this idiom, and differences between period indexes are
    as good a signal as differences between pitches.
    """
    ticks_per_bar = composition.ticks_per_beat * beats_per_bar
    if ticks_per_bar <= 0:
        return frozenset()
    low = start_bar * ticks_per_bar
    high = low + n_bars * ticks_per_bar

    per_channel: dict[str, list[int]] = defaultdict(list)
    for event in sorted(composition.sounding_events(), key=lambda e: e.tick):
        if not (low <= event.tick < high):
            continue
        value = event.pitch if isinstance(event, Note) else event.value
        per_channel[event.channel_id].append(int(value))

    grams: set[int] = set()
    for channel_id, values in per_channel.items():
        collapsed = [
            value for index, value in enumerate(values) if index == 0 or value != values[index - 1]
        ]
        intervals = [b - a for a, b in zip(collapsed, collapsed[1:], strict=False)]
        for i in range(len(intervals) - ngram + 1):
            key = f"{channel_id}:" + ",".join(str(v) for v in intervals[i : i + ngram])
            grams.add(zlib.crc32(key.encode()))
    return frozenset(grams)


def similarity(a: frozenset[int], b: frozenset[int]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    return intersection / len(a | b) if intersection else 0.0


def analyze_structure(
    composition: Composition,
    *,
    segment_bars: int = SEGMENT_BARS,
    beats_per_bar: int = 4,
    n_bars: int | None = None,
) -> list[SegmentAnalysis]:
    """Cut a composition into segments and label them by repetition."""
    if segment_bars < 1:
        raise ValueError(f"segment_bars must be positive, got {segment_bars}")

    ticks_per_bar = composition.ticks_per_beat * beats_per_bar
    if ticks_per_bar <= 0:
        return []
    total_bars = n_bars if n_bars is not None else _bar_count(composition, ticks_per_bar)
    if total_bars <= 0:
        return []

    starts = list(range(0, total_bars, segment_bars))
    fingerprints = [
        segment_fingerprint(
            composition,
            start,
            min(segment_bars, total_bars - start),
            beats_per_bar=beats_per_bar,
        )
        for start in starts
    ]

    # The piece's own baseline, so the threshold adapts to how self-similar
    # this particular music is.
    pairwise = [
        similarity(fingerprints[i], fingerprints[j])
        for i in range(len(fingerprints))
        for j in range(i + 1, len(fingerprints))
    ]
    # With fewer than three pairwise comparisons the median is not a baseline,
    # it is one of the values being tested — and `1.5 x itself` can never be
    # met, so a two-segment window would report no repetition however literal.
    if len(pairwise) >= 3:
        threshold = max(RETURN_FLOOR, median(pairwise) * RETURN_RELATIVE)
    else:
        threshold = RETURN_FLOOR

    analyses: list[SegmentAnalysis] = []
    known: list[tuple[str, frozenset[int]]] = []

    for index, (start, fingerprint) in enumerate(zip(starts, fingerprints, strict=True)):
        span = min(segment_bars, total_bars - start)
        if not fingerprint:
            # A silent or near-silent segment carries no evidence; let it
            # continue whatever section it sits inside.
            label = analyses[-1].label if analyses else SECTION_LETTERS[0]
            analyses.append(SegmentAnalysis(index, start, span, label, 0.0, 0))
            continue

        best_label, best_score = "", 0.0
        for label, seen in known:
            score = similarity(fingerprint, seen)
            if score > best_score:
                best_label, best_score = label, score

        used = {label for label, _ in known}
        if best_score >= threshold and best_label:
            label, matched = best_label, best_score
        elif len(used) < len(SECTION_LETTERS):
            label, matched = SECTION_LETTERS[len(used)], 0.0
            known.append((label, fingerprint))
        else:
            label, matched = (best_label or SECTION_LETTERS[-1]), best_score
        analyses.append(SegmentAnalysis(index, start, span, label, matched, len(fingerprint)))
    return analyses


def sections_from_analysis(analyses: list[SegmentAnalysis]) -> list[SectionSpan]:
    """Collapse per-segment labels into spans, marking where form changes."""
    spans: list[SectionSpan] = []
    for analysis in analyses:
        if spans and spans[-1].label == analysis.label:
            continue
        spans.append(SectionSpan(start_bar=analysis.start_bar, label=analysis.label))
    return spans


def derive_sections(
    composition: Composition,
    *,
    segment_bars: int = SEGMENT_BARS,
    beats_per_bar: int = 4,
    n_bars: int | None = None,
) -> list[SectionSpan]:
    """Infer a composition's section spans from its own repetition."""
    return sections_from_analysis(
        analyze_structure(
            composition,
            segment_bars=segment_bars,
            beats_per_bar=beats_per_bar,
            n_bars=n_bars,
        )
    )


def form_string(analyses: list[SegmentAnalysis]) -> str:
    """The form as a readable string, e.g. `AABA`."""
    return "".join(analysis.label for analysis in analyses)


def _bar_count(composition: Composition, ticks_per_bar: int) -> int:
    end = composition.end_tick
    return -(-end // ticks_per_bar) if end > 0 else 0
