"""Does the second half of a piece bring back the first half's material?

Built for the paired 8-vs-16-bar experiment, where the first measurement said
generated 16-bar pieces restate almost nothing: median per-bar overlap 0.115,
one exactly-repeated bar across 24 pieces. That number is uninterpretable on
its own. Real NES music may not literally repeat bars either -- a tracker
author writing a 16-bar loop has every reason to vary the second half -- so the
corpus has to be measured on the identical metric before any conclusion is
drawn about the model.

Three measures, because "the melody comes back" has three different meanings
and they fail apart:

**Exact restatement** is per-bar overlap of (channel, slot, voice) triples.
It answers "is bar i+8 the same bar as bar i". This is the strictest reading
and the one most likely to be absent from good music.

**Contour recurrence** is transposition- and rhythm-invariant: the melody is
reduced to a sequence of interval *signs*, and the question is what fraction
of the second half's shapes already occurred in the first. A figure that
returns a third higher, or augmented, or reharmonized, counts here and not
above. This is the measure that matches what a listener means by a theme
returning.

**Interval recurrence** sits between them: exact interval sequences, so
transposition-invariant but not contour-flattened. A melody answered at a
different pitch level scores here; one merely sharing an up-down-up shape does
not.

Bar boundaries come from the grid, never from `end_tick / n_bars`. Dividing by
the event span puts every boundary in the wrong place whenever the last bar
ends early, which silently scrambles the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rse.events.schema import Composition, Note

#: Default contour/interval n-gram length. Four intervals spans five notes,
#: which is long enough to be a recognisable figure rather than a coincidence
#: and short enough to survive the variation that makes a return musical.
NGRAM = 4


def ticks_per_bar(*, ticks_per_step: int = 120, steps_per_bar: int = 16) -> int:
    """Grid ticks in one bar. The frozen D4 representation gives 120 * 16."""
    return int(ticks_per_step) * int(steps_per_bar)


def bar_of(tick: int, *, bar_ticks: int) -> int:
    return int(tick // bar_ticks)


def voice_of(event: Any) -> Any:
    """Pitch on the pitched channels, hardware period on the noise channel.

    Never compared across channels, so one field is safe: a drum pattern gets
    to count as restated material like anything else.
    """
    voice = getattr(event, "pitch", None)
    return getattr(event, "value", None) if voice is None else voice


def bar_fingerprints(
    composition: Composition, *, n_bars: int, bar_ticks: int
) -> list[frozenset[tuple[Any, int, Any]]]:
    """Per-bar content as (channel, slot-within-bar, voice) triples."""
    bars: list[set[tuple[Any, int, Any]]] = [set() for _ in range(n_bars)]
    for event in composition.sounding_events():
        bar = bar_of(event.tick, bar_ticks=bar_ticks)
        if not 0 <= bar < n_bars:
            continue
        slot = int(event.tick - bar * bar_ticks)
        bars[bar].add((event.channel_id, slot, voice_of(event)))
    return [frozenset(bar) for bar in bars]


def jaccard(left: frozenset[Any], right: frozenset[Any]) -> float:
    if not left and not right:
        return 1.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def lead_melody(composition: Composition) -> tuple[str, list[tuple[int, int]]]:
    """The lead channel's (tick, pitch) pairs in time order.

    The channel is chosen over the whole piece so both halves are read from
    the same voice. A theme that returns on a different channel is a real
    musical event but a different claim, and conflating the two would let an
    unrelated bass line count as a returning melody.
    """
    from rse.conditioning import lead_channel

    channel = lead_channel(composition)
    if not channel:
        return "", []
    notes = sorted(
        (
            event
            for event in composition.sounding_events(channel)
            if isinstance(event, Note)
        ),
        key=lambda event: event.tick,
    )
    return channel, [(note.tick, note.pitch) for note in notes]


def intervals(pitches: list[int]) -> list[int]:
    return [b - a for a, b in zip(pitches, pitches[1:], strict=False)]


def contour(pitches: list[int]) -> list[int]:
    """Interval signs: the shape, independent of size and of key."""
    return [(1 if d > 0 else -1 if d < 0 else 0) for d in intervals(pitches)]


def ngrams(sequence: list[Any], n: int) -> set[tuple[Any, ...]]:
    if len(sequence) < n:
        return set()
    return {tuple(sequence[i : i + n]) for i in range(len(sequence) - n + 1)}


def recurrence(first: list[Any], second: list[Any], *, n: int = NGRAM) -> float | None:
    """Fraction of the second half's n-grams that already occur in the first.

    `None` when either half is too short to hold an n-gram, which is a
    different outcome from zero recurrence and must not be averaged in as if
    it were.
    """
    later = ngrams(second, n)
    if not later or not ngrams(first, n):
        return None
    return len(later & ngrams(first, n)) / len(later)


@dataclass
class HalvesReport:
    """How much of the first half returns in the second."""

    n_bars: int = 0
    exact_restatement_mean: float = 0.0
    exact_restatement_bars: int = 0
    contour_recurrence: float | None = None
    interval_recurrence: float | None = None
    pitch_overlap: float | None = None
    n_melody_first: int = 0
    n_melody_second: int = 0
    new_pitches: int = 0
    lost_pitches: int = 0
    empty_bars_first: int = 0
    empty_bars_second: int = 0
    lead_channel: str = ""
    per_bar: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_bars": self.n_bars,
            "exact_restatement_mean": round(self.exact_restatement_mean, 4),
            "exact_restatement_bars": self.exact_restatement_bars,
            "contour_recurrence": None
            if self.contour_recurrence is None
            else round(self.contour_recurrence, 4),
            "interval_recurrence": None
            if self.interval_recurrence is None
            else round(self.interval_recurrence, 4),
            "pitch_overlap": None
            if self.pitch_overlap is None
            else round(self.pitch_overlap, 4),
            "n_melody_first": self.n_melody_first,
            "n_melody_second": self.n_melody_second,
            "new_pitches": self.new_pitches,
            "lost_pitches": self.lost_pitches,
            "empty_bars_first": self.empty_bars_first,
            "empty_bars_second": self.empty_bars_second,
            "lead_channel": self.lead_channel,
            "per_bar": [round(v, 4) for v in self.per_bar],
        }


def _report(
    first: list[frozenset[tuple[Any, int, Any]]],
    second: list[frozenset[tuple[Any, int, Any]]],
    *,
    melody_first: list[int],
    melody_second: list[int],
    channel: str,
    ngram: int,
) -> HalvesReport:
    """The shared core, so corpus spans and generated pieces are measured
    by exactly the same code rather than by two implementations that agree
    until one of them is edited."""
    per_bar = [jaccard(a, b) for a, b in zip(first, second, strict=True)]
    pitches_first = {v for bar in first for (c, _s, v) in bar if c != "NO"}
    pitches_second = {v for bar in second for (c, _s, v) in bar if c != "NO"}
    union = pitches_first | pitches_second

    return HalvesReport(
        n_bars=len(first) + len(second),
        exact_restatement_mean=sum(per_bar) / len(per_bar) if per_bar else 0.0,
        exact_restatement_bars=sum(1 for value in per_bar if value >= 0.999),
        contour_recurrence=recurrence(
            contour(melody_first), contour(melody_second), n=ngram
        ),
        interval_recurrence=recurrence(
            intervals(melody_first), intervals(melody_second), n=ngram
        ),
        pitch_overlap=(len(pitches_first & pitches_second) / len(union))
        if union
        else None,
        n_melody_first=len(melody_first),
        n_melody_second=len(melody_second),
        new_pitches=len(pitches_second - pitches_first),
        lost_pitches=len(pitches_first - pitches_second),
        empty_bars_first=sum(1 for bar in first if not bar),
        empty_bars_second=sum(1 for bar in second if not bar),
        lead_channel=channel,
        per_bar=per_bar,
    )


def analyse_halves(
    composition: Composition,
    *,
    n_bars: int = 16,
    bar_ticks: int | None = None,
    ngram: int = NGRAM,
) -> HalvesReport:
    """Compare the two halves of one `n_bars`-bar composition.

    Used for the generated pieces, which are a single composition.
    """
    if n_bars % 2:
        raise ValueError(f"n_bars must be even to split in half, got {n_bars}")
    bar_ticks = ticks_per_bar() if bar_ticks is None else bar_ticks
    half = n_bars // 2
    bars = bar_fingerprints(composition, n_bars=n_bars, bar_ticks=bar_ticks)
    channel, melody = lead_melody(composition)
    split = half * bar_ticks
    return _report(
        bars[:half],
        bars[half:],
        melody_first=[p for tick, p in melody if tick < split],
        melody_second=[p for tick, p in melody if tick >= split],
        channel=channel,
        ngram=ngram,
    )


def analyse_adjacent(
    first: Composition,
    second: Composition,
    *,
    n_bars: int = 8,
    bar_ticks: int | None = None,
    ngram: int = NGRAM,
) -> HalvesReport:
    """Compare two adjacent `n_bars`-bar spans held as separate compositions.

    Used for the corpus control, where a 16-bar span is two consecutive cache
    windows from one piece. Merging them into one composition first would mean
    re-deriving tick offsets, so instead each window keeps its own
    bar-zero-based grid and bar *i* of the first is compared with bar *i* of
    the second -- which is the same comparison `analyse_halves` makes.

    The lead channel is taken from the first span and applied to both, so a
    theme that reappears on a different voice is not counted as a return. That
    matches how the generated pieces are measured.
    """
    bar_ticks = ticks_per_bar() if bar_ticks is None else bar_ticks
    channel, melody_first = lead_melody(first)
    if channel:
        notes = sorted(
            (
                event
                for event in second.sounding_events(channel)
                if isinstance(event, Note)
            ),
            key=lambda event: event.tick,
        )
        melody_second = [(note.tick, note.pitch) for note in notes]
    else:
        melody_second = []
    return _report(
        bar_fingerprints(first, n_bars=n_bars, bar_ticks=bar_ticks),
        bar_fingerprints(second, n_bars=n_bars, bar_ticks=bar_ticks),
        melody_first=[p for _tick, p in melody_first],
        melody_second=[p for _tick, p in melody_second],
        channel=channel,
        ngram=ngram,
    )
