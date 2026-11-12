"""The frozen distributional statistics and selection objective.

One implementation, used for the reference population, the development
candidates and any later reporting, so a generated arm and the corpus can
never be measured by different code.

Per §3 and §4 of the frozen procedure:

- **attacks per second** — note starts / span, per excerpt, then median
- **mean simultaneity** — mean notes per distinct onset, per excerpt, then
  median
- **interval histogram** — per voice, consecutive pitched starts, clipped to
  +/-12 semitones, pooled across excerpts
- **objective** — relative difference on the two scalars plus total variation
  on the histogram, all weights 1, lower is closer to the corpus

**Per-excerpt then median, never pooled.** Level 2 pooled decoded event lists
and reported 296 attacks/s against a true corpus range of 13-26, because every
decoded item restarts its clock near zero.
"""

from __future__ import annotations

import statistics as st
from typing import Any, Sequence

CLIP = 12


def excerpt_statistics(notes: Sequence[dict[str, Any]], *,
                       grid_hz: float = 4000.0) -> dict[str, Any] | None:
    """Statistics for ONE excerpt, or None if it has no notes.

    Returning None is what makes Amendment A1 enforceable: a zero-note
    decoded performance has no defined statistics, so it cannot be scored and
    its candidate is ineligible.
    """
    if not notes:
        return None
    span_ticks = max(n["start_tick"] for n in notes)
    span = span_ticks / grid_hz
    if span <= 0:
        return None
    onsets: dict[int, int] = {}
    for note in notes:
        onsets[note["start_tick"]] = onsets.get(note["start_tick"], 0) + 1
    by_voice: dict[int, list[float]] = {}
    for note in notes:
        if note["midi"] is None:
            continue
        by_voice.setdefault(note["slot"], []).append(float(note["midi"]))
    intervals: dict[int, int] = {}
    for pitches in by_voice.values():
        for a, b in zip(pitches, pitches[1:]):
            step = max(-CLIP, min(CLIP, int(round(b - a))))
            intervals[step] = intervals.get(step, 0) + 1
    return {
        "n_notes": len(notes),
        "span_sec": span,
        "attacks_per_sec": len(notes) / span,
        "mean_simultaneity": sum(onsets.values()) / len(onsets),
        "interval_counts": intervals,
    }


def aggregate(per_excerpt: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Median of the scalars; pooled, normalised interval histogram."""
    if not per_excerpt:
        raise ValueError("no excerpts to aggregate")
    pooled: dict[int, int] = {}
    for stats in per_excerpt:
        for step, count in stats["interval_counts"].items():
            pooled[int(step)] = pooled.get(int(step), 0) + count
    total = sum(pooled.values())
    return {
        "n_excerpts": len(per_excerpt),
        "attacks_per_sec": st.median(s["attacks_per_sec"] for s in per_excerpt),
        "mean_simultaneity": st.median(s["mean_simultaneity"]
                                       for s in per_excerpt),
        "interval_histogram": {str(k): v / total for k, v in sorted(pooled.items())}
                              if total else {},
        "n_intervals": total,
    }


def objective(candidate: dict[str, Any], reference: dict[str, Any]) -> float:
    """The frozen objective. Lower is closer to the corpus."""
    if reference["attacks_per_sec"] <= 0 or reference["mean_simultaneity"] <= 0:
        return float("inf")
    density = abs(candidate["attacks_per_sec"] - reference["attacks_per_sec"]) \
        / reference["attacks_per_sec"]
    simultaneity = abs(candidate["mean_simultaneity"]
                       - reference["mean_simultaneity"]) \
        / reference["mean_simultaneity"]
    keys = set(candidate["interval_histogram"]) | set(
        reference["interval_histogram"])
    variation = 0.5 * sum(
        abs(candidate["interval_histogram"].get(k, 0.0)
            - reference["interval_histogram"].get(k, 0.0)) for k in keys)
    return density + simultaneity + variation


def tie_break_key(row: dict[str, Any]) -> tuple:
    """Declared ordering: lowest distance, then lowest temperature, then
    highest top-p. Distances compared at 6 decimal places."""
    return (round(row["objective"], 6), row["temperature"], -row["top_p"])
