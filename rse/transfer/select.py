"""Ranking generated pieces by how close they sit to the corpus.

The lesson this encodes, learned the hard way: **musical quality is closeness
to the corpus distribution, not the maximum of any metric.** Of twelve Fermata
generations, the two a listener picked out had IOI entropy 1.22 against a
corpus median of 1.38, while the ten they passed over averaged 1.79 and
*overshot* it. Pushing a metric up had been treated as improvement all through
this project; the only human judgement available says otherwise.

So every rhythmic and melodic measure here is scored as a **two-sided**
distance from the corpus, normalized by the corpus's own interquartile range
so that each metric contributes in units of how much real NES music varies.
A piece at the corpus median scores 0.

Motif participation is the one exception, and it is weighted highest. It was
the strongest discriminator by a wide margin -- 0.71 median for the preferred
pieces against 0.15 for the rest, with four of ten at exactly zero -- so it is
scored as a two-sided distance like the others but given more weight, rather
than being maximized.

The evidence base is two preferred pieces. That is enough to build a ranking
worth testing and nowhere near enough to trust it, which is why
`scripts/fermata_select.py` renders both the top and the bottom of the ranking
for blind comparison. If the ranking does not predict preference on a fresh
batch, it is wrong and should be discarded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Metric -> weight. Motif participation carries most because it separated the
#: preferred pieces most sharply; the rest are the corpus-relative measures
#: that also tracked the judgement.
WEIGHTS: dict[str, float] = {
    "motif_note_participation": 3.0,
    "notes_per_bar": 2.0,
    "ioi_entropy": 2.0,
    "stepwise_fraction": 1.0,
    "ngram_repetition_rate": 1.0,
    "pitch_class_entropy": 1.0,
}

#: A piece must clear these to be ranked at all. They are hard musical
#: requirements rather than preferences: a piece with no percussion when the
#: plan asked for drums, or one that uses a single channel, is not an NES
#: arrangement regardless of how its metrics score.
MIN_CHANNELS = 3
MIN_EVENTS = 40


@dataclass
class Reference:
    """Corpus median and IQR per metric, from `rhythm_reference.json`."""

    median: dict[str, float] = field(default_factory=dict)
    iqr: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_rhythm_reference(cls, payload: dict[str, Any], *, split: str = "train") -> Reference:
        overall = payload["splits"][split]["overall"]
        median: dict[str, float] = {}
        iqr: dict[str, float] = {}
        for metric, quantiles in overall.items():
            if not quantiles:
                continue
            median[metric] = quantiles["median"]
            spread = quantiles["p75"] - quantiles["p25"]
            # A degenerate IQR would make that metric dominate the score.
            iqr[metric] = spread if spread > 1e-6 else 1.0
        return cls(median=median, iqr=iqr)

    def add(self, metric: str, median: float, p25: float, p75: float) -> None:
        self.median[metric] = median
        spread = p75 - p25
        self.iqr[metric] = spread if spread > 1e-6 else 1.0

    def distance(self, metric: str, value: float) -> float | None:
        """Absolute deviation from the corpus median, in IQR units."""
        if metric not in self.median:
            return None
        return abs(value - self.median[metric]) / self.iqr[metric]


@dataclass
class Scored:
    name: str
    score: float
    eligible: bool
    reason: str
    distances: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 4),
            "eligible": self.eligible,
            "reason": self.reason,
            "distances": {k: round(v, 4) for k, v in self.distances.items()},
            "metrics": dict(self.metrics),
        }


def score_piece(
    metrics: dict[str, Any],
    reference: Reference,
    *,
    weights: dict[str, float] | None = None,
) -> Scored:
    """Weighted corpus distance. Lower is closer to real NES music."""
    weights = weights or WEIGHTS
    name = str(metrics.get("name", "?"))

    channels = metrics.get("n_channels_used")
    if channels is None:
        channels = len(metrics.get("channels_used", []) or [])
    events = metrics.get("n_events", 0)
    if channels < MIN_CHANNELS:
        return Scored(name, float("inf"), False, f"only {channels} channels", metrics=metrics)
    if events < MIN_EVENTS:
        return Scored(name, float("inf"), False, f"only {events} events", metrics=metrics)

    distances: dict[str, float] = {}
    total = 0.0
    used = 0.0
    for metric, weight in weights.items():
        value = metrics.get(metric)
        if not isinstance(value, (int, float)):
            continue
        distance = reference.distance(metric, float(value))
        if distance is None:
            continue
        distances[metric] = distance
        total += weight * distance
        used += weight
    if not used:
        return Scored(name, float("inf"), False, "no comparable metrics", metrics=metrics)
    return Scored(name, total / used, True, "ok", distances=distances, metrics=metrics)


def rank(
    pieces: list[dict[str, Any]],
    reference: Reference,
    *,
    weights: dict[str, float] | None = None,
) -> list[Scored]:
    """Closest to the corpus first; ineligible pieces last."""
    scored = [score_piece(p, reference, weights=weights) for p in pieces]
    return sorted(scored, key=lambda s: (not s.eligible, s.score))
