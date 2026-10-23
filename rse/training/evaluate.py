"""Musical evaluation.

Proposal 11.2 lists six measurable qualities plus human listening. Validation
loss is in the trainer; the rest are here, as explicit proxies rather than
vague claims.

Every metric is a *proxy*, and the docstrings say what each one actually
measures, because the gap between "in-scale note fraction" and "sounds
harmonically coherent" is where this kind of evaluation misleads people.

Proposal 11.3 makes the comparison against the rule-based generator a
requirement, not a courtesy: if the model cannot beat the rules, the machine
learning has not earned its place. `compare` is that test.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from rse.data import dedupe
from rse.events.convert import composition_to_intake
from rse.events.schema import Composition, Note
from rse.generation.loops import validate_loop
from rse.generation.theory import SCALES
from rse.hardware.constraints import ConstraintEngine
from rse.hardware.profile import HardwareProfile

#: Scales considered when inferring a piece's key.
KEY_CANDIDATES = ("major", "minor", "harmonic_minor")

#: Melodic n-gram length for the internal repetition measure.
REPETITION_NGRAM = 4

#: Jaccard similarity against the corpus at or above which a sample is flagged
#: for review as potentially memorized (MVP criterion 10).
MEMORIZATION_THRESHOLD = 0.6

#: A healthy share of bars that repeat earlier material. Outside this band the
#: piece is either a loop of one bar or has no structure at all.
REPETITION_BAND = (0.1, 0.85)


@dataclass
class MusicMetrics:
    """Measured properties of one composition."""

    n_notes: int = 0
    n_bars: int = 0
    notes_per_bar: float = 0.0
    n_channels_used: int = 0

    # Harmonic consistency
    key: str = "unknown"
    in_scale_fraction: float = 0.0
    pitch_class_entropy: float = 0.0

    # Repetition balance
    repeated_bar_fraction: float = 0.0
    unique_bar_fraction: float = 0.0
    ngram_repetition_rate: float = 0.0

    # Melodic shape
    stepwise_fraction: float = 0.0
    large_leap_fraction: float = 0.0

    # Technical
    hardware_compliant: bool = False
    #: Whether it was compliant *before* the constraint engine touched it.
    #: Informative for model output, which carries no guarantee; always true
    #: for the rule-based generator, which is compliant by construction.
    raw_hardware_compliant: bool = True
    loop_ok: bool = False
    grid_aligned_fraction: float = 0.0

    # Originality
    max_corpus_similarity: float | None = None
    flagged_as_memorized: bool = False

    warnings: list[str] = field(default_factory=list)

    @property
    def repetition_in_band(self) -> bool:
        low, high = REPETITION_BAND
        return low <= self.repeated_bar_fraction <= high

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_notes": self.n_notes,
            "n_bars": self.n_bars,
            "notes_per_bar": round(self.notes_per_bar, 2),
            "n_channels_used": self.n_channels_used,
            "key": self.key,
            "in_scale_fraction": round(self.in_scale_fraction, 4),
            "pitch_class_entropy": round(self.pitch_class_entropy, 4),
            "repeated_bar_fraction": round(self.repeated_bar_fraction, 4),
            "unique_bar_fraction": round(self.unique_bar_fraction, 4),
            "ngram_repetition_rate": round(self.ngram_repetition_rate, 4),
            "repetition_in_band": self.repetition_in_band,
            "stepwise_fraction": round(self.stepwise_fraction, 4),
            "large_leap_fraction": round(self.large_leap_fraction, 4),
            "hardware_compliant": self.hardware_compliant,
            "raw_hardware_compliant": self.raw_hardware_compliant,
            "loop_ok": self.loop_ok,
            "grid_aligned_fraction": round(self.grid_aligned_fraction, 4),
            "max_corpus_similarity": (
                None if self.max_corpus_similarity is None else round(self.max_corpus_similarity, 4)
            ),
            "flagged_as_memorized": self.flagged_as_memorized,
            "warnings": self.warnings[:10],
        }


def infer_key(pitches: list[int]) -> tuple[str, float]:
    """Best-fitting key, and the fraction of notes inside it.

    A simple template match over all 12 tonics and three scales. It does not
    model modulation, so a piece that changes key scores lower than it deserves
    — the number is a consistency proxy, not a key-detection claim.
    """
    if not pitches:
        return "unknown", 0.0
    classes = Counter(pitch % 12 for pitch in pitches)
    total = sum(classes.values())
    best_name = "unknown"
    best_score = 0.0
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    for scale_name in KEY_CANDIDATES:
        intervals = SCALES[scale_name]
        for tonic in range(12):
            members = {(tonic + interval) % 12 for interval in intervals}
            score = sum(count for pitch_class, count in classes.items() if pitch_class in members)
            fraction = score / total
            if fraction > best_score:
                best_score = fraction
                best_name = f"{names[tonic]} {scale_name}"
    return best_name, best_score


def pitch_class_entropy(pitches: list[int]) -> float:
    """Shannon entropy of pitch-class use, in bits.

    Low means a few notes dominate; high means all twelve are used evenly,
    which in this idiom usually means aimlessness rather than sophistication.
    """
    if not pitches:
        return 0.0
    counts = Counter(pitch % 12 for pitch in pitches)
    total = sum(counts.values())
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def bar_signatures(
    composition: Composition, *, beats_per_bar: int = 4
) -> list[tuple[tuple[int, int, int], ...]]:
    """A comparable fingerprint of each bar's content."""
    ticks_per_bar = composition.ticks_per_beat * beats_per_bar
    if ticks_per_bar <= 0:
        return []
    bars: dict[int, list[tuple[int, int, int]]] = {}
    for event in composition.sounding_events():
        bar, offset = divmod(event.tick, ticks_per_bar)
        value = event.pitch if isinstance(event, Note) else event.value
        channel = composition.channel_ids.index(event.channel_id)
        bars.setdefault(bar, []).append((offset, channel, value))
    if not bars:
        return []
    return [tuple(sorted(bars.get(bar, []))) for bar in range(max(bars) + 1)]


def repetition_metrics(signatures: list[tuple]) -> tuple[float, float]:
    """`(fraction of bars repeating an earlier bar, fraction unique)`."""
    if not signatures:
        return 0.0, 0.0
    seen: set[tuple] = set()
    repeated = 0
    for signature in signatures:
        if signature in seen:
            repeated += 1
        seen.add(signature)
    return repeated / len(signatures), len(seen) / len(signatures)


def ngram_repetition_rate(pitches: list[int], n: int = REPETITION_NGRAM) -> float:
    """Share of melodic n-grams that occur more than once.

    Measured on pitch intervals, so it is transposition-invariant: a sequence
    repeated a fifth higher counts as repetition, which is what a listener
    hears.
    """
    if len(pitches) < n + 1:
        return 0.0
    intervals = [b - a for a, b in zip(pitches, pitches[1:], strict=False)]
    grams = [tuple(intervals[i : i + n]) for i in range(len(intervals) - n + 1)]
    if not grams:
        return 0.0
    counts = Counter(grams)
    return sum(count for count in counts.values() if count > 1) / len(grams)


def melodic_shape(pitches: list[int]) -> tuple[float, float]:
    """`(stepwise fraction, large-leap fraction)` of consecutive intervals."""
    if len(pitches) < 2:
        return 0.0, 0.0
    intervals = [abs(b - a) for a, b in zip(pitches, pitches[1:], strict=False)]
    stepwise = sum(1 for i in intervals if 0 < i <= 2)
    leaps = sum(1 for i in intervals if i > 7)
    return stepwise / len(intervals), leaps / len(intervals)


def grid_alignment(
    composition: Composition, *, steps_per_bar: int = 16, beats_per_bar: int = 4
) -> float:
    """Fraction of notes landing exactly on the grid."""
    events = composition.sounding_events()
    if not events:
        return 0.0
    step = composition.ticks_per_beat * beats_per_bar // steps_per_bar
    if step <= 0:
        return 0.0
    return sum(1 for e in events if e.tick % step == 0) / len(events)


def melody_pitches(composition: Composition, channel_id: str | None = None) -> list[int]:
    """Pitches in time order, from pitched channels only.

    Noise values are hardware periods and must never be treated as pitches, so
    they are excluded here as everywhere else.
    """
    events = [
        e
        for e in composition.sounding_events()
        if isinstance(e, Note) and (channel_id is None or e.channel_id == channel_id)
    ]
    return [e.pitch for e in sorted(events, key=lambda e: (e.tick, e.channel_id))]


def evaluate_composition(
    composition: Composition,
    profile: HardwareProfile,
    *,
    corpus_fingerprints: list[set[int]] | None = None,
    beats_per_bar: int = 4,
) -> MusicMetrics:
    """Measure one composition against every proxy in proposal 11.2."""
    metrics = MusicMetrics()
    sounding = composition.sounding_events()
    metrics.n_notes = len(sounding)
    metrics.n_channels_used = len({e.channel_id for e in sounding})

    signatures = bar_signatures(composition, beats_per_bar=beats_per_bar)
    metrics.n_bars = len(signatures)
    metrics.notes_per_bar = metrics.n_notes / metrics.n_bars if metrics.n_bars else 0.0
    metrics.repeated_bar_fraction, metrics.unique_bar_fraction = repetition_metrics(signatures)

    # Harmony is judged on the lead line, where a key is actually expressed,
    # rather than on every channel at once.
    lead = melody_pitches(composition, "P1") or melody_pitches(composition)
    metrics.key, metrics.in_scale_fraction = infer_key(lead)
    metrics.pitch_class_entropy = pitch_class_entropy(lead)
    metrics.ngram_repetition_rate = ngram_repetition_rate(lead)
    metrics.stepwise_fraction, metrics.large_leap_fraction = melodic_shape(lead)
    metrics.grid_aligned_fraction = grid_alignment(composition, beats_per_bar=beats_per_bar)

    engine = ConstraintEngine(profile)
    metrics.hardware_compliant = not engine.validate(composition)
    loop_report = validate_loop(composition_to_intake(composition, profile=profile))
    metrics.loop_ok = loop_report.ok
    if not loop_report.ok:
        metrics.warnings.extend(loop_report.issues[:3])

    if corpus_fingerprints:
        own = dedupe.fingerprint(composition_to_intake(composition, profile=profile))
        if own:
            metrics.max_corpus_similarity = max(
                dedupe.jaccard(own, other) for other in corpus_fingerprints
            )
            metrics.flagged_as_memorized = metrics.max_corpus_similarity >= MEMORIZATION_THRESHOLD
    return metrics


def corpus_fingerprints(
    compositions: list[Composition], profile: HardwareProfile
) -> list[set[int]]:
    """Melodic fingerprints of a training corpus, for originality screening."""
    out = []
    for composition in compositions:
        fingerprint = dedupe.fingerprint(composition_to_intake(composition, profile=profile))
        if fingerprint:
            out.append(fingerprint)
    return out


@dataclass
class Aggregate:
    """Mean metrics over a set of compositions, for comparing systems."""

    label: str
    n: int = 0
    means: dict[str, float] = field(default_factory=dict)
    rates: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "n": self.n,
            "means": {k: round(v, 4) for k, v in self.means.items()},
            "rates": {k: round(v, 4) for k, v in self.rates.items()},
        }


#: Numeric fields averaged when aggregating.
_MEAN_FIELDS = (
    "n_notes",
    "n_bars",
    "notes_per_bar",
    "n_channels_used",
    "in_scale_fraction",
    "pitch_class_entropy",
    "repeated_bar_fraction",
    "unique_bar_fraction",
    "ngram_repetition_rate",
    "stepwise_fraction",
    "large_leap_fraction",
    "grid_aligned_fraction",
)

#: Boolean fields reported as a rate when aggregating.
_RATE_FIELDS = (
    "hardware_compliant",
    "raw_hardware_compliant",
    "loop_ok",
    "repetition_in_band",
    "flagged_as_memorized",
)


def aggregate(label: str, metrics: list[MusicMetrics]) -> Aggregate:
    """Average a set of per-composition metrics."""
    result = Aggregate(label=label, n=len(metrics))
    if not metrics:
        return result
    for name in _MEAN_FIELDS:
        values = [float(getattr(m, name)) for m in metrics]
        result.means[name] = sum(values) / len(values)
    for name in _RATE_FIELDS:
        values = [bool(getattr(m, name)) for m in metrics]
        result.rates[name] = sum(values) / len(values)
    similarities = [m.max_corpus_similarity for m in metrics if m.max_corpus_similarity is not None]
    if similarities:
        result.means["max_corpus_similarity"] = sum(similarities) / len(similarities)
        result.means["worst_corpus_similarity"] = max(similarities)
    return result


def compare(baseline: Aggregate, candidate: Aggregate) -> dict[str, Any]:
    """Put the model beside the rule-based baseline (proposal 11.3).

    No single verdict is produced, because these proxies do not justify one.
    What it gives is the honest comparison: where the model is ahead, where it
    is behind, and whether it clears the technical bar at all.
    """
    deltas = {
        name: round(candidate.means.get(name, 0.0) - baseline.means.get(name, 0.0), 4)
        for name in sorted(set(baseline.means) | set(candidate.means))
    }
    rate_deltas = {
        name: round(candidate.rates.get(name, 0.0) - baseline.rates.get(name, 0.0), 4)
        for name in sorted(set(baseline.rates) | set(candidate.rates))
    }
    technical_bar = {
        "hardware_compliant": candidate.rates.get("hardware_compliant", 0.0) >= 1.0,
        "loop_ok": candidate.rates.get("loop_ok", 0.0) >= 1.0,
        "no_memorization_flags": candidate.rates.get("flagged_as_memorized", 1.0) == 0.0,
    }
    return {
        "baseline": baseline.to_dict(),
        "candidate": candidate.to_dict(),
        "mean_deltas": deltas,
        "rate_deltas": rate_deltas,
        "technical_bar": technical_bar,
        "clears_technical_bar": all(technical_bar.values()),
    }
