"""Conditioning features and labels derived from musical evidence.

Three independent axes, each labelled from measurement rather than from track
titles, franchises or assumptions about composers:

- **motif recurrence** -- how much of the melody participates in ideas that
  return at separated positions.
- **rhythmic character** -- whether there is a stable pulse carrying real
  activity, not merely a high event count.
- **phrase behaviour** -- whether recurring phrase families come back altered.

Each axis can be `unknown`. A piece with too few melodic notes to form a motif
is not low-recurrence, and a piece with too few phrases is not lacking
variation; recording those as negatives would teach the model that sparse
music means "low", which is a different claim from the one the data supports.
Measured across NES-MDB, 22% of windows genuinely never restate a motif and
those are sparser but *more* stepwise than the rest, so low recurrence tracks
with melodic motion rather than against it.

Design decisions that came out of measurement
---------------------------------------------
**Exact pitch is the primary matcher, intervals are supplementary.** Across the
full train split the transposition-invariant and exact-pitch recurrence rates
differ by a median of 0.003, so transposition-invariance buys essentially
nothing on this corpus.

**Recurrence counts notes, not n-grams, and requires separation.** Adjacent
overlapping n-grams inflate a gram-based rate: a run of five identical notes
produces two "recurring" 4-grams without any musical return. Each participating
note is counted once and a match must start at least `MIN_MOTIF_SEPARATION`
notes away.

**Recurrence and bar duplication stay independent.** Real NES music at maximum
recurrence still varies nearly every bar -- consecutive bar duplication reaches
only 0.04 -- while the model's locked-loop failure duplicates adjacent bars
outright. Conflating them is what let a model satisfy "repetitive" by looping.

**Variation peaks at moderate recurrence.** Across recurrence bands the
variation rate runs 0.228, 0.291, 0.232, 0.163, 0.056: heavily repeating music
returns verbatim. So `motif=high` and `phrase=variation` are not the same
request and must be separately controllable.

**The lead is not always P1.** The melodic line is chosen by behaviour rather
than assumed, since pulse roles swap and the triangle sometimes carries melody.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from rse.events.pulse import detect_pulse
from rse.events.schema import Composition, Note

#: Version stamped into every record. Bump when a feature's meaning changes,
#: so a label set can never be silently mixed across definitions.
FEATURE_VERSION = "rse-conditioning-v1"

#: Motif lengths searched, in notes.
MOTIF_MIN = 4
MOTIF_MAX = 8

#: A recurrence must begin at least this many notes after the occurrence it
#: matches. Without it, overlapping adjacent windows count as a return.
MIN_MOTIF_SEPARATION = 4

#: Melodic notes needed before a motif label means anything.
MIN_MELODY_NOTES = 24

#: Onsets needed before a rhythm label means anything.
MIN_RHYTHM_ONSETS = 24

#: Phrase families needed before a phrase label means anything.
MIN_PHRASE_FAMILIES = 2

#: Phrases are this many fitted bars long.
PHRASE_BARS = 2

#: Similarity bands for phrase comparison, calibrated against the corpus
#: rather than chosen. Over 8338 phrase pairs from 89 train pieces, pairs
#: sharing nothing sit at a median similarity of 0.168 with a 90th
#: percentile of 0.427, so 0.45 is the point above which resemblance is not
#: chance; the 99th percentile is 0.941, so 0.92 marks a restatement rather
#: than a variation. Above `SAME` a pair is the same phrase again; between
#: the two it is a variant; below `RELATED` there is no family.
PHRASE_SAME = 0.92
PHRASE_RELATED = 0.45

UNKNOWN = "unknown"


@dataclass
class ConditioningFeatures:
    """Raw measurements behind the labels, kept so a label can be re-derived."""

    # motif
    motif_note_participation: float = 0.0
    motif_exact_ngram: float = 0.0
    motif_interval_ngram: float = 0.0
    n_melody_notes: int = 0
    lead_channel: str = ""
    # rhythm
    pulse_strength: float = 0.0
    pulse_period_sec: float = 0.0
    onset_rate: float = 0.0
    active_fraction: float = 0.0
    accent_regularity: float = 0.0
    #: -1.0 means not measurable: the piece lacks a triangle or noise voice.
    bass_percussion_agreement: float = -1.0
    n_onsets: int = 0
    # phrase
    phrase_variation_rate: float = 0.0
    n_phrase_families: int = 0
    n_phrases: int = 0
    # structure, kept independent of motif on purpose
    bars_consecutive_dup: float = 0.0
    bars_nonconsecutive_reuse: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in asdict(self).items()}


@dataclass
class ConditioningRecord:
    """Features, eligibility and labels for one musical unit."""

    piece_id: str
    split: str
    window_index: int
    features: ConditioningFeatures
    eligibility: dict[str, bool] = field(default_factory=dict)
    conditioning: dict[str, str] = field(default_factory=dict)
    feature_version: str = FEATURE_VERSION
    #: NES-MDB's ingested records carry game and track name only, so this stays
    #: None. Composer disjointness is a property of the official splits rather
    #: than something checkable per piece.
    composer: str | None = None
    game: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "piece_id": self.piece_id,
            "split": self.split,
            "window_index": self.window_index,
            "game": self.game,
            "composer": self.composer,
            "features": self.features.to_dict(),
            "eligibility": self.eligibility,
            "conditioning": self.conditioning,
            "feature_version": self.feature_version,
        }


# -- lead channel --------------------------------------------------------------


def lead_channel(composition: Composition) -> str:
    """Which channel carries the melody.

    Chosen by behaviour, not by assuming P1: pulse roles swap between pieces
    and the triangle sometimes carries the tune. Scored on note count, mean
    register and how much the line actually moves, since an accompaniment
    tends to sit still or hammer one pitch.
    """
    best, best_score = "", -1.0
    for channel in ("P1", "P2", "TR"):
        notes = [e for e in composition.sounding_events(channel) if isinstance(e, Note)]
        if len(notes) < 8:
            continue
        pitches = [n.pitch for n in notes]
        moves = [abs(b - a) for a, b in zip(pitches, pitches[1:], strict=False)]
        movement = statistics.fmean(moves) if moves else 0.0
        distinct = len(set(pitches)) / len(pitches)
        register = statistics.fmean(pitches) / 108.0
        score = distinct * 2.0 + register + min(movement, 6.0) / 6.0
        if score > best_score:
            best, best_score = channel, score
    return best


def melody_line(composition: Composition) -> tuple[str, list[int]]:
    """The lead channel and its pitches in time order."""
    channel = lead_channel(composition)
    if not channel:
        return "", []
    notes = sorted(
        (e for e in composition.sounding_events(channel) if isinstance(e, Note)),
        key=lambda e: e.tick,
    )
    return channel, [n.pitch for n in notes]


# -- motif recurrence ----------------------------------------------------------


def motif_note_participation(pitches: list[int]) -> float:
    """Share of melodic notes taking part in a motif that returns elsewhere.

    Counts each note once however many matches it belongs to, and requires a
    match to start at least `MIN_MOTIF_SEPARATION` notes away, so overlapping
    adjacent windows do not register as a return. A gram-based rate counts a
    run of five identical notes as two recurring 4-grams; this does not.
    """
    if len(pitches) < MOTIF_MIN * 2:
        return 0.0
    participating: set[int] = set()
    for width in range(MOTIF_MAX, MOTIF_MIN - 1, -1):
        if len(pitches) < width * 2:
            continue
        seen: dict[tuple[int, ...], list[int]] = {}
        for start in range(len(pitches) - width + 1):
            gram = tuple(pitches[start : start + width])
            # A motif needs internal movement. A run of one repeated pitch
            # satisfies any separation rule without being a musical return,
            # and eight identical notes scored full recurrence.
            if len(set(gram)) < 2:
                continue
            seen.setdefault(gram, []).append(start)
        for starts in seen.values():
            if len(starts) < 2:
                continue
            separated = [starts[0]]
            for start in starts[1:]:
                if start - separated[-1] >= max(width, MIN_MOTIF_SEPARATION):
                    separated.append(start)
            if len(separated) < 2:
                continue
            for start in separated:
                participating.update(range(start, start + width))
    return len(participating) / len(pitches)


def _ngram_rate(sequence: list[int], width: int) -> float:
    if len(sequence) < width:
        return 0.0
    grams = [tuple(sequence[i : i + width]) for i in range(len(sequence) - width + 1)]
    counts = Counter(grams)
    return sum(c for c in counts.values() if c > 1) / len(grams)


# -- rhythmic character --------------------------------------------------------


def rhythmic_features(composition: Composition) -> dict[str, float]:
    """Pulse stability, activity, accent regularity and bass/percussion agreement.

    Times come from the composition's own mapping. For the NES-MDB MIDI release
    that is 22050 ticks per beat at a nominal 120 BPM, so 44100 ticks per
    second — the nominal tempo is an artefact and is never used as a tempo.

    Pulse strength reuses `detect_pulse`, which is already validated against
    this corpus: it recovers the tracker row at a median 7.41 NTSC frames and
    its grid was confirmed by listening.
    """
    time = composition.time
    sounding = composition.sounding_events()
    if len(sounding) < MIN_RHYTHM_ONSETS:
        return {}
    starts = sorted(time.seconds(e.tick) for e in sounding)
    span = max(starts[-1] - starts[0], 1e-9)

    pulse = detect_pulse(composition)
    # Normalized so a very dense track is not called driving on density alone.
    strength = pulse.strength if pulse.found else 0.0

    # Active fraction: share of pulse periods containing at least one onset.
    period = pulse.period_sec if pulse.found and pulse.period_sec > 0 else 0.1
    slots = max(1, int(span / period))
    occupied = len({int((t - starts[0]) / period) for t in starts})
    active = min(1.0, occupied / slots)

    # Accent regularity: do the louder notes land consistently on the same
    # position within the beat? Measuring the spread of gaps between accents
    # instead clamped to 0.0 for almost every piece, since real accent spacing
    # varies. Velocity is the APU volume register, so this is real accent
    # information rather than an inferred dynamic.
    mean_velocity = statistics.fmean(e.velocity for e in sounding)
    accent = 0.0
    if pulse.found and pulse.period_ticks > 0:
        loud_rows = [
            int(round((e.tick - pulse.phase_ticks) / pulse.period_ticks)) % 4
            for e in sounding
            if e.velocity >= mean_velocity
        ]
        if len(loud_rows) >= 8:
            counts = Counter(loud_rows)
            accent = max(counts.values()) / len(loud_rows)

    # Bass and percussion agreement: share of noise onsets landing near a
    # triangle onset, which is how NES arrangements lock a kit to the bass.
    tri = sorted(time.seconds(e.tick) for e in composition.sounding_events("TR"))
    noise = sorted(time.seconds(e.tick) for e in composition.sounding_events("NO"))
    # None, not 0.0, when either voice is absent: a piece with no
    # percussion has no coordination to measure, and recording that as
    # zero agreement would be a different claim.
    agreement: float | None = None
    if tri and noise:
        tolerance = period / 4
        hits = 0
        for onset in noise:
            nearest = min(tri, key=lambda t: abs(t - onset))
            if abs(nearest - onset) <= tolerance:
                hits += 1
        agreement = hits / len(noise)

    return {
        "pulse_strength": strength,
        "pulse_period_sec": pulse.period_sec if pulse.found else 0.0,
        "onset_rate": len(starts) / span,
        "active_fraction": active,
        "accent_regularity": accent,
        "bass_percussion_agreement": agreement if agreement is not None else -1.0,
        "n_onsets": float(len(starts)),
    }


# -- phrase behaviour ----------------------------------------------------------


def _similarity(left: list[tuple[int, int]], right: list[tuple[int, int]]) -> float:
    """Normalized alignment similarity over (pitch, duration) pairs.

    The substitution cost is graded rather than all-or-nothing. Requiring exact
    tuple equality scored real restatements at a median similarity of 0.022,
    because a motif returning with any rhythmic change counted as entirely
    different -- which would have made every phrase its own family and the
    variation rate meaningless. Partial credit is the point: a variation shares
    pitches while altering rhythm, or the reverse.
    """
    if not left or not right:
        return 0.0
    rows, cols = len(left), len(right)
    previous = list(range(cols + 1))
    for i in range(1, rows + 1):
        current = [float(i)] + [0.0] * cols
        for j in range(1, cols + 1):
            lp, ld = left[i - 1]
            rp, rd = right[j - 1]
            if lp == rp and ld == rd:
                cost = 0.0
            elif lp == rp:
                cost = 0.35
            elif ld == rd:
                cost = 0.7
            else:
                cost = 1.0
            current[j] = min(previous[j] + 1.0, current[j - 1] + 1.0, previous[j - 1] + cost)
        previous = current
    return max(0.0, 1.0 - previous[cols] / max(rows, cols))


def phrase_features(composition: Composition) -> dict[str, float]:
    """Recurring phrase families, and how many come back altered.

    Phrases are fixed spans of the piece's own fitted bars rather than guessed
    from rests: a rest can be articulation and a phrase can turn without one.

    A family needs at least two phrases above `PHRASE_RELATED`, so two merely
    different phrases never count as variation -- a variation has to share
    recognizable material. Within a family, a pair above `PHRASE_SAME` is a
    restatement and anything between the thresholds is a variant.
    """
    fit = detect_pulse(composition)
    if not fit.found or fit.period_ticks <= 0:
        return {}
    bar_ticks = fit.period_ticks * 16
    span_ticks = bar_ticks * PHRASE_BARS
    channel, _ = melody_line(composition)
    if not channel:
        return {}
    notes = sorted(
        (e for e in composition.sounding_events(channel) if isinstance(e, Note)),
        key=lambda e: e.tick,
    )
    if len(notes) < MIN_MELODY_NOTES:
        return {}

    phrases: list[list[tuple[int, int]]] = []
    end = composition.end_tick
    start = notes[0].tick
    while start < end:
        inside = [n for n in notes if start <= n.tick < start + span_ticks]
        if len(inside) >= 4:
            phrases.append(
                [(n.pitch, int(round(n.duration / max(fit.period_ticks, 1e-9)))) for n in inside]
            )
        start += span_ticks
    if len(phrases) < 2:
        return {}

    families: list[list[int]] = []
    for index in range(len(phrases)):
        placed = False
        for family in families:
            if _similarity(phrases[index], phrases[family[0]]) >= PHRASE_RELATED:
                family.append(index)
                placed = True
                break
        if not placed:
            families.append([index])

    recurring = [f for f in families if len(f) >= 2]
    if not recurring:
        return {"phrase_variation_rate": 0.0, "n_phrase_families": 0, "n_phrases": len(phrases)}
    varied = 0
    for family in recurring:
        anchor = phrases[family[0]]
        if any(_similarity(anchor, phrases[i]) < PHRASE_SAME for i in family[1:]):
            varied += 1
    return {
        "phrase_variation_rate": varied / len(recurring),
        "n_phrase_families": float(len(recurring)),
        "n_phrases": float(len(phrases)),
    }


# -- assembly ------------------------------------------------------------------


def extract_features(composition: Composition) -> ConditioningFeatures:
    """Measure all three axes. Emptiness is reported, never defaulted to zero."""
    from rse.training.evaluate import bar_signatures

    features = ConditioningFeatures()
    channel, pitches = melody_line(composition)
    features.lead_channel = channel
    features.n_melody_notes = len(pitches)
    if pitches:
        features.motif_note_participation = motif_note_participation(pitches)
        features.motif_exact_ngram = _ngram_rate(pitches, 4)
        intervals = [b - a for a, b in zip(pitches, pitches[1:], strict=False)]
        features.motif_interval_ngram = _ngram_rate(intervals, 4)

    for key, value in rhythmic_features(composition).items():
        setattr(features, key, int(value) if key == "n_onsets" else value)
    for key, value in phrase_features(composition).items():
        setattr(
            features,
            key,
            int(value) if key in ("n_phrase_families", "n_phrases") else value,
        )

    signatures = bar_signatures(composition)
    if len(signatures) > 1:
        features.bars_consecutive_dup = sum(
            1 for i in range(1, len(signatures)) if signatures[i] == signatures[i - 1]
        ) / (len(signatures) - 1)
        features.bars_nonconsecutive_reuse = sum(
            1
            for i in range(len(signatures))
            if any(signatures[i] == signatures[j] for j in range(i + 2, len(signatures)))
        ) / len(signatures)
    return features


def eligibility(features: ConditioningFeatures) -> dict[str, bool]:
    """Whether each axis has enough evidence to carry a label."""
    return {
        "motif": features.n_melody_notes >= MIN_MELODY_NOTES,
        "rhythm": features.n_onsets >= MIN_RHYTHM_ONSETS,
        "phrase": features.n_phrase_families >= MIN_PHRASE_FAMILIES,
    }


@dataclass
class Thresholds:
    """Label boundaries, fitted on the training split and then frozen.

    Fitted from eligible training examples only, and never refitted against
    validation or test. The zero-recurrence population is handled by an
    absolute floor rather than by forcing equal-sized bins: 22% of the corpus
    genuinely never restates a motif, and a tercile split would relabel much
    of that as medium.
    """

    motif_low: float = 0.15
    motif_high: float = 0.45
    rhythm_sparse_onset: float = 8.0
    rhythm_driving_onset: float = 14.0
    rhythm_driving_pulse: float = 0.55
    phrase_stable: float = 0.25
    phrase_variation: float = 0.60
    fitted_on: str = ""
    n_fitted: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def label(features: ConditioningFeatures, thresholds: Thresholds) -> dict[str, str]:
    """Labels for all three axes, `unknown` where evidence is insufficient."""
    ok = eligibility(features)
    out: dict[str, str] = {}

    if not ok["motif"]:
        out["motif"] = UNKNOWN
    elif features.motif_note_participation >= thresholds.motif_high:
        out["motif"] = "high"
    elif features.motif_note_participation <= thresholds.motif_low:
        out["motif"] = "low"
    else:
        out["motif"] = "medium"

    if not ok["rhythm"]:
        out["rhythm"] = UNKNOWN
    elif (
        features.onset_rate >= thresholds.rhythm_driving_onset
        and features.pulse_strength >= thresholds.rhythm_driving_pulse
    ):
        # Both conditions, deliberately: a dense scatter of noise triggers has
        # a high onset rate without any pulse, and is not driving.
        out["rhythm"] = "driving"
    elif features.onset_rate <= thresholds.rhythm_sparse_onset:
        out["rhythm"] = "sparse"
    else:
        out["rhythm"] = "moderate"

    if not ok["phrase"]:
        out["phrase"] = UNKNOWN
    elif features.phrase_variation_rate >= thresholds.phrase_variation:
        out["phrase"] = "variation"
    elif features.phrase_variation_rate <= thresholds.phrase_stable:
        out["phrase"] = "stable"
    else:
        out["phrase"] = "contrast"
    return out


def fit_thresholds(features: list[ConditioningFeatures], *, split: str = "train") -> Thresholds:
    """Fit label boundaries from eligible examples in one split.

    Motif boundaries come from the terciles of the *non-zero* population with
    an absolute floor for the zero mode, because the corpus is multimodal and
    equal-sized bins would split a genuine mode across two labels.
    """
    eligible_motif = [f.motif_note_participation for f in features if eligibility(f)["motif"]]
    eligible_rhythm = [f for f in features if eligibility(f)["rhythm"]]
    eligible_phrase = [f.phrase_variation_rate for f in features if eligibility(f)["phrase"]]

    thresholds = Thresholds(fitted_on=split, n_fitted=len(features))
    nonzero = sorted(v for v in eligible_motif if v > 0.01)
    if len(nonzero) >= 10:
        thresholds.motif_low = round(nonzero[int(0.33 * (len(nonzero) - 1))], 4)
        thresholds.motif_high = round(nonzero[int(0.67 * (len(nonzero) - 1))], 4)
    if len(eligible_rhythm) >= 10:
        rates = sorted(f.onset_rate for f in eligible_rhythm)
        strengths = sorted(f.pulse_strength for f in eligible_rhythm)
        thresholds.rhythm_sparse_onset = round(rates[int(0.33 * (len(rates) - 1))], 4)
        thresholds.rhythm_driving_onset = round(rates[int(0.67 * (len(rates) - 1))], 4)
        thresholds.rhythm_driving_pulse = round(strengths[int(0.50 * (len(strengths) - 1))], 4)
    if len(eligible_phrase) >= 10:
        ordered = sorted(eligible_phrase)
        thresholds.phrase_stable = round(ordered[int(0.33 * (len(ordered) - 1))], 4)
        thresholds.phrase_variation = round(ordered[int(0.67 * (len(ordered) - 1))], 4)
    return thresholds


def agreement(left: dict[str, str], right: dict[str, str]) -> dict[str, bool]:
    """Whether two views of the same music produce the same labels.

    Gate 2 exists because of a measured hazard: round-tripping drops IOI
    entropy from 2.37 to 0.47 on matched surviving notes, so a rhythmic
    distinction present in the source may be absent from what the model sees.
    Labelling a piece `driving` on evidence the tokenizer destroys would teach
    the model nothing.
    """
    return {axis: left.get(axis) == right.get(axis) for axis in ("motif", "rhythm", "phrase")}


def describe_math(value: float) -> float:
    """Guard against NaN leaking into a label record."""
    return 0.0 if value is None or math.isnan(value) else value
