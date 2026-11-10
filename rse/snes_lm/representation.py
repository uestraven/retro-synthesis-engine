"""A token representation for snesmdb events.

Separate from `rse.tokenizers` on purpose. That package holds
`rse-repr/1.0.0-d4k4-tempo1.01`, which is frozen, NES-specific (four channels,
duty-cycle timbre, vocabulary 555) and must not be modified. Nothing here
touches it.

Design decisions, each with the measurement or constraint behind it:

**Time is a delta from the previous event, bucketed by TRAIN-SPLIT quantiles.**
The release publishes a per-track row grid read from the SPC700 timer register,
which is musically the right target, and it is not usable: it is found for only
60.2% of test tracks against 68.9% of train (a split-dependent asymmetry), and
only about 51% of the grids it does find are plausible, with `rows_per_second`
spanning 1.18 to 500. Unquantised `onset_sec` covers every event and assumes no
grid. See `docs/EXPERIMENT_PREREGISTRATION.md` in the snesmdb repository.

**Quantiles are fitted on train only.** The test split is sealed until model
selection is final, and a bucket boundary fitted on it would be a leak.

**Four tokens per event, fixed stride: TIME, VOICE, PITCH, DUR.** A fixed
stride makes the registered "structurally valid generated sequences" criterion
decidable rather than a matter of opinion: a sequence is valid when its tokens
fall in that cycle with each drawn from the right group.

**`origin == "legato_step"` is excluded.** A legato step is a pitch change
inside a sounding note, not an attack. The existing interval baseline excluded
it by definition, and the registration requires baselines on an identical
representation, so this follows suit rather than inventing a second population.

**Unpitched events are kept, with an explicit token.** `pitch_status` other
than `estimated` means there is no pitch -- noise, pitch modulation, or an
unrecoverable estimate. Dropping those events would delete real musical
activity (percussion above all); giving them a fabricated pitch would be worse.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

#: Tokens per event. TIME, VOICE, PITCH, DUR.
STRIDE = 4

#: Semitone range kept as pitch tokens. `midi_note` is a per-sample ESTIMATE,
#: fractional and not clamped to MIDI's domain -- measured -36.74 to +131.09
#: across the corpus -- so anything outside a musical register is far more
#: likely an octave error in the estimate than a note someone wrote. Such
#: events become PITCH_OOR rather than being clamped into a plausible-looking
#: wrong note or silently dropped.
PITCH_MIN, PITCH_MAX = 12, 108

VOICES = 8


@dataclass(frozen=True)
class Vocab:
    """Token ids, grouped so structural validity is checkable."""

    time_buckets: tuple[float, ...]
    dur_buckets: tuple[float, ...]
    #: Positive deltas are capped here before bucketing; see `fit_vocab`.
    max_delta_sec: float = 2.0
    #: Representative value for each bucket, for DECODE only, fitted on train.
    #:
    #: Decode originally returned each bucket's UPPER EDGE, which is wrong by
    #: however wide the bucket is. For durations that was severe: quantile
    #: spacing put the top bucket at 1.4871 to 141.4345 s, a 95x span, so 1 in
    #: 32 notes decoded as a 141-second note. Measured over 2,483,636 train
    #: notes, the upper-edge convention gives a mean error of 4,351.6 ms and a
    #: p99 of 139,534.8 ms; each bucket's observed MEDIAN gives 49.3 ms and
    #: 703.6 ms -- 88x and 198x better.
    #:
    #: This is a DECODE-SIDE correction only. Encoding is unchanged, the
    #: bucket edges are unchanged, and Level 1 is unaffected because its
    #: likelihood is computed on token ids and never decodes. Empty tuples
    #: fall back to the upper edge, preserving old behaviour.
    time_representatives: tuple[float, ...] = ()
    dur_representatives: tuple[float, ...] = ()

    def time_value(self, index: int) -> float:
        """Seconds for a positive time-bucket index (0-based into buckets)."""
        if self.time_representatives and index < len(self.time_representatives):
            return self.time_representatives[index]
        return self.time_buckets[index]

    def dur_value(self, index: int) -> float:
        if self.dur_representatives and index < len(self.dur_representatives):
            return self.dur_representatives[index]
        return self.dur_buckets[index]

    PAD: int = 0
    BOS: int = 1
    EOS: int = 2

    @property
    def n_special(self) -> int:
        return 3

    @property
    def time_base(self) -> int:
        return self.n_special

    @property
    def time_same(self) -> int:
        """Delta exactly zero: this event shares its onset with the previous.

        A dedicated token rather than a bucket, because 51.9% of consecutive
        pairs are this case and it is categorical, not continuous.
        """
        return self.time_base

    @property
    def n_time(self) -> int:
        return len(self.time_buckets) + 1  # positive buckets plus TIME_SAME

    @property
    def voice_base(self) -> int:
        return self.time_base + self.n_time

    @property
    def pitch_base(self) -> int:
        return self.voice_base + VOICES

    @property
    def n_pitch(self) -> int:
        # the semitone range, plus PITCH_NONE and PITCH_OOR
        return (PITCH_MAX - PITCH_MIN + 1) + 2

    @property
    def pitch_none(self) -> int:
        return self.pitch_base + (PITCH_MAX - PITCH_MIN + 1)

    @property
    def pitch_oor(self) -> int:
        return self.pitch_none + 1

    @property
    def dur_base(self) -> int:
        return self.pitch_base + self.n_pitch

    @property
    def n_dur(self) -> int:
        return len(self.dur_buckets) + 1  # buckets plus DUR_NONE

    @property
    def dur_none(self) -> int:
        return self.dur_base + len(self.dur_buckets)

    @property
    def size(self) -> int:
        return self.dur_base + self.n_dur

    # --- group membership, used by the validity checker ---

    def group(self, token: int) -> str:
        if token < self.n_special:
            return ("pad", "bos", "eos")[token]
        if token < self.voice_base:
            return "time"
        if token < self.pitch_base:
            return "voice"
        if token < self.dur_base:
            return "pitch"
        if token < self.size:
            return "dur"
        return "invalid"

    def to_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "n_time": self.n_time,
            "n_voice": VOICES,
            "n_pitch": self.n_pitch,
            "n_dur": self.n_dur,
            "pitch_min": PITCH_MIN,
            "pitch_max": PITCH_MAX,
            "stride": STRIDE,
            "max_delta_sec": self.max_delta_sec,
            "time_representatives": list(self.time_representatives),
            "dur_representatives": list(self.dur_representatives),
            "time_buckets": list(self.time_buckets),
            "dur_buckets": list(self.dur_buckets),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Vocab":
        return cls(
            time_buckets=tuple(data["time_buckets"]),
            dur_buckets=tuple(data["dur_buckets"]),
            max_delta_sec=float(data.get("max_delta_sec", 2.0)),
            time_representatives=tuple(data.get("time_representatives", ())),
            dur_representatives=tuple(data.get("dur_representatives", ())),
        )


def _bucket(value: float, edges: Sequence[float]) -> int:
    """Index of the bucket `value` falls in. Edges are upper bounds."""
    low, high = 0, len(edges)
    while low < high:
        mid = (low + high) // 2
        if value <= edges[mid]:
            high = mid
        else:
            low = mid + 1
    return min(low, len(edges) - 1)


def _quantile_edges(values: list[float], n: int) -> tuple[float, ...]:
    """Quantile bucket upper bounds, deduplicated and sorted."""
    values = sorted(v for v in values if v is not None)
    if not values:
        return (0.0,)
    out: list[float] = []
    for i in range(1, n + 1):
        index = min(len(values) - 1, (i * len(values)) // n)
        edge = values[index]
        if not out or edge > out[-1]:
            out.append(edge)
    return tuple(out)


def _log_edges(lo: float, hi: float, n: int) -> tuple[float, ...]:
    """Geometrically spaced upper bounds from `lo` to `hi`."""
    if n < 2 or hi <= lo:
        return (hi,)
    ratio = (hi / lo) ** (1.0 / (n - 1))
    return tuple(lo * ratio**i for i in range(n))


def fit_vocab(
    deltas: Iterable[float],
    durations: Iterable[float],
    *,
    n_time: int = 96,
    n_dur: int = 32,
    max_delta_sec: float = 2.0,
    min_delta_sec: float = 0.0005,
) -> Vocab:
    """Fit bucket edges. TRAIN SPLIT ONLY.

    Three corrections, each forced by a measurement rather than taste. The
    error figures are positive-delta quantisation error over all 1,224,840
    positive train deltas.

    **Zero gets its own token and is excluded from the fit.** 51.9% of
    consecutive key-on pairs across the full corpus share an onset exactly,
    because SNES drivers key several voices on one sequencer tick. Fitting
    quantiles over that spent half the buckets on a single value and collapsed
    48 requested buckets to 25. Zero is categorical -- "same tick" -- not a
    point on a continuum.

    **Long gaps are capped**, folding into the top bucket. Only 0.072% of
    positive deltas exceed 2 s, and a multi-second silence carries little
    musical information where a 20 ms note spacing carries a lot.

    **Time buckets are LOG spaced, not quantile spaced.** This is the change
    that mattered. Quantiles put buckets where the mass is, and the mass is at
    small deltas, so the top bucket spanned 0.4 s to 2 s:

        quantile 48 + cap   p99  1489 ms   relative p99  2.918
        quantile 96 + cap   p99   114 ms   relative p99  0.488
        log 96 + cap        p99    31 ms   relative p99  0.090

    Log spacing bounds the RELATIVE error, which is the musically meaningful
    invariant -- timing is heard as ratios, so 10 ms matters at a 50 ms spacing
    and is inaudible at 1 s. The original scheme had a 10,806 ms p99 and a
    428 ms mean; this has 31 ms and 5.1 ms.

    Durations keep quantile spacing: their distribution is not dominated by a
    single repeated value, so quantiles are not pathological there.
    """
    positive = [d for d in deltas if d is not None and d > 0.0]
    observed_max = max(positive) if positive else max_delta_sec
    return Vocab(
        time_buckets=_log_edges(
            min_delta_sec, min(max_delta_sec, max(observed_max, min_delta_sec * 2)), n_time
        ),
        dur_buckets=_quantile_edges([d for d in durations if d is not None], n_dur),
        max_delta_sec=max_delta_sec,
    )


@dataclass
class EncodedTrack:
    tokens: list[int]
    n_events: int
    n_oor: int = 0
    n_unpitched: int = 0
    n_dur_none: int = 0


def attacks(events: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Key-on events in time order. Legato steps are excluded by definition."""
    out = [e for e in events if e.get("origin") == "key_on"]
    # Sort by onset then voice so simultaneous attacks have a defined order.
    # SNES drivers key several voices on one sequencer tick, so ties are the
    # common case rather than an edge case.
    out.sort(key=lambda e: (e["onset_sec"], e["voice"]))
    return out


def encode(events: Sequence[dict[str, Any]], vocab: Vocab) -> EncodedTrack:
    """Encode one track's key-on events as a token sequence."""
    rows = attacks(events)
    tokens = [vocab.BOS]
    previous = 0.0
    stats = {"oor": 0, "unpitched": 0, "dur_none": 0}

    for event in rows:
        onset = float(event["onset_sec"])
        delta = max(0.0, onset - previous)
        previous = onset
        if delta == 0.0:
            tokens.append(vocab.time_same)
        else:
            capped = min(delta, vocab.max_delta_sec)
            tokens.append(
                vocab.time_base + 1 + _bucket(capped, vocab.time_buckets)
            )

        voice = int(event["voice"])
        tokens.append(vocab.voice_base + (voice if 0 <= voice < VOICES else 0))

        if event.get("pitch_status") != "estimated" or event.get("midi_note") is None:
            tokens.append(vocab.pitch_none)
            stats["unpitched"] += 1
        else:
            note = round(float(event["midi_note"]))
            if PITCH_MIN <= note <= PITCH_MAX:
                tokens.append(vocab.pitch_base + (note - PITCH_MIN))
            else:
                tokens.append(vocab.pitch_oor)
                stats["oor"] += 1

        duration = event.get("duration_sec")
        if duration is None:
            tokens.append(vocab.dur_none)
            stats["dur_none"] += 1
        else:
            tokens.append(vocab.dur_base + _bucket(float(duration), vocab.dur_buckets))

    tokens.append(vocab.EOS)
    return EncodedTrack(
        tokens=tokens,
        n_events=len(rows),
        n_oor=stats["oor"],
        n_unpitched=stats["unpitched"],
        n_dur_none=stats["dur_none"],
    )


def decode(tokens: Sequence[int], vocab: Vocab) -> list[dict[str, Any]]:
    """Tokens back to events, with times reconstructed at bucket resolution.

    Lossy by construction -- a bucket is a range -- so a round trip reproduces
    bucket INDICES exactly and wall-clock times only to the bucket's width.
    The round-trip test asserts the former.
    """
    out: list[dict[str, Any]] = []
    body = [t for t in tokens if t not in (vocab.PAD, vocab.BOS, vocab.EOS)]
    clock = 0.0
    for i in range(0, len(body) - STRIDE + 1, STRIDE):
        chunk = body[i : i + STRIDE]
        if [vocab.group(t) for t in chunk] != ["time", "voice", "pitch", "dur"]:
            continue
        time_index = chunk[0] - vocab.time_base
        clock += 0.0 if time_index == 0 else vocab.time_value(time_index - 1)
        note: float | None = None
        if chunk[2] == vocab.pitch_none:
            status = "unpitched"
        elif chunk[2] == vocab.pitch_oor:
            status = "out_of_range"
        else:
            status = "estimated"
            note = float(chunk[2] - vocab.pitch_base + PITCH_MIN)
        duration = (
            None
            if chunk[3] == vocab.dur_none
            else vocab.dur_value(chunk[3] - vocab.dur_base)
        )
        out.append(
            {
                "onset_sec": clock,
                "duration_sec": duration,
                "voice": chunk[1] - vocab.voice_base,
                "midi_note": note,
                "pitch_status": status,
                "time_bucket": time_index,
            }
        )
    return out


def structurally_valid(tokens: Sequence[int], vocab: Vocab) -> tuple[bool, str]:
    """Whether a sequence follows the TIME, VOICE, PITCH, DUR cycle.

    This is the registered generation-validity criterion, made decidable. A
    model that emits a pitch where a duration belongs has produced something
    that is not an event stream, whatever its loss.
    """
    body = [t for t in tokens if t not in (vocab.PAD, vocab.BOS, vocab.EOS)]
    if not body:
        return False, "empty"
    if len(body) % STRIDE:
        return False, f"length {len(body)} is not a multiple of {STRIDE}"
    expected = ("time", "voice", "pitch", "dur")
    for index, token in enumerate(body):
        want = expected[index % STRIDE]
        got = vocab.group(token)
        if got != want:
            return False, f"position {index}: expected {want}, got {got}"
    return True, "ok"


@dataclass
class CorpusStats:
    n_tracks: int = 0
    n_events: int = 0
    n_tokens: int = 0
    n_oor: int = 0
    n_unpitched: int = 0
    n_dur_none: int = 0
    per_split: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_tracks": self.n_tracks,
            "n_events": self.n_events,
            "n_tokens": self.n_tokens,
            "n_out_of_range_pitch": self.n_oor,
            "n_unpitched": self.n_unpitched,
            "n_null_duration": self.n_dur_none,
            "per_split": self.per_split,
        }


def write_vocab(path: str, vocab: Vocab) -> None:
    with open(path, "w") as handle:
        json.dump(vocab.to_dict(), handle, indent=1)


def read_vocab(path: str) -> Vocab:
    with open(path) as handle:
        return Vocab.from_dict(json.load(handle))
