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
    def n_time(self) -> int:
        return len(self.time_buckets)

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
            "time_buckets": list(self.time_buckets),
            "dur_buckets": list(self.dur_buckets),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Vocab":
        return cls(
            time_buckets=tuple(data["time_buckets"]),
            dur_buckets=tuple(data["dur_buckets"]),
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


def fit_vocab(
    deltas: Iterable[float],
    durations: Iterable[float],
    *,
    n_time: int = 48,
    n_dur: int = 32,
) -> Vocab:
    """Fit bucket edges by quantile. TRAIN SPLIT ONLY.

    Quantiles rather than fixed or log spacing: the delta distribution is
    whatever these drivers happen to produce, and a fixed grid was already
    rejected on measurement. Quantiles put resolution where the data is.
    """
    def edges(values: list[float], n: int) -> tuple[float, ...]:
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

    return Vocab(
        time_buckets=edges(list(deltas), n_time),
        dur_buckets=edges(list(durations), n_dur),
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
        tokens.append(vocab.time_base + _bucket(delta, vocab.time_buckets))

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
        clock += vocab.time_buckets[time_index]
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
            else vocab.dur_buckets[chunk[3] - vocab.dur_base]
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
