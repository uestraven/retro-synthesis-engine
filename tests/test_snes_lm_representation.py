"""The snesmdb token representation.

Measured properties of the corpus that these tests encode, so a later change
that breaks one is visible:

* 51.3% of consecutive key-on pairs are SIMULTANEOUS (delta exactly 0), and
  events per tick equals distinct voices per tick exactly -- no voice ever
  keys twice on one tick. Simultaneity is voice-parallel chords, nothing else.
* Nonzero deltas have a 94.1 ms median, which independently corroborates the
  ~11 rows/sec median of the release's own onset-grid scan.
* 21.4% of train key-ons are unpitched and 1.17% fall outside MIDI 12-108.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from rse.snes_lm import representation as rep


def vocab() -> rep.Vocab:
    return rep.fit_vocab(
        deltas=[0.0, 0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
        durations=[0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6],
        n_time=6,
        n_dur=4,
    )


def event(onset, voice, note, duration=0.1, origin="key_on", status="estimated"):
    return {
        "onset_sec": Decimal(str(onset)),
        "duration_sec": None if duration is None else Decimal(str(duration)),
        "voice": voice,
        "midi_note": note,
        "pitch_status": status,
        "origin": origin,
    }


# --- vocabulary layout ----------------------------------------------------


def test_the_token_groups_tile_the_vocabulary_without_overlap() -> None:
    """Structural validity is decided by group membership, so the groups must
    partition the id space exactly."""
    v = vocab()
    groups = [v.group(t) for t in range(v.size)]
    assert "invalid" not in groups
    assert groups[: v.n_special] == ["pad", "bos", "eos"]
    assert groups.count("time") == v.n_time
    assert groups.count("voice") == rep.VOICES
    assert groups.count("pitch") == v.n_pitch
    assert groups.count("dur") == v.n_dur
    assert v.group(v.size) == "invalid"


def test_bucket_edges_are_sorted_and_unique() -> None:
    """Quantile edges collapse when a value repeats, and the delta
    distribution is over half exact zeros -- 48 requested time buckets
    produced 25. Duplicate edges would make a bucket unreachable."""
    v = vocab()
    assert list(v.time_buckets) == sorted(set(v.time_buckets))
    assert list(v.dur_buckets) == sorted(set(v.dur_buckets))


def test_fitting_is_deterministic() -> None:
    a, b = vocab(), vocab()
    assert a.time_buckets == b.time_buckets
    assert a.dur_buckets == b.dur_buckets


# --- encoding -------------------------------------------------------------


def test_a_track_encodes_to_four_tokens_per_event_plus_delimiters() -> None:
    v = vocab()
    enc = rep.encode([event(0.0, 0, 60), event(0.1, 1, 64)], v)
    assert enc.n_events == 2
    assert len(enc.tokens) == 2 * rep.STRIDE + 2  # BOS + events + EOS
    assert enc.tokens[0] == v.BOS and enc.tokens[-1] == v.EOS


def test_legato_steps_are_excluded_by_definition() -> None:
    """A legato step is a pitch change inside a sounding note, not an attack.
    The existing interval baseline excluded it, and the registration requires
    baselines on an identical representation."""
    v = vocab()
    events = [
        event(0.0, 0, 60),
        event(0.05, 0, 62, origin="legato_step"),
        event(0.1, 0, 64),
    ]
    enc = rep.encode(events, v)
    assert enc.n_events == 2


def test_simultaneous_events_are_ordered_by_voice_deterministically() -> None:
    """Over half of all consecutive pairs share an onset, so tie order is the
    common case. Without a deterministic rule the same track would encode
    differently run to run."""
    v = vocab()
    forwards = rep.encode([event(1.0, 5, 60), event(1.0, 2, 64)], v).tokens
    backwards = rep.encode([event(1.0, 2, 64), event(1.0, 5, 60)], v).tokens
    assert forwards == backwards
    decoded = rep.decode(forwards, v)
    assert [d["voice"] for d in decoded] == [2, 5]


def test_an_unpitched_event_is_kept_with_its_own_token() -> None:
    """Dropping them would delete real musical activity -- percussion above
    all, 21.4% of train key-ons -- and inventing a pitch would be worse."""
    v = vocab()
    enc = rep.encode([event(0.0, 0, None, status="noise")], v)
    assert v.pitch_none in enc.tokens
    assert enc.n_unpitched == 1
    assert rep.decode(enc.tokens, v)[0]["pitch_status"] == "unpitched"


def test_a_pitch_outside_the_musical_range_is_flagged_not_clamped() -> None:
    """`midi_note` is a per-sample estimate measured from -36.74 to +131.09.
    Outside a musical register it is far more likely an octave error than a
    note someone wrote, so clamping would manufacture a plausible wrong note."""
    v = vocab()
    enc = rep.encode([event(0.0, 0, 130.0), event(0.1, 0, -20.0)], v)
    assert enc.n_oor == 2
    assert all(d["pitch_status"] == "out_of_range" for d in rep.decode(enc.tokens, v))


def test_a_null_duration_is_representable() -> None:
    """159,584 rows have no duration: legato steps, plus 4,674 key-ons still
    sounding when the render window ended."""
    v = vocab()
    enc = rep.encode([event(0.0, 0, 60, duration=None)], v)
    assert v.dur_none in enc.tokens
    assert rep.decode(enc.tokens, v)[0]["duration_sec"] is None


# --- round trip -----------------------------------------------------------


def test_encode_decode_preserves_voice_pitch_and_bucket_indices() -> None:
    """The representation is lossy in wall-clock time by construction -- a
    bucket is a range -- so what must survive exactly is the discrete content:
    voice, semitone, and which bucket each time and duration landed in."""
    v = vocab()
    events = [
        event(0.0, 0, 60),
        event(0.0, 3, 67),      # simultaneous
        event(0.25, 1, 55, duration=0.4),
        event(1.25, 7, 72, duration=None),
    ]
    enc = rep.encode(events, v)
    decoded = rep.decode(enc.tokens, v)
    assert len(decoded) == len(events)
    assert [d["voice"] for d in decoded] == [0, 3, 1, 7]
    assert [d["midi_note"] for d in decoded] == [60.0, 67.0, 55.0, 72.0]
    # re-encoding the decoded events must give the same tokens
    again = rep.encode(
        [
            event(d["onset_sec"], d["voice"], d["midi_note"], d["duration_sec"])
            for d in decoded
        ],
        v,
    )
    assert [t for t in again.tokens if v.group(t) in ("voice", "pitch")] == [
        t for t in enc.tokens if v.group(t) in ("voice", "pitch")
    ]


def test_decoded_onsets_are_monotonic() -> None:
    v = vocab()
    enc = rep.encode([event(0.0, 0, 60), event(0.3, 1, 61), event(1.9, 2, 62)], v)
    onsets = [d["onset_sec"] for d in rep.decode(enc.tokens, v)]
    assert onsets == sorted(onsets)


# --- the registered validity criterion ------------------------------------


def test_encoder_output_is_structurally_valid() -> None:
    v = vocab()
    enc = rep.encode([event(0.0, 0, 60), event(0.2, 4, 65)], v)
    ok, why = rep.structurally_valid(enc.tokens, v)
    assert ok, why


@pytest.mark.parametrize(
    "mutate,expect",
    [
        (lambda t, v: t[:-2], "not a multiple"),
        (lambda t, v: [v.BOS, v.voice_base, v.voice_base + 1,
                       v.pitch_base, v.dur_base, v.EOS], "expected time"),
        (lambda t, v: [v.BOS, v.time_base, v.time_base,
                       v.pitch_base, v.dur_base, v.EOS], "expected voice"),
    ],
)
def test_malformed_sequences_are_rejected_with_a_reason(mutate, expect) -> None:
    """A model that emits a pitch where a duration belongs has produced
    something that is not an event stream, whatever its loss. The registered
    criterion needs this to be decidable, not a judgement."""
    v = vocab()
    tokens = rep.encode([event(0.0, 0, 60), event(0.2, 1, 61)], v).tokens
    ok, why = rep.structurally_valid(mutate(tokens, v), v)
    assert not ok
    assert expect in why


def test_an_empty_sequence_is_not_valid() -> None:
    v = vocab()
    ok, why = rep.structurally_valid([v.BOS, v.EOS], v)
    assert not ok and why == "empty"
