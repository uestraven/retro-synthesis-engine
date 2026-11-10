"""Matched statistical baselines.

Two of these tests exist because they caught real bugs that would have
weakened the baselines and so flattered the model. Both bugs were found by
asking a question with only one possible answer, not by inspection:

* a stride-aware unigram scored 18.0 bits/token against a flat unigram's 6.2,
  which is impossible for a model with strictly more information — the slot
  convention was rotated by one between `fit` and `score`;
* an interpolated n-gram scored 6.274 against its own backoff's 4.209, also
  impossible — a frequently-seen context was trusted even when it assigned the
  target zero mass.
"""

from __future__ import annotations

import math

import pytest

from rse.snes_lm import baselines as bl
from rse.snes_lm import representation as rep


def vocab() -> rep.Vocab:
    return rep.fit_vocab(
        deltas=[0.0, 0.01, 0.05, 0.1, 0.5, 1.0],
        durations=[0.05, 0.1, 0.4],
        n_time=8,
        n_dur=4,
    )


def track(v: rep.Vocab, pitches, voices=None, deltas=None) -> list[int]:
    voices = voices or [0] * len(pitches)
    deltas = deltas or [0.1] * len(pitches)
    clock = 0.0
    events = []
    for pitch, voice, delta in zip(pitches, voices, deltas):
        clock += delta
        events.append(
            {
                "onset_sec": clock,
                "duration_sec": 0.1,
                "voice": voice,
                "midi_note": pitch,
                "pitch_status": "estimated",
                "origin": "key_on",
            }
        )
    return rep.encode(events, v).tokens


def corpus(v: rep.Vocab) -> list[list[int]]:
    return [
        track(v, [60, 62, 64, 65, 67, 65, 64, 62] * 3),
        track(v, [48, 52, 55, 52] * 4, voices=[1] * 16),
        track(v, [72, 71, 69, 67] * 3, voices=[2] * 12),
    ]


def scored(v, model, seqs):
    return bl.score(model, [("t", "g", s) for s in seqs], v).bits_per_token


# --- the slot convention --------------------------------------------------


def test_the_first_real_token_is_a_time_slot() -> None:
    """Targets are indexed over a body that INCLUDES BOS, so index 1 is the
    first real token and must be slot 0. `fit` and `logprob` must agree on
    this; they did not, and the stride-aware unigram predicted the wrong token
    group at every position."""
    assert bl._slot(1) == 0
    assert bl._slot(2) == 1
    assert bl._slot(4) == 3
    assert bl._slot(5) == 0


def test_knowing_the_stride_slot_cannot_make_a_unigram_worse() -> None:
    """The check that caught the rotation. A stride-aware unigram has strictly
    more information than a flat one, so it must score at least as well."""
    v = vocab()
    seqs = corpus(v)
    flat = scored(v, bl.FlatUnigram(v).fit(seqs), seqs)
    aware = scored(v, bl.StrideUnigram(v).fit(seqs), seqs)
    assert aware < flat, f"stride-aware {aware:.3f} must beat flat {flat:.3f}"


def test_the_grammar_is_worth_about_two_bits() -> None:
    """Measured on the full corpus: flat 6.2025, stride-aware 4.2025, a gap of
    exactly 2.000 bits = log2(STRIDE). The stride-aware unigram confines mass
    to a quarter of the vocabulary, so this is structure rather than music --
    which is why it, not the flat unigram, is the registered floor."""
    assert math.isclose(math.log2(rep.STRIDE), 2.0)


# --- interpolation --------------------------------------------------------


def test_an_interpolated_ngram_never_loses_to_its_own_backoff_on_train() -> None:
    """The check that caught the zero-mass collapse. On the data it was fitted
    to, a model that interpolates from a backoff must not be worse than that
    backoff. (On held-out data it legitimately can be, which is why this is
    asserted on train.)"""
    v = vocab()
    seqs = corpus(v)
    backoff = scored(v, bl.StrideUnigram(v).fit(seqs), seqs)
    for order in (1, 2, 3):
        model = bl.SlotNGram(v, order=order, discount=1.0).fit(seqs)
        got = scored(v, model, seqs)
        assert got <= backoff + 1e-9, (
            f"order {order}: {got:.4f} worse than backoff {backoff:.4f}"
        )


def test_every_log_probability_is_finite() -> None:
    """A deterministic rule predicting one token with probability 1 incurs
    infinite cross-entropy the moment it is wrong, which makes it unscorable
    rather than bad. Every heuristic must be mixed with a backoff."""
    v = vocab()
    seqs = corpus(v)
    floor = bl.StrideUnigram(v).fit(seqs)
    models = [
        floor,
        bl.SlotNGram(v, order=2, discount=2.0).fit(seqs),
        bl.CopyPreviousEvent(v, floor, 0.3, name="copy"),
        bl.CopyAtLag(v, floor, 0.3, 16),
        bl.IntervalReversal(v, floor, 0.3),
    ]
    for model in models:
        for sequence in seqs:
            body = [t for t in sequence if t != v.PAD]
            for value in model.stream(body):
                assert math.isfinite(value), model.name


def test_a_heuristic_with_zero_weight_is_exactly_its_backoff() -> None:
    """Pins the mixture: lam=0 must recover the backoff, so any difference
    measured at lam>0 is the heuristic's contribution and nothing else."""
    v = vocab()
    seqs = corpus(v)
    floor = bl.StrideUnigram(v).fit(seqs)
    base = scored(v, floor, seqs)
    for model in (
        bl.CopyPreviousEvent(v, floor, 0.0, name="copy"),
        bl.CopyAtLag(v, floor, 0.0, 16),
        bl.IntervalReversal(v, floor, 0.0),
    ):
        assert math.isclose(scored(v, model, seqs), base, rel_tol=1e-9), model.name


# --- the heuristics do what they say --------------------------------------


def test_copy_previous_event_reads_four_positions_back() -> None:
    v = vocab()
    floor = bl.StrideUnigram(v).fit(corpus(v))
    model = bl.CopyPreviousEvent(v, floor, 0.5, name="copy")
    history = [v.BOS, v.time_same, v.voice_base, v.pitch_base + 10, v.dur_base]
    assert model.predict(history) == v.time_same


def test_interval_reversal_only_fires_on_pitch_slots() -> None:
    """It predicts a pitch, so firing elsewhere would put a pitch token where
    a duration belongs."""
    v = vocab()
    floor = bl.StrideUnigram(v).fit(corpus(v))
    model = bl.IntervalReversal(v, floor, 0.5)
    rising = [
        v.BOS,
        v.time_same, v.voice_base, v.pitch_base + 10, v.dur_base,
        v.time_same, v.voice_base, v.pitch_base + 14, v.dur_base,
    ]
    assert bl._slot(len(rising)) == 0
    assert model.predict(rising) is None            # time slot: declines
    assert model.predict(rising + [v.time_same, v.voice_base]) == v.pitch_base + 10


def test_interval_reversal_streaming_matches_its_own_predict() -> None:
    """The streaming path carries the last two pitches instead of rescanning,
    because rescanning is quadratic and tracks reach 57,510 tokens. It must
    agree with the straightforward version."""
    v = vocab()
    seqs = corpus(v)
    floor = bl.StrideUnigram(v).fit(seqs)
    model = bl.IntervalReversal(v, floor, 0.4)
    body = [t for t in seqs[0] if t != v.PAD]
    streamed = model.stream(body)
    direct = [model.logprob(body[:i], body[i]) for i in range(1, len(body))]
    assert len(streamed) == len(direct)
    for a, b in zip(streamed, direct):
        assert math.isclose(a, b, rel_tol=1e-9)


# --- scoring accounting ---------------------------------------------------


def test_scoring_counts_every_target_after_bos_and_no_padding() -> None:
    v = vocab()
    seqs = corpus(v)
    model = bl.StrideUnigram(v).fit(seqs)
    result = bl.score(model, [("t", "g", seqs[0] + [v.PAD] * 10)], v)
    assert result.n_tokens == len(seqs[0]) - 1   # every token after BOS
    assert result.per_track and len(result.per_track) == 1


def test_per_game_figures_are_retained_for_the_descriptive_report() -> None:
    """The registration reports all eight test games descriptively and
    bootstraps over tracks, so both breakdowns have to survive scoring."""
    v = vocab()
    seqs = corpus(v)
    model = bl.StrideUnigram(v).fit(seqs)
    result = bl.score(
        model,
        [("a", "GameA", seqs[0]), ("b", "GameB", seqs[1]), ("c", "GameA", seqs[2])],
        v,
    )
    assert set(result.per_game) == {"GameA", "GameB"}
    assert len(result.per_track) == 3
