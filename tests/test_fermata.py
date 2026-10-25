"""The reconstructed Fermata codec.

Tests run against a synthetic vocabulary with the same families as the real
one, so they need neither the 403 MB checkpoint nor a network. The grammar
itself was recovered from the checkpoint's own sampled output, and the
perplexity evidence for it is recorded in `docs/fermata_phase4_results.md`.
"""

from __future__ import annotations

import json

import pytest

from rse.transfer.fermata import SUBPOS, Vocab, decode, encode
from rse.transfer.score import PERCUSSION, Score, ScoreNote, ScorePercussion, Track


def _vocab() -> Vocab:
    tokens = [
        "<PAD>",
        "<BOS>",
        "<EOS>",
        "<MASK>",
        "<INFILL>",
        "<CONTINUE>",
        "<BAR>",
        "<PLAN>",
        "<PLAN_END>",
        "<DRUM_TRACK>",
        "<GENRE_soundtrack>",
        "<GENRE_classical>",
        "<ENSEMBLE_band>",
        "<KEY_0:major>",
        "<KEY_0:none>",
        "<KEY_0:minor>",
        "<METER_4/4>",
        "<METER_3/4>",
        "<BPE_0>",
        "<BPE_1>",
    ]
    tokens += [f"<SUBPOS_{i}>" for i in range(SUBPOS)]
    tokens += [f"<BEAT_{i}>" for i in range(8)]
    tokens += [f"<TRK_{i}>" for i in range(4)]
    tokens += [f"<PITCH_{i}>" for i in range(128)]
    tokens += [f"<DRUM_NOTE_{i}>" for i in (36, 38, 40, 42)]
    tokens += [f"<VELOCITY_{i}>" for i in range(16)]
    # Sparse durations, like the real 811-of-14400 set.
    tokens += [f"<DURATION_{d}>" for d in (1, 3, 6, 12, 24, 48, 96)]
    tokens += [f"<TRACK_{p}>" for p in (0, 38, 80, 81)]
    tokens += [f"<TEMPO_{t}>" for t in (100, 120, 150, 200)]
    return Vocab(token_to_id={t: i for i, t in enumerate(tokens)})


def _score(**kw) -> Score:
    melody = Track(role="melody", program=80, channel_id="P1")
    melody.notes = [
        ScoreNote(tick=0, duration=12, pitch=60, velocity=96),
        ScoreNote(tick=12, duration=6, pitch=64, velocity=96),
        ScoreNote(tick=24, duration=12, pitch=67, velocity=96),
    ]
    drums = Track(role=PERCUSSION, channel_id="NO")
    drums.percussion = [
        ScorePercussion(tick=0, duration=6, key=36, velocity=100),
        ScorePercussion(tick=48, duration=6, key=38, velocity=100),
    ]
    return Score(
        ticks_per_quarter=SUBPOS,
        tempo_bpm=150.0,
        tracks=[melody, drums],
        **kw,
    )


# -- vocabulary handling ------------------------------------------------------


def test_bpe_tokens_are_identified_for_suppression():
    """The model trained on the atomic stream, so emitting or scoring a BPE
    token feeds it an id it never saw."""
    vocab = _vocab()
    assert vocab.bpe_ids
    assert all(vocab.id_to_token[i].startswith("<BPE_") for i in vocab.bpe_ids)


def test_duration_snaps_to_the_nearest_representable_value():
    """Only 811 of 14,400 lengths exist as tokens in the real vocabulary, so a
    duration must be snapped and the error reported rather than assumed
    expressible."""
    vocab = _vocab()
    assert vocab.nearest_duration(12) == (12, 0)
    assert vocab.nearest_duration(13) == (12, 1)
    assert vocab.nearest_duration(20) == (24, 4)
    assert vocab.nearest_duration(0) == (1, 0)


def test_an_unknown_token_is_an_explicit_error():
    with pytest.raises(KeyError, match="not in Fermata's vocabulary"):
        _vocab().id_of("<NOPE_1>")


# -- encoding -----------------------------------------------------------------


def test_encode_emits_the_recovered_grammar():
    """Header, then a plan, then bar-major note groups with DURATION before
    VELOCITY -- the order the checkpoint's own output showed."""
    vocab = _vocab()
    ids, report = encode(_score(), vocab=vocab)
    names = [vocab.id_to_token[i] for i in ids]
    assert names[0] == "<BOS>"
    assert names[1:3] == ["<GENRE_soundtrack>", "<ENSEMBLE_band>"]
    assert names[3].startswith("<TEMPO_")
    assert names[4].startswith("<KEY_")
    assert names[5].startswith("<METER_")
    assert names[6] == "<PLAN>"
    assert names[7:9] == ["<TRACK_80>", "<DRUM_TRACK>"]
    assert names[9] == "<PLAN_END>"
    assert names[10] == "<BAR>"
    assert names[-1] == "<EOS>"
    first = names.index("<TRK_0>")
    assert names[first + 1].startswith("<BEAT_")
    assert names[first + 2].startswith("<SUBPOS_")
    assert names[first + 3].startswith("<PITCH_")
    assert names[first + 4].startswith("<DURATION_")
    assert names[first + 5].startswith("<VELOCITY_")
    assert report.n_notes == 3
    assert report.n_percussion == 2


def test_encode_never_emits_a_bpe_token():
    vocab = _vocab()
    ids, _ = encode(_score(), vocab=vocab)
    assert not any(vocab.id_to_token[i].startswith("<BPE_") for i in ids)


def test_encode_refuses_a_grid_other_than_24():
    """Resampling here would hide the quantization the study measures."""
    score = _score()
    score.ticks_per_quarter = 48
    with pytest.raises(ValueError, match="24 ticks per quarter"):
        encode(score, vocab=_vocab())


def test_beat_resets_each_bar():
    """`<BEAT_>` is the beat within the bar, not an absolute index. The
    checkpoint's sampled stream runs BEAT_0 .. BEAT_1 .. <BAR> BEAT_0."""
    vocab = _vocab()
    track = Track(role="melody", program=80)
    # One note in bar 0 beat 0, one in bar 1 beat 0.
    track.notes = [
        ScoreNote(tick=0, duration=6, pitch=60),
        ScoreNote(tick=SUBPOS * 4, duration=6, pitch=62),
    ]
    score = Score(ticks_per_quarter=SUBPOS, tempo_bpm=120.0, tracks=[track])
    names = [vocab.id_to_token[i] for i in encode(score, vocab=vocab)[0]]
    assert names.count("<BAR>") == 2
    assert [n for n in names if n.startswith("<BEAT_")] == ["<BEAT_0>", "<BEAT_0>"]


def test_an_unmappable_drum_key_is_counted_not_dropped_silently():
    vocab = _vocab()
    drums = Track(role=PERCUSSION)
    drums.percussion = [ScorePercussion(tick=0, duration=6, key=99)]
    score = Score(ticks_per_quarter=SUBPOS, tempo_bpm=120.0, tracks=[drums])
    _ids, report = encode(score, vocab=vocab)
    assert report.n_drum_unmapped == 1
    assert report.warnings


def test_a_pitch_outside_the_vocabulary_is_counted():
    vocab = _vocab()
    track = Track(role="melody", program=80)
    track.notes = [ScoreNote(tick=0, duration=6, pitch=200)]
    score = Score(ticks_per_quarter=SUBPOS, tempo_bpm=120.0, tracks=[track])
    _ids, report = encode(score, vocab=vocab)
    assert report.n_pitch_out_of_range == 1


def test_snapped_durations_are_reported_with_their_error():
    vocab = _vocab()
    track = Track(role="melody", program=80)
    track.notes = [ScoreNote(tick=0, duration=20, pitch=60)]
    score = Score(ticks_per_quarter=SUBPOS, tempo_bpm=120.0, tracks=[track])
    _ids, report = encode(score, vocab=vocab)
    assert report.n_duration_snapped == 1
    assert report.duration_error_units == 4


# -- decoding -----------------------------------------------------------------


def test_round_trip_preserves_every_event():
    vocab = _vocab()
    score = _score()
    ids, _ = encode(score, vocab=vocab)
    back, report = decode(ids, vocab=vocab)
    assert report.hit_eos
    assert back.n_events == score.n_events
    assert report.had_plan


def test_round_trip_preserves_pitch_order_within_a_track():
    """An NES arpeggio is a sequence on one voice. Reordering it would change
    its musical meaning, and merging it into a chord would imply polyphony the
    APU cannot play."""
    vocab = _vocab()
    ids, _ = encode(_score(), vocab=vocab)
    back, _ = decode(ids, vocab=vocab)
    melodic = [t for t in back.tracks if t.notes][0]
    ordered = sorted(melodic.notes, key=lambda n: n.tick)
    assert [n.pitch for n in ordered] == [60, 64, 67]
    assert len({n.tick for n in ordered}) == 3


def test_percussion_stays_percussion_through_the_plan():
    vocab = _vocab()
    ids, _ = encode(_score(), vocab=vocab)
    back, _ = decode(ids, vocab=vocab)
    drums = [t for t in back.tracks if t.role == PERCUSSION]
    assert len(drums) == 1
    assert len(drums[0].percussion) == 2
    assert drums[0].notes == []


def test_a_stream_without_a_plan_cannot_type_its_tracks():
    """Track identity is not recoverable from note tokens alone. The decoder
    must say so rather than guessing, since guessing is how percussion would
    silently become a pitched part."""
    vocab = _vocab()
    names = [
        "<BOS>",
        "<BAR>",
        "<TRK_0>",
        "<BEAT_0>",
        "<SUBPOS_0>",
        "<PITCH_60>",
        "<DURATION_6>",
        "<VELOCITY_8>",
        "<EOS>",
    ]
    _score_out, report = decode([vocab.id_of(n) for n in names], vocab=vocab)
    assert report.had_plan is False
    assert any("never declared" in w for w in report.warnings)


def test_bpe_tokens_in_a_stream_are_skipped_and_counted():
    vocab = _vocab()
    names = [
        "<BOS>",
        "<PLAN>",
        "<TRACK_80>",
        "<PLAN_END>",
        "<BAR>",
        "<TRK_0>",
        "<BPE_0>",
        "<BEAT_0>",
        "<SUBPOS_0>",
        "<PITCH_60>",
        "<DURATION_6>",
        "<VELOCITY_8>",
        "<EOS>",
    ]
    _s, report = decode([vocab.id_of(n) for n in names], vocab=vocab)
    assert report.n_skipped >= 1
    assert report.n_events == 1


def test_decode_reports_a_stream_with_no_music():
    vocab = _vocab()
    names = ["<BOS>", "<PLAN>", "<TRACK_80>", "<PLAN_END>", "<EOS>"]
    _s, report = decode([vocab.id_of(n) for n in names], vocab=vocab)
    assert report.n_events == 0
    assert any("no <BAR>" in w for w in report.warnings)


def test_decode_recovers_tempo_and_meter():
    vocab = _vocab()
    score = _score(beats_per_bar=3)
    ids, _ = encode(score, vocab=vocab)
    back, _ = decode(ids, vocab=vocab)
    assert back.tempo_bpm == 150.0
    assert back.beats_per_bar == 3


def test_encode_is_deterministic():
    vocab = _vocab()
    first, _ = encode(_score(), vocab=vocab)
    second, _ = encode(_score(), vocab=vocab)
    assert first == second


def test_reports_are_json_serializable():
    vocab = _vocab()
    ids, enc = encode(_score(), vocab=vocab)
    _s, dec = decode(ids, vocab=vocab)
    assert json.loads(json.dumps(enc.to_dict()))["n_events"] == 5
    assert json.loads(json.dumps(dec.to_dict()))["n_events"] == 5
    # Every artifact is marked as coming from a reconstruction.
    assert "reconstruction" in enc.to_dict()
    assert "reconstruction" in dec.to_dict()


# -- eight-bar generation -----------------------------------------------------


def test_trim_cuts_at_the_bar_boundary_keeping_the_last_bar_whole():
    """Fermata emits `<BAR>` at the start of each bar, so the token that must
    not appear is the ninth. Everything from it onward goes, which keeps every
    event of the eighth bar while removing the opening of the next."""
    from rse.transfer.fermata import BAR_LIMIT, trim_to_bars

    vocab = _vocab()
    bar, eos = vocab.id_of("<BAR>"), vocab.id_of("<EOS>")
    pitch = vocab.id_of("<PITCH_60>")
    ids = [vocab.id_of("<BOS>")] + [x for _ in range(10) for x in (bar, pitch)]

    result = trim_to_bars(ids, vocab=vocab, n_bars=8)
    assert result.reason == BAR_LIMIT
    assert result.n_bars == 8
    assert result.complete
    assert result.ids.count(bar) == 8
    assert result.ids[-1] == eos
    # The eighth bar's note survives: 8 bars x (BAR, PITCH) plus BOS and EOS.
    assert result.ids.count(pitch) == 8
    assert result.raw_ids == ids


def test_a_natural_eos_is_distinguished_from_a_bar_limit():
    """The model choosing to end is a different artefact from being cut off,
    and conflating them is how an earlier evaluation reported completion for
    output that was mostly empty."""
    from rse.transfer.fermata import NATURAL_EOS, trim_to_bars

    vocab = _vocab()
    bar, eos = vocab.id_of("<BAR>"), vocab.id_of("<EOS>")
    pitch = vocab.id_of("<PITCH_60>")
    ids = [vocab.id_of("<BOS>")] + [x for _ in range(5) for x in (bar, pitch)] + [eos]
    result = trim_to_bars(ids, vocab=vocab, n_bars=8)
    assert result.reason == NATURAL_EOS
    assert result.n_bars == 5
    assert result.complete, "the model ended deliberately, so this is valid"


def test_a_budget_stop_is_never_counted_complete():
    """An incomplete sequence must not pass as a valid composition however
    many bars preceded the cut."""
    from rse.transfer.fermata import TOKEN_BUDGET, trim_to_bars

    vocab = _vocab()
    bar, pitch = vocab.id_of("<BAR>"), vocab.id_of("<PITCH_60>")
    ids = [vocab.id_of("<BOS>")] + [x for _ in range(3) for x in (bar, pitch)]
    result = trim_to_bars(ids, vocab=vocab, n_bars=8)
    assert result.reason == TOKEN_BUDGET
    assert result.n_bars == 3
    assert not result.complete


def test_trimming_leaves_the_unrestricted_stream_available():
    """The raw output is kept so restricted and unrestricted modes can be
    compared rather than one replacing the other."""
    from rse.transfer.fermata import trim_to_bars

    vocab = _vocab()
    bar, pitch = vocab.id_of("<BAR>"), vocab.id_of("<PITCH_60>")
    ids = [vocab.id_of("<BOS>")] + [x for _ in range(12) for x in (bar, pitch)]
    result = trim_to_bars(ids, vocab=vocab, n_bars=8)
    assert len(result.raw_ids) == len(ids)
    assert len(result.ids) < len(ids)


def test_trimmed_stream_still_decodes():
    from rse.transfer.fermata import decode, trim_to_bars

    vocab = _vocab()
    ids, _report = encode(_score(), vocab=vocab)
    trimmed = trim_to_bars(ids, vocab=vocab, n_bars=1)
    score, report = decode(trimmed.ids, vocab=vocab)
    assert report.hit_eos
    assert score.n_events >= 1


# -- scenario-aware ensemble validation --------------------------------------


def test_two_channels_is_correct_when_two_tracks_were_requested():
    """The defect this replaces: a blanket three-channel minimum discarded
    seven of sixty pieces for correctly obeying a two-track `sparse` prompt."""
    from rse.events.schema import Composition, Note
    from rse.transfer.ensemble import check_ensemble

    melody = Track(role="melody", program=80)
    melody.notes = [ScoreNote(tick=0, duration=12, pitch=72)]
    bass = Track(role="bass", program=38)
    bass.notes = [ScoreNote(tick=0, duration=12, pitch=43)]
    score = Score(ticks_per_quarter=SUBPOS, tempo_bpm=120.0, tracks=[melody, bass])
    arranged = Composition(
        hardware_profile="nes",
        ticks_per_beat=480,
        events=[
            Note(channel_id="P1", tick=0, pitch=72, duration=60, velocity=96),
            Note(channel_id="TR", tick=0, pitch=43, duration=60, velocity=96),
        ],
    )
    check = check_ensemble(
        scenario="sparse",
        plan=["<TRACK_80>", "<TRACK_38>"],
        score=score,
        arranged=arranged,
    )
    assert check.n_requested_tracks == 2
    assert check.requested_percussion is False
    assert check.percussion_as_requested
    assert check.missing_tracks == 0
    assert check.honours_request, "obeying a two-track prompt is not a failure"


def test_missing_percussion_is_a_failure_when_drums_were_requested():
    """The same two channels, asked for differently, is a different verdict."""
    from rse.events.schema import Composition, Note
    from rse.transfer.ensemble import check_ensemble

    melody = Track(role="melody", program=80)
    melody.notes = [ScoreNote(tick=0, duration=12, pitch=72)]
    score = Score(ticks_per_quarter=SUBPOS, tempo_bpm=120.0, tracks=[melody])
    arranged = Composition(
        hardware_profile="nes",
        ticks_per_beat=480,
        events=[Note(channel_id="P1", tick=0, pitch=72, duration=60, velocity=96)],
    )
    check = check_ensemble(
        scenario="action",
        plan=["<TRACK_80>", "<TRACK_81>", "<TRACK_38>", "<DRUM_TRACK>"],
        score=score,
        arranged=arranged,
    )
    assert check.requested_percussion is True
    assert check.percussion_sounding is False
    assert not check.percussion_as_requested
    assert check.missing_tracks == 3
    assert not check.honours_request


def test_ensemble_check_reports_each_stage_separately():
    """Requested, generated, mapped and sounding diverge for different
    reasons, and one channel count hides all of them."""
    from rse.events.schema import Composition, Noise, Note
    from rse.transfer.ensemble import check_ensemble

    melody = Track(role="melody", program=80)
    melody.notes = [ScoreNote(tick=0, duration=12, pitch=72)]
    drums = Track(role=PERCUSSION)
    drums.percussion = [ScorePercussion(tick=0, duration=6, key=36)]
    score = Score(ticks_per_quarter=SUBPOS, tempo_bpm=120.0, tracks=[melody, drums])
    arranged = Composition(
        hardware_profile="nes",
        ticks_per_beat=480,
        events=[
            Note(channel_id="P1", tick=0, pitch=72, duration=60, velocity=96),
            Noise(channel_id="NO", tick=0, value=1, duration=30, velocity=96),
        ],
    )
    check = check_ensemble(
        scenario="town",
        plan=["<TRACK_80>", "<DRUM_TRACK>"],
        score=score,
        arranged=arranged,
    )
    row = check.to_dict()
    assert row["n_requested_tracks"] == 2
    assert row["n_generated_tracks"] == 2
    assert row["channels_sounding"] == ["P1", "NO"]
    assert row["percussion_sounding"] is True
    assert row["honours_request"] is True
