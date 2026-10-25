"""The four codecs, and the safeguards Branch B depends on."""

from __future__ import annotations

import pytest

from rse.codec.benchmark import (
    enforce_monophony,
    ioi_entropy,
    match_events,
    overlapping_events,
)
from rse.events.schema import Composition, Noise, Note
from rse.tokenizers.tokenizer import GridFit, RseTokenizer
from rse.tokenizers.vocab import CODECS, MOTIF, OFFSET, TokenizerConfig, build_vocabulary

K_VALUES = (2, 4, 8)


def _piece() -> Composition:
    """Events deliberately off the row grid, so a sub-row codec has work to do.

    The row is 100 ticks. Onsets at 0, 25, 50, 75 put one event exactly on the
    row and three at quarter-row offsets, which only k>=4 can place exactly.
    """
    events: list[Note | Noise] = []
    for bar in range(4):
        base = bar * 1600
        for index, offset in enumerate((0, 25, 50, 75)):
            events.append(
                Note(
                    channel_id="P1" if index % 2 == 0 else "P2",
                    tick=base + index * 100 + offset,
                    pitch=60 + index,
                    duration=60,
                    velocity=96,
                )
            )
        events.append(Noise(channel_id="NO", tick=base + 400, value=4, duration=50, velocity=96))
    return Composition(hardware_profile="nes", ticks_per_beat=480, events=events)


def _fit() -> GridFit:
    return GridFit(step_ticks=100.0, phase_ticks=0.0, aligned=True, period_sec=0.1)


def _tokenizer(codec: str, k: int) -> RseTokenizer:
    if codec == "A":
        return RseTokenizer(TokenizerConfig())
    return RseTokenizer(TokenizerConfig(codec=codec, sub_row_divisions=k))


# -- safeguard C: codec A is untouched ----------------------------------------


def test_codec_a_vocabulary_is_unchanged():
    """The foundation's checkpoint indexes this table; a shift of one id makes
    it meaningless."""
    vocab = build_vocabulary(TokenizerConfig())
    assert len(vocab) == 272
    assert vocab.tokens[-5:] == (
        "Motif_high",
        "Motif_medium",
        "Motif_low",
        "Motif_unknown",
        "Motif_unspecified",
    )
    assert not any(t.startswith(OFFSET) for t in vocab.tokens)


def test_codec_a_ids_are_preserved_in_every_other_codec():
    """New families are appended, so a token's id never moves. Without this a
    checkpoint migrated between codecs would read a Pitch as a Position."""
    base = build_vocabulary(TokenizerConfig())
    for codec in ("B",):
        wider = build_vocabulary(TokenizerConfig(codec=codec, sub_row_divisions=8))
        assert wider.tokens[: len(base)] == base.tokens


def test_slot_of_delegates_to_row_of_at_one_division():
    """Equivalence by construction rather than by coincidence."""
    fit = _fit()
    for tick in (0, 37, 99, 100, 101, 1234, 5555):
        assert fit.slot_of(tick, 1) == fit.row_of(tick)
        assert fit.tick_of_slot(tick, 1) == fit.tick_of(tick)


def test_codec_a_round_trip_is_unaffected_by_the_codec_machinery():
    piece, fit = _piece(), _fit()
    tokenizer = _tokenizer("A", 1)
    encoded = tokenizer.encode(piece, fit=fit)
    decoded = tokenizer.decode(encoded.ids)
    assert decoded.composition.sounding_events()
    # Every onset lands on a row multiple of the nominal step.
    step = tokenizer.ticks_per_step(decoded.composition.ticks_per_beat)
    assert all(e.tick % step == 0 for e in decoded.composition.sounding_events())


# -- safeguard: every codec round-trips ---------------------------------------


@pytest.mark.parametrize("codec", CODECS)
@pytest.mark.parametrize("k", K_VALUES)
def test_every_codec_round_trips(codec, k):
    if codec == "A" and k != K_VALUES[0]:
        pytest.skip("codec A has no k")
    tokenizer = _tokenizer(codec, k)
    encoded = tokenizer.encode(_piece(), fit=_fit())
    decoded = tokenizer.decode(encoded.ids)
    assert decoded.n_tokens_skipped == 0, "a codec must decode its own output"
    assert decoded.composition.sounding_events()


@pytest.mark.parametrize("k", K_VALUES)
def test_codec_b_emits_offset_tokens_and_c_does_not(k):
    b = _tokenizer("B", k).encode(_piece(), fit=_fit())
    c = _tokenizer("C", k).encode(_piece(), fit=_fit())
    assert any(t.startswith(OFFSET) for t in b.tokens)
    assert not any(t.startswith(OFFSET) for t in c.tokens)
    # C pays one token fewer per event group, which is the whole claim that it
    # is the more efficient way to buy the same onset resolution.
    assert len(c.ids) < len(b.ids)


# -- item 4: B and C must be equivalent in what they reconstruct --------------


@pytest.mark.parametrize("k", K_VALUES)
def test_codecs_b_and_c_reconstruct_identical_events(k):
    """B and C quantize onsets identically and collapse identically, so they
    must differ only in token cost. If their decoded streams disagree, one of
    them has a defect and no fidelity number from either can be trusted."""
    piece, fit = _piece(), _fit()
    b = _tokenizer("B", k)
    c = _tokenizer("C", k)
    b_events = b.decode(b.encode(piece, fit=fit).ids).composition.sounding_events()
    c_events = c.decode(c.encode(piece, fit=fit).ids).composition.sounding_events()

    assert len(b_events) == len(c_events)
    for left, right in zip(b_events, c_events, strict=True):
        assert left.channel_id == right.channel_id
        assert left.tick == right.tick
        assert left.duration == right.duration
        assert type(left) is type(right)
        if isinstance(left, Note):
            assert left.pitch == right.pitch
        else:
            assert left.value == right.value


@pytest.mark.parametrize("k", K_VALUES)
def test_codec_d_matches_b_on_onsets_but_not_on_durations(k):
    """D differs from B and C in duration resolution alone. Onsets equal means
    the comparison isolates articulation; durations differing means D is
    actually doing the thing it costs vocabulary for."""
    piece, fit = _piece(), _fit()
    b = _tokenizer("B", k)
    d = _tokenizer("D", k)
    b_events = b.decode(b.encode(piece, fit=fit).ids).composition.sounding_events()
    d_events = d.decode(d.encode(piece, fit=fit).ids).composition.sounding_events()
    assert [e.tick for e in b_events] == [e.tick for e in d_events]
    assert [e.duration for e in b_events] != [e.duration for e in d_events]


# -- sub-row codecs actually improve onset placement --------------------------


@pytest.mark.parametrize("k", K_VALUES)
def test_sub_row_codecs_reduce_onset_error(k):
    """The point of the exercise: finer onsets must place events closer to
    where they were, or the extra tokens buy nothing."""
    piece, fit = _piece(), _fit()
    a = _tokenizer("A", 1).encode(piece, fit=fit)
    c = _tokenizer("C", k).encode(piece, fit=fit)
    assert c.total_start_error_ticks < a.total_start_error_ticks


def test_finer_divisions_never_increase_onset_error():
    piece, fit = _piece(), _fit()
    errors = [_tokenizer("C", k).encode(piece, fit=fit).total_start_error_ticks for k in K_VALUES]
    assert errors == sorted(errors, reverse=True)


@pytest.mark.parametrize("k", K_VALUES)
def test_sub_row_codecs_keep_more_events(k):
    """Collapsing per sub-row slot rather than per row is what preserves an
    arpeggio's frames."""
    piece, fit = _piece(), _fit()
    a = _tokenizer("A", 1).encode(piece, fit=fit)
    c = _tokenizer("C", k).encode(piece, fit=fit)
    assert c.n_rows_collapsed <= a.n_rows_collapsed
    assert c.n_encoded >= a.n_encoded


# -- item 5: monophony across durations, not just onsets ----------------------


@pytest.mark.parametrize("codec", CODECS)
@pytest.mark.parametrize("k", K_VALUES)
def test_no_overlapping_notes_survive_monophony_enforcement(codec, k):
    """The collapse key guarantees one onset per channel per slot, which is not
    the same as non-overlapping notes: a long note followed by a nearer one
    overlaps. Real hardware steals the voice, so this must hold at every
    resolution -- D especially, where finer onsets bring notes closer."""
    if codec == "A" and k != K_VALUES[0]:
        pytest.skip("codec A has no k")
    tokenizer = _tokenizer(codec, k)
    decoded = tokenizer.decode(tokenizer.encode(_piece(), fit=_fit()).ids)
    repaired, _truncated = enforce_monophony(decoded.composition)
    assert overlapping_events(repaired) == []


def test_monophony_enforcement_detects_an_overlap_it_must_repair():
    """Guards the guard: a test that only ran on already-clean input would
    pass against an `enforce_monophony` that did nothing."""
    comp = Composition(
        hardware_profile="nes",
        ticks_per_beat=480,
        events=[
            Note(channel_id="P1", tick=0, pitch=60, duration=400, velocity=96),
            Note(channel_id="P1", tick=100, pitch=64, duration=400, velocity=96),
        ],
    )
    assert len(overlapping_events(comp)) == 1
    repaired, truncated = enforce_monophony(comp)
    assert truncated == 1
    assert overlapping_events(repaired) == []
    assert [e.duration for e in repaired.sounding_events()] == [100, 400]


def test_monophony_leaves_separate_channels_alone():
    comp = Composition(
        hardware_profile="nes",
        ticks_per_beat=480,
        events=[
            Note(channel_id="P1", tick=0, pitch=60, duration=400, velocity=96),
            Note(channel_id="P2", tick=100, pitch=64, duration=400, velocity=96),
        ],
    )
    repaired, truncated = enforce_monophony(comp)
    assert truncated == 0
    assert [e.duration for e in repaired.sounding_events()] == [400, 400]


def test_simultaneous_writes_on_one_channel_keep_the_later_event_audible():
    """Two writes at one tick cannot both sound. The earlier is reduced to a
    length of one rather than deleted, so event counts stay comparable across
    codecs and the loss appears as duration error instead of as a
    disappearing note."""
    comp = Composition(
        hardware_profile="nes",
        ticks_per_beat=480,
        events=[
            Note(channel_id="P1", tick=0, pitch=60, duration=200, velocity=96),
            Note(channel_id="P1", tick=0, pitch=67, duration=200, velocity=96),
        ],
    )
    repaired, truncated = enforce_monophony(comp)
    assert truncated == 1
    assert sorted(e.duration for e in repaired.sounding_events()) == [1, 200]


# -- item 1: one-to-one matching ----------------------------------------------


def test_matching_is_one_to_one():
    """One reconstructed note must not be credited with recovering four
    originals, or collapsing an arpeggio would look free."""
    original = [
        Note(channel_id="P1", tick=t, pitch=60, duration=10, velocity=96) for t in (0, 10, 20, 30)
    ]
    reconstructed = [Note(channel_id="P1", tick=0, pitch=60, duration=10, velocity=96)]
    result = match_events(original, reconstructed, tolerance=200)
    assert len(result.pairs) == 1
    assert len(result.unmatched_original) == 3
    assert result.spurious_reconstructed == []


def test_spurious_events_are_reported_separately_from_missing_ones():
    """Netting extra against missing would let a codec that invents a note
    offset one it dropped."""
    original = [Note(channel_id="P1", tick=0, pitch=60, duration=10, velocity=96)]
    reconstructed = [
        Note(channel_id="P1", tick=0, pitch=72, duration=10, velocity=96),
        Note(channel_id="P1", tick=50, pitch=74, duration=10, velocity=96),
    ]
    result = match_events(original, reconstructed, tolerance=200)
    assert result.pairs == []
    assert len(result.unmatched_original) == 1
    assert len(result.spurious_reconstructed) == 2


def test_matching_requires_the_same_channel_and_value():
    """A note recovered at the wrong pitch is not that note."""
    original = [Note(channel_id="P1", tick=0, pitch=60, duration=10, velocity=96)]
    assert (
        match_events(
            original,
            [Note(channel_id="P2", tick=0, pitch=60, duration=10, velocity=96)],
            tolerance=200,
        ).pairs
        == []
    )
    assert (
        match_events(
            original,
            [Note(channel_id="P1", tick=0, pitch=61, duration=10, velocity=96)],
            tolerance=200,
        ).pairs
        == []
    )


def test_matching_prefers_the_closest_correspondence():
    original = [
        Note(channel_id="P1", tick=0, pitch=60, duration=10, velocity=96),
        Note(channel_id="P1", tick=100, pitch=60, duration=10, velocity=96),
    ]
    reconstructed = [
        Note(channel_id="P1", tick=95, pitch=60, duration=10, velocity=96),
        Note(channel_id="P1", tick=5, pitch=60, duration=10, velocity=96),
    ]
    result = match_events(original, reconstructed, tolerance=200)
    assert result.pairs == [(0, 1), (1, 0)]


def test_matching_is_deterministic():
    original = [
        Note(channel_id="P1", tick=t, pitch=60, duration=10, velocity=96) for t in range(0, 200, 10)
    ]
    reconstructed = [
        Note(channel_id="P1", tick=t + 3, pitch=60, duration=10, velocity=96)
        for t in range(0, 200, 10)
    ]
    first = match_events(original, reconstructed, tolerance=50)
    second = match_events(original, reconstructed, tolerance=50)
    assert first.pairs == second.pairs


def test_ioi_entropy_is_zero_for_a_perfectly_regular_stream():
    assert ioi_entropy([0.0, 0.1, 0.2, 0.3, 0.4]) == pytest.approx(0.0)


def test_ioi_entropy_rises_with_rhythmic_variety():
    regular = ioi_entropy([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    varied = ioi_entropy([0.0, 0.05, 0.3, 0.35, 0.8, 1.6])
    assert varied > regular


# -- config guards ------------------------------------------------------------


def test_codec_a_rejects_a_sub_row_setting_it_would_ignore():
    """Otherwise a config reports as "codec A at k=8" in the results table and
    is read as a fidelity result."""
    with pytest.raises(ValueError, match="codec A has no sub-row"):
        TokenizerConfig(codec="A", sub_row_divisions=8)


def test_sub_row_codecs_reject_k_of_one():
    for codec in ("B", "C", "D"):
        with pytest.raises(ValueError, match="sub-row codec"):
            TokenizerConfig(codec=codec, sub_row_divisions=1)


def test_an_unknown_codec_is_rejected():
    with pytest.raises(ValueError, match="unknown codec"):
        TokenizerConfig(codec="E", sub_row_divisions=2)


def test_duration_ceiling_scales_so_the_longest_note_is_the_same_length():
    """Leaving `max_duration_steps` alone would cut codec D's longest note to
    a kth of the others and read as an articulation win for everyone else."""
    a = TokenizerConfig()
    d = TokenizerConfig(codec="D", sub_row_divisions=8)
    assert d.duration_token_max == a.max_duration_steps * 8
    c = TokenizerConfig(codec="C", sub_row_divisions=8)
    assert c.duration_token_max == a.max_duration_steps


def test_motif_slot_is_present_in_every_codec():
    for config in (
        TokenizerConfig(),
        TokenizerConfig(codec="B", sub_row_divisions=4),
        TokenizerConfig(codec="D", sub_row_divisions=4),
    ):
        tokens = build_vocabulary(config).tokens
        assert any(t.startswith(MOTIF) for t in tokens)


def test_matching_needs_a_shared_time_scale():
    """The bug this guards against produced a 0.05 event recall and a 4.4
    second onset error on real pieces. A source piece runs at 44100 ticks per
    second while `decode` rebuilds on a nominal 480-ticks-per-beat grid, so
    comparing raw ticks measures a unit difference, not a timing error."""
    original = [Note(channel_id="P1", tick=44100, pitch=60, duration=4410, velocity=96)]
    reconstructed = [Note(channel_id="P1", tick=480, pitch=60, duration=48, velocity=96)]

    # Raw ticks: the same event looks 43620 ticks apart and matches nothing.
    assert match_events(original, reconstructed, tolerance=1000).pairs == []

    # In seconds, both are at 1.0 s and pair up.
    paired = match_events(
        original,
        reconstructed,
        tolerance=0.2,
        original_scale=1.0 / 44100,
        reconstructed_scale=1.0 / 480,
    )
    assert paired.pairs == [(0, 0)]
