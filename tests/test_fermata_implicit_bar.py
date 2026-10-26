"""A stream that forgets its opening `<BAR>` still has a first bar.

Fermata's grammar is `<BAR> <TRK_n> (<BEAT_b> <SUBPOS_s> <PITCH_p>
<DURATION_d> <VELOCITY_v>)*`, but the model sometimes emits notes before any
`<BAR>`. `decode` tracked bars from -1, so those notes landed on negative
ticks; the D4 round trip then shifts the minimum tick to zero, which moved the
whole piece one bar later. An eight-bar request came back spanning nine, with
the last bar's onsets past the window.

4 of the 30 streams in `listen_batch2` are affected -- 61 of 4769 events,
1.28%, across four of six scenarios -- and nothing downstream caught it:
hardware validation checks channels, pitches and polyphony, never bar
structure.

The fix reads a leading note as an implicit opening bar, and `trim_to_bars`
counts it, so the two agree about how many bars the stream holds.

`test_implicit_first_bar_does_not_shift_the_piece` is the fixture that fails
under the old behaviour: it asserts no negative ticks and an eight-bar span.
"""

from __future__ import annotations

import pytest

from rse.transfer.fermata import SUBPOS, Vocab, decode, trim_to_bars

BEATS_PER_BAR = 4
TICKS_PER_BAR = SUBPOS * BEATS_PER_BAR


def build_vocab() -> Vocab:
    """A vocabulary holding just the tokens these streams use."""
    tokens = [
        "<PAD>",
        "<BOS>",
        "<EOS>",
        "<BAR>",
        "<PLAN>",
        "<PLAN_END>",
        "<DRUM_TRACK>",
        "<METER_4/4>",
        "<TEMPO_120>",
        "<TRACK_80>",
        "<TRK_0>",
    ]
    tokens += [f"<BEAT_{b}>" for b in range(8)]
    tokens += [f"<SUBPOS_{s}>" for s in range(SUBPOS)]
    tokens += [f"<PITCH_{p}>" for p in (60, 62, 64, 67)]
    tokens += [f"<DURATION_{d}>" for d in (6, 12, 24)]
    tokens += [f"<VELOCITY_{v}>" for v in (8, 16)]
    return Vocab(token_to_id={t: i for i, t in enumerate(tokens)})


def note(beat: int, pitch: int) -> list[str]:
    return [f"<BEAT_{beat}>", "<SUBPOS_0>", f"<PITCH_{pitch}>", "<DURATION_6>", "<VELOCITY_8>"]


def header() -> list[str]:
    return [
        "<BOS>",
        "<TEMPO_120>",
        "<METER_4/4>",
        "<PLAN>",
        "<TRACK_80>",
        "<PLAN_END>",
    ]


def stream(*, leading: bool, n_explicit_bars: int) -> list[str]:
    """`leading=True` omits the opening `<BAR>`, as the model sometimes does."""
    tokens = header()
    if leading:
        tokens += ["<TRK_0>", *note(0, 60), *note(2, 62)]
    for index in range(n_explicit_bars):
        tokens += ["<BAR>", "<TRK_0>", *note(0, 64), *note(2, 67)]
        del index
    return tokens


def ids_of(vocab: Vocab, tokens: list[str]) -> list[int]:
    return [vocab.id_of(t) for t in tokens]


def test_well_formed_stream_is_unchanged() -> None:
    """The ordinary case must not move: bar 0 starts at tick 0."""
    vocab = build_vocab()
    ids = ids_of(vocab, stream(leading=False, n_explicit_bars=4))
    score, report = decode(ids, vocab=vocab)
    assert report.implicit_first_bar is False
    assert report.n_events_before_first_bar == 0
    assert report.n_bars == 4
    ticks = sorted(n.tick for t in score.tracks for n in t.notes)
    assert min(ticks) == 0
    assert max(ticks) < 4 * TICKS_PER_BAR


def test_leading_notes_are_attributed_to_an_implicit_first_bar() -> None:
    vocab = build_vocab()
    ids = ids_of(vocab, stream(leading=True, n_explicit_bars=4))
    score, report = decode(ids, vocab=vocab)
    assert report.implicit_first_bar is True
    assert report.n_events_before_first_bar == 2
    # One implicit bar plus four explicit ones.
    assert report.n_bars == 5
    ticks = sorted(n.tick for t in score.tracks for n in t.notes)
    assert min(ticks) >= 0, "leading notes must not produce negative ticks"
    assert max(ticks) < 5 * TICKS_PER_BAR


def test_implicit_first_bar_does_not_shift_the_piece() -> None:
    """The regression fixture.

    Under the old behaviour the leading notes sat at tick -96 and -48, and the
    D4 round trip's shift-to-zero moved everything one bar later.
    """
    vocab = build_vocab()
    ids = ids_of(vocab, stream(leading=True, n_explicit_bars=7))
    score, _report = decode(ids, vocab=vocab)
    ticks = sorted(n.tick for t in score.tracks for n in t.notes)
    assert min(ticks) == 0
    # Eight bars of music: one implicit, seven explicit.
    assert max(ticks) < 8 * TICKS_PER_BAR
    assert max(ticks) >= 7 * TICKS_PER_BAR


@pytest.mark.parametrize("requested", [2, 4, 8])
def test_trim_counts_the_implicit_bar(requested: int) -> None:
    """Asking for N bars gives N bars of music, implicit one included.

    With a leading note the Nth bar is delimited by the (N-1)th explicit
    `<BAR>`, so the trim must stop one `<BAR>` earlier than it otherwise
    would. Without this, an eight-bar request returned nine bars.
    """
    vocab = build_vocab()
    ids = ids_of(vocab, stream(leading=True, n_explicit_bars=requested + 3))
    trimmed = trim_to_bars(ids, vocab=vocab, n_bars=requested)
    assert trimmed.complete
    assert trimmed.n_bars == requested
    score, report = decode(trimmed.ids, vocab=vocab)
    assert report.n_bars == requested
    ticks = [n.tick for t in score.tracks for n in t.notes]
    assert min(ticks) == 0
    assert max(ticks) < requested * TICKS_PER_BAR


@pytest.mark.parametrize("requested", [2, 4, 8])
def test_trim_unaffected_without_a_leading_note(requested: int) -> None:
    vocab = build_vocab()
    ids = ids_of(vocab, stream(leading=False, n_explicit_bars=requested + 3))
    trimmed = trim_to_bars(ids, vocab=vocab, n_bars=requested)
    assert trimmed.complete
    assert trimmed.n_bars == requested
    _score, report = decode(trimmed.ids, vocab=vocab)
    assert report.n_bars == requested
    assert report.implicit_first_bar is False


def test_cross_bar_sustain_is_preserved() -> None:
    """A long note keeps its length; only its bar attribution changed.

    The fix must not shorten anything -- `trim_to_bars` deliberately keeps a
    note whose duration runs past the end, since duration is a property of the
    note rather than a position.
    """
    vocab = build_vocab()
    tokens = header() + [
        "<TRK_0>",
        "<BEAT_0>",
        "<SUBPOS_0>",
        "<PITCH_60>",
        "<DURATION_24>",
        "<VELOCITY_8>",
        "<BAR>",
        "<TRK_0>",
        *note(0, 64),
    ]
    score, report = decode(ids_of(vocab, tokens), vocab=vocab)
    assert report.implicit_first_bar is True
    sustained = [n for t in score.tracks for n in t.notes if n.pitch == 60]
    assert len(sustained) == 1
    assert sustained[0].duration == 24
    assert sustained[0].tick == 0


def test_no_events_before_bar_means_no_implicit_bar() -> None:
    """A `<TRK_>` token alone before the first `<BAR>` is not a note.

    Only an actual onset implies a dropped bar; structural tokens do not, and
    treating them as one would add a phantom bar to well-formed streams.
    """
    vocab = build_vocab()
    tokens = header() + ["<TRK_0>", "<BAR>", "<TRK_0>", *note(0, 60)]
    score, report = decode(ids_of(vocab, tokens), vocab=vocab)
    assert report.implicit_first_bar is False
    assert report.n_bars == 1
    assert [n.tick for t in score.tracks for n in t.notes] == [0]
