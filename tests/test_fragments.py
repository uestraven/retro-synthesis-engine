"""Bar-boundary splitting: nothing lost, nothing duplicated, nothing inherited."""

from __future__ import annotations

import pytest

from rse.events.schema import Composition, Note
from rse.tokenizers.frozen import frozen_config
from rse.tokenizers.tokenizer import GridFit, RseTokenizer
from rse.training.fragments import (
    FragmentError,
    events_by_bar,
    split_window,
    verify_split,
)

ROW = 100.0
ROWS_PER_BAR = 16
BAR_TICKS = ROW * ROWS_PER_BAR


def _tokenizer() -> RseTokenizer:
    return RseTokenizer(frozen_config())


def _fit() -> GridFit:
    return GridFit(step_ticks=ROW, phase_ticks=0.0, aligned=True, period_sec=0.1)


def _dense(n_bars: int, per_bar: int = 40) -> Composition:
    """A window dense enough to overflow a small context."""
    events: list[Note] = []
    for bar in range(n_bars):
        for index in range(per_bar):
            events.append(
                Note(
                    channel_id=("P1", "P2", "TR")[index % 3],
                    tick=int(bar * BAR_TICKS + index * (BAR_TICKS / per_bar)),
                    pitch=60 + (index % 24),
                    duration=40,
                    velocity=96,
                )
            )
    return Composition(hardware_profile="nes", ticks_per_beat=480, events=events)


def test_a_window_that_fits_is_returned_whole():
    tokenizer, fit = _tokenizer(), _fit()
    window = _dense(8, per_bar=4)
    fragments = split_window(
        window, tokenizer, fit=fit, context=4096, window_bars=8, composition_id="p"
    )
    assert len(fragments) == 1
    assert fragments[0].n_fragments == 1
    assert fragments[0].is_fragment is False
    assert fragments[0].n_bars == 8


def test_an_overlong_window_is_split_at_bar_boundaries():
    tokenizer, fit = _tokenizer(), _fit()
    window = _dense(8, per_bar=40)
    fragments = split_window(
        window, tokenizer, fit=fit, context=600, window_bars=8, composition_id="p"
    )
    assert len(fragments) > 1
    assert all(len(f.ids) <= 600 for f in fragments)
    assert all(f.is_fragment for f in fragments)
    # Bars are contiguous and in order, with no overlap.
    assert [f.start_bar for f in fragments] == sorted(f.start_bar for f in fragments)
    covered: list[int] = []
    for f in fragments:
        covered.extend(range(f.start_bar, f.start_bar + f.n_bars))
    assert covered == list(range(8))


def test_the_split_loses_and_duplicates_nothing():
    tokenizer, fit = _tokenizer(), _fit()
    window = _dense(8, per_bar=40)
    fragments = split_window(
        window, tokenizer, fit=fit, context=600, window_bars=8, composition_id="p"
    )
    report = verify_split(window, fragments, fit=fit, tokenizer=tokenizer, window_bars=8)
    assert report["n_events"] == len(window.sounding_events())
    assert report["bars_covered"] == 8
    assert report["max_tokens"] <= 600


def test_events_are_grouped_by_onset_so_each_belongs_to_one_bar():
    """A note sounding across a bar line stays in the bar it started in. That
    is what makes the fragment union exact rather than overlapping."""
    tokenizer, fit = _tokenizer(), _fit()
    crossing = Note(
        channel_id="P1",
        tick=int(BAR_TICKS - ROW),
        pitch=60,
        duration=int(ROW * 4),
        velocity=96,
    )
    window = Composition(hardware_profile="nes", ticks_per_beat=480, events=[crossing])
    grouped = events_by_bar(
        window,
        fit=fit,
        rows_per_bar=ROWS_PER_BAR,
        divisions=tokenizer.config.onset_divisions,
    )
    assert sorted(grouped) == [0]
    assert len(grouped[0]) == 1


def test_a_note_crossing_a_fragment_boundary_keeps_its_full_duration():
    """Duration is a token, not an end position, so decode reproduces the
    audible continuation. Clipping it would quietly shorten the music."""
    tokenizer, fit = _tokenizer(), _fit()
    long_note = Note(channel_id="TR", tick=0, pitch=48, duration=int(BAR_TICKS * 3), velocity=96)
    window = Composition(
        hardware_profile="nes",
        ticks_per_beat=480,
        events=[long_note, *_dense(8, per_bar=40).events],
    )
    fragments = split_window(
        window, tokenizer, fit=fit, context=600, window_bars=8, composition_id="p"
    )
    first = fragments[0]
    decoded = tokenizer.decode(first.ids).composition
    kept = [e for e in decoded.sounding_events() if e.channel_id == "TR" and e.tick == 0]
    assert kept, "the long note should survive in the fragment it starts in"
    # Its length must exceed the fragment, not be trimmed to it.
    assert kept[0].duration > first.n_bars * ROWS_PER_BAR


def test_a_single_bar_too_large_for_the_context_is_an_error():
    """Silent truncation is the failure this module exists to prevent."""
    tokenizer, fit = _tokenizer(), _fit()
    window = _dense(2, per_bar=60)
    with pytest.raises(FragmentError, match="refusing to truncate"):
        split_window(window, tokenizer, fit=fit, context=80, window_bars=2)


def test_fragments_carry_provenance_and_never_inherit_labels():
    tokenizer, fit = _tokenizer(), _fit()
    window = _dense(8, per_bar=40)
    fragments = split_window(
        window,
        tokenizer,
        fit=fit,
        context=600,
        window_bars=8,
        composition_id="nesmdb:abc",
        source_start_bar=24,
        split="train",
    )
    for index, fragment in enumerate(fragments):
        row = fragment.to_dict()
        assert row["composition_id"] == "nesmdb:abc"
        assert row["source_start_bar"] == 24
        assert row["source_n_bars"] == 8
        assert row["fragment_index"] == index
        assert row["split"] == "train"
        assert row["labels_inherited"] is False
        assert row["is_fragment"] is True


def test_a_fragment_key_cannot_collide_with_its_parent_window():
    """A label manifest keyed on (composition_id, start_bar) must not match a
    fragment to the whole-window label it was cut from. Gate 3A measured that
    a cropped window changes motif class, so inheriting would train against a
    label describing music the fragment does not contain."""
    tokenizer, fit = _tokenizer(), _fit()
    fragments = split_window(
        _dense(8, per_bar=40),
        tokenizer,
        fit=fit,
        context=600,
        window_bars=8,
        composition_id="p",
        source_start_bar=16,
    )
    keys = [(f.composition_id, f.to_dict()["start_bar"]) for f in fragments]
    assert len(set(keys)) == len(keys), "fragment keys must be unique"
    # Only the fragment that genuinely starts at the parent's first bar shares
    # its key, and it is marked a fragment so a join can exclude it.
    parent_key = ("p", 16)
    sharing = [f for f in fragments if (f.composition_id, f.to_dict()["start_bar"]) == parent_key]
    assert all(f.is_fragment for f in sharing)


def test_splitting_is_deterministic():
    tokenizer, fit = _tokenizer(), _fit()
    window = _dense(8, per_bar=40)
    first = split_window(window, tokenizer, fit=fit, context=600, window_bars=8)
    second = split_window(window, tokenizer, fit=fit, context=600, window_bars=8)
    assert [(f.start_bar, f.n_bars, f.ids) for f in first] == [
        (f.start_bar, f.n_bars, f.ids) for f in second
    ]


def test_greedy_splitting_takes_as_many_bars_as_fit():
    """Maximizing bars per fragment keeps each one as close to the original
    musical unit as possible, and keeps the fragment count down."""
    tokenizer, fit = _tokenizer(), _fit()
    window = _dense(8, per_bar=40)
    tight = split_window(window, tokenizer, fit=fit, context=600, window_bars=8)
    loose = split_window(window, tokenizer, fit=fit, context=1200, window_bars=8)
    assert len(loose) < len(tight)


def test_verify_split_rejects_a_gap():
    tokenizer, fit = _tokenizer(), _fit()
    window = _dense(8, per_bar=40)
    fragments = split_window(window, tokenizer, fit=fit, context=600, window_bars=8)
    with pytest.raises(FragmentError, match="missing"):
        verify_split(window, fragments[:-1], fit=fit, tokenizer=tokenizer, window_bars=8)


def test_verify_split_rejects_a_duplicate():
    tokenizer, fit = _tokenizer(), _fit()
    window = _dense(8, per_bar=40)
    fragments = split_window(window, tokenizer, fit=fit, context=600, window_bars=8)
    with pytest.raises(FragmentError, match="duplicated"):
        verify_split(
            window,
            [*fragments, fragments[0]],
            fit=fit,
            tokenizer=tokenizer,
            window_bars=8,
        )


def test_an_event_rounding_past_the_last_bar_is_kept():
    """`slot_of` rounds to the nearest slot, so an event just inside the final
    bar can round up past it. `verify_split` caught exactly one event of 1504
    disappearing this way on real data; it belongs to the last bar."""
    tokenizer, fit = _tokenizer(), _fit()
    divisions = tokenizer.config.onset_divisions
    # A tick that rounds to the first slot of bar 8 in an 8-bar window.
    late = int(8 * BAR_TICKS - ROW / (divisions * 4))
    window = Composition(
        hardware_profile="nes",
        ticks_per_beat=480,
        events=[Note(channel_id="P1", tick=late, pitch=72, duration=20, velocity=96)],
    )
    unclamped = events_by_bar(window, fit=fit, rows_per_bar=ROWS_PER_BAR, divisions=divisions)
    clamped = events_by_bar(
        window, fit=fit, rows_per_bar=ROWS_PER_BAR, divisions=divisions, n_bars=8
    )
    assert max(unclamped) == 8, "the fixture must actually overshoot"
    assert max(clamped) == 7
    assert sum(len(v) for v in clamped.values()) == 1
