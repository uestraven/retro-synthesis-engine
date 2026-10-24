from __future__ import annotations

import pytest

from rse.events.convert import composition_from_intake
from rse.events.schema import Composition, Noise, Note
from rse.generation.plan import make_plan
from rse.generation.procedural import generate
from rse.tokenizers.tokenizer import RseTokenizer
from rse.training.dataset import (
    TRAIN,
    VALID,
    assign_splits,
    bar_count,
    build_windows,
    load_windows,
    save_windows,
    slice_bars,
    split_windows,
    windows_from_composition,
)


@pytest.fixture
def tokenizer():
    return RseTokenizer()


@pytest.fixture
def composition(nes_profile):
    generated = generate(nes_profile, make_plan(seed=3, n_bars=16, energy="medium"))
    return composition_from_intake(generated.record)


# -- slicing ------------------------------------------------------------------


def test_bar_count(composition):
    assert bar_count(composition) == 16


def test_slicing_rebases_to_tick_zero(composition):
    window = slice_bars(composition, 4, 4)
    assert window.sounding_events()
    assert min(e.tick for e in window.sounding_events()) < composition.ticks_per_beat * 4
    assert bar_count(window) == 4


def test_slicing_takes_only_the_requested_bars(composition):
    window = slice_bars(composition, 0, 2)
    ticks_per_bar = composition.ticks_per_beat * 4
    assert all(e.tick < 2 * ticks_per_bar for e in window.sounding_events())


def test_notes_are_cut_at_the_window_edge():
    """A window must be self-contained enough to render on its own."""
    long_note = Composition(
        hardware_profile="nes",
        ticks_per_beat=480,
        events=[Note(channel_id="P1", tick=0, pitch=60, duration=480 * 16, velocity=100)],
    )
    window = slice_bars(long_note, 0, 2)
    assert window.sounding_events()[0].end_tick == 480 * 8


def test_only_the_first_window_inherits_the_loop(composition):
    assert slice_bars(composition, 0, 16).loop_start_tick == 0
    # A slice from the middle of a loop is not itself a loop.
    assert slice_bars(composition, 4, 4).loop_start_tick is None
    assert slice_bars(composition, 4, 4).loop_end_tick is None


def test_slicing_preserves_noise_semantics(composition):
    window = slice_bars(composition, 0, 4)
    assert any(isinstance(e, Noise) for e in window.events)
    assert all(not isinstance(e, Note) or e.channel_id != "NO" for e in window.events)


# -- windowing ----------------------------------------------------------------


def test_windows_are_bar_aligned_and_tokenized(composition, tokenizer):
    windows = windows_from_composition(composition, tokenizer, split=TRAIN, window_bars=8)
    assert windows
    assert all(w.n_bars == 8 for w in windows)
    assert all(w.start_bar % 8 == 0 for w in windows)
    assert all(len(w.ids) > 10 for w in windows)


def test_window_length_conditioning_matches_the_window(composition, tokenizer):
    """A model conditioned on 8 bars must be trained on real 8-bar sequences."""
    window = windows_from_composition(composition, tokenizer, split=TRAIN, window_bars=8)[0]
    tokens = [tokenizer.vocabulary.token_of(i) for i in window.ids]
    assert "Length_8" in tokens
    assert tokens.count("Bar") == 8


def test_overlapping_windows_with_a_hop(nes_profile, tokenizer):
    """Needs a piece long enough in its own bars.

    Windows are cut on the fitted row now, and this generator's onsets sit on
    eighths, so a 32-bar plan is 16 fitted bars. Asking for 8-bar windows of
    the 8-bar fixture gave exactly one window and never exercised the hop.
    """
    from rse.events.convert import composition_from_intake
    from rse.generation.plan import make_plan
    from rse.generation.procedural import generate

    longer = composition_from_intake(generate(nes_profile, make_plan(seed=7, n_bars=32)).record)
    windows = windows_from_composition(longer, tokenizer, split=TRAIN, window_bars=8, hop_bars=4)
    assert [w.start_bar for w in windows] == [0, 4, 8]


def test_a_short_composition_still_yields_one_window(tokenizer):
    short = Composition(
        hardware_profile="nes",
        ticks_per_beat=480,
        events=[Note(channel_id="P1", tick=0, pitch=60, duration=240, velocity=100)],
    )
    windows = windows_from_composition(short, tokenizer, split=TRAIN, window_bars=16)
    assert len(windows) == 1


def test_an_empty_composition_yields_nothing(tokenizer):
    empty = Composition(hardware_profile="nes", ticks_per_beat=480)
    assert windows_from_composition(empty, tokenizer, split=TRAIN) == []


def test_build_windows_reports_statistics(composition, tokenizer):
    entries = [(composition, TRAIN, "a"), (composition, VALID, "b")]
    windows, stats = build_windows(entries, tokenizer, window_bars=8)
    assert stats.n_compositions == 2
    assert stats.n_windows == len(windows)
    assert stats.per_split == {TRAIN: len(windows) // 2, VALID: len(windows) // 2}
    assert stats.max_tokens > 0
    assert stats.mean_tokens > 0


def test_windows_persist(composition, tokenizer, tmp_path):
    windows, _ = build_windows([(composition, TRAIN, "a")], tokenizer, window_bars=8)
    path = save_windows(tmp_path / "w.jsonl", windows)
    restored = load_windows(path)
    assert [w.to_dict() for w in restored] == [w.to_dict() for w in windows]


def test_split_windows_groups_by_split(composition, tokenizer):
    entries = [(composition, TRAIN, "a"), (composition, VALID, "b")]
    windows, _ = build_windows(entries, tokenizer, window_bars=8)
    grouped = split_windows(windows)
    assert set(grouped) >= {TRAIN, VALID}
    assert all(w.split == TRAIN for w in grouped[TRAIN])


# -- splits -------------------------------------------------------------------


def test_splits_are_assigned_per_composition_not_per_window():
    """Overlapping windows in train and valid would make validation loss a lie."""
    identifiers = [f"piece_{i}" for i in range(30)]
    assignment = assign_splits(identifiers, seed=0)
    assert set(assignment) == set(identifiers)
    assert set(assignment.values()) == {TRAIN, VALID, "test"}


def test_splits_are_deterministic():
    identifiers = [f"p{i}" for i in range(20)]
    assert assign_splits(identifiers, seed=7) == assign_splits(identifiers, seed=7)
    assert assign_splits(identifiers, seed=7) != assign_splits(identifiers, seed=8)


def test_a_tiny_corpus_stays_entirely_in_train():
    assert set(assign_splits(["only", "two"]).values()) == {TRAIN}


def test_impossible_fractions_are_refused():
    with pytest.raises(ValueError, match="leave something for training"):
        assign_splits(["a", "b", "c"], valid_fraction=0.6, test_fraction=0.5)


def test_windows_are_cut_on_the_pieces_own_bars(nes_profile, tokenizer):
    """Slicing used ticks_per_beat * 4, which is not a bar for a sampled
    source: on exprsco it is 16 frames, about 0.67 s, against roughly 4 s for
    a real bar. A 16-bar window then held a couple of fitted bars of music and
    the encoder padded the rest, so Length_16 described almost nothing."""
    from rse.events.convert import composition_from_intake
    from rse.generation.plan import make_plan
    from rse.generation.procedural import generate
    from rse.training.dataset import bar_count

    composition = composition_from_intake(
        generate(nes_profile, make_plan(seed=11, n_bars=32)).record
    )
    fit = tokenizer.fit_grid(composition)
    assert fit.aligned
    fitted = bar_count(composition, ticks_per_bar=fit.step_ticks * tokenizer.config.steps_per_bar)
    nominal = bar_count(composition)
    assert fitted != nominal, "fixture cannot distinguish the two grids"

    windows = windows_from_composition(composition, tokenizer, split=TRAIN, window_bars=4)
    # Window starts must stop at the fitted bar count, not the nominal one.
    assert max(w.start_bar for w in windows) < fitted


def test_a_window_inherits_its_parents_row(nes_profile, tokenizer):
    """A short slice can fit a different row from the piece it came from, and a
    corpus whose windows disagree about the grid is the confusion the fitted
    grid exists to remove."""
    from rse.events.convert import composition_from_intake
    from rse.generation.plan import make_plan
    from rse.generation.procedural import generate

    composition = composition_from_intake(
        generate(nes_profile, make_plan(seed=13, n_bars=32)).record
    )
    parent = tokenizer.fit_grid(composition)
    windows = windows_from_composition(composition, tokenizer, split=TRAIN, window_bars=4)
    tempo = f"Tempo_{tokenizer.tempo_token_value(parent.bpm)}"
    for window in windows:
        tokens = [tokenizer.vocabulary.token_of(i) for i in window.ids]
        assert tempo in tokens
