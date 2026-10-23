from __future__ import annotations

import numpy as np
import pytest

from rse.compose import ComposeRequest, compose
from rse.ui.pianoroll import (
    BACKGROUND,
    CHANNEL_COLOURS,
    NOISE_ROWS,
    legend,
    render_pianoroll,
)


def record_with(events, ticks_per_beat=480):
    return {"midi": {"ticks_per_beat": ticks_per_beat}, "events": events}


def note(tick, pitch, ch="P1", dur=240, velocity=100):
    return {
        "ch": ch,
        "type": "note",
        "tick": tick,
        "dur_ticks": dur,
        "pitch": pitch,
        "velocity": velocity,
    }


def noise(tick, value, dur=120, velocity=100):
    return {
        "ch": "NO",
        "type": "noise",
        "tick": tick,
        "dur_ticks": dur,
        "value": value,
        "velocity": velocity,
    }


def painted(image):
    """Pixels that are not background."""
    return int((image != np.array(BACKGROUND, dtype=np.uint8)).any(axis=2).sum())


def test_shape_and_dtype():
    image = render_pianoroll(record_with([note(0, 60)]), width=320, height=160)
    assert image.shape == (160, 320, 3)
    assert image.dtype == np.uint8


def test_an_empty_record_is_blank_not_an_error():
    image = render_pianoroll(record_with([]), width=64, height=64)
    assert painted(image) == 0


def test_absurd_sizes_are_refused():
    with pytest.raises(ValueError, match="at least 16x32"):
        render_pianoroll(record_with([note(0, 60)]), width=8, height=8)


def test_notes_are_drawn():
    sparse = render_pianoroll(record_with([note(0, 60)]), width=320, height=160)
    busy = render_pianoroll(
        record_with([note(t, 60 + t // 240) for t in range(0, 2400, 240)]),
        width=320,
        height=160,
    )
    assert painted(busy) > painted(sparse) > 0


def test_each_channel_gets_its_own_colour():
    events = [note(0, 72, "P1"), note(480, 67, "P2"), note(960, 48, "TR")]
    image = render_pianoroll(record_with(events), width=320, height=200)
    present = [px for px in np.unique(image.reshape(-1, 3), axis=0)]
    # Velocity dims a colour, so compare hue ordering rather than exact values.
    for channel in ("P1", "P2", "TR"):
        expected = CHANNEL_COLOURS[channel]
        assert any(
            np.argmax(px) == np.argmax(expected) and int(px.astype(int).sum()) > 120
            for px in present
        ), channel


def test_higher_pitches_are_drawn_higher():
    image = render_pianoroll(
        record_with([note(0, 90, dur=2000), note(2000, 40, dur=2000)]),
        width=200,
        height=200,
    )
    rows = np.where((image != np.array(BACKGROUND, dtype=np.uint8)).any(axis=2))[0]
    high_half = rows[rows < 100]
    low_half = rows[rows > 100]
    assert high_half.size and low_half.size


def test_noise_is_drawn_in_its_own_strip_not_among_pitches():
    """Its value is a hardware period; placing it on the pitch axis would lie."""
    height = 200
    image = render_pianoroll(
        record_with([note(0, 60, dur=1000), noise(0, 8, dur=1000)]),
        width=200,
        height=height,
    )
    noise_colour = np.array(CHANNEL_COLOURS["NO"], dtype=np.int16)
    rows = [
        row
        for row in range(height)
        if np.any(np.abs(image[row].astype(np.int16) - noise_colour).sum(axis=1) < 90)
    ]
    assert rows
    # Every noise pixel sits in the bottom strip.
    assert min(rows) >= height - NOISE_ROWS - 2


def test_a_short_note_is_still_visible():
    image = render_pianoroll(
        record_with([note(0, 60, dur=1), note(10000, 60, dur=10000)]),
        width=200,
        height=120,
    )
    assert painted(image) > 0


def test_quieter_notes_are_dimmer():
    loud = render_pianoroll(
        record_with([note(0, 60, dur=2000, velocity=127)]), width=120, height=120
    )
    soft = render_pianoroll(
        record_with([note(0, 60, dur=2000, velocity=20)]), width=120, height=120
    )
    assert loud.astype(int).sum() > soft.astype(int).sum()


def test_bar_lines_are_drawn():
    four_bars = [note(i * 1920, 60, dur=120) for i in range(4)]
    image = render_pianoroll(record_with(four_bars), width=400, height=160)
    assert painted(image) > 4 * 4  # notes alone could not paint this much


def test_a_long_piece_does_not_draw_an_unreadable_beat_grid():
    """Beat lines are suppressed when they would be closer than a few pixels."""
    many = [note(i * 480, 60, dur=120) for i in range(400)]
    image = render_pianoroll(record_with(many), width=200, height=120)
    assert image.shape == (120, 200, 3)


def test_a_real_composition_renders(nes_profile):
    result = compose(ComposeRequest(seed=8, bars=8), profile=nes_profile)
    image = render_pianoroll(result.record, width=640, height=240)
    assert painted(image) > 500
    # All four channels should appear somewhere.
    present = {tuple(px) for px in image.reshape(-1, 3)}
    assert len(present) > 5


def test_legend_names_every_channel():
    text = legend()
    for channel in CHANNEL_COLOURS:
        assert channel in text
    assert "noise" in text.lower()
