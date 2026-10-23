"""Piano-roll rendering (proposal 10.3).

Plain NumPy, returning an RGB array. No plotting library, for two reasons: it
adds no dependency to an application that already renders audio with NumPy
alone, and a function that returns an array is testable by asserting on
pixels rather than by looking at a picture.

The noise channel gets its own strip at the bottom rather than a position on
the pitch axis, because its value is a hardware period and placing it among
pitches would draw a lie.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Colours per channel, RGB. Pulse channels warm, triangle cool, noise grey —
#: distinguishable without relying on hue alone, since the roll is also read
#: by position.
CHANNEL_COLOURS: dict[str, tuple[int, int, int]] = {
    "P1": (255, 196, 64),
    "P2": (255, 112, 96),
    "TR": (96, 192, 255),
    "NO": (168, 168, 176),
}
DEFAULT_COLOUR = (220, 220, 220)

BACKGROUND = (18, 18, 24)
BAR_LINE = (44, 44, 56)
BEAT_LINE = (30, 30, 38)
NOISE_DIVIDER = (60, 60, 72)

#: Rows reserved for the noise strip, one per possible value.
NOISE_ROWS = 16

#: Smallest pitch span to display, so a two-note piece is not drawn as two
#: enormous blocks.
MIN_PITCH_SPAN = 24


def render_pianoroll(
    record: dict[str, Any],
    *,
    width: int = 960,
    height: int = 320,
    beats_per_bar: int = 4,
) -> np.ndarray:
    """Draw an intake record as an RGB piano roll."""
    if width < 16 or height < 32:
        raise ValueError(f"piano roll must be at least 16x32, got {width}x{height}")

    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = BACKGROUND

    events = [e for e in record.get("events", ()) if e.get("type") in ("note", "noise")]
    if not events:
        return image

    ticks_per_beat = record.get("midi", {}).get("ticks_per_beat", 480)
    ticks_per_bar = ticks_per_beat * beats_per_bar
    end_tick = max(e["tick"] + e.get("dur_ticks", 0) for e in events) or 1

    noise_height = NOISE_ROWS if height >= 96 else 0
    pitch_height = height - noise_height - (1 if noise_height else 0)

    _draw_grid(image, end_tick, ticks_per_bar, ticks_per_beat, width, pitch_height)

    pitched = [e for e in events if e.get("type") == "note"]
    if pitched:
        low, high = _pitch_window(pitched)
        for event in pitched:
            _draw_note(
                image,
                x0=_x(event["tick"], end_tick, width),
                x1=_x(event["tick"] + event.get("dur_ticks", 0), end_tick, width),
                y0=_pitch_y(event["pitch"], low, high, pitch_height),
                row_height=max(1, pitch_height // (high - low + 1)),
                colour=CHANNEL_COLOURS.get(event["ch"], DEFAULT_COLOUR),
                velocity=event.get("velocity", 96),
            )

    if noise_height:
        image[pitch_height, :] = NOISE_DIVIDER
        for event in (e for e in events if e.get("type") == "noise"):
            value = int(np.clip(event.get("value", 1), 1, NOISE_ROWS))
            row = pitch_height + 1 + (NOISE_ROWS - value)
            _draw_note(
                image,
                x0=_x(event["tick"], end_tick, width),
                x1=_x(event["tick"] + event.get("dur_ticks", 0), end_tick, width),
                y0=min(row, height - 1),
                row_height=1,
                colour=CHANNEL_COLOURS.get(event["ch"], DEFAULT_COLOUR),
                velocity=event.get("velocity", 96),
            )
    return image


def _pitch_window(events: list[dict[str, Any]]) -> tuple[int, int]:
    """Pitch range to display, widened to a sensible minimum."""
    pitches = [e["pitch"] for e in events]
    low, high = min(pitches), max(pitches)
    span = high - low + 1
    if span < MIN_PITCH_SPAN:
        pad = (MIN_PITCH_SPAN - span) // 2
        low, high = low - pad, high + (MIN_PITCH_SPAN - span - pad)
    return low, high


def _x(tick: int, end_tick: int, width: int) -> int:
    return int(np.clip(round(tick / end_tick * (width - 1)), 0, width - 1))


def _pitch_y(pitch: int, low: int, high: int, pitch_height: int) -> int:
    """Higher pitches sit higher in the image."""
    span = max(1, high - low + 1)
    position = (high - pitch) / span
    return int(np.clip(round(position * (pitch_height - 1)), 0, max(0, pitch_height - 1)))


def _draw_grid(
    image: np.ndarray,
    end_tick: int,
    ticks_per_bar: int,
    ticks_per_beat: int,
    width: int,
    pitch_height: int,
) -> None:
    """Beat and bar lines, so phrase structure is visible at a glance."""
    if ticks_per_beat <= 0 or pitch_height <= 0:
        return
    n_beats = int(end_tick // ticks_per_beat) + 1
    # A dense grid on a long piece is noise; draw beats only when they are
    # far enough apart to read.
    draw_beats = n_beats and (width / n_beats) >= 6
    for beat in range(n_beats):
        tick = beat * ticks_per_beat
        column = _x(tick, end_tick, width)
        if tick % ticks_per_bar == 0:
            image[:pitch_height, column] = BAR_LINE
        elif draw_beats:
            image[:pitch_height, column] = BEAT_LINE


def _draw_note(
    image: np.ndarray,
    *,
    x0: int,
    x1: int,
    y0: int,
    row_height: int,
    colour: tuple[int, int, int],
    velocity: int,
) -> None:
    """Fill a note's rectangle, dimmed by how quiet it is."""
    height, width, _ = image.shape
    x1 = max(x1, x0 + 1)  # a very short note must still be visible
    x0 = int(np.clip(x0, 0, width - 1))
    x1 = int(np.clip(x1, x0 + 1, width))
    y0 = int(np.clip(y0, 0, height - 1))
    y1 = int(np.clip(y0 + max(1, row_height), y0 + 1, height))

    scale = 0.45 + 0.55 * min(1.0, max(0, velocity) / 127.0)
    shade = np.array([int(c * scale) for c in colour], dtype=np.uint8)
    image[y0:y1, x0:x1] = shade


def legend() -> str:
    """Markdown legend matching the colours used above."""
    swatches = []
    for channel, (red, green, blue) in CHANNEL_COLOURS.items():
        swatches.append(f"<span style='color:rgb({red},{green},{blue})'>■</span> {channel}")
    return " &nbsp; ".join(swatches) + " &nbsp;&nbsp; *(noise on its own strip below)*"
