"""Tempo on the NES is a frame count, not a number of beats per minute.

The console has no tempo register. A music driver runs from the NMI, once per
video frame, and advances the tracker one row every N frames -- N an integer,
because there is no fraction of a frame to wait. So with four rows to the beat
the reachable tempos are a ladder, not a dial:

    BPM = 60 * frame_rate / (rows_per_beat * frames_per_row)

At NTSC's 60.0988 Hz that gives 225.37, 180.30, 150.25, 128.78, 112.69 and so
on down. 170 BPM is not on it, and no NES ever played it.

RSE composed in float seconds and therefore produced tempos the hardware
cannot express: `action` asked for 168 BPM, which is 5.37 frames per row --
7% away from anything real. Everything downstream was consistent with that
number, so nothing flagged it; the renderer simply played a tempo no console
could output, and the generator reported it as if it could.

`frames_per_row` is the source of truth here and BPM is derived from it. That
ordering is the point: a snapped integer cannot drift from the number shown
the way a float BPM "corrected on the way out" eventually does.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Video frame rates. The NES music driver ticks once per frame.
NTSC_FRAME_RATE = 60.0988
PAL_FRAME_RATE = 50.0070

#: The D4 grid: four tracker rows to the beat.
ROWS_PER_BEAT = 4

#: Playable range of frames per row. Below 3 the driver has no time to update
#: every channel between rows; above 16 the music is slower than anything in
#: the corpus.
MIN_FRAMES_PER_ROW = 3
MAX_FRAMES_PER_ROW = 16


def bpm_for_frames(
    frames_per_row: int,
    *,
    frame_rate: float = NTSC_FRAME_RATE,
    rows_per_beat: int = ROWS_PER_BEAT,
) -> float:
    """Tempo produced by advancing a row every `frames_per_row` frames."""
    if frames_per_row <= 0:
        raise ValueError(f"frames_per_row must be positive, got {frames_per_row}")
    return 60.0 * frame_rate / (rows_per_beat * frames_per_row)


def frames_for_bpm(
    bpm: float,
    *,
    frame_rate: float = NTSC_FRAME_RATE,
    rows_per_beat: int = ROWS_PER_BEAT,
) -> float:
    """Frames per row a tempo would need. Fractional means unplayable."""
    if bpm <= 0:
        raise ValueError(f"bpm must be positive, got {bpm}")
    return 60.0 * frame_rate / (rows_per_beat * bpm)


def row_seconds(frames_per_row: int, *, frame_rate: float = NTSC_FRAME_RATE) -> float:
    """How long one tracker row lasts."""
    return frames_per_row / frame_rate


@dataclass(frozen=True)
class Tempo:
    """A tempo the hardware can actually produce."""

    frames_per_row: int
    frame_rate: float = NTSC_FRAME_RATE
    rows_per_beat: int = ROWS_PER_BEAT

    @property
    def bpm(self) -> float:
        return bpm_for_frames(
            self.frames_per_row,
            frame_rate=self.frame_rate,
            rows_per_beat=self.rows_per_beat,
        )

    @property
    def row_sec(self) -> float:
        return row_seconds(self.frames_per_row, frame_rate=self.frame_rate)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "frames_per_row": self.frames_per_row,
            "bpm": round(self.bpm, 2),
            "row_ms": round(self.row_sec * 1000, 3),
        }

    def __str__(self) -> str:
        return f"{self.bpm:.2f} BPM ({self.frames_per_row} frames/row)"


def ladder(
    *,
    frame_rate: float = NTSC_FRAME_RATE,
    rows_per_beat: int = ROWS_PER_BEAT,
    min_frames: int = MIN_FRAMES_PER_ROW,
    max_frames: int = MAX_FRAMES_PER_ROW,
) -> list[Tempo]:
    """Every playable tempo, fastest first."""
    return [
        Tempo(frames_per_row=frames, frame_rate=frame_rate, rows_per_beat=rows_per_beat)
        for frames in range(min_frames, max_frames + 1)
    ]


def snap(
    bpm: float,
    *,
    frame_rate: float = NTSC_FRAME_RATE,
    rows_per_beat: int = ROWS_PER_BEAT,
    min_frames: int = MIN_FRAMES_PER_ROW,
    max_frames: int = MAX_FRAMES_PER_ROW,
) -> Tempo:
    """Nearest playable tempo to the one requested.

    Nearest in frames rather than in BPM, because frames are what the hardware
    counts and the BPM ladder is not evenly spaced: the gap from 180.30 to
    150.25 is 30 BPM, while 100.16 to 90.15 is 10. Rounding the frame count
    treats a request the way the driver would.
    """
    wanted = frames_for_bpm(bpm, frame_rate=frame_rate, rows_per_beat=rows_per_beat)
    frames = min(max_frames, max(min_frames, round(wanted)))
    return Tempo(
        frames_per_row=int(frames), frame_rate=frame_rate, rows_per_beat=rows_per_beat
    )


def is_playable(
    bpm: float,
    *,
    tolerance_bpm: float = 0.01,
    frame_rate: float = NTSC_FRAME_RATE,
    rows_per_beat: int = ROWS_PER_BEAT,
) -> bool:
    """Whether a tempo sits on the ladder, within a rounding tolerance."""
    nearest = snap(bpm, frame_rate=frame_rate, rows_per_beat=rows_per_beat)
    return abs(nearest.bpm - bpm) <= tolerance_bpm
