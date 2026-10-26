"""Does the loop actually join? Measured, not asserted.

"Loopable" has been claimed three times on evidence that amounted to setting
two integers. The faults found by actually looking were different each time:

1. a note sustaining past the loop point, cut mid-sound on every wrap
   (`past_loop_end`, found by the constraint engine, not repaired by it);
2. an energy cliff -- sections opening at ~1526 RMS against a 4918 ending, so
   the wrap dropped into a hole;
3. nothing at all wrong with timing, which is what made the first two easy to
   miss.

So this reports three independent things rather than a verdict. Timing
continuity is a hard pass/fail: a note crossing the boundary is a defect. The
other two are measurements with no threshold attached, because "how big a
level jump is audible" is a judgement and inventing a cutoff for it would
repeat the mistake of every metric in this project that failed to predict what
a listener heard.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rse.events.schema import Composition


@dataclass
class LoopDiagnostic:
    """What the loop boundary looks like, symbolically and in the audio."""

    loop_start_tick: int = 0
    loop_end_tick: int = 0
    #: Events whose duration crosses the loop point. A hard defect: on the
    #: wrap they are cut mid-sound.
    n_crossing_events: int = 0
    crossing: list[dict[str, Any]] = field(default_factory=list)
    #: Events starting at or after the loop end, which will never sound.
    n_past_end: int = 0
    #: Bars with no onset immediately before the loop point, and immediately
    #: after it. A silent final bar makes the loop stutter.
    final_bar_onsets: int = 0
    first_bar_onsets: int = 0
    #: Audio measurements, when a rendered file is supplied.
    first_bar_rms: float | None = None
    final_bar_rms: float | None = None
    energy_ratio: float | None = None
    boundary_step: int | None = None
    peak: int | None = None

    @property
    def timing_continuous(self) -> bool:
        """Hard criterion: nothing is cut by the wrap, nothing is stranded."""
        return self.n_crossing_events == 0 and self.n_past_end == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "loop_start_tick": self.loop_start_tick,
            "loop_end_tick": self.loop_end_tick,
            "timing_continuous": self.timing_continuous,
            "n_crossing_events": self.n_crossing_events,
            "crossing": self.crossing[:10],
            "n_past_end": self.n_past_end,
            "final_bar_onsets": self.final_bar_onsets,
            "first_bar_onsets": self.first_bar_onsets,
            "first_bar_rms": None if self.first_bar_rms is None else round(self.first_bar_rms, 1),
            "final_bar_rms": None if self.final_bar_rms is None else round(self.final_bar_rms, 1),
            "energy_ratio": None if self.energy_ratio is None else round(self.energy_ratio, 3),
            "boundary_step": self.boundary_step,
            "peak": self.peak,
            "notes": self.notes(),
        }

    def notes(self) -> list[str]:
        """Plain statements about the boundary, with no pass/fail invented."""
        out: list[str] = []
        if self.n_crossing_events:
            out.append(
                f"{self.n_crossing_events} event(s) sustain past the loop point "
                "and will be cut on the wrap"
            )
        if self.n_past_end:
            out.append(
                f"{self.n_past_end} event(s) start at or after the loop end and "
                "never sound"
            )
        if not self.final_bar_onsets:
            out.append("the final bar has no onsets; the loop will stutter")
        if self.energy_ratio is not None:
            if self.energy_ratio < 0.5:
                out.append(
                    f"the first bar is much quieter than the last "
                    f"(ratio {self.energy_ratio:.2f}); the wrap drops in level"
                )
            elif self.energy_ratio > 2.0:
                out.append(
                    f"the first bar is much louder than the last "
                    f"(ratio {self.energy_ratio:.2f}); the wrap jumps in level"
                )
        if self.boundary_step is not None and self.peak:
            share = self.boundary_step / self.peak
            if share > 0.1:
                out.append(
                    f"sample discontinuity at the boundary is {share:.0%} of "
                    "peak, which may click"
                )
        if not out:
            out.append("no timing or level discontinuity found at the boundary")
        return out


def check(
    composition: Composition,
    *,
    bar_ticks: int,
    wav_path: Path | None = None,
) -> LoopDiagnostic:
    """Inspect the loop boundary symbolically, and in the audio if given."""
    start = composition.loop_start_tick or 0
    end = composition.loop_end_tick or composition.end_tick or 0
    diagnostic = LoopDiagnostic(loop_start_tick=start, loop_end_tick=end)
    if end <= start:
        return diagnostic

    for event in composition.sounding_events():
        finish = event.tick + getattr(event, "duration", 0)
        if event.tick >= end:
            diagnostic.n_past_end += 1
        elif finish > end:
            diagnostic.n_crossing_events += 1
            diagnostic.crossing.append(
                {
                    "channel": event.channel_id,
                    "tick": event.tick,
                    "ends": finish,
                    "overshoot": finish - end,
                }
            )
        if end - bar_ticks <= event.tick < end:
            diagnostic.final_bar_onsets += 1
        if start <= event.tick < start + bar_ticks:
            diagnostic.first_bar_onsets += 1

    if wav_path is not None and Path(wav_path).exists():
        with wave.open(str(wav_path), "rb") as handle:
            frames = handle.readframes(handle.getnframes())
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float64)
        if samples.size:
            total_bars = max(1, round((end - start) / bar_ticks))
            per_bar = samples.size / total_bars
            head = samples[: int(per_bar)]
            tail = samples[-int(per_bar) :]
            diagnostic.first_bar_rms = float(np.sqrt((head**2).mean()))
            diagnostic.final_bar_rms = float(np.sqrt((tail**2).mean()))
            if diagnostic.final_bar_rms > 0:
                diagnostic.energy_ratio = (
                    diagnostic.first_bar_rms / diagnostic.final_bar_rms
                )
            # The wrap puts the last sample next to the first one.
            diagnostic.boundary_step = int(abs(samples[-1] - samples[0]))
            diagnostic.peak = int(np.abs(samples).max())
    return diagnostic
