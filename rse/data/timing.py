"""Tick-to-musical-position and tick-to-seconds conversion.

Source timing is kept at its original tick resolution everywhere; these helpers
only *derive* beat, bar and second positions alongside it. Quantization is a
separate, optional, additive step (see `normalize.quantize_beat`) so that no
original timing is ever overwritten (proposal 5.3).
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

DEFAULT_TEMPO_US_PER_BEAT = 500_000  # MIDI default: 120 BPM


@dataclass(frozen=True)
class TempoChange:
    tick: int
    us_per_beat: int

    @property
    def bpm(self) -> float:
        return 60_000_000.0 / self.us_per_beat


@dataclass(frozen=True)
class TimeSignatureChange:
    tick: int
    numerator: int
    denominator: int

    @property
    def quarters_per_bar(self) -> float:
        return self.numerator * 4.0 / self.denominator


class MusicalTime:
    """Maps ticks onto beats, bars and seconds for one composition."""

    def __init__(
        self,
        ticks_per_beat: int,
        tempo_changes: list[TempoChange] | None = None,
        time_signatures: list[TimeSignatureChange] | None = None,
    ) -> None:
        if ticks_per_beat <= 0:
            raise ValueError(f"ticks_per_beat must be positive, got {ticks_per_beat}")
        self.ticks_per_beat = ticks_per_beat

        tempos = sorted(tempo_changes or [], key=lambda t: t.tick)
        if not tempos or tempos[0].tick > 0:
            tempos.insert(0, TempoChange(0, DEFAULT_TEMPO_US_PER_BEAT))
        self.tempo_changes = tempos

        sigs = sorted(time_signatures or [], key=lambda s: s.tick)
        if not sigs or sigs[0].tick > 0:
            sigs.insert(0, TimeSignatureChange(0, 4, 4))
        self.time_signatures = sigs

        # Precompute cumulative seconds and bars at each change point so lookups
        # stay O(log n) over a whole dataset pass.
        self._tempo_ticks = [t.tick for t in tempos]
        self._tempo_seconds: list[float] = [0.0]
        for prev, nxt in zip(tempos, tempos[1:], strict=False):
            span_beats = (nxt.tick - prev.tick) / ticks_per_beat
            self._tempo_seconds.append(
                self._tempo_seconds[-1] + span_beats * prev.us_per_beat / 1_000_000.0
            )

        self._sig_ticks = [s.tick for s in sigs]
        self._sig_bars: list[float] = [0.0]
        for prev, nxt in zip(sigs, sigs[1:], strict=False):
            span_quarters = (nxt.tick - prev.tick) / ticks_per_beat
            self._sig_bars.append(self._sig_bars[-1] + span_quarters / prev.quarters_per_bar)

    def beat(self, tick: int) -> float:
        """Position in quarter notes from the start of the composition."""
        return tick / self.ticks_per_beat

    def seconds(self, tick: int) -> float:
        i = max(0, bisect_right(self._tempo_ticks, tick) - 1)
        base = self.tempo_changes[i]
        span_beats = (tick - base.tick) / self.ticks_per_beat
        return self._tempo_seconds[i] + span_beats * base.us_per_beat / 1_000_000.0

    def bar_position(self, tick: int) -> tuple[int, float]:
        """`(bar index, position within bar in quarter notes)` for a tick."""
        i = max(0, bisect_right(self._sig_ticks, tick) - 1)
        sig = self.time_signatures[i]
        quarters_since = (tick - sig.tick) / self.ticks_per_beat
        bars_since = quarters_since / sig.quarters_per_bar
        absolute_bars = self._sig_bars[i] + bars_since
        bar_index = int(absolute_bars)
        position_in_bar = (absolute_bars - bar_index) * sig.quarters_per_bar
        return bar_index, position_in_bar

    def time_signature_at(self, tick: int) -> TimeSignatureChange:
        i = max(0, bisect_right(self._sig_ticks, tick) - 1)
        return self.time_signatures[i]

    def tempo_at(self, tick: int) -> TempoChange:
        i = max(0, bisect_right(self._tempo_ticks, tick) - 1)
        return self.tempo_changes[i]

    @property
    def is_constant_tempo(self) -> bool:
        return len(self.tempo_changes) == 1

    @property
    def is_constant_meter(self) -> bool:
        return len(self.time_signatures) == 1
