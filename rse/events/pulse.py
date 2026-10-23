"""Finding a composition's beat.

Why this exists
---------------
NES-MDB's timing is in audio frames with no usable tempo: every file claims
120 BPM and the real rate is unrecoverable from the metadata. RSE therefore
tokenizes position as "frame index modulo 16", which means `Position_7` is a
downbeat in one piece and an off-beat in the next. A model asked to learn
rhythm from tokens whose meaning shifts between training examples cannot, and
every checkpoint so far has been described by listeners as having timing that
"feels off" at every sampling temperature.

The rhythm is in the data. Searching each piece for its own period and phase,
real NES-MDB drum onsets concentrate 2.46x above uniform — *more* strongly
than the rule-based generator's deliberately metrical patterns at 2.11x. What
is missing is alignment: the grid is anchored to an arbitrary zero rather than
to the music's pulse.

Status: not production-ready
----------------------------
The detector is validated against a known-metrical control — it recovers the
rule-based generator's period 4, phase 0 exactly, with every onset on one
position of four. On real NES-MDB it finds a pulse in 91% of pieces at a median
concentration of 2.84x uniform.

It is **not yet trustworthy enough to rebuild the tokenizer on**. The most
common period across the corpus is 19 frames and only 66% of estimates are
even numbers; 19 and 11 are primes, which are not musical periods. So roughly a
third of pieces get an implausible estimate, and aligning the grid to a wrong
period would replace one arbitrary grid with another rather than fixing
anything.

Making this reliable is a real beat-tracking problem: tempo changes mid-piece,
pieces with no percussion, and swung or triplet subdivisions all need handling.
Until then this module is a measurement tool, not part of the pipeline.

Method
------
Autocorrelation of the onset sequence, not a histogram search across periods.
A histogram scored per period is biased toward small periods — with a period of
3 every onset lands close to some phase, so concentration looks high. That bias
produced a measurement of 1.04x and a wrong conclusion that the corpus was
arrhythmic. Autocorrelation measures self-similarity at a lag, which does not
reward small lags for free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rse.events.schema import Composition, Noise, Note

#: Frames per second in the exprsco grid.
FRAME_RATE_HZ = 24.0

#: Candidate pulse periods, in frames. 4 frames is 167 ms (a fast 16th at
#: ~90 BPM); 48 frames is 2 s. Below 4 the estimate stops being a beat and
#: becomes an ornament rate.
MIN_PERIOD = 4
MAX_PERIOD = 48

#: Onsets needed before an estimate is worth making.
MIN_ONSETS = 16

#: Peak-to-mean autocorrelation below which the estimate is not trusted.
MIN_CONFIDENCE = 1.15


@dataclass(frozen=True)
class PulseEstimate:
    """A composition's detected beat."""

    period_frames: int
    phase_frames: int
    #: Autocorrelation peak relative to the mean; 1.0 is no structure at all.
    confidence: float
    n_onsets: int

    @property
    def found(self) -> bool:
        return self.period_frames > 0 and self.confidence >= MIN_CONFIDENCE

    @property
    def period_sec(self) -> float:
        return self.period_frames / FRAME_RATE_HZ

    @property
    def bpm(self) -> float:
        """The pulse as a tempo.

        This is the beat the onsets imply, which may be a bar, a beat or a
        subdivision — autocorrelation cannot tell which, and nothing here
        pretends otherwise. It is still vastly better than the 120 BPM the
        files all falsely claim.
        """
        return 60.0 / self.period_sec if self.period_frames else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "period_frames": self.period_frames,
            "phase_frames": self.phase_frames,
            "period_sec": round(self.period_sec, 4),
            "bpm": round(self.bpm, 2),
            "confidence": round(self.confidence, 4),
            "n_onsets": self.n_onsets,
            "found": self.found,
        }


def onset_frames(composition: Composition, *, percussion_only: bool = False) -> list[int]:
    """Frame indexes at which anything starts.

    Percussion marks the beat most plainly, but many pieces have no drums at
    all, so by default every channel contributes.
    """
    ticks_per_frame = composition.ticks_per_beat // 4
    if ticks_per_frame <= 0:
        return []
    wanted: tuple[type, ...] = (Noise,) if percussion_only else (Note, Noise)
    return sorted(
        {
            event.tick // ticks_per_frame
            for event in composition.sounding_events()
            if isinstance(event, wanted)
        }
    )


def detect_pulse(
    composition: Composition,
    *,
    min_period: int = MIN_PERIOD,
    max_period: int = MAX_PERIOD,
) -> PulseEstimate:
    """Estimate a composition's beat period and phase."""
    onsets = onset_frames(composition, percussion_only=True)
    # Fall back to every channel when there is too little percussion to judge.
    if len(onsets) < MIN_ONSETS:
        onsets = onset_frames(composition)
    if len(onsets) < MIN_ONSETS:
        return PulseEstimate(0, 0, 0.0, len(onsets))

    length = onsets[-1] + 1
    signal = np.zeros(length, dtype=np.float64)
    signal[np.asarray(onsets)] = 1.0

    period, confidence = _dominant_lag(signal, min_period, min(max_period, length // 2))
    if period <= 0:
        return PulseEstimate(0, 0, 0.0, len(onsets))
    phase = _best_phase(onsets, period)
    return PulseEstimate(period, phase, confidence, len(onsets))


def _dominant_lag(signal: np.ndarray, min_period: int, max_period: int) -> tuple[int, float]:
    """The lag at which the onset sequence most resembles itself."""
    if max_period < min_period:
        return 0, 0.0
    centred = signal - signal.mean()
    size = 1 << (2 * len(centred) - 1).bit_length()
    spectrum = np.fft.rfft(centred, size)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum))[: len(centred)]
    if correlation[0] <= 0:
        return 0, 0.0
    window = correlation[min_period : max_period + 1]
    if window.size == 0:
        return 0, 0.0

    baseline = float(np.mean(np.abs(window)))
    if baseline <= 0:
        return 0, 0.0

    # Comb scoring: a real beat is also echoed at twice and three times its
    # period, so sum the correlation at a lag's multiples. Scoring bare lags
    # instead let noise win — the most common period across the corpus came out
    # as 19 frames, and 19 and 11 are primes, which are not musical periods.
    best_period, best_score = 0, -np.inf
    for candidate in range(min_period, max_period + 1):
        score = 0.0
        weight_total = 0.0
        for harmonic in (1, 2, 3, 4):
            lag = candidate * harmonic
            if lag >= len(correlation):
                break
            weight = 1.0 / harmonic
            score += weight * float(correlation[lag])
            weight_total += weight
        if weight_total <= 0:
            continue
        score /= weight_total
        if score > best_score:
            best_period, best_score = candidate, score
    if best_period <= 0:
        return 0, 0.0
    period = best_period
    peak = float(correlation[period])

    return period, peak / baseline


def _best_phase(onsets: list[int], period: int) -> int:
    """The offset that puts the most onsets on the beat."""
    counts = np.bincount(np.asarray(onsets) % period, minlength=period)
    return int(np.argmax(counts))


def align_to_pulse(frame: int, pulse: PulseEstimate) -> int:
    """A frame's position within the beat, counted from the detected phase."""
    if not pulse.found:
        return frame
    return (frame - pulse.phase_frames) % pulse.period_frames


def concentration(onsets: list[int], pulse: PulseEstimate) -> float:
    """How strongly onsets land on one position, relative to uniform.

    1.0 means evenly spread over every position in the beat; higher means
    metrical. Used to validate the detector rather than by the tokenizer.
    """
    if not pulse.found or not onsets:
        return 0.0
    counts = np.bincount(
        (np.asarray(onsets) - pulse.phase_frames) % pulse.period_frames,
        minlength=pulse.period_frames,
    )
    shares = counts / counts.sum()
    return float(shares.max() * pulse.period_frames)
