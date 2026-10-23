"""Loop validation, musical and acoustic.

Proposal 7.5 requires both: a loop can be metrically perfect and still click,
and it can be click-free and still leave an audible gap. So the musical side
checks the score and the acoustic side checks the rendered samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Amplitude below which a sample counts as silence.
SILENCE_FLOOR = 1e-4

#: Window used to measure how loud the loop's start and end are, in seconds.
LOUDNESS_WINDOW_SEC = 0.025

#: Loudness mismatch across the seam, in dB, above which the repeat pumps
#: audibly. A starting value, not a measured one: tune it once there are real
#: loops to listen to.
MAX_SEAM_LOUDNESS_DB = 10.0

#: Headroom on the raw-sample seam step. Chip audio steps full-scale between
#: adjacent samples constantly — that is what a square wave *is* — so the seam
#: only indicates a click if it is more abrupt than the transitions already in
#: the signal. This check almost never fires, and that is the correct
#: behaviour: an earlier version compared the seam against a fixed threshold
#: and flagged every piece that simply began on a downbeat.
SEAM_STEP_TOLERANCE = 1.1

#: Trailing silence, in seconds, above which a loop has an audible gap.
MAX_TRAILING_SILENCE_SEC = 0.05


@dataclass
class LoopReport:
    """Whether a composition loops cleanly, and if not, why not."""

    n_bars: int = 0
    total_ticks: int = 0
    notes_crossing_boundary: int = 0
    resolves_to_tonic: bool = False
    has_loop_markers: bool = False
    seam_step: float | None = None
    max_adjacent_step: float | None = None
    seam_loudness_db: float | None = None
    trailing_silence_sec: float | None = None
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "n_bars": self.n_bars,
            "total_ticks": self.total_ticks,
            "notes_crossing_boundary": self.notes_crossing_boundary,
            "resolves_to_tonic": self.resolves_to_tonic,
            "has_loop_markers": self.has_loop_markers,
            "seam_step": None if self.seam_step is None else round(self.seam_step, 5),
            "max_adjacent_step": (
                None if self.max_adjacent_step is None else round(self.max_adjacent_step, 5)
            ),
            "seam_loudness_db": (
                None if self.seam_loudness_db is None else round(self.seam_loudness_db, 2)
            ),
            "trailing_silence_sec": (
                None if self.trailing_silence_sec is None else round(self.trailing_silence_sec, 4)
            ),
            "issues": self.issues,
        }


def seam_step(samples: np.ndarray) -> float:
    """The amplitude jump from the last sample back to the first, over peak.

    This is what a listener hears as a click when playback wraps.
    """
    array = np.asarray(samples)
    if array.size < 2:
        return 0.0
    peak = float(np.max(np.abs(array)))
    if peak <= 0:
        return 0.0
    return abs(float(array[-1]) - float(array[0])) / peak


def max_adjacent_step(samples: np.ndarray) -> float:
    """The largest jump between neighbouring samples, over peak.

    The yardstick for the seam: a transition no sharper than this is one the
    signal already contains everywhere.
    """
    array = np.asarray(samples)
    if array.size < 2:
        return 0.0
    peak = float(np.max(np.abs(array)))
    if peak <= 0:
        return 0.0
    return float(np.max(np.abs(np.diff(array))) / peak)


def seam_loudness_db(samples: np.ndarray, sample_rate: int) -> float:
    """How much louder the loop's first moment is than its last, in dB.

    A large positive value means the repeat slams in; a large negative one
    means it dies away before looping. Either pumps.
    """
    array = np.asarray(samples, dtype=np.float64)
    window = max(1, int(LOUDNESS_WINDOW_SEC * sample_rate))
    if array.size < window * 2:
        return 0.0
    start = float(np.sqrt(np.mean(array[:window] ** 2)))
    end = float(np.sqrt(np.mean(array[-window:] ** 2)))
    return 20.0 * np.log10(max(start, 1e-9) / max(end, 1e-9))


def trailing_silence_sec(samples: np.ndarray, sample_rate: int) -> float:
    """How much silence sits at the end of the audio."""
    array = np.abs(np.asarray(samples))
    if array.size == 0:
        return 0.0
    sounding = np.flatnonzero(array > SILENCE_FLOOR)
    if sounding.size == 0:
        return array.size / sample_rate
    return float(array.size - 1 - sounding[-1]) / sample_rate


def tile_for_audition(samples: np.ndarray, repeats: int = 3) -> np.ndarray:
    """Repeat a loop end to end so the seam can be heard rather than measured.

    The seam is the one part of a loop that no single playthrough reveals: a
    piece can measure clean and still pump or click on the wrap. Playing it
    three times is how a person checks.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    array = np.asarray(samples)
    if array.size == 0 or repeats == 1:
        return array
    return np.tile(array, repeats)


def validate_loop(
    record: dict[str, Any],
    *,
    samples: np.ndarray | None = None,
    sample_rate: int | None = None,
    tonic: int | None = None,
) -> LoopReport:
    """Check that a composition is fit to loop.

    Pass rendered `samples` to include the acoustic checks; without them only
    the score is inspected.
    """
    report = LoopReport()
    events = record.get("events", ())
    notes = [e for e in events if e.get("type") in ("note", "noise")]

    plan = record.get("generation", {}).get("plan")
    ticks_per_beat = record.get("midi", {}).get("ticks_per_beat", 480)
    if plan:
        report.n_bars = plan["n_bars"]
        report.total_ticks = plan["n_bars"] * plan["beats_per_bar"] * plan["ticks_per_beat"]
        if tonic is None:
            tonic = plan["key"]["tonic"]
    else:
        report.n_bars = record.get("measurements", {}).get("n_bars", 0)
        beats_per_bar = 4
        signatures = record.get("time_signatures", ())
        if signatures:
            beats_per_bar = signatures[0].get("numerator", 4)
        report.total_ticks = report.n_bars * beats_per_bar * ticks_per_beat

    if not notes:
        report.issues.append("composition contains no notes")
        return report

    if report.total_ticks > 0:
        crossing = [n for n in notes if n["tick"] + n.get("dur_ticks", 0) > report.total_ticks]
        report.notes_crossing_boundary = len(crossing)
        if crossing:
            report.issues.append(
                f"{len(crossing)} note(s) extend past the loop boundary at tick "
                f"{report.total_ticks}; on repeat they would overlap bar 1"
            )

    report.has_loop_markers = bool(record.get("loop_markers"))
    if not report.has_loop_markers:
        report.issues.append("no loop markers; the loop boundary is implicit")

    if tonic is not None and report.total_ticks > 0:
        bar_ticks = report.total_ticks // max(1, report.n_bars)
        final_bar = [n for n in notes if n["tick"] >= report.total_ticks - bar_ticks]
        pitched = [n for n in final_bar if n.get("type") == "note"]
        report.resolves_to_tonic = any((n["pitch"] - tonic) % 12 == 0 for n in pitched)
        if pitched and not report.resolves_to_tonic:
            report.issues.append(
                "the final bar does not touch the tonic, so the return to bar 1 "
                "will not feel resolved"
            )

    if samples is not None:
        rate = sample_rate or 44_100
        report.seam_step = seam_step(samples)
        report.trailing_silence_sec = trailing_silence_sec(samples, rate)
        report.max_adjacent_step = max_adjacent_step(samples)
        report.seam_loudness_db = seam_loudness_db(samples, rate)
        if report.seam_step > SEAM_STEP_TOLERANCE * report.max_adjacent_step:
            report.issues.append(
                f"the loop seam steps by {report.seam_step:.0%} of peak amplitude, "
                f"sharper than the {report.max_adjacent_step:.0%} the signal "
                "contains elsewhere; it will click"
            )
        if abs(report.seam_loudness_db) > MAX_SEAM_LOUDNESS_DB:
            direction = "slams in" if report.seam_loudness_db > 0 else "dies away"
            report.issues.append(
                f"the loop {direction} across the seam "
                f"({report.seam_loudness_db:+.1f} dB); the repeat will pump"
            )
        if report.trailing_silence_sec > MAX_TRAILING_SILENCE_SEC:
            report.issues.append(
                f"{report.trailing_silence_sec:.2f}s of silence at the end leaves an "
                "audible gap on repeat; render loops with tail_sec=0"
            )
    return report
