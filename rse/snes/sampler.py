"""SNES-inspired preview renderer: sampled voices, stereo, eight at a time.

**This is a preview, not emulation.** It plays PCM with linear interpolation,
not BRR with the S-DSP's Gaussian filter; its envelopes are computed in
floating point, not from the hardware's rate tables; and it enforces no ARAM
budget. The eight-voice ceiling, the key-on/key-off envelope stages and the
per-voice left/right levels are modelled because they shape the arrangement.
Nothing here may be described as authentic hardware output.

Two things in the brief are easy to get wrong and are done deliberately.

**Release tails occupy voices.** A note's musical duration ends at key-off, but
the sample keeps sounding through its release. The allocator already accounts
for that, and this renderer honours the allocator's assignment rather than
re-deciding: if a voice's previous note is still releasing, that voice is not
available, and retriggering it would cut the tail off mid-decay. So the
renderer trusts `VoiceAssignment` and asserts the schedule is consistent.

**Stereo is a property of the buffer.** Each voice has independent left and
right levels, mixed into a two-channel buffer -- the same shape the S-DSP's
per-voice VOL(L)/VOL(R) registers produce. Mono duplicated into two channels
would be a stereo file with no stereo in it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rse.platform.roles import AllocationReport, Role, VoiceAssignment
from rse.snes.samples import SampleAsset, midi_to_hz

#: Output rate of the preview. Not the S-DSP's 32 kHz mix rate; the preview
#: renders at CD rate so its output needs no resampling before export.
OUTPUT_RATE = 44100

#: Headroom before the mix is normalised, so eight voices summing do not clip.
MIX_PEAK = 0.89


@dataclass(frozen=True)
class VoiceSettings:
    """How one voice plays its instrument."""

    instrument: str
    #: -1 hard left, 0 centre, +1 hard right.
    pan: float = 0.0
    #: Linear gain applied on top of the instrument's calibration.
    gain: float = 1.0
    #: Echo send, 0-1. Declared and reported; the preview applies a simple
    #: delay rather than the S-DSP's FIR echo.
    echo: float = 0.0

    def levels(self) -> tuple[float, float]:
        """Left and right levels, constant-power panned.

        Constant power rather than linear, so a voice panned centre is not
        3 dB louder than the same voice panned hard left -- which is exactly
        the class of balance defect the NES work ran into.
        """
        angle = (max(-1.0, min(1.0, self.pan)) + 1.0) * math.pi / 4.0
        return math.cos(angle) * self.gain, math.sin(angle) * self.gain


@dataclass
class RenderedSong:
    """A finished stereo render and what it cost."""

    #: `(frames, 2)` float buffer in [-1, 1].
    samples: np.ndarray
    sample_rate: int
    #: Per-voice stems, each `(frames, 2)`.
    stems: dict[str, np.ndarray] = field(default_factory=dict)
    allocation: AllocationReport | None = None
    peak_voices: int = 0
    clipped_frames: int = 0
    notes_rendered: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return self.samples.shape[0] / self.sample_rate

    def to_dict(self) -> dict[str, Any]:
        left = self.samples[:, 0]
        right = self.samples[:, 1]
        return {
            "renderer": "snes_sampler (SNES-INSPIRED PREVIEW, not hardware)",
            "sample_rate": self.sample_rate,
            "channels": 2,
            "seconds": round(self.seconds, 3),
            "notes_rendered": self.notes_rendered,
            "peak_voices": self.peak_voices,
            "clipped_frames": self.clipped_frames,
            "peak_left": round(float(np.abs(left).max()), 4),
            "peak_right": round(float(np.abs(right).max()), 4),
            "rms_left": round(float(np.sqrt(np.mean(left**2))), 5),
            "rms_right": round(float(np.sqrt(np.mean(right**2))), 5),
            "stereo_correlation": round(_correlation(left, right), 4),
            "allocation": self.allocation.to_dict() if self.allocation else None,
            "warnings": list(self.warnings),
        }


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    """1.0 means the channels are identical -- i.e. not actually stereo."""
    if left.size == 0:
        return 0.0
    a = left - left.mean()
    b = right - right.mean()
    denominator = math.sqrt(float((a**2).sum()) * float((b**2).sum()))
    return float((a * b).sum() / denominator) if denominator > 1e-12 else 0.0


def envelope_for(
    asset: SampleAsset, *, hold_frames: int, rate: int
) -> np.ndarray:
    """ADSR over the note, key-on through key-off and release.

    Attack and decay run from key-on; sustain holds until key-off at
    `hold_frames`; release then falls to silence. The returned curve is longer
    than `hold_frames` by the release, which is the tail the allocator
    reserved the voice for.
    """
    envelope = asset.instrument.envelope
    attack = max(1, int(envelope.attack * rate))
    decay = max(1, int(envelope.decay * rate))
    release = max(1, int(envelope.release * rate))
    hold = max(1, hold_frames)

    curve = np.zeros(hold + release, dtype=np.float64)
    # The attack RATE is fixed by the instrument, so a note shorter than its
    # own attack is cut off part-way up rather than reaching full level. The
    # first version scaled the ramp to the note instead, which gave every
    # short note a full-amplitude attack and flattened the dynamics -- a 0.1 s
    # note on a 0.16 s attack reported level 1.000 instead of 0.63.
    attack_ramp = np.linspace(0.0, 1.0, attack, endpoint=False)
    end = min(attack, hold)
    curve[:end] = attack_ramp[:end]
    # Decay to the sustain level, also at a fixed rate.
    if hold > attack:
        decay_ramp = np.linspace(1.0, envelope.sustain, decay, endpoint=False)
        decay_end = min(attack + decay, hold)
        span = decay_end - attack
        if span > 0:
            curve[attack:decay_end] = decay_ramp[:span]
        if hold > decay_end:
            curve[decay_end:hold] = envelope.sustain
    level_at_keyoff = curve[hold - 1] if hold > 0 else envelope.sustain
    # Release from wherever the envelope had got to, not from the sustain
    # level: a note released during its attack must not jump up first.
    curve[hold:] = np.linspace(level_at_keyoff, 0.0, release)
    return curve


def _resample(asset: SampleAsset, pitch: int, frames: int) -> np.ndarray:
    """Play the sample at `pitch`, looping if it loops.

    Linear interpolation. The S-DSP uses a 4-tap Gaussian filter whose
    characteristic softening is part of the console's sound; this is
    documented as a difference rather than approximated.
    """
    ratio = midi_to_hz(pitch) / midi_to_hz(asset.instrument.root_note)
    step = ratio * asset.rate / OUTPUT_RATE
    data = asset.data.astype(np.float64)
    loops = asset.loop_end > asset.loop_start

    positions = np.arange(frames, dtype=np.float64) * step
    if loops:
        loop_len = asset.loop_end - asset.loop_start
        past = positions >= asset.loop_start
        positions = np.where(
            past,
            asset.loop_start + np.mod(positions - asset.loop_start, loop_len),
            positions,
        )
    else:
        # One-shot: silence past the end rather than wrapping.
        positions = np.clip(positions, 0, data.size - 1)

    low = np.floor(positions).astype(np.int64)
    high = np.minimum(low + 1, data.size - 1)
    frac = positions - low
    out = data[low] * (1.0 - frac) + data[high] * frac
    if not loops:
        out[np.arange(frames, dtype=np.float64) * step >= data.size] = 0.0
    return out


def render(
    assignments: list[VoiceAssignment],
    *,
    bank: dict[str, SampleAsset],
    voices: dict[Role, VoiceSettings],
    seconds_per_tick: float,
    allocation: AllocationReport | None = None,
    tail_sec: float = 0.6,
    rate: int = OUTPUT_RATE,
    peak: float | None = MIX_PEAK,
) -> RenderedSong:
    """Mix the allocator's assignments into a stereo buffer.

    Consumes `VoiceAssignment`s, so the audible result and the allocation
    report describe the same thing. A note the allocator dropped is not
    rendered, which is the point: the report is the explanation for what is
    missing.
    """
    if not assignments:
        return RenderedSong(
            samples=np.zeros((1, 2), dtype=np.float32),
            sample_rate=rate,
            allocation=allocation,
        )

    end_tick = max(a.note.end_tick for a in assignments)
    frames = int(math.ceil(end_tick * seconds_per_tick * rate)) + int(tail_sec * rate)
    stems: dict[str, np.ndarray] = {}
    warnings: list[str] = []

    # A voice's occupancy, so a retrigger that would cut a live release tail
    # is caught rather than silently truncating the previous note.
    occupied: dict[str, int] = {}

    for assignment in sorted(assignments, key=lambda a: (a.note.tick, a.voice)):
        note = assignment.note
        settings = voices.get(note.role)
        if settings is None:
            warnings.append(f"no voice settings for role {note.role.value}")
            continue
        asset = bank.get(settings.instrument)
        if asset is None:
            warnings.append(f"no sample for instrument {settings.instrument!r}")
            continue

        previous_end = occupied.get(assignment.voice)
        if previous_end is not None and note.tick < previous_end:
            warnings.append(
                f"voice {assignment.voice} retriggered at tick {note.tick} "
                f"while its previous note was still releasing until "
                f"{previous_end}; the allocator should have reserved it"
            )
        occupied[assignment.voice] = note.end_tick

        start = int(note.tick * seconds_per_tick * rate)
        hold = max(1, int(note.duration * seconds_per_tick * rate))
        curve = envelope_for(asset, hold_frames=hold, rate=rate)
        wave = _resample(asset, note.value, len(curve)) * curve
        # Velocity, the instrument's calibration, and the voice's own gain.
        wave *= (note.velocity / 127.0) * asset.instrument.gain
        left, right = settings.levels()

        stem = stems.setdefault(
            assignment.voice, np.zeros((frames, 2), dtype=np.float64)
        )
        stop = min(frames, start + len(wave))
        if stop > start:
            span = stop - start
            stem[start:stop, 0] += wave[:span] * left
            stem[start:stop, 1] += wave[:span] * right

    mix = np.zeros((frames, 2), dtype=np.float64)
    for stem in stems.values():
        mix += stem

    overshoot = int(np.count_nonzero(np.abs(mix) > 1.0))
    if peak is not None:
        loudest = float(np.abs(mix).max())
        if loudest > 0:
            # One scalar for both channels: scaling them independently would
            # destroy the panning the renderer exists to produce.
            scale = peak / loudest
            mix *= scale
            stems = {k: v * scale for k, v in stems.items()}

    # Clipping is what survives into the buffer, not what the sum reached
    # before normalisation scaled it back. Counting the overshoot reported
    # thousands of "clipped" frames in a file whose peak was 0.89.
    clipped = int(np.count_nonzero(np.abs(mix) > 1.0))
    if overshoot and not clipped:
        warnings.append(
            f"the raw mix exceeded full scale on {overshoot} samples and was "
            f"scaled to {peak}; no clipping reached the output"
        )
    return RenderedSong(
        samples=mix.astype(np.float32),
        sample_rate=rate,
        stems={k: v.astype(np.float32) for k, v in stems.items()},
        allocation=allocation,
        peak_voices=allocation.peak_voices if allocation else len(stems),
        clipped_frames=clipped,
        notes_rendered=len(assignments),
        warnings=warnings,
    )
