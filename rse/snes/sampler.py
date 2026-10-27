"""SNES-inspired preview renderer: sampled voices, stereo, eight at a time.

**This is a preview, not emulation.** What IS real: the samples are BRR-encoded
and decoded by `rse.snes.brr`, so they carry the format's quantisation; the
ARAM budget is computed from measured encoded sizes by `rse.snes.aram`; the
eight-voice ceiling, the key-on/key-off envelope stages and the per-voice
left/right levels are all modelled.

What is NOT: interpolation is a 4-point Catmull-Rom kernel plus a documented
low-pass, not the S-DSP's own 4-tap Gaussian table, which I do not have.
Envelopes are computed in floating point rather than from the hardware's rate
tables. There is no SPC export and no emulator has run any of this -- a
correct byte count is not an SPC file. Nothing here may be described as
authentic hardware output.

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

#: Output high-pass, in Hz.
#:
#: The NES renderer has one and this did not, which showed: the finished mix
#: carried +0.012 of DC and 3.2% of its energy below 60 Hz, peaking at 5-7 Hz.
#: That is sub-audible rumble from envelope and summing offsets, and it reads
#: as muddiness rather than as a tone. Real SNES output is high-passed too.
OUTPUT_HIGHPASS_HZ = 30.0


def _highpass(signal: np.ndarray, cutoff_hz: float, rate: int) -> np.ndarray:
    """One-pole high-pass, applied per channel.

    Deliberately gentle: 30 Hz removes DC and the sub-audible rumble without
    touching the bass instrument, whose lowest note is around 65 Hz.
    """
    if cutoff_hz <= 0 or signal.shape[0] < 2:
        return signal
    alpha = 1.0 / (1.0 + 2.0 * math.pi * cutoff_hz / rate)
    out = np.empty_like(signal)
    previous_in = signal[0].copy()
    previous_out = np.zeros(signal.shape[1], dtype=signal.dtype)
    out[0] = previous_out
    for index in range(1, signal.shape[0]):
        current = signal[index]
        previous_out = alpha * (previous_out + current - previous_in)
        out[index] = previous_out
        previous_in = current
    return out


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


#: Fractional resolution of the interpolation table.
INTERP_STEPS = 256


def _catmull_rom_table(steps: int = INTERP_STEPS) -> np.ndarray:
    """A 4-point interpolating kernel, one row per fractional position.

    Catmull-Rom, which is a genuine interpolator: at an integer position it
    reduces to the sample itself, so the data is reproduced exactly and only
    the positions between samples are estimated.

    This replaces a Gaussian-shaped kernel I wrote first and had to throw
    away. Its row at frac=0 was [0.24, 0.48, 0.24, 0.03], so it smeared an
    impulse across three taps instead of reproducing it -- a low-pass filter
    misnamed as an interpolator. It measured WORSE than plain linear
    interpolation on inharmonic energy, which is how the mistake surfaced.

    The S-DSP's own 4-tap table does roll off the top end, and that softening
    is part of the console's character. It is modelled HERE as a separate,
    explicitly labelled low-pass rather than by pretending this kernel is the
    hardware's table, which I do not have.
    """
    offsets = np.arange(steps, dtype=np.float64) / steps
    table = np.empty((steps, 4), dtype=np.float64)
    for index, t in enumerate(offsets):
        t2, t3 = t * t, t * t * t
        table[index] = (
            -0.5 * t3 + t2 - 0.5 * t,
            1.5 * t3 - 2.5 * t2 + 1.0,
            -1.5 * t3 + 2.0 * t2 + 0.5 * t,
            0.5 * t3 - 0.5 * t2,
        )
    return table


INTERP = _catmull_rom_table()

#: Cutoff of the softening filter that stands in for the S-DSP's 4-tap
#: table's top-end rolloff. A documented approximation of the character, not
#: the hardware's coefficients.
DSP_SOFTEN_HZ = 11000.0


def _interpolate(
    data: np.ndarray,
    positions: np.ndarray,
    *,
    loop: tuple[int, int] | None = None,
) -> np.ndarray:
    """Sample `data` at fractional `positions` with the 4-point kernel.

    `loop` is the `(start, end)` the player wraps around, and passing it is
    not optional for a looping sample. A 4-point kernel reads taps at -1, 0,
    +1 and +2 around each position, so the last two reads before a wrap need
    samples from the START of the loop. Without `loop` they were clamped to
    the buffer's final sample instead, repeating it -- a step at every wrap,
    in the rendered output, on every held note.

    It survived three rounds of loop testing because every one of those tests
    examined the SAMPLE DATA. The data was periodic and the decoded BRR
    repeated bit-identically; the defect was downstream of both, in the
    player reading it. Nothing was wrong with the loop. The interpolator just
    did not know there was one.

    Taps wrap once they are inside the loop region, which also means the -1
    tap at exactly `start` reads `end - 1` rather than the attack's last
    sample. That is correct for every pass after the first, and harmless on
    the first, because the lead-in is generated continuously into the loop --
    the two candidate samples are nearly equal by construction.
    """
    base = np.floor(positions).astype(np.int64)
    frac = positions - base
    rows = INTERP[
        np.minimum((frac * INTERP_STEPS).astype(np.int64), INTERP_STEPS - 1)
    ]
    size = data.size
    out = np.zeros(positions.size, dtype=np.float64)
    for tap, offset in enumerate((-1, 0, 1, 2)):
        index = base + offset
        if loop is not None:
            start, end = loop
            length = end - start
            if length > 0:
                index = np.where(
                    index >= start, start + np.mod(index - start, length), index
                )
        index = np.clip(index, 0, size - 1)
        out += data[index] * rows[:, tap]
    return out


def _soften(signal: np.ndarray, cutoff_hz: float, rate: int) -> np.ndarray:
    """One-pole low-pass, standing in for the DSP table's top-end rolloff."""
    if cutoff_hz <= 0 or signal.shape[0] < 2:
        return signal
    alpha = (2.0 * math.pi * cutoff_hz / rate) / (
        1.0 + 2.0 * math.pi * cutoff_hz / rate
    )
    out = np.empty_like(signal)
    previous = signal[0]
    out[0] = previous
    for index in range(1, signal.shape[0]):
        previous = previous + alpha * (signal[index] - previous)
        out[index] = previous
    return out


def _resample(
    asset: SampleAsset,
    pitch: int,
    frames: int,
    *,
    pitched: bool = True,
    interpolation: str = "catmull",
) -> np.ndarray:
    """Play the sample at `pitch`, looping if it loops.

    `pitched=False` plays the sample at its own root note, untransposed. That
    is what a percussion hit needs: its `value` selects an instrument rather
    than naming a pitch -- on NES it is a noise period index -- and resampling
    by it pitched a kick down 2.9 octaves, a snare 4.5 and a hat 6, stretching
    a 70 ms hat into a 6.17-second sub-bass drone. With 327 percussion notes
    the result was a continuous bed of overlapping drones, heard as a note
    that was both off-key and off-beat.

    `interpolation` is "catmull" for the 4-point kernel or "linear" for the
    simpler one. Neither is the S-DSP's own Gaussian table; see
    `_catmull_rom_table` for what is and is not claimed.
    """
    if not pitched:
        pitch = asset.instrument.root_note
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

    # The loop bounds go to the interpolator, not just to the position
    # arithmetic. Wrapping the positions is only half the job: the kernel's
    # taps have to wrap too, or the samples it reads at the boundary come
    # from the end of the buffer instead of the start of the loop.
    bounds = (asset.loop_start, asset.loop_end) if loops else None
    if interpolation == "catmull":
        out = _interpolate(data, positions, loop=bounds)
    else:
        low = np.floor(positions).astype(np.int64)
        high = low + 1
        if loops:
            length = asset.loop_end - asset.loop_start
            high = np.where(
                high >= asset.loop_start,
                asset.loop_start + np.mod(high - asset.loop_start, length),
                high,
            )
        high = np.minimum(high, data.size - 1)
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
    interpolation: str = "catmull",
    soften: bool = True,
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
        wave = _resample(
            asset, note.value, len(curve), pitched=note.pitched,
            interpolation=interpolation,
        ) * curve
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

    if soften:
        mix = _soften(mix, DSP_SOFTEN_HZ, rate)
    mix = _highpass(mix, OUTPUT_HIGHPASS_HZ, rate)

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
