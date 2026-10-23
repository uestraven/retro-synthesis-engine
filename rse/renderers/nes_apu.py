"""A native NES APU synthesizer.

Written in place of the alternatives because it needs no external binary, no
subprocess and no download: it runs offline on Windows, Linux and macOS, and
every part of it is unit-testable. For the record, the alternatives surveyed
were the NES-MDB synthesis package (its `apu.py` holds register tables but no
synthesizer; audio goes through an external `vgm2wav` binary, and two of its
modules are Python 2), FamiStudio's CLI and Furnace's CLI (both require a
download and a subprocess per render). See docs/renderer-survey.md.

What this models
----------------
- Pulse channels: four duty-cycle sequences, 4-bit volume, 11-bit timer.
- Triangle: the 32-step sequence, no volume control, 11-bit timer.
- Noise: the 15-bit LFSR with both tap modes and the NTSC period table.
- Pitches quantized to what the hardware timer can actually produce, so notes
  detune away from equal temperament exactly as they do on the console.
- The APU's nonlinear mixing of pulse against triangle/noise.

What it does not model
----------------------
Hardware envelopes, sweep units, length and linear counters, DPCM, and the
440 Hz output high-pass. Note on/off comes from the score, and volume from
velocity. These are the gaps Phase 3 closes; they are listed in the survey doc
with what each one costs.

Register layout and the length-counter table were cross-checked against
nesmdb's `apu.py` (MIT licensed).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from rse.hardware.profile import HardwareProfile
from rse.renderers.base import EnvelopePoint, RenderNote, RenderResult, RenderSettings

RENDERER_ID = "nes_apu/0.1.0"

#: CPU clock rates. The APU derives every period from these.
NTSC_CPU_HZ = 21_477_272.0 / 12.0  # ~1789772.67
PAL_CPU_HZ = 1_662_607.0

#: The four pulse duty sequences, one row each, 8 steps per cycle.
DUTY_SEQUENCES = np.array(
    [
        [0, 1, 0, 0, 0, 0, 0, 0],  # 12.5%
        [0, 1, 1, 0, 0, 0, 0, 0],  # 25%
        [0, 1, 1, 1, 1, 0, 0, 0],  # 50%
        [1, 0, 0, 1, 1, 1, 1, 1],  # 25% inverted
    ],
    dtype=np.float32,
)

#: The triangle channel's fixed 32-step sequence, 15 down to 0 and back.
TRIANGLE_SEQUENCE = np.array(list(range(15, -1, -1)) + list(range(0, 16)), dtype=np.float32)

#: Noise timer periods in CPU cycles, NTSC. Index 0..15.
NOISE_PERIODS_NTSC = (
    4,
    8,
    16,
    32,
    64,
    96,
    128,
    160,
    202,
    254,
    380,
    508,
    762,
    1016,
    2034,
    4068,
)

#: Below this timer value a pulse channel is silenced by the hardware.
MIN_PULSE_TIMER = 8
#: The triangle's practical floor; lower values are ultrasonic buzz.
MIN_TRIANGLE_TIMER = 2

#: 4-bit volume range shared by the pulse and noise channels.
MAX_VOLUME = 15

#: Default duty per pulse channel: a hollow 50% lead against a thinner 25%
#: counterpoint, which is the most common NES pairing. Overridable.
DEFAULT_DUTY = {"P1": 2, "P2": 1}

#: The console's output stage high-pass, in Hz. The real hardware adds a second
#: stage at 440 Hz; it is omitted because it strips most of the triangle bass,
#: and emulators conventionally leave it out for the same reason.
OUTPUT_HIGHPASS_HZ = 90.0


class NesApuRenderer:
    """Renders RSE notes to audio through a model of the NES APU."""

    renderer_id = RENDERER_ID

    def __init__(
        self,
        profile: HardwareProfile,
        *,
        duty_by_channel: dict[str, int] | None = None,
        region: str = "NTSC",
    ) -> None:
        if profile.profile_id != "nes":
            raise ValueError(f"{RENDERER_ID} renders the 'nes' profile, not {profile.profile_id!r}")
        region = region.upper()
        if region not in ("NTSC", "PAL"):
            raise ValueError(f"region must be NTSC or PAL, got {region!r}")
        self.profile = profile
        self.profile_id = profile.profile_id
        self.region = region
        self.cpu_hz = NTSC_CPU_HZ if region == "NTSC" else PAL_CPU_HZ
        self.duty_by_channel = {**DEFAULT_DUTY, **(duty_by_channel or {})}

    def render(
        self,
        notes: list[RenderNote],
        settings: RenderSettings | None = None,
        envelopes: dict[str, list[EnvelopePoint]] | None = None,
    ) -> RenderResult:
        """Render notes to audio.

        `envelopes` supplies per-channel volume over time, from the source's
        own controller data. Without it every note holds its onset velocity for
        its whole duration, which is what makes a decaying drum hit sound like
        static.
        """
        settings = settings or RenderSettings()
        warnings: list[str] = []
        internal_rate = settings.sample_rate * settings.oversample

        end_sec = max((n.end_sec for n in notes), default=0.0) + settings.tail_sec
        n_internal = int(math.ceil(end_sec * internal_rate))
        # Keep the buffer a whole number of oversample blocks so decimation is
        # an exact reshape rather than a lossy truncation.
        n_internal += (-n_internal) % settings.oversample
        n_internal = max(n_internal, settings.oversample)

        buffers = {cid: np.zeros(n_internal, dtype=np.float32) for cid in self.profile.channel_ids}

        for note in notes:
            if note.channel_id not in buffers:
                warnings.append(f"no channel {note.channel_id!r} in the {self.profile_id} profile")
                continue
            self._render_note(note, buffers, internal_rate, warnings, envelopes)

        mixed = self._mix(buffers)
        if settings.apply_output_filters:
            mixed = _highpass(mixed, OUTPUT_HIGHPASS_HZ, internal_rate)

        samples = _decimate(mixed, settings.oversample)
        stems = {
            cid: _decimate(buf, settings.oversample) for cid, buf in buffers.items() if np.any(buf)
        }

        if settings.peak is not None:
            loudest = float(np.max(np.abs(samples))) if samples.size else 0.0
            if loudest > 0:
                scale = settings.peak / loudest
                samples = samples * scale
                stems = {cid: stem * scale for cid, stem in stems.items()}

        return RenderResult(
            samples=samples.astype(np.float32),
            sample_rate=settings.sample_rate,
            settings=settings,
            renderer_id=self.renderer_id,
            stems={cid: stem.astype(np.float32) for cid, stem in stems.items()},
            warnings=warnings,
        )

    # -- per-channel synthesis ------------------------------------------------

    def _render_note(
        self,
        note: RenderNote,
        buffers: dict[str, np.ndarray],
        rate: int,
        warnings: list[str],
        envelopes: dict[str, list[EnvelopePoint]] | None = None,
    ) -> None:
        spec = self.profile.channel(note.channel_id)
        if not spec.in_range(note.value):
            warnings.append(
                f"{note.channel_id} value {note.value} is outside the profile range "
                f"({spec.value_min}-{spec.value_max}); skipped"
            )
            return

        buffer = buffers[note.channel_id]
        start = int(round(note.start_sec * rate))
        stop = int(round(note.end_sec * rate))
        start = max(0, min(start, buffer.size))
        stop = max(start, min(stop, buffer.size))
        if stop == start:
            return

        count = stop - start
        index = np.arange(count, dtype=np.float64)
        volume = _velocity_to_volume(note.velocity)

        if spec.type == "pulse":
            wave = self._pulse_wave(note, index, rate, volume, warnings)
        elif spec.type == "triangle":
            wave = self._triangle_wave(note, index, rate, warnings)
        elif spec.type == "noise":
            wave = self._noise_wave(note, index, rate, volume)
        else:  # pragma: no cover - the NES profile has no other channel types
            warnings.append(f"no synthesis for channel type {spec.type!r}")
            return

        if wave is None:
            return

        # The triangle channel has no volume register, so an envelope must not
        # be applied to it — on real hardware it is on at full level or off.
        if envelopes and spec.volume_levels > 1:
            gain = _sample_envelope(envelopes.get(note.channel_id), note.start_sec, count, rate)
            if gain is not None:
                wave = wave * gain
        # Assignment, not addition: a hardware channel plays one note at a
        # time, so a later note replaces an earlier one rather than summing.
        buffer[start:stop] = wave

    def _pulse_wave(
        self,
        note: RenderNote,
        index: np.ndarray,
        rate: int,
        volume: float,
        warnings: list[str],
    ) -> np.ndarray | None:
        timer = pulse_timer(midi_to_hz(note.value), self.cpu_hz)
        if timer < MIN_PULSE_TIMER:
            warnings.append(
                f"{note.channel_id} pitch {note.value} needs timer {timer}, below the "
                f"hardware minimum of {MIN_PULSE_TIMER}; silenced as the console would"
            )
            return None
        duty = self.duty_by_channel.get(note.channel_id, 2)
        # The pulse timer decrements once per APU cycle (two CPU cycles) and
        # advances the duty sequencer when it wraps. Stepping phase in CPU
        # cycles rather than in output samples puts the waveform transitions
        # where the hardware puts them, instead of quantizing them to the
        # render rate.
        step = self._sequencer_step(index, rate, 2 * (timer + 1), 8)
        return DUTY_SEQUENCES[duty][step] * volume

    def _triangle_wave(
        self, note: RenderNote, index: np.ndarray, rate: int, warnings: list[str]
    ) -> np.ndarray | None:
        timer = triangle_timer(midi_to_hz(note.value), self.cpu_hz)
        if timer < MIN_TRIANGLE_TIMER:
            warnings.append(
                f"TR pitch {note.value} needs timer {timer}, below the practical "
                f"minimum of {MIN_TRIANGLE_TIMER}; silenced"
            )
            return None
        # The triangle timer decrements every CPU cycle, so its sequencer steps
        # twice as often as a pulse channel's for the same timer value.
        step = self._sequencer_step(index, rate, timer + 1, 32)
        # The triangle has no volume register: it is on at full level or off.
        return TRIANGLE_SEQUENCE[step]

    def _noise_wave(
        self, note: RenderNote, index: np.ndarray, rate: int, volume: float
    ) -> np.ndarray:
        # Profile values run 1..16; the hardware period table is indexed 0..15.
        period = NOISE_PERIODS_NTSC[min(max(note.value - 1, 0), 15)]
        sequence = _lfsr_sequence(mode=0)
        position = self._sequencer_step(index, rate, period, sequence.size)
        return sequence[position] * volume

    def _sequencer_step(
        self, index: np.ndarray, rate: int, cycles_per_step: int, n_steps: int
    ) -> np.ndarray:
        """Which sequencer step is active at each output sample.

        Phase is counted in CPU cycles, which is the clock the APU's timers
        actually divide, so the step boundaries land where the hardware puts
        them rather than on render-rate boundaries.
        """
        cpu_cycles = index * (self.cpu_hz / rate)
        return (cpu_cycles / cycles_per_step).astype(np.int64) % n_steps

    # -- mixing ---------------------------------------------------------------

    def _mix(self, buffers: dict[str, np.ndarray]) -> np.ndarray:
        """Combine channels with the APU's documented nonlinear mixer.

        Two pulse channels at full volume are quieter than the sum of each
        alone, and the pulse group mixes independently of triangle and noise.
        This nonlinearity is a real part of how the console sounds.
        """
        pulse_sum = np.zeros_like(next(iter(buffers.values())))
        tnd = np.zeros_like(pulse_sum)
        for cid, buffer in buffers.items():
            channel_type = self.profile.channel(cid).type
            if channel_type == "pulse":
                pulse_sum = pulse_sum + buffer
            elif channel_type == "triangle":
                tnd = tnd + buffer / 8227.0
            elif channel_type == "noise":
                tnd = tnd + buffer / 12241.0

        pulse_out = np.where(
            pulse_sum > 0, 95.88 / (8128.0 / np.maximum(pulse_sum, 1e-9) + 100.0), 0.0
        )
        tnd_out = np.where(tnd > 0, 159.79 / (1.0 / np.maximum(tnd, 1e-12) + 100.0), 0.0)
        return (pulse_out + tnd_out).astype(np.float32)


# -- hardware period maths ----------------------------------------------------


def midi_to_hz(note: int) -> float:
    """Equal-tempered frequency for a MIDI note number."""
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def pulse_timer(frequency: float, cpu_hz: float = NTSC_CPU_HZ) -> int:
    """The 11-bit timer value a pulse channel needs for `frequency`."""
    if frequency <= 0:
        return 0
    return int(max(0, min(2047, round(cpu_hz / (16.0 * frequency) - 1))))


def triangle_timer(frequency: float, cpu_hz: float = NTSC_CPU_HZ) -> int:
    """The 11-bit timer value the triangle channel needs for `frequency`."""
    if frequency <= 0:
        return 0
    return int(max(0, min(2047, round(cpu_hz / (32.0 * frequency) - 1))))


def pulse_frequency(timer: int, cpu_hz: float = NTSC_CPU_HZ) -> float:
    """The frequency a pulse channel actually produces for a timer value."""
    return cpu_hz / (16.0 * (timer + 1))


def triangle_frequency(timer: int, cpu_hz: float = NTSC_CPU_HZ) -> float:
    return cpu_hz / (32.0 * (timer + 1))


def pitch_error_cents(note: int, cpu_hz: float = NTSC_CPU_HZ) -> float:
    """How far a pulse channel's nearest timer lands from equal temperament."""
    wanted = midi_to_hz(note)
    actual = pulse_frequency(pulse_timer(wanted, cpu_hz), cpu_hz)
    return 1200.0 * math.log2(actual / wanted)


@lru_cache(maxsize=2)
def _lfsr_sequence(mode: int = 0) -> np.ndarray:
    """One full period of the noise channel's shift register output.

    The register is 15 bits with feedback from bit 0 against bit 1 (mode 0) or
    bit 6 (mode 1), and the mixer is gated off while bit 0 is set. Precomputing
    the whole period turns noise synthesis into an array lookup.
    """
    tap = 6 if mode else 1
    length = 93 if mode else 32_767  # mode 1 collapses to a short metallic loop
    register = 1
    out = np.empty(length, dtype=np.float32)
    for i in range(length):
        out[i] = 0.0 if (register & 1) else 1.0
        feedback = (register & 1) ^ ((register >> tap) & 1)
        register = (register >> 1) | (feedback << 14)
    return out


# -- output stage -------------------------------------------------------------


def _decimate(signal: np.ndarray, factor: int) -> np.ndarray:
    """Average each block of `factor` samples down to one.

    Box-averaging is the anti-aliasing low-pass: the APU's stepped waveforms
    carry harmonics well above Nyquist, and without this they fold back as
    audible whistles.
    """
    if factor == 1:
        return signal
    usable = signal.size - (signal.size % factor)
    return signal[:usable].reshape(-1, factor).mean(axis=1).astype(np.float32)


def _highpass(signal: np.ndarray, cutoff_hz: float, rate: int) -> np.ndarray:
    """Remove DC and low bass by subtracting a moving average.

    An approximation of the console's output-stage high-pass, chosen because it
    runs in two passes over the array instead of a per-sample feedback loop.
    """
    window = int(round(0.443 * rate / cutoff_hz))
    if window < 2 or window >= signal.size:
        return signal - float(signal.mean()) if signal.size else signal
    return (signal - _moving_average(signal, window)).astype(np.float32)


def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average, edge-padded, computed with a prefix sum."""
    half = window // 2
    padded = np.pad(signal.astype(np.float64), (half, window - half - 1), mode="edge")
    cumulative = np.cumsum(padded)
    cumulative = np.concatenate(([0.0], cumulative))
    return ((cumulative[window:] - cumulative[:-window]) / window)[: signal.size]


def _sample_envelope(
    points: list[EnvelopePoint] | None,
    start_sec: float,
    count: int,
    rate: int,
) -> np.ndarray | None:
    """Per-sample gain for a note, held stepwise between breakpoints.

    The volume register steps rather than glides, so a breakpoint's value holds
    until the next write. Returns None when there is nothing to apply, leaving
    the caller's velocity-derived constant level in place.
    """
    if not points:
        return None
    times = np.fromiter((point.sec for point in points), dtype=np.float64, count=len(points))
    gains = np.fromiter((point.gain for point in points), dtype=np.float32, count=len(points))
    sample_times = start_sec + np.arange(count, dtype=np.float64) / rate
    # `side="right" - 1` gives the last breakpoint at or before each sample.
    index = np.searchsorted(times, sample_times, side="right") - 1
    if int(index.max()) < 0:
        return None
    # Before the first breakpoint, hold full level rather than silence, so a
    # note starting ahead of any controller write is still audible.
    return np.where(index < 0, np.float32(1.0), gains[np.clip(index, 0, None)]).astype(np.float32)


def _velocity_to_volume(velocity: int) -> float:
    """Map MIDI velocity onto the APU's 4-bit volume register.

    A stand-in for the hardware envelope: NES-MDB carries per-channel volume as
    controller data, and wiring that through is Phase 3 work.
    """
    if velocity <= 0:
        return 0.0
    return float(max(1, min(MAX_VOLUME, round(velocity * MAX_VOLUME / 127))))
