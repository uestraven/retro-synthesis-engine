"""Reading NES-MDB's expressive score (`exprsco`) release.

This is the release to train on. The MIDI release is a transcription of VGM
register writes at audio-sample resolution: its `ticks_per_beat` is 22050 with
tempo pinned at 120 BPM for all 5278 files, the gcd of note onsets is one tick,
and no uniform grid explains the data — the best candidate (1/240 s) covers
10.9% of onsets. Quantizing that onto a musical grid would destroy the rhythm.

`exprsco` is the same music on a regular 24 Hz grid, and it is APU-native:
velocity *is* the 4-bit volume register and timbre *is* the pulse duty cycle.
It also ships NES-MDB's official composer-disjoint splits.

Format: a pickle of `(rate, nsamps, uint8 array[frames, 4, 3])`, where the
channel axis is P1, P2, TR, NO and the last axis is `[pitch, velocity,
timbre]`. A sustained note is held by repeating its pitch across frames.

Those pickles are untrusted input, so they are read with an unpickler
restricted to the numpy array-reconstruction globals. Plain `pickle.load` on a
downloaded file executes whatever the file says to.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rse.data.midi_io import Control, IntakeComposition, Note
from rse.data.roles import BY_TRACK_NAME, AssignmentResult, ChannelAssignment, SourceVoice
from rse.data.timing import MusicalTime, TempoChange, TimeSignatureChange
from rse.hardware.profile import HardwareProfile

#: Ticks per 24 Hz frame, and the resolution that makes one frame land exactly
#: on one tokenizer grid step. Frames map onto the grid with no quantization
#: error at all, which is the whole reason for preferring this release.
FRAME_TICKS = 120
TICKS_PER_BEAT = 480

#: With one frame per sixteenth note at 24 frames per second, a quarter note
#: lasts 1/6 s — so the tempo that makes wall-clock duration come out right is
#: 360 BPM. The number is strange because the source has no real tempo: it is a
#: fixed-rate score, not a performance in a key and metre. What matters is that
#: rendered audio has the correct duration, and this makes it exact.
FRAME_RATE_HZ = 24.0
TEMPO_US_PER_BEAT = round(60_000_000 / 360)

#: Controllers used to carry the per-frame envelope, matching the numbers
#: NES-MDB's own MIDI release uses: 11 for volume, 12 for timbre.
CC_VOLUME = 11
CC_TIMBRE = 12

#: The triangle channel has no volume register, so exprsco stores velocity 0
#: for it throughout. Notes still need an audible velocity downstream.
TRIANGLE_VELOCITY = 100

#: Channel axis order in the exprsco array.
CHANNEL_ORDER = ("P1", "P2", "TR", "NO")

#: 4-bit volume register maximum.
MAX_REGISTER_VOLUME = 15


class ExprscoError(ValueError):
    """Raised when a file cannot be read as an exprsco score."""


class UnsafePickleError(ExprscoError, pickle.UnpicklingError):
    """Raised when a pickle tries to pull in something other than numpy.

    Inherits from both so that callers catching either an exprsco problem or an
    unpickling problem see it, and so a security refusal is distinguishable
    from a merely corrupt file.
    """


class _SafeUnpickler(pickle.Unpickler):
    """Allows only the globals numpy needs to rebuild an array."""

    ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy._core.multiarray", "_reconstruct"),
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise UnsafePickleError(
            f"refused to load {module}.{name} from an exprsco pickle; only numpy "
            "array reconstruction is permitted"
        )


@dataclass(frozen=True)
class ExprscoScore:
    """One parsed exprsco file."""

    rate: float
    n_samples: int
    array: np.ndarray

    @property
    def n_frames(self) -> int:
        return int(self.array.shape[0])

    @property
    def duration_sec(self) -> float:
        return self.n_frames / self.rate if self.rate else 0.0


def load_exprsco(path: Path) -> ExprscoScore:
    """Read an exprsco pickle, safely."""
    path = Path(path)
    try:
        with path.open("rb") as handle:
            payload = _SafeUnpickler(handle).load()
    except UnsafePickleError:
        raise  # a refusal, not a corruption; let it speak for itself
    except Exception as exc:
        raise ExprscoError(f"{path.name}: {type(exc).__name__}: {exc}") from exc

    if not isinstance(payload, tuple) or len(payload) != 3:
        raise ExprscoError(f"{path.name}: expected a 3-tuple, got {type(payload).__name__}")
    rate, n_samples, array = payload
    if not isinstance(array, np.ndarray):
        raise ExprscoError(f"{path.name}: third element is not a numpy array")
    if array.ndim != 3 or array.shape[1] != 4 or array.shape[2] != 3:
        raise ExprscoError(
            f"{path.name}: expected an array shaped (frames, 4, 3), got {array.shape}"
        )
    return ExprscoScore(rate=float(rate), n_samples=int(n_samples), array=array)


def score_to_composition(
    score: ExprscoScore, path: Path, profile: HardwareProfile
) -> IntakeComposition:
    """Convert a score into the same structure the MIDI parser produces.

    Reusing `IntakeComposition` means exprsco records go through exactly the
    same normalization, measurement and compliance checks as MIDI ones, so
    `hardware_compliant` means the same thing for both.
    """
    array = score.array
    warnings: list[str] = []
    notes: list[Note] = []
    controls: list[Control] = []

    for channel_index, channel_id in enumerate(CHANNEL_ORDER):
        if channel_id not in profile.channel_ids:
            warnings.append(f"profile has no channel {channel_id!r}; its frames were skipped")
            continue
        spec = profile.channel(channel_id)
        pitched = spec.pitched
        channel_notes, channel_controls = _channel_events(
            array[:, channel_index, :], channel_id, channel_index, pitched=pitched
        )
        notes.extend(channel_notes)
        controls.extend(channel_controls)

    notes.sort(key=lambda n: (n.tick, n.channel_id or "", n.pitch))
    controls.sort(key=lambda c: (c.tick, c.channel_id or "", c.type))

    voices = [
        SourceVoice(
            key=(index, index),
            track_index=index,
            midi_channel=index,
            track_name=channel_id,
            instrument_name=None,
            n_notes=sum(1 for n in notes if n.channel_id == channel_id),
        )
        for index, channel_id in enumerate(CHANNEL_ORDER)
    ]
    assignment = AssignmentResult(
        assignments={
            voice.key: ChannelAssignment(
                CHANNEL_ORDER[index],
                profile.channel(CHANNEL_ORDER[index]).role
                if CHANNEL_ORDER[index] in profile.channel_ids
                else None,
                BY_TRACK_NAME,
            )
            for index, voice in enumerate(voices)
        },
        warnings=[],
    )

    time = MusicalTime(
        TICKS_PER_BEAT,
        [TempoChange(0, TEMPO_US_PER_BEAT)],
        [TimeSignatureChange(0, 4, 4)],
    )
    if score.rate and abs(score.rate - FRAME_RATE_HZ) > 1e-6:
        warnings.append(
            f"frame rate is {score.rate} Hz, not the expected {FRAME_RATE_HZ} Hz; "
            "timing will be wrong"
        )

    return IntakeComposition(
        source_path=path,
        midi_format=-1,  # not from a MIDI file
        ticks_per_beat=TICKS_PER_BEAT,
        n_tracks=len(CHANNEL_ORDER),
        time=time,
        voices=voices,
        assignment=assignment,
        notes=notes,
        controls=controls,
        text_events=[],
        key_signatures=[],
        warnings=warnings,
    )


def _channel_events(
    frames: np.ndarray, channel_id: str, channel_index: int, *, pitched: bool
) -> tuple[list[Note], list[Control]]:
    """Notes and envelope controllers for one channel's frame column.

    A note runs for as long as its pitch stays the same and non-zero. Velocity
    and timbre changes *within* a note become controller events rather than
    being thrown away — that envelope is most of what makes NES instruments
    sound alive, and discarding it is the mistake proposal 5.2 warns about.
    """
    notes: list[Note] = []
    controls: list[Control] = []

    pitches = frames[:, 0].astype(int)
    velocities = frames[:, 1].astype(int)
    timbres = frames[:, 2].astype(int)

    start = None
    for index in range(len(pitches) + 1):
        pitch = int(pitches[index]) if index < len(pitches) else 0
        previous = int(pitches[index - 1]) if index > 0 else 0

        if pitch != previous:
            if previous > 0 and start is not None:
                notes.append(
                    Note(
                        channel_id=channel_id,
                        tick=start * FRAME_TICKS,
                        dur_ticks=(index - start) * FRAME_TICKS,
                        pitch=previous,
                        velocity=_note_velocity(velocities[start], pitched=pitched),
                        track_index=channel_index,
                        midi_channel=channel_index,
                    )
                )
            start = index if pitch > 0 else None

        if index >= len(pitches):
            break
        # Envelope data, recorded whenever it moves.
        if index == 0 or velocities[index] != velocities[index - 1]:
            controls.append(
                Control(
                    channel_id=channel_id,
                    tick=index * FRAME_TICKS,
                    type="control_change",
                    data={"control": CC_VOLUME, "value": _scale_volume(velocities[index])},
                    track_index=channel_index,
                    midi_channel=channel_index,
                )
            )
        if index == 0 or timbres[index] != timbres[index - 1]:
            controls.append(
                Control(
                    channel_id=channel_id,
                    tick=index * FRAME_TICKS,
                    type="control_change",
                    data={"control": CC_TIMBRE, "value": int(timbres[index])},
                    track_index=channel_index,
                    midi_channel=channel_index,
                )
            )
    return notes, controls


def _note_velocity(register_volume: int, *, pitched: bool) -> int:
    """MIDI velocity for a 4-bit register value.

    The triangle channel stores zero throughout because it has no volume
    register, so it gets a fixed audible velocity instead of being silenced.
    """
    value = int(register_volume)
    if value <= 0:
        return TRIANGLE_VELOCITY if pitched else 1
    return _scale_volume(value)


def _scale_volume(register_volume: int) -> int:
    value = int(register_volume)
    if value <= 0:
        return 0
    return max(1, min(127, round(value * 127 / MAX_REGISTER_VOLUME)))


def split_from_path(path: Path) -> str | None:
    """NES-MDB's own composer-disjoint split, from the directory name."""
    for part in Path(path).parts:
        lowered = part.lower()
        if lowered in ("train", "training"):
            return "train"
        if lowered in ("valid", "validation", "val"):
            return "valid"
        if lowered == "test":
            return "test"
    return None
