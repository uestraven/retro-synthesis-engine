"""WAV export.

Uses the standard library's `wave` module, so rendering and export add no
dependency beyond NumPy.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np

#: 16-bit PCM: the full-scale positive value. -32768 is reachable, +32768 is not.
INT16_PEAK = 32767


def to_int16(samples: Any) -> np.ndarray:
    """Convert float samples in [-1, 1] to clipped 16-bit PCM."""
    array = np.asarray(samples, dtype=np.float64)
    return np.clip(array * INT16_PEAK, -32768, INT16_PEAK).astype("<i2")


def write_wav(path: Path, samples: Any, sample_rate: int) -> Path:
    """Write 16-bit PCM audio, mono or stereo.

    Mono is a 1-D array, stereo an `(n, 2)` array of `[left, right]` frames.
    The channel count comes from the DATA, so a caller cannot write a stereo
    buffer under a mono header: that mismatch plays at half speed and is the
    sort of defect that survives a listening test as "sounds a bit odd".

    An `(n, 1)` array is accepted and written as mono, so a caller that always
    produces 2-D buffers does not have to special-case a single channel. What
    is NOT done is duplicating mono into two channels -- that is a stereo file
    with no stereo in it, and the SNES renderer's whole point is per-voice
    left/right levels.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim == 1:
        channels = 1
    elif array.ndim == 2:
        if array.shape[1] not in (1, 2):
            raise ValueError(
                f"expected 1 or 2 channels, got {array.shape[1]}; a stereo "
                "buffer is (frames, 2)"
            )
        channels = int(array.shape[1])
        if channels == 1:
            array = array[:, 0]
    else:
        raise ValueError(f"samples must be 1-D or 2-D, got {array.ndim}-D")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = to_int16(array)
    if channels == 2:
        # WAV stores stereo interleaved: L R L R ...
        pcm = pcm.reshape(-1)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(channels)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(pcm.tobytes())
    return path


def write_stems(directory: Path, stems: dict[str, Any], sample_rate: int) -> dict[str, Path]:
    """Write one WAV per channel, for inspecting channels in isolation."""
    directory = Path(directory)
    return {
        channel_id: write_wav(directory / f"{channel_id}.wav", samples, sample_rate)
        for channel_id, samples in sorted(stems.items())
    }


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit WAV back as floats in [-1, 1]. Used by the tests.

    Mono returns 1-D; stereo returns `(frames, 2)`, de-interleaved.
    """
    with wave.open(str(path), "rb") as fh:
        if fh.getsampwidth() != 2:
            raise ValueError(f"{path} is not 16-bit PCM")
        frames = fh.readframes(fh.getnframes())
        rate = fh.getframerate()
        channels = fh.getnchannels()
    array = np.frombuffer(frames, dtype="<i2").astype(np.float32) / INT16_PEAK
    if channels > 1:
        # De-interleave rather than discarding the other channels. Keeping
        # only the left channel here would hide every stereo defect from the
        # tests that read a file back.
        array = array.reshape(-1, channels)
    return array, rate
