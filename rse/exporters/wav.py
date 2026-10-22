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
    """Write mono 16-bit PCM audio."""
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = to_int16(samples)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
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
    """Read a mono 16-bit WAV back as floats in [-1, 1]. Used by the tests."""
    with wave.open(str(path), "rb") as fh:
        if fh.getsampwidth() != 2:
            raise ValueError(f"{path} is not 16-bit PCM")
        frames = fh.readframes(fh.getnframes())
        rate = fh.getframerate()
        channels = fh.getnchannels()
    array = np.frombuffer(frames, dtype="<i2").astype(np.float32) / INT16_PEAK
    if channels > 1:
        array = array.reshape(-1, channels)[:, 0]
    return array, rate
