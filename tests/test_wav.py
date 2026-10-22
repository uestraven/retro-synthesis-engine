from __future__ import annotations

import numpy as np
import pytest

from rse.exporters.wav import INT16_PEAK, read_wav, to_int16, write_stems, write_wav


def test_float_to_int16_scales_to_full_scale():
    pcm = to_int16(np.array([0.0, 1.0, -1.0], dtype=np.float32))
    assert pcm.tolist() == [0, INT16_PEAK, -INT16_PEAK]
    assert pcm.dtype.str == "<i2"


def test_out_of_range_samples_are_clipped_not_wrapped():
    pcm = to_int16(np.array([2.5, -2.5], dtype=np.float32))
    assert pcm.tolist() == [INT16_PEAK, -32768]


def test_wav_round_trips(tmp_path):
    original = np.sin(np.linspace(0, 40 * np.pi, 4410)).astype(np.float32) * 0.5
    path = write_wav(tmp_path / "tone.wav", original, 44_100)
    restored, rate = read_wav(path)
    assert rate == 44_100
    assert restored.size == original.size
    # 16-bit quantization is the only loss.
    assert np.abs(restored - original).max() < 2.0 / INT16_PEAK


def test_wav_header_describes_mono_16_bit(tmp_path):
    import wave

    write_wav(tmp_path / "a.wav", np.zeros(1000, dtype=np.float32), 22_050)
    with wave.open(str(tmp_path / "a.wav"), "rb") as fh:
        assert fh.getnchannels() == 1
        assert fh.getsampwidth() == 2
        assert fh.getframerate() == 22_050
        assert fh.getnframes() == 1000


def test_parent_directories_are_created(tmp_path):
    path = write_wav(tmp_path / "deep" / "nested" / "a.wav", np.zeros(10, dtype=np.float32), 8_000)
    assert path.is_file()


def test_invalid_sample_rate_is_refused(tmp_path):
    with pytest.raises(ValueError, match="sample_rate must be positive"):
        write_wav(tmp_path / "a.wav", np.zeros(10, dtype=np.float32), 0)


def test_stems_are_written_one_file_per_channel(tmp_path):
    stems = {
        "P1": np.zeros(100, dtype=np.float32),
        "NO": np.ones(100, dtype=np.float32) * 0.1,
    }
    paths = write_stems(tmp_path / "stems", stems, 44_100)
    assert set(paths) == {"P1", "NO"}
    assert all(path.is_file() for path in paths.values())
    assert paths["P1"].name == "P1.wav"


def test_reading_a_non_16_bit_file_is_refused(tmp_path):
    import wave

    path = tmp_path / "eight.wav"
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(1)
        fh.setframerate(8_000)
        fh.writeframes(b"\x00" * 10)
    with pytest.raises(ValueError, match="not 16-bit"):
        read_wav(path)
