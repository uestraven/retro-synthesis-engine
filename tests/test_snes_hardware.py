"""BRR, interpolation and the ARAM budget: measured, not estimated.

What changed from the preview: samples are really BRR-encoded, the byte counts
are measurements rather than geometry, and interpolation is a genuine 4-point
kernel. What is still NOT claimed, and is asserted to be labelled as such: the
S-DSP's own Gaussian table, an SPC file, and any hardware verification.
"""

from __future__ import annotations

import numpy as np
import pytest

from rse.snes.aram import (
    ARAM_BYTES,
    DIRECTORY_ENTRY_BYTES,
    ECHO_BYTES_PER_STEP,
    budget_for,
    echo_bytes_for,
    largest_echo_that_fits,
)
from rse.snes.brr import (
    BYTES_PER_BLOCK,
    FILTERS,
    MAX_RANGE,
    SAMPLES_PER_BLOCK,
    bytes_for,
    decode,
    encode,
    round_trip_error,
)
from rse.snes.samples import build_bank

# --- BRR format -----------------------------------------------------------


def test_nine_bytes_hold_sixteen_samples() -> None:
    """The format's whole geometry."""
    assert SAMPLES_PER_BLOCK == 16
    assert BYTES_PER_BLOCK == 9
    assert bytes_for(16) == 9
    assert bytes_for(17) == 18
    assert bytes_for(1600) == 900


def test_compression_is_exactly_the_format_ratio() -> None:
    """32 bytes of 16-bit PCM become 9 bytes of BRR."""
    encoded = encode(np.zeros(1600))
    assert encoded.size_bytes == 900
    assert encoded.size_bytes / (1600 * 2) == pytest.approx(9 / 32, abs=1e-9)


def test_length_is_padded_to_whole_blocks() -> None:
    """The hardware reads blocks, not samples."""
    for frames in (1, 15, 16, 17, 31, 33):
        encoded = encode(np.zeros(frames))
        assert encoded.frames % SAMPLES_PER_BLOCK == 0
        assert encoded.frames >= frames
        assert encoded.size_bytes % BYTES_PER_BLOCK == 0


def test_the_four_filters_are_the_hardware_predictors() -> None:
    assert FILTERS[0] == (0.0, 0.0)
    assert FILTERS[1] == (15 / 16, 0.0)
    assert FILTERS[2] == (61 / 32, -15 / 16)
    assert FILTERS[3] == (115 / 64, -13 / 16)


def test_the_range_never_exceeds_twelve() -> None:
    """13-15 behave oddly on hardware and are not emitted."""
    data = np.sin(np.linspace(0, 60, 320))
    encoded = encode(data)
    for index in range(encoded.blocks):
        header = encoded.data[index * BYTES_PER_BLOCK]
        assert (header >> 4) & 0xF <= MAX_RANGE


def test_the_first_block_uses_filter_zero() -> None:
    """It has no predictor history."""
    encoded = encode(np.sin(np.linspace(0, 30, 160)))
    assert (encoded.data[0] >> 2) & 0x3 == 0


def test_the_loop_block_uses_filter_zero() -> None:
    """Filters 1-3 predict from the previous two decoded samples, so the block
    at the loop point would predict from whatever preceded the wrap. Leaving a
    predictive filter there clicks on every repeat."""
    data = np.sin(np.linspace(0, 60, 320))
    encoded = encode(data, loop_start=64)
    assert encoded.loop_block == 4
    header = encoded.data[encoded.loop_block * BYTES_PER_BLOCK]
    assert (header >> 2) & 0x3 == 0


def test_the_loop_point_rounds_down_to_a_block() -> None:
    """The hardware's loop address is a block address. Rounding up would skip
    audio."""
    encoded = encode(np.zeros(320), loop_start=70)
    assert encoded.loop_block == 4  # 70 // 16
    assert encoded.loop_offset_bytes == 36


def test_the_end_and_loop_flags_are_set() -> None:
    looping = encode(np.zeros(320), loop_start=64)
    final = looping.data[(looping.blocks - 1) * BYTES_PER_BLOCK]
    assert final & 0x01, "end flag"
    assert final & 0x02, "loop flag"
    one_shot = encode(np.zeros(320))
    final = one_shot.data[(one_shot.blocks - 1) * BYTES_PER_BLOCK]
    assert final & 0x01
    assert not final & 0x02


def test_a_one_shot_declares_no_loop() -> None:
    encoded = encode(np.zeros(64))
    assert encoded.loop_block is None
    assert encoded.loop_offset_bytes is None


def test_encoding_is_lossy_and_the_loss_is_reported() -> None:
    """A 4-bit residual over a 2-tap predictor cannot be exact, and how lossy
    depends entirely on the material."""
    smooth = np.sin(np.linspace(0, 20, 640)) * 0.8
    noise = np.random.default_rng(0).standard_normal(640) * 0.3
    assert round_trip_error(smooth) < round_trip_error(noise), (
        "noise must cost more than a smooth wave: it is what a linear "
        "predictor cannot predict"
    )


def test_the_real_bank_survives_a_round_trip() -> None:
    bank = build_bank(brr=False)
    for name, asset in bank.items():
        loop = asset.loop_start if asset.loop_end > asset.loop_start else None
        error = round_trip_error(asset.data, loop_start=loop)
        assert error < 0.02, f"{name}: {error * 100:.2f}% RMS"


def test_decoding_is_deterministic() -> None:
    data = np.sin(np.linspace(0, 40, 320)) * 0.6
    encoded = encode(data)
    assert np.array_equal(decode(encoded), decode(encoded))


def test_brr_quantisation_reaches_the_bank() -> None:
    """The point of BRR beyond byte counts: a preview playing clean PCM does
    not sound like a SNES."""
    clean = build_bank(brr=False)
    quantised = build_bank(brr=True)
    for name in clean:
        difference = float(
            np.sqrt(
                np.mean(
                    (
                        quantised[name].data.astype(np.float64)
                        - clean[name].data.astype(np.float64)
                    )
                    ** 2
                )
            )
        )
        assert difference > 0, f"{name} was not quantised"
        assert difference < 0.03, f"{name}: {difference * 100:.2f}% is too lossy"


def test_loops_still_wrap_seamlessly_after_encoding() -> None:
    """Encoding must not undo the periodicity fix."""
    bank = build_bank(brr=True)
    for name in ("brass", "strings", "choir", "bass"):
        asset = bank[name]
        loop = asset.data[asset.loop_start : asset.loop_end].astype(np.float64)
        doubled = np.concatenate([loop, loop])
        wrap = abs(doubled[loop.size] - doubled[loop.size - 1])
        largest = float(np.abs(np.diff(loop)).max())
        assert wrap <= largest * 1.05, name


# --- interpolation --------------------------------------------------------


def test_the_kernel_is_a_real_interpolator() -> None:
    """The first attempt was a Gaussian-shaped low-pass whose row at frac=0
    was [0.24, 0.48, 0.24, 0.03] -- it smeared an impulse across three taps
    instead of reproducing it, and measured WORSE than plain linear."""
    from rse.snes.sampler import INTERP, _interpolate

    assert np.allclose(INTERP[0], [0.0, 1.0, 0.0, 0.0], atol=1e-9)
    impulse = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    out = _interpolate(impulse, np.arange(6, dtype=np.float64))
    assert np.allclose(out, impulse, atol=1e-9)


def test_every_kernel_row_is_a_partition_of_unity() -> None:
    from rse.snes.sampler import INTERP

    assert np.allclose(INTERP.sum(axis=1), 1.0, atol=1e-9)


def test_four_point_interpolation_beats_linear() -> None:
    from rse.snes.sampler import OUTPUT_RATE, _resample
    from rse.snes.samples import midi_to_hz

    # The CLEAN bank: BRR quantisation adds noise of its own, which swamps the
    # interpolation difference and made this comparison measure the wrong
    # thing entirely.
    asset = build_bank(brr=False)["brass"]

    def inharmonic(interpolation: str, pitch: int) -> float:
        wave = _resample(
            asset, pitch, int(0.4 * OUTPUT_RATE), interpolation=interpolation
        )
        magnitude = np.abs(np.fft.rfft(wave * np.hanning(len(wave))))
        freqs = np.fft.rfftfreq(len(wave), 1 / OUTPUT_RATE)
        f0 = midi_to_hz(pitch)
        harmonic = np.zeros_like(magnitude, dtype=bool)
        index = 1
        while f0 * index < OUTPUT_RATE / 2:
            harmonic |= np.abs(freqs - f0 * index) < f0 * 0.25
            index += 1
        audible = freqs > 60
        return magnitude[audible & ~harmonic].sum() / magnitude[audible].sum()

    better = sum(
        inharmonic("catmull", pitch) <= inharmonic("linear", pitch) + 1e-4
        for pitch in (60, 72, 84, 88)
    )
    assert better >= 3, "the 4-point kernel should not be worse than linear"


def test_the_gaussian_table_is_not_claimed() -> None:
    """It is a Gaussian-shaped approximation of the character, not the
    hardware's 512-entry table, and the code says so."""
    from pathlib import Path

    source = Path("rse/snes/sampler.py").read_text()
    assert "Catmull-Rom" in source
    assert "which I do not have" in source
    assert "not the S-DSP's own 4-tap Gaussian table" in source
    assert "no emulator has run any of this" in source


# --- the ARAM budget ------------------------------------------------------


def test_the_budget_is_the_whole_sixty_four_kibibytes() -> None:
    assert ARAM_BYTES == 65536


def test_echo_is_charged_against_the_same_memory_as_samples() -> None:
    """The tradeoff the hardware imposes: a 240 ms echo is 30 KiB, nearly half
    the budget, paid for in instruments."""
    assert echo_bytes_for(16) == ECHO_BYTES_PER_STEP
    assert echo_bytes_for(240) == 15 * ECHO_BYTES_PER_STEP
    assert echo_bytes_for(240) / ARAM_BYTES > 0.45
    assert echo_bytes_for(0) == 0


def test_echo_rounds_up_to_the_dsp_step() -> None:
    assert echo_bytes_for(1) == ECHO_BYTES_PER_STEP
    assert echo_bytes_for(17) == 2 * ECHO_BYTES_PER_STEP


def test_the_budget_uses_measured_encoded_sizes() -> None:
    bank = build_bank(brr=False)
    encoded = {
        name: encode(
            asset.data,
            loop_start=asset.loop_start if asset.loop_end > asset.loop_start else None,
        )
        for name, asset in bank.items()
    }
    budget = budget_for(encoded, sequence_bytes=3000, echo_delay_ms=64)
    assert budget.sample_bytes == sum(e.size_bytes for e in encoded.values())
    assert budget.directory_bytes == len(encoded) * DIRECTORY_ENTRY_BYTES
    assert budget.fits
    assert budget.remaining == ARAM_BYTES - budget.total


def test_the_budget_says_which_figures_are_assumed() -> None:
    """RSE has no SPC700 driver, so the driver and runtime figures are
    assumptions and must be labelled."""
    budget = budget_for({}, sequence_bytes=0)
    report = budget.to_dict()
    assert "driver_bytes" in report["assumed"]
    assert "runtime_bytes" in report["assumed"]
    assert "no SPC700 driver" in report["caveat"]
    assert "no hardware compatibility" in report["caveat"].lower()


def test_an_oversized_budget_is_reported_not_hidden() -> None:
    encoded = {"huge": encode(np.zeros(200_000))}
    budget = budget_for(encoded, echo_delay_ms=240)
    assert not budget.fits
    assert budget.remaining < 0
    assert "OVER BUDGET" in budget.report()


def test_the_longest_echo_that_fits_is_computable() -> None:
    """The question in the other direction: given an instrument set, how much
    room is there for a room?"""
    bank = build_bank(brr=False)
    encoded = {
        name: encode(asset.data) for name, asset in bank.items()
    }
    budget = budget_for(encoded, sequence_bytes=3000)
    longest = largest_echo_that_fits(budget)
    assert longest > 0
    with_echo = budget_for(encoded, sequence_bytes=3000, echo_delay_ms=longest)
    assert with_echo.fits
    over = budget_for(encoded, sequence_bytes=3000, echo_delay_ms=longest + 16)
    assert not over.fits


def test_the_real_track_fits_in_aram() -> None:
    from rse.procedural.compose import compose
    from rse.snes.arrange import allocate, arrange

    result = compose(style_name="boss", seed=4242, target_seconds=60.0)
    song = arrange(result)
    allocation = allocate(song)
    bank = build_bank(brr=False)
    encoded = {
        name: encode(
            asset.data,
            loop_start=asset.loop_start if asset.loop_end > asset.loop_start else None,
        )
        for name, asset in bank.items()
    }
    budget = budget_for(
        encoded,
        sequence_bytes=len(allocation.assignments) * 4,
        echo_delay_ms=64,
    )
    assert budget.fits, budget.report()
    assert budget.total / ARAM_BYTES < 0.75
