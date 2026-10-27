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
    decode_looping,
    encode,
    predict,
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


# --- BRR loop playback ----------------------------------------------------
#
# The filter-0-at-the-loop-point rule was documented here as a hardware
# requirement, on the reasoning that the predictor has no history across a
# wrap. That was wrong: the S-DSP carries its two decoded-sample registers
# straight through the loop. What filter 0 actually buys is REPEATABILITY --
# the loop block decodes from its own nibbles alone, so pass 2 and pass 5 are
# identical. These tests measure that property rather than asserting the
# incorrect justification.


def _looping_assets() -> list[str]:
    return ["brass", "strings", "choir", "bass"]


def test_every_repetition_decodes_identically() -> None:
    """The point of the filter-0 policy, measured over five passes.

    `decode_looping` deliberately carries the predictor registers across the
    wrap, as the hardware does. If the loop block used a predictive filter it
    would decode differently on pass 1 (preceded by the attack) than on pass 2
    (preceded by the end of the loop), and filters 2 and 3 have poles close
    enough to the unit circle that the difference need not decay.
    """
    bank = build_bank(brr=False)
    for name in _looping_assets():
        asset = bank[name]
        encoded = encode(asset.data, loop_start=asset.loop_start)
        assert encoded.loop_block is not None
        start = encoded.loop_block * SAMPLES_PER_BLOCK
        period = encoded.frames - start
        played = decode_looping(encoded, frames=start + period * 5)
        passes = [played[start + i * period : start + (i + 1) * period] for i in range(5)]
        for index, repetition in enumerate(passes[1:], start=1):
            drift = float(np.abs(repetition - passes[0]).max())
            assert drift == 0.0, f"{name} pass {index + 1} drifted by {drift}"


def test_the_wrap_is_no_louder_a_step_than_the_loop_itself() -> None:
    """A discontinuity test that is relative, because an absolute threshold
    would pass a bass and fail a hat for reasons that have nothing to do with
    the loop. The step across the boundary is compared to the distribution of
    steps inside the loop; a click shows up as a wrap step far outside it.
    """
    bank = build_bank(brr=False)
    for name in _looping_assets():
        asset = bank[name]
        encoded = encode(asset.data, loop_start=asset.loop_start)
        assert encoded.loop_block is not None
        start = encoded.loop_block * SAMPLES_PER_BLOCK
        period = encoded.frames - start
        played = decode_looping(encoded, frames=start + period * 3)
        inside = np.abs(np.diff(played[start : start + period]))
        for repeat in (1, 2):
            boundary = start + period * repeat
            wrap = abs(float(played[boundary] - played[boundary - 1]))
            assert wrap <= float(inside.max()) * 1.5, (
                f"{name}: wrap step {wrap:.5f} against a largest in-loop step "
                f"of {inside.max():.5f}"
            )


def test_rounding_the_loop_point_down_costs_at_most_a_block() -> None:
    """Rounding down can only ever replay up to fifteen extra samples, and on
    this bank it is at most two -- so the periodicity the samples were built
    for survives the block quantisation.
    """
    bank = build_bank(brr=False)
    for name in _looping_assets():
        asset = bank[name]
        encoded = encode(asset.data, loop_start=asset.loop_start)
        assert encoded.loop_block is not None
        aligned = encoded.loop_block * SAMPLES_PER_BLOCK
        assert aligned <= asset.loop_start
        assert asset.loop_start - aligned < SAMPLES_PER_BLOCK
        raw = np.asarray(asset.data, dtype=np.float64)
        intended = abs(float(raw[asset.loop_start] - raw[asset.loop_end - 1]))
        actual = abs(float(raw[aligned] - raw[-1]))
        # Rounding must not make the source discontinuity meaningfully worse.
        assert actual <= intended + 0.002, name


def test_the_loop_block_is_not_a_worse_approximation_than_the_rest() -> None:
    """The cost of the policy, measured rather than assumed.

    Forcing filter 0 means the loop block spends its 4-bit residual on an
    absolute value instead of a prediction error, which could in principle
    make it the noisiest block in the sample. It does not: its peak error is
    within the range the predictive blocks already produce.
    """
    bank = build_bank(brr=False)
    for name in _looping_assets():
        asset = bank[name]
        encoded = encode(asset.data, loop_start=asset.loop_start)
        assert encoded.loop_block is not None
        decoded = decode(encoded, frames=len(asset.data))
        error = np.abs(decoded - np.asarray(asset.data, dtype=np.float64))
        block = np.arange(
            encoded.loop_block * SAMPLES_PER_BLOCK,
            (encoded.loop_block + 1) * SAMPLES_PER_BLOCK,
        )
        elsewhere = np.delete(np.arange(error.size), block)
        assert float(error[block].max()) <= float(error[elsewhere].max()) * 1.1, name


def test_a_one_shot_stops_instead_of_looping() -> None:
    """Percussion has no loop, so END without LOOP must end the voice. A
    decoder that wrapped anyway would turn a kick into a drone -- which is
    the shape of a bug this project has already shipped once, when a hat was
    stretched 88x by being treated as pitched.
    """
    bank = build_bank(brr=False)
    for name in ("kick", "snare", "hat"):
        asset = bank[name]
        assert asset.loop_end <= asset.loop_start, f"{name} should be a one-shot"
        encoded = encode(asset.data)
        assert encoded.loop_block is None
        last = encoded.data[(encoded.blocks - 1) * BYTES_PER_BLOCK]
        assert last & 0x01, f"{name} has no END flag"
        assert not last & 0x02, f"{name} should not set LOOP"
        played = decode_looping(encoded, frames=encoded.frames * 3)
        tail = played[encoded.frames :]
        assert float(np.abs(tail).max()) == 0.0, f"{name} kept playing past END"


def test_the_integer_shift_forms_are_the_documented_fractions() -> None:
    """The coefficients, checked algebraically against the shift forms.

    `predict` runs the shifts; `FILTERS` records the fractions. They must be
    the same algebra, or one of them is wrong about what the S-DSP computes:
      filter 1   p1 - p1/16                  = 15/16 p1
      filter 2   2p1 - 3p1/32 - p2 + p2/16   = 61/32 p1 - 15/16 p2
      filter 3   2p1 - 13p1/64 - p2 + 3p2/16 = 115/64 p1 - 13/16 p2
    """
    assert FILTERS[1] == (1 - 1 / 16, 0.0)
    assert FILTERS[2] == (2 - 3 / 32, -(1 - 1 / 16))
    assert FILTERS[3] == (2 - 13 / 64, -(1 - 3 / 16))
    assert predict(0, 1000, -2000) == 0
    # On values divisible by the shift denominators the two forms must agree
    # exactly, since there is nothing left to truncate.
    for p1, p2 in ((64, 64), (-1024, 512), (4096, -4096)):
        for index in (1, 2, 3):
            c1, c2 = FILTERS[index]
            assert predict(index, p1, p2) == int(c1 * p1 + c2 * p2), (index, p1, p2)


def _decode_with_float_fractions(sample: object) -> np.ndarray:
    """The decoder RSE used to have: the fractions, multiplied in float.

    Kept as a measurement instrument rather than a reference. It is the
    approximation, not the authority -- the hardware truncates each shift,
    float multiplication rounds to nearest, and the result feeds back.
    """
    data = sample.data  # type: ignore[attr-defined]
    frames = sample.frames  # type: ignore[attr-defined]
    out = np.zeros(frames, dtype=np.float64)
    p1 = p2 = 0
    position = 0
    for index in range(len(data) // BYTES_PER_BLOCK):
        offset = index * BYTES_PER_BLOCK
        header = data[offset]
        shift = (header >> 4) & 0xF
        c1, c2 = FILTERS[(header >> 2) & 0x3]
        for byte in data[offset + 1 : offset + BYTES_PER_BLOCK]:
            for nibble in ((byte >> 4) & 0xF, byte & 0xF):
                if position >= frames:
                    return out / 32767.0
                value = nibble - 16 if nibble >= 8 else nibble
                decoded = int(
                    max(-32768, min(32767, round(c1 * p1 + c2 * p2 + (value << shift))))
                )
                out[position] = decoded
                p2, p1 = p1, decoded
                position += 1
    return out / 32767.0


def test_float_fractions_are_not_a_substitute_for_the_integer_arithmetic() -> None:
    """Why `predict` exists, measured.

    A half-unit rounding difference per sample would be inaudible if it stayed
    a half-unit. It does not: the prediction feeds back, and filters 2 and 3
    have poles near the unit circle, so the float decoder drifts far past its
    own rounding error. This test records that the gap is real and large
    enough to matter -- if a future change replaced `predict` with the float
    multiply again, the tests above would still pass and this one would be the
    only thing left saying it was wrong.
    """
    bank = build_bank(brr=False)
    worst_overall = 0.0
    for name, asset in bank.items():
        loop = asset.loop_start if asset.loop_end > asset.loop_start else None
        encoded = encode(asset.data, loop_start=loop)
        gap = float(np.abs(decode(encoded) - _decode_with_float_fractions(encoded)).max())
        worst_overall = max(worst_overall, gap)
        # A single sample's rounding difference is at most 1/32767 of full
        # scale. Anything beyond that is accumulation through the feedback.
        assert gap >= 0.0, name
    assert worst_overall > 10 / 32767, (
        "the float decoder no longer drifts, which means either predict() or "
        f"the float reference changed: worst gap {worst_overall}"
    )


def test_filters_used_across_the_bank_are_not_all_filter_zero() -> None:
    """If the encoder only ever chose filter 0, the predictor -- and so most
    of the compression story -- would be untested by everything above.
    """
    bank = build_bank(brr=False)
    seen: set[int] = set()
    for asset in bank.values():
        loop = asset.loop_start if asset.loop_end > asset.loop_start else None
        encoded = encode(asset.data, loop_start=loop)
        for index in range(encoded.blocks):
            header = encoded.data[index * BYTES_PER_BLOCK]
            seen.add((header >> 2) & 0x3)
    assert len(seen) > 1, f"only filter(s) {sorted(seen)} were ever chosen"


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


def test_the_bank_is_in_tune_after_block_alignment() -> None:
    """The regression the alignment fix caused, and the check that caught it.

    Making the loop length a whole number of BRR blocks removed a dropout,
    but it also moved the loop length, and `loop_partials` derives the
    fundamental FROM that length -- so rounding 1957 samples to 1952 put
    every sustained instrument 4.4 cents sharp. `align_loop` searches block
    counts instead of rounding to the nearest, which brings it under a cent.

    Asserted as absolute pitch rather than as agreement between instruments,
    because a uniform offset would be inaudible and still wrong: the melody
    has to line up with the NES rendering of the same composition.
    """
    from rse.snes.samples import BANK_RATE, midi_to_hz

    bank = build_bank(brr=False)
    for name in ("brass", "strings", "choir", "bass"):
        asset = bank[name]
        region = asset.data[asset.loop_start :]
        wanted = midi_to_hz(asset.instrument.root_note)
        # The loop plays `cycles` whole cycles in `len(region)` samples, so
        # this IS the pitch, not an approximation of it.
        cycles = round(wanted * len(region) / BANK_RATE)
        actual = cycles * BANK_RATE / len(region)
        cents = abs(1200 * np.log2(actual / wanted))
        assert cents < 1.0, f"{name} is {cents:.2f} cents out"


def test_aligning_the_loop_leaves_nothing_for_the_encoder_to_round() -> None:
    """The two conditions together, since either alone is insufficient.

    Length a multiple of 16 means no zero padding lands inside the loop;
    loop start a multiple of 16 means the encoder's round-down is a no-op.
    The bank had the first and not the second, and the result was six zeros
    and a five-sample skip per repetition.
    """
    bank = build_bank(brr=False)
    for name, asset in bank.items():
        looping = asset.loop_end > asset.loop_start
        encoded = encode(
            asset.data, loop_start=asset.loop_start if looping else None
        )
        assert encoded.frames == len(asset.data), f"{name} was padded"
        assert len(asset.data) % SAMPLES_PER_BLOCK == 0, name
        if looping:
            assert asset.loop_start % SAMPLES_PER_BLOCK == 0, name
            assert encoded.loop_block == asset.loop_start // SAMPLES_PER_BLOCK


def test_catmull_rom_overshoot_is_bounded_and_measured() -> None:
    """Catmull-Rom is not interpolation-with-no-downside.

    Its kernel has negative lobes, so it can exceed the range of the samples
    it interpolates between -- the price of reproducing them exactly at
    integer positions, which the earlier Gaussian-shaped low-pass avoided by
    not being an interpolator at all. The S-DSP's own 4-tap Gaussian
    deliberately attenuates instead, so this is a difference in CHARACTER as
    well as accuracy, and it belongs in the preview label rather than being
    presented as closer-to-hardware.

    Measured here rather than bounded by theory because the magnitude depends
    entirely on the material: overshoot is small on the pitched instruments
    and largest on the snare, whose spectrum is mostly noise.
    """
    from rse.snes.sampler import _resample

    bank = build_bank(brr=True)
    worst_name, worst = "", 0.0
    for name, asset in bank.items():
        source = float(np.abs(np.asarray(asset.data)).max())
        for semitones in (-12, -5, 0, 3, 7, 12, 19):
            pitch = asset.instrument.root_note + semitones
            peak = float(np.abs(_resample(asset, pitch, 40000)).max())
            if peak / source - 1.0 > worst:
                worst_name, worst = name, peak / source - 1.0

    # Real, so it must not be asserted away to zero...
    assert worst > 0.0, "no overshoot at all would mean this is not Catmull-Rom"
    # ...but bounded well inside the headroom the mix leaves.
    assert worst < 0.10, f"{worst_name} overshot by {worst * 100:.2f}%"
    assert worst_name == "snare", (
        f"expected the noisiest sample to overshoot most, got {worst_name}"
    )


def test_overshoot_cannot_clip_the_output() -> None:
    """Why the overshoot above is a curiosity rather than a defect.

    Per-voice overshoot would matter if voices were written straight to the
    file. They are not: the mix is scaled by one scalar after summing, so the
    output peak is set by the normaliser and not by the interpolator. This
    test exists because the previous clipping report counted PRE-normalisation
    overshoot and was therefore measuring something the listener never hears.
    """
    from rse.song.snes_generate import generate_snes

    song = generate_snes(style="boss", seed=803, target_seconds=20.0)
    peak = float(np.abs(song.samples).max())
    assert peak <= 1.0, f"output peak {peak} exceeds full scale"
    assert song.audio["clipped_frames"] == 0
    # The normaliser's target, so a change to it shows up here rather than as
    # a quiet or clipped download.
    assert 0.85 <= peak <= 0.9, peak


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
    # Asserting the substance, not the sentence: the caveat was reworded when
    # the evidence classes went in, and a test pinned to its old phrasing
    # failed without anything being wrong.
    assert report["hardware_budgeted"] is False
    assert "no emulator has run any of this" in report["caveat"].lower()


def test_every_line_of_the_budget_declares_its_evidence_class() -> None:
    """The review question that prompted this: is "measured" load-bearing?

    It was being applied to the whole total when it is only true of three of
    the six lines. A caller reading `total_bytes` has no way to know which,
    so every line now carries measured / modelled / assumed and the breakdown
    and the evidence map must agree on the key set.
    """
    budget = budget_for({"x": encode(np.zeros(64))}, sequence_bytes=120)
    report = budget.to_dict()
    assert set(report["evidence"]) == set(report["breakdown"])
    assert report["evidence"]["samples"] == "measured"
    assert report["evidence"]["echo"] == "measured"
    assert report["evidence"]["directory"] == "measured"
    # RSE emits no sequence bytes at all, so this one is arithmetic about a
    # driver format it does not implement.
    assert report["evidence"]["sequence"] == "modelled"
    assert report["evidence"]["driver"] == "assumed"
    assert report["evidence"]["runtime"] == "assumed"


def test_measured_bytes_are_a_strict_subset_of_the_total() -> None:
    bank = build_bank(brr=False)
    encoded = {name: encode(asset.data) for name, asset in bank.items()}
    budget = budget_for(encoded, sequence_bytes=3000, echo_delay_ms=64)
    assert budget.measured_bytes < budget.total
    assert (
        budget.measured_bytes
        == budget.sample_bytes + budget.directory_bytes + budget.echo_bytes
    )
    assert budget.to_dict()["measured_bytes"] == budget.measured_bytes


def test_hardware_budgeting_is_not_claimed_while_anything_is_unmodelled() -> None:
    """The flag that must not drift into True by accident.

    `fits` and `hardware_budgeted` answer different questions, and the easy
    mistake is to let the first imply the second: a track can fit comfortably
    inside the represented allocations while the representation is missing an
    instrument table, pattern data, page alignment, the SPC700's stack and a
    DSP register shadow -- all of which a real image needs.
    """
    from rse.snes.aram import UNMODELLED

    assert UNMODELLED, "if this is empty the claim below needs re-examining"
    budget = budget_for({"x": encode(np.zeros(64))})
    assert budget.fits
    assert not budget.hardware_budgeted
    report = budget.to_dict()
    assert report["hardware_budgeted"] is False
    assert len(report["unmodelled"]) == len(UNMODELLED)
    for entry in report["unmodelled"]:
        assert entry["name"] and entry["why_it_matters"]
    # The text report has to say it too, since that is what a human reads.
    assert "UNMODELLED" in budget.report()
    assert "hardware_budgeted: False" in budget.report()


def test_the_caveat_tells_a_caller_not_to_add_the_echo_twice() -> None:
    """A reviewer read the 47% figure as excluding echo and worked out what
    enabling a 240 ms buffer would cost on top: 61,705 bytes, leaving almost
    nothing. The echo was already inside the total, so the real figure for
    cavern is lower -- but the report gave no way to tell, which is the
    defect. It says so now.
    """
    encoded = {"x": encode(np.zeros(64))}
    dry = budget_for(encoded, echo_delay_ms=0)
    cavern = budget_for(encoded, echo_delay_ms=240)
    assert cavern.total == dry.total + echo_bytes_for(240)
    assert cavern.echo_bytes == 30720
    assert "do not add it again" in cavern.to_dict()["caveat"]


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
