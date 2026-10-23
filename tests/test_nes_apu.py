"""Tests for the NES APU renderer.

Audio cannot be checked by ear here, so correctness is established numerically:
the fundamental frequency is measured out of the rendered signal with an FFT,
duty ratios are measured off the raw waveform, and the mixer's nonlinearity is
checked against the documented formula.
"""

from __future__ import annotations

import numpy as np
import pytest

from rse.renderers.base import RenderNote, RenderSettings
from rse.renderers.nes_apu import (
    DUTY_SEQUENCES,
    MAX_VOLUME,
    NOISE_PERIODS_NTSC,
    NTSC_CPU_HZ,
    TRIANGLE_SEQUENCE,
    NesApuRenderer,
    _lfsr_sequence,
    _velocity_to_volume,
    midi_to_hz,
    pitch_error_cents,
    pulse_frequency,
    pulse_timer,
    triangle_frequency,
    triangle_timer,
)

A4 = 69  # MIDI note number for A440


@pytest.fixture
def renderer(nes_profile):
    return NesApuRenderer(nes_profile)


def note(channel, value, *, start=0.0, end=1.0, velocity=127, pitched=True):
    return RenderNote(
        channel_id=channel,
        start_sec=start,
        end_sec=end,
        value=value,
        velocity=velocity,
        pitched=pitched,
    )


def dominant_hz(samples, sample_rate):
    """The strongest non-DC frequency component of a signal."""
    windowed = samples * np.hanning(len(samples))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(windowed), 1.0 / sample_rate)
    spectrum[freqs < 20] = 0.0  # ignore DC and sub-audible content
    return float(freqs[int(np.argmax(spectrum))])


def measured_hz(samples, sample_rate, expected_hz):
    """Measure pitch by waveform period, searching near `expected_hz`.

    Autocorrelation rather than the strongest FFT bin, because a narrow pulse
    wave's second harmonic can exceed its fundamental — at 12.5% duty it does.
    The search is windowed around the expected period so the usual
    autocorrelation subharmonic error cannot pick a multiple of the true lag.
    """
    signal = np.asarray(samples, dtype=np.float64)
    signal = signal - signal.mean()
    size = 1 << (2 * len(signal) - 1).bit_length()
    spectrum = np.fft.rfft(signal, size)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum))[: len(signal)]

    target_lag = sample_rate / expected_hz
    low = max(1, int(target_lag * 0.75))
    high = min(len(correlation) - 2, int(target_lag * 1.34) + 2)
    lag = low + int(np.argmax(correlation[low:high]))
    # Parabolic interpolation: the true period is rarely a whole sample.
    before, peak, after = correlation[lag - 1], correlation[lag], correlation[lag + 1]
    curvature = before - 2 * peak + after
    if curvature:
        lag += 0.5 * (before - after) / curvature
    return sample_rate / lag


def inharmonic_fraction(samples, sample_rate, fundamental_hz):
    """Share of spectral energy that is not near a harmonic of the fundamental.

    Aliased partials land at frequencies unrelated to the fundamental, so this
    rises with aliasing. Useful for comparing settings, not as an absolute.
    """
    windowed = samples * np.hanning(len(samples))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(windowed), 1.0 / sample_rate)
    audible = freqs > 20
    harmonic = np.zeros_like(freqs, dtype=bool)
    for k in range(1, int(sample_rate / 2 / fundamental_hz) + 1):
        harmonic |= np.abs(freqs - k * fundamental_hz) < max(8.0, 0.01 * k * fundamental_hz)
    return float(spectrum[audible & ~harmonic].sum() / spectrum[audible].sum())


# -- hardware period maths ----------------------------------------------------


def test_midi_to_hz_anchors_on_a440():
    assert midi_to_hz(A4) == pytest.approx(440.0)
    assert midi_to_hz(A4 + 12) == pytest.approx(880.0)
    assert midi_to_hz(A4 - 12) == pytest.approx(220.0)


def test_pulse_timer_round_trips_through_the_hardware_divider():
    timer = pulse_timer(440.0)
    # f = CPU / (16 * (t + 1))
    assert timer == round(NTSC_CPU_HZ / (16 * 440.0) - 1)
    assert pulse_frequency(timer) == pytest.approx(440.0, abs=0.5)


def test_triangle_runs_an_octave_lower_for_the_same_timer():
    assert triangle_frequency(100) == pytest.approx(pulse_frequency(100) / 2)
    assert triangle_timer(220.0) == pytest.approx(pulse_timer(440.0), abs=1)


def test_timer_is_clamped_to_eleven_bits():
    assert pulse_timer(1.0) == 2047  # absurdly low note saturates the timer
    assert pulse_timer(0.0) == 0
    assert pulse_timer(-5.0) == 0


def test_the_hardware_cannot_play_perfect_equal_temperament():
    # Quantizing to an integer timer detunes notes, and more so as pitch rises,
    # because the same timer step spans more cents up there. This is audible
    # character, not a bug to correct.
    low = abs(pitch_error_cents(45))
    high = abs(pitch_error_cents(100))
    assert low < 1.0
    assert high > low
    # Still musically usable across the profile's range.
    assert max(abs(pitch_error_cents(n)) for n in range(33, 109)) < 50.0


# -- waveform tables ----------------------------------------------------------


def test_duty_sequences_have_the_documented_ratios():
    ratios = [row.mean() for row in DUTY_SEQUENCES]
    assert ratios == pytest.approx([0.125, 0.25, 0.5, 0.75])


def test_triangle_sequence_is_a_full_32_step_ramp():
    assert len(TRIANGLE_SEQUENCE) == 32
    assert TRIANGLE_SEQUENCE[0] == 15 and TRIANGLE_SEQUENCE[15] == 0
    assert TRIANGLE_SEQUENCE[16] == 0 and TRIANGLE_SEQUENCE[-1] == 15


def test_noise_period_table_is_monotonic_and_sixteen_long():
    assert len(NOISE_PERIODS_NTSC) == 16
    assert list(NOISE_PERIODS_NTSC) == sorted(NOISE_PERIODS_NTSC)


def test_lfsr_has_the_full_15_bit_period_and_is_balanced():
    sequence = _lfsr_sequence(mode=0)
    assert sequence.size == 32_767
    # A maximal-length LFSR spends almost exactly half its period gated off.
    assert sequence.mean() == pytest.approx(0.5, abs=0.001)
    assert set(np.unique(sequence)) == {0.0, 1.0}


def test_lfsr_mode_one_is_the_short_metallic_loop():
    assert _lfsr_sequence(mode=1).size == 93


def test_velocity_maps_onto_four_bit_volume():
    assert _velocity_to_volume(127) == MAX_VOLUME
    assert _velocity_to_volume(0) == 0.0
    # Any audible velocity must stay audible rather than rounding to silence.
    assert _velocity_to_volume(1) == 1.0


# -- rendered audio -----------------------------------------------------------


@pytest.mark.parametrize("channel", ["P1", "P2"])
def test_pulse_channels_render_the_requested_pitch(renderer, channel):
    result = renderer.render([note(channel, A4)])
    measured = measured_hz(result.samples, result.sample_rate, 440.0)
    assert measured == pytest.approx(440.0, abs=2.0)


def test_triangle_renders_the_requested_pitch(renderer):
    result = renderer.render([note("TR", A4)])
    assert measured_hz(result.samples, result.sample_rate, 440.0) == pytest.approx(440.0, abs=2.0)


@pytest.mark.parametrize("midi_note", [33, 45, 57, 69, 81, 93, 104, 108])
def test_pitch_tracks_the_whole_profile_range(renderer, midi_note):
    """The rendered pitch must match what the hardware timer can produce.

    Not equal temperament: at MIDI 93 the nearest timer gives 1747.8 Hz rather
    than 1760 Hz, and reproducing that detuning is the point.
    """
    wanted = midi_to_hz(midi_note)
    achievable = pulse_frequency(pulse_timer(wanted))
    result = renderer.render([note("P1", midi_note)])
    measured = measured_hz(result.samples, result.sample_rate, wanted)
    assert measured == pytest.approx(achievable, rel=0.01)


def test_noise_has_no_dominant_tone(renderer):
    result = renderer.render([note("NO", 8, pitched=False)])
    windowed = result.samples * np.hanning(len(result.samples))
    spectrum = np.abs(np.fft.rfft(windowed))
    spectrum[:20] = 0.0
    # Broadband: the strongest bin holds only a small share of total energy,
    # unlike a tone where it dominates.
    assert spectrum.max() / spectrum.sum() < 0.01


def test_lower_noise_values_are_brighter(renderer):
    """Low period index means a faster shift rate, so more high-frequency energy."""

    def centroid(value):
        result = renderer.render([note("NO", value, pitched=False)])
        spectrum = np.abs(np.fft.rfft(result.samples))
        freqs = np.fft.rfftfreq(len(result.samples), 1.0 / result.sample_rate)
        return float((spectrum * freqs).sum() / spectrum.sum())

    assert centroid(1) > centroid(16)


def test_duty_cycle_changes_the_timbre_not_the_pitch(nes_profile):
    pitches, spectra = [], []
    for duty in (0, 1, 2):
        renderer = NesApuRenderer(nes_profile, duty_by_channel={"P1": duty})
        result = renderer.render([note("P1", A4)])
        pitches.append(measured_hz(result.samples, result.sample_rate, 440.0))
        spectra.append(np.abs(np.fft.rfft(result.samples)))
    assert all(p == pytest.approx(440.0, abs=2.0) for p in pitches)
    # 12.5% and 50% duty differ in harmonic content: the 50% square has no
    # even harmonics, so the spectra must not match.
    assert not np.allclose(spectra[0][:2000], spectra[2][:2000], rtol=0.5)


def test_silence_when_there_are_no_notes(renderer):
    result = renderer.render([])
    assert result.samples.size > 0
    assert not np.any(result.samples)


def test_note_timing_is_respected(renderer):
    settings = RenderSettings(peak=None)
    result = renderer.render([note("P1", A4, start=1.0, end=1.5)], settings)
    rate = result.sample_rate
    before = result.samples[: int(0.9 * rate)]
    during = result.samples[int(1.1 * rate) : int(1.4 * rate)]
    after = result.samples[int(1.6 * rate) :]
    assert np.abs(before).max() < 1e-3
    assert np.abs(during).max() > 1e-2
    assert np.abs(after).max() < 1e-3


def test_duration_covers_the_last_note_plus_a_tail(renderer):
    settings = RenderSettings(tail_sec=0.5)
    result = renderer.render([note("P1", A4, start=0.0, end=2.0)], settings)
    assert result.duration_sec == pytest.approx(2.5, abs=0.01)


def test_quieter_velocity_renders_quieter(renderer):
    settings = RenderSettings(peak=None)  # normalization would hide this
    loud = renderer.render([note("P1", A4, velocity=127)], settings)
    soft = renderer.render([note("P1", A4, velocity=32)], settings)
    assert np.abs(soft.samples).max() < np.abs(loud.samples).max()


def test_the_mixer_is_nonlinear(renderer):
    settings = RenderSettings(peak=None)
    one = renderer.render([note("P1", A4)], settings)
    two = renderer.render([note("P1", A4), note("P2", A4 + 7)], settings)
    peak_one = float(np.abs(one.samples).max())
    peak_two = float(np.abs(two.samples).max())
    # Two pulses are louder than one, but by distinctly less than double.
    assert peak_one < peak_two < 2 * peak_one


def test_stems_are_produced_per_sounding_channel(renderer):
    result = renderer.render([note("P1", A4), note("TR", A4 - 24), note("NO", 8, pitched=False)])
    assert set(result.stems) == {"P1", "TR", "NO"}
    assert all(stem.size == result.samples.size for stem in result.stems.values())
    assert "P2" not in result.stems  # silent channels are not emitted


def test_a_channel_plays_one_note_at_a_time(renderer):
    """Overlapping notes on one channel replace, never sum — as on hardware."""
    settings = RenderSettings(peak=None)
    single = renderer.render([note("P1", A4, start=0.0, end=1.0)], settings)
    overlapped = renderer.render(
        [note("P1", A4, start=0.0, end=1.0), note("P1", A4 + 5, start=0.5, end=1.0)],
        settings,
    )
    assert np.abs(overlapped.samples).max() == pytest.approx(np.abs(single.samples).max(), rel=0.2)


def test_rendering_is_deterministic(renderer):
    notes = [note("P1", A4), note("NO", 4, pitched=False)]
    first = renderer.render(notes)
    second = renderer.render(notes)
    assert np.array_equal(first.samples, second.samples)


def test_out_of_range_values_are_skipped_with_a_warning(renderer):
    result = renderer.render([note("NO", 99, pitched=False)])
    assert not np.any(result.samples)
    assert any("outside the profile range" in w for w in result.warnings)


def test_subsonic_pitch_is_silenced_like_the_hardware(renderer):
    # Timer saturation means very low pitches cannot be produced; the console
    # silences the channel rather than playing something wrong.
    result = renderer.render([note("P1", 33)])
    if result.warnings:
        assert any("below the hardware minimum" in w for w in result.warnings)


def test_unknown_channel_is_reported_not_crashed(renderer):
    result = renderer.render([note("FM1", A4)])
    assert any("no channel" in w for w in result.warnings)


def test_peak_normalization_hits_its_target(renderer):
    result = renderer.render([note("P1", A4)], RenderSettings(peak=0.5))
    assert float(np.abs(result.samples).max()) == pytest.approx(0.5, abs=1e-4)


def test_output_stays_within_full_scale(renderer):
    notes = [
        note("P1", A4),
        note("P2", A4 + 4),
        note("TR", A4 - 24),
        note("NO", 6, pitched=False),
    ]
    result = renderer.render(notes)
    assert float(np.abs(result.samples).max()) <= 1.0


def test_oversampling_suppresses_aliasing(renderer):
    """A high note is where undersampling folds harmonics back audibly."""
    high = note("P1", 104)
    fundamental = pulse_frequency(pulse_timer(midi_to_hz(104)))
    ratios = [
        inharmonic_fraction(
            renderer.render([high], RenderSettings(oversample=factor, peak=None)).samples,
            44_100,
            fundamental,
        )
        for factor in (1, 2, 4, 8)
    ]
    # Monotonically cleaner as oversampling rises.
    assert ratios == sorted(ratios, reverse=True)
    assert ratios[-1] < ratios[0] * 0.8


def test_filters_remove_dc_offset(renderer):
    filtered = renderer.render([note("P1", A4)], RenderSettings(peak=None))
    unfiltered = renderer.render(
        [note("P1", A4)], RenderSettings(peak=None, apply_output_filters=False)
    )
    # The raw APU mixer output is unipolar, so it carries a large DC offset.
    assert abs(float(unfiltered.samples.mean())) > 0.01
    assert abs(float(filtered.samples.mean())) < 0.005


def test_renderer_refuses_a_profile_it_does_not_implement(nes_profile):
    from dataclasses import replace

    with pytest.raises(ValueError, match="renders the 'nes' profile"):
        NesApuRenderer(replace(nes_profile, profile_id="genesis"))


def test_renderer_refuses_an_unknown_region(nes_profile):
    with pytest.raises(ValueError, match="region must be NTSC or PAL"):
        NesApuRenderer(nes_profile, region="SECAM")


def test_pal_runs_at_a_different_clock(nes_profile):
    pal = NesApuRenderer(nes_profile, region="PAL")
    assert pal.cpu_hz != NTSC_CPU_HZ
    result = pal.render([note("P1", A4)])
    # Same requested pitch; PAL's slower clock gives a slightly different timer.
    assert dominant_hz(result.samples, result.sample_rate) == pytest.approx(440.0, abs=3.0)


def test_render_settings_validate_their_inputs():
    with pytest.raises(ValueError, match="sample_rate"):
        RenderSettings(sample_rate=0)
    with pytest.raises(ValueError, match="oversample"):
        RenderSettings(oversample=0)
    with pytest.raises(ValueError, match="peak"):
        RenderSettings(peak=1.5)


def test_result_metadata_excludes_the_audio(renderer):
    payload = renderer.render([note("P1", A4)]).to_dict()
    assert payload["renderer_id"].startswith("nes_apu/")
    assert payload["channels_rendered"] == ["P1"]
    assert payload["settings"]["sample_rate"] == 44_100
    assert "samples" not in payload


# -- volume envelopes ---------------------------------------------------------


def test_an_envelope_shapes_the_note(renderer):
    """Without this, a decaying drum hit renders as flat static."""
    from rse.renderers.base import EnvelopePoint

    # The control must switch off the default percussive decay, or it is not
    # a flat reference.
    settings = RenderSettings(peak=None)
    flat = renderer.render(
        [note("NO", 8, end=0.5, pitched=False)],
        RenderSettings(peak=None, default_noise_decay=False),
    )
    shaped = renderer.render(
        [note("NO", 8, end=0.5, pitched=False)],
        settings,
        envelopes={"NO": [EnvelopePoint(0.0, 1.0), EnvelopePoint(0.1, 0.25)]},
    )
    rate = flat.sample_rate
    early = slice(int(0.01 * rate), int(0.08 * rate))
    late = slice(int(0.15 * rate), int(0.4 * rate))

    # Flat holds its level; shaped falls after the breakpoint.
    assert np.abs(flat.samples[early]).max() == pytest.approx(
        np.abs(flat.samples[late]).max(), rel=0.2
    )
    assert np.abs(shaped.samples[late]).max() < 0.5 * np.abs(shaped.samples[early]).max()


def test_the_triangle_ignores_envelopes(renderer):
    """It has no volume register; exprsco stores zero velocity for it, so
    applying an envelope would silence the bass line entirely."""
    from rse.renderers.base import EnvelopePoint

    settings = RenderSettings(peak=None)
    silencing = {"TR": [EnvelopePoint(0.0, 0.0)]}
    plain = renderer.render([note("TR", A4 - 24, end=0.5)], settings)
    with_envelope = renderer.render([note("TR", A4 - 24, end=0.5)], settings, silencing)
    assert np.abs(with_envelope.samples).max() > 0
    assert np.array_equal(plain.samples, with_envelope.samples)


def test_no_envelope_leaves_rendering_unchanged(renderer):
    settings = RenderSettings(peak=None)
    a = renderer.render([note("P1", A4)], settings)
    b = renderer.render([note("P1", A4)], settings, envelopes={})
    c = renderer.render([note("P1", A4)], settings, envelopes={"P2": []})
    assert np.array_equal(a.samples, b.samples)
    assert np.array_equal(a.samples, c.samples)


def test_a_note_before_the_first_breakpoint_stays_audible(renderer):
    """Otherwise a note starting ahead of any controller write is silenced."""
    from rse.renderers.base import EnvelopePoint

    result = renderer.render(
        [note("P1", A4, start=0.0, end=0.2)],
        RenderSettings(peak=None),
        envelopes={"P1": [EnvelopePoint(0.5, 0.1)]},
    )
    assert np.abs(result.samples[: int(0.1 * result.sample_rate)]).max() > 0.01


def test_the_register_steps_rather_than_glides(renderer):
    """The APU's volume register holds its value until the next write."""
    from rse.renderers.base import EnvelopePoint

    result = renderer.render(
        [note("P1", A4, end=0.4)],
        RenderSettings(peak=None, apply_output_filters=False),
        envelopes={"P1": [EnvelopePoint(0.0, 1.0), EnvelopePoint(0.2, 0.5)]},
    )
    rate = result.sample_rate
    first = np.abs(result.samples[int(0.05 * rate) : int(0.15 * rate)]).max()
    second = np.abs(result.samples[int(0.25 * rate) : int(0.35 * rate)]).max()
    assert second == pytest.approx(first * 0.5, rel=0.15)


def test_envelopes_are_read_from_a_record(nes_profile):
    """CC 11 is what both the exprsco reader and NES-MDB's MIDI use."""
    from rse.renderers.base import envelopes_from_record

    record = {
        "midi": {"ticks_per_beat": 480},
        "tempo_map": [{"tick": 0, "us_per_beat": 500_000}],
        "events": [
            {"ch": "NO", "type": "control", "tick": 0, "control": 11, "value": 127},
            {"ch": "NO", "type": "control", "tick": 240, "control": 11, "value": 32},
            {"ch": "NO", "type": "control", "tick": 480, "control": 12, "value": 2},
        ],
    }
    envelopes = envelopes_from_record(record)
    assert list(envelopes) == ["NO"]
    # CC 12 is timbre, not volume, and must not become an envelope point.
    assert [round(p.gain, 3) for p in envelopes["NO"]] == [1.0, 0.252]
    assert envelopes["NO"][1].sec == pytest.approx(0.25)


def test_an_unpitched_note_decays_by_default(renderer):
    """Model output carries no controller data, so without a default decay a
    generated drum renders at one constant level — heard as static."""
    hit = [note("NO", 13, end=5 / 24, pitched=False)]
    flat = renderer.render(hit, RenderSettings(peak=None, default_noise_decay=False))
    shaped = renderer.render(hit, RenderSettings(peak=None))

    def per_frame(result):
        rate = result.sample_rate
        return [
            float(np.abs(result.stems["NO"][int(k * rate / 24) : int((k + 1) * rate / 24)]).max())
            for k in range(5)
        ]

    constant = per_frame(flat)
    decayed = per_frame(shaped)
    assert constant[0] == pytest.approx(constant[-1], rel=0.01)
    # Falls to roughly the measured 0.43 of onset by the fifth frame.
    assert decayed[-1] / decayed[0] == pytest.approx(0.43, abs=0.08)


def test_pitched_channels_never_get_the_default_decay(renderer):
    """Measured pulse notes rise then hold (1.00 1.23 1.22 1.19), so decaying
    them would invent behaviour the hardware does not have."""
    for channel, value in (("P1", A4), ("P2", A4), ("TR", A4 - 24)):
        held = [note(channel, value, end=0.4)]
        flat = renderer.render(held, RenderSettings(peak=None, default_noise_decay=False))
        shaped = renderer.render(held, RenderSettings(peak=None))
        assert np.array_equal(flat.samples, shaped.samples), channel


def test_source_envelopes_take_precedence_over_the_default(renderer):
    """Real data knows better than a default shape."""
    from rse.renderers.base import EnvelopePoint

    hit = [note("NO", 13, end=0.3, pitched=False)]
    default = renderer.render(hit, RenderSettings(peak=None))
    explicit = renderer.render(
        hit,
        RenderSettings(peak=None),
        envelopes={"NO": [EnvelopePoint(0.0, 1.0)]},
    )
    # A flat explicit envelope must override the decaying default.
    assert np.abs(explicit.samples).max() > np.abs(default.samples[-1000:]).max()


def test_the_default_decay_is_reported_in_settings():
    assert RenderSettings().to_dict()["default_noise_decay"] is True
    assert RenderSettings(default_noise_decay=False).to_dict()["default_noise_decay"] is False
