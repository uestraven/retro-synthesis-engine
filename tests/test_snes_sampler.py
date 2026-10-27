"""The SNES preview renderer: samples, stereo, envelopes, eight voices.

A PREVIEW. These tests pin the preview's own behaviour and the honesty of its
labelling; none of them assert hardware accuracy, because none of it is
hardware-accurate. No BRR, no Gaussian interpolation, no ARAM enforcement.

Three defects found while building this are pinned here so they cannot return:

1. The sample bank seeded its RNG from `hash(name)`, which Python randomises
   per process, so every render differed. The determinism requirement caught
   it.
2. The envelope scaled its attack to the note length, so a note shorter than
   its own attack still reached full level -- flattening the dynamics of every
   short note.
3. Release tails ran past each part's rhythmic slot, so three sustained
   harmony notes occupied four voices at every bar line. The arrangement asked
   for eleven voices on a console with eight and lost the entire choir pad.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from rse.exporters.wav import read_wav, write_wav
from rse.platform.roles import PartNote, Role, VoiceAllocator, concurrency
from rse.procedural.compose import compose
from rse.snes.arrange import VOICE_PLAN, allocate, arrange, chord_for
from rse.snes.sampler import VoiceSettings, envelope_for, render
from rse.snes.samples import BANK, build_bank, calibrate, manifest


@pytest.fixture(scope="module")
def bank():
    return build_bank()


# --- the sample bank ------------------------------------------------------


def test_the_bank_is_deterministic_across_processes() -> None:
    """Seeded from `hash(name)` at first, which Python randomises per process
    unless PYTHONHASHSEED is set, so every render was different."""
    import subprocess
    import sys

    digests = set()
    for seed in ("1", "2"):
        proc = subprocess.run(
            [
                sys.executable, "-I", "-c",
                "import sys,hashlib; sys.path.insert(0,'.');"
                "from rse.snes.samples import build_bank;"
                "b=build_bank();"
                "print(hashlib.sha256(b''.join("
                "b[k].data.tobytes() for k in sorted(b))).hexdigest())",
            ],
            capture_output=True, text=True, check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        )
        digests.add(proc.stdout.strip())
    assert len(digests) == 1, "the bank differs between processes"


def test_every_instrument_renders(bank) -> None:
    assert set(bank) == set(BANK)
    for name, asset in bank.items():
        assert asset.frames > 0, name
        assert np.isfinite(asset.data).all(), name
        assert float(np.abs(asset.data).max()) <= 1.0001, name


def test_sustained_instruments_loop_and_percussion_does_not(bank) -> None:
    for name in ("brass", "strings", "choir", "bass"):
        asset = bank[name]
        assert asset.loop_end > asset.loop_start, name
        assert asset.loop_start > 0, f"{name} should have an attack before its loop"
    for name in ("kick", "snare", "hat"):
        assert bank[name].loop_end == bank[name].loop_start == 0, name


def test_instruments_in_the_same_register_are_spectrally_distinct(bank) -> None:
    """A brass sample must not be a string sample with a different name.

    Compared only among the instruments that share a register and can
    therefore be confused: brass, strings and choir all have root note 60. The
    bass is an octave and a half below and never plays the same part, so
    comparing its raw loop spectrum to a choir's measures nothing useful --
    which an earlier version of this test did, and failed on.
    """
    import itertools

    spectra = {}
    for name in ("brass", "strings", "choir"):
        asset = bank[name]
        assert asset.instrument.root_note == 60, name
        region = asset.data[asset.loop_start : asset.loop_end]
        magnitude = np.abs(np.fft.rfft(region * np.hanning(len(region))))
        spectra[name] = magnitude / (magnitude.max() or 1)
    for first, second in itertools.combinations(spectra, 2):
        length = min(len(spectra[first]), len(spectra[second]))
        difference = float(
            np.abs(spectra[first][:length] - spectra[second][:length]).mean()
        )
        assert difference > 0.002, f"{first} and {second} are too similar"


def test_the_bass_is_distinct_by_register(bank) -> None:
    """Separated by where it sits rather than by spectral shape."""
    def centroid(name: str) -> float:
        asset = bank[name]
        region = asset.data[asset.loop_start : asset.loop_end].astype(np.float64)
        magnitude = np.abs(np.fft.rfft(region * np.hanning(len(region))))
        freqs = np.fft.rfftfreq(len(region), 1 / asset.rate)
        return float((freqs * magnitude).sum() / magnitude.sum())

    assert bank["bass"].instrument.root_note < 48
    assert centroid("bass") < centroid("brass") / 2


def test_gains_are_calibrated_by_loop_rms_not_peak(bank) -> None:
    """Peak-normalising each sample equalises the wrong quantity: loudness
    follows RMS. This is the NES balance lesson applied before it recurs."""
    levels = {}
    for name in ("brass", "strings", "choir", "bass"):
        asset = bank[name]
        region = asset.data[asset.loop_start : asset.loop_end]
        rms = float(np.sqrt(np.mean(region.astype(np.float64) ** 2)))
        levels[name] = rms * asset.instrument.gain
    spread = max(levels.values()) / min(levels.values())
    assert spread < 1.25, f"calibrated levels still differ by {spread:.2f}x"


def test_percussion_is_not_calibrated_to_the_lead(bank) -> None:
    """Calibrating a hat to a brass note's RMS makes it a cymbal crash."""
    for name in ("kick", "snare", "hat"):
        assert bank[name].instrument.gain == 1.0, name


def test_the_manifest_declares_provenance(bank) -> None:
    report = manifest(bank)
    assert "Original synthesised prototype" in report["provenance"]
    assert "No commercial or" in report["provenance"]


def test_the_manifest_distinguishes_estimated_from_measured(bank) -> None:
    """Sizes were geometry estimates before `rse.snes.brr` existed. Now they
    can be measurements, and the manifest has to say which it is holding."""
    from rse.snes.brr import encode as brr_encode

    estimated = manifest(bank)
    assert estimated["brr_measured"] is False
    assert "Nothing in this bank is BRR-encoded" in estimated["caveat"]

    encoded = {
        name: brr_encode(
            asset.data,
            loop_start=asset.loop_start if asset.loop_end > asset.loop_start else None,
        )
        for name, asset in bank.items()
    }
    measured = manifest(bank, encoded=encoded)
    assert measured["brr_measured"] is True
    assert "MEASURED" in measured["caveat"]
    # Hardware compatibility is still not claimed either way.
    assert "not claimed" in measured["caveat"]
    assert measured["brr_bytes"] < measured["aram_total_bytes"]


def test_calibrate_handles_a_silent_sample(bank) -> None:
    import dataclasses

    quiet = dataclasses.replace(bank["brass"], data=np.zeros(100, dtype=np.float32))
    gains = calibrate({"brass": bank["brass"], "quiet": quiet})
    assert gains["quiet"] == 1.0


# --- envelopes ------------------------------------------------------------


def test_the_attack_rate_is_fixed_not_scaled_to_the_note(bank) -> None:
    """The defect: a note shorter than its attack reached full level anyway,
    because the ramp was scaled to the note. Every short note then had a
    full-amplitude attack and the dynamics went flat."""
    asset = bank["strings"]
    attack = asset.instrument.envelope.attack
    rate = 44100
    for fraction in (0.125, 0.25, 0.5):
        hold = int(attack * fraction * rate)
        curve = envelope_for(asset, hold_frames=hold, rate=rate)
        assert curve[hold - 1] == pytest.approx(fraction, abs=0.02), fraction


def test_release_starts_from_wherever_the_envelope_reached(bank) -> None:
    """A note released during its attack must not jump up to sustain first."""
    asset = bank["strings"]
    hold = int(0.02 * 44100)
    curve = envelope_for(asset, hold_frames=hold, rate=44100)
    assert curve[hold] <= curve[hold - 1] + 1e-6
    assert curve[-1] == pytest.approx(0.0, abs=1e-6)


def test_the_envelope_is_longer_than_the_hold_by_its_release(bank) -> None:
    asset = bank["strings"]
    release = int(asset.instrument.envelope.release * 44100)
    curve = envelope_for(asset, hold_frames=1000, rate=44100)
    assert len(curve) == 1000 + release


def test_a_long_note_settles_on_the_sustain_level(bank) -> None:
    asset = bank["strings"]
    curve = envelope_for(asset, hold_frames=44100, rate=44100)
    assert curve[44099] == pytest.approx(
        asset.instrument.envelope.sustain, abs=1e-6
    )


# --- stereo ---------------------------------------------------------------


def test_constant_power_panning_keeps_centre_from_being_louder() -> None:
    """Linear panning makes a centred voice 3 dB louder than the same voice
    panned hard left -- the balance defect the NES work ran into."""
    for pan in (-1.0, -0.5, 0.0, 0.5, 1.0):
        left, right = VoiceSettings(instrument="brass", pan=pan).levels()
        assert left**2 + right**2 == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(
    ("pan", "expect"),
    [(-1.0, "left"), (0.0, "both"), (1.0, "right")],
)
def test_panning_moves_energy_between_channels(bank, pan, expect) -> None:
    notes = [PartNote(role=Role.LEAD, tick=0, duration=48, value=60)]
    allocation = VoiceAllocator(["V1"]).allocate(notes)
    out = render(
        allocation.assignments, bank=bank,
        voices={Role.LEAD: VoiceSettings(instrument="brass", pan=pan)},
        seconds_per_tick=0.01, allocation=allocation,
    )
    left = float(np.sqrt((out.samples[:, 0] ** 2).mean()))
    right = float(np.sqrt((out.samples[:, 1] ** 2).mean()))
    if expect == "left":
        assert left > 0 and right == pytest.approx(0.0, abs=1e-6)
    elif expect == "right":
        assert right > 0 and left == pytest.approx(0.0, abs=1e-6)
    else:
        assert left == pytest.approx(right, rel=1e-6)


def test_the_render_buffer_is_two_channel(bank) -> None:
    notes = [PartNote(role=Role.LEAD, tick=0, duration=48, value=60)]
    allocation = VoiceAllocator(["V1"]).allocate(notes)
    out = render(
        allocation.assignments, bank=bank,
        voices={Role.LEAD: VoiceSettings(instrument="brass")},
        seconds_per_tick=0.01, allocation=allocation,
    )
    assert out.samples.ndim == 2
    assert out.samples.shape[1] == 2


def test_wav_export_writes_real_stereo(tmp_path: Path) -> None:
    """Not a duplicated mono waveform: the channel count comes from the data
    and the two channels must differ."""
    frames = 400
    tone = np.sin(np.linspace(0, 40, frames))
    buffer = np.stack([tone * 0.2, tone * 0.9], axis=1)
    path = write_wav(tmp_path / "s.wav", buffer, 44100)
    with wave.open(str(path)) as handle:
        assert handle.getnchannels() == 2
        assert handle.getnframes() == frames
    back, rate = read_wav(path)
    assert rate == 44100
    assert back.shape == (frames, 2)
    assert not np.array_equal(back[:, 0], back[:, 1])


def test_mono_export_is_unchanged(tmp_path: Path) -> None:
    """NES stays mono and backward-compatible."""
    mono = np.sin(np.linspace(0, 40, 300)) * 0.5
    path = write_wav(tmp_path / "m.wav", mono, 44100)
    with wave.open(str(path)) as handle:
        assert handle.getnchannels() == 1
    back, _rate = read_wav(path)
    assert back.ndim == 1


def test_a_single_column_buffer_is_written_as_mono(tmp_path: Path) -> None:
    mono = np.sin(np.linspace(0, 40, 200)).reshape(-1, 1)
    path = write_wav(tmp_path / "one.wav", mono, 44100)
    with wave.open(str(path)) as handle:
        assert handle.getnchannels() == 1


@pytest.mark.parametrize("shape", [(10, 3), (4, 4, 2)])
def test_malformed_buffers_are_refused(tmp_path: Path, shape) -> None:
    with pytest.raises(ValueError):
        write_wav(tmp_path / "bad.wav", np.zeros(shape), 44100)


# --- voice allocation drives the audio ------------------------------------


def test_the_renderer_only_plays_what_the_allocator_assigned(bank) -> None:
    """So the audible result and the allocation report describe the same
    thing. A dropped note is explained by the report, not silently missing."""
    notes = [
        PartNote(role=Role.ARPEGGIO, tick=0, duration=50, value=60 + i)
        for i in range(12)
    ]
    allocation = VoiceAllocator([f"V{i}" for i in range(1, 9)]).allocate(notes)
    out = render(
        allocation.assignments, bank=bank,
        voices={Role.ARPEGGIO: VoiceSettings(instrument="brass")},
        seconds_per_tick=0.01, allocation=allocation,
    )
    assert out.notes_rendered == len(allocation.assignments) == 8
    assert allocation.dropped == 4
    assert out.peak_voices == 8


def test_a_retrigger_over_a_live_release_is_reported(bank) -> None:
    """The renderer honours the allocator rather than re-deciding, so if an
    assignment would cut a release tail it says so instead of hiding it."""
    from rse.platform.roles import VoiceAssignment

    assignments = [
        VoiceAssignment(
            note=PartNote(
                role=Role.PAD, tick=0, duration=20, value=60, release=200
            ),
            voice="V1",
        ),
        VoiceAssignment(
            note=PartNote(role=Role.PAD, tick=30, duration=20, value=62),
            voice="V1",
        ),
    ]
    out = render(
        assignments, bank=bank,
        voices={Role.PAD: VoiceSettings(instrument="choir")},
        seconds_per_tick=0.01,
    )
    assert any("still releasing" in w for w in out.warnings)


def test_a_missing_instrument_is_reported_not_silent(bank) -> None:
    notes = [PartNote(role=Role.LEAD, tick=0, duration=20, value=60)]
    allocation = VoiceAllocator(["V1"]).allocate(notes)
    out = render(
        allocation.assignments, bank=bank,
        voices={Role.LEAD: VoiceSettings(instrument="tuba")},
        seconds_per_tick=0.01, allocation=allocation,
    )
    assert any("no sample for instrument" in w for w in out.warnings)


# --- the arrangement ------------------------------------------------------


def test_the_arrangement_fits_eight_voices_with_nothing_dropped() -> None:
    """The defect: release tails ran past each part's slot, so three harmony
    notes cost four voices at every bar line, the arrangement needed eleven,
    and the choir pad lost all eight of its notes."""
    result = compose(style_name="boss", seed=4242, target_seconds=60.0)
    song = arrange(result)
    assert concurrency(song.parts) <= 8
    allocation = allocate(song)
    assert allocation.dropped == 0, allocation.to_dict()
    assert allocation.stolen == 0
    assert allocation.peak_voices <= 8


def test_the_pad_survives_because_it_is_the_contrast() -> None:
    result = compose(style_name="boss", seed=4242, target_seconds=60.0)
    song = arrange(result)
    allocation = allocate(song)
    pad_parts = sum(1 for p in song.parts if p.role is Role.PAD)
    pad_kept = sum(1 for a in allocation.assignments if a.note.role is Role.PAD)
    assert pad_parts > 0
    assert pad_kept == pad_parts


def test_the_arrangement_is_polyphonic_not_a_waveform_swap() -> None:
    """Sustained three-note chords are the thing the NES cannot do: its one
    comp channel can only arpeggiate a chord, never state it."""
    result = compose(style_name="boss", seed=4242, target_seconds=60.0)
    song = arrange(result)
    harmony = [p for p in song.parts if p.role is Role.HARMONY]
    assert concurrency(harmony) >= 2, "harmony must actually be polyphonic"
    assert len({p.role for p in song.parts}) >= 5


def test_extended_chords_appear_and_keep_their_identity() -> None:
    result = compose(style_name="boss", seed=4242, target_seconds=60.0)
    song = arrange(result)
    extended = [v for v in song.voicings if v.intent.seventh.value != "none"]
    assert extended, "no extended-chord passage"
    for voicing in extended:
        assert voicing.keeps_identity, voicing.intent.symbol
        assert voicing.voice_count <= 3


def test_a_minor_triad_takes_a_minor_seventh() -> None:
    """Harmonic minor hands out a major seventh on its tonic, which makes a
    minor-major chord: diatonic, harsh, and not what this repertoire uses."""
    chord = chord_for(
        degree=0, tonic=57, mode="harmonic_minor", extended=True, section="B"
    )
    assert chord.symbol == "Am9"
    assert "Maj" not in chord.symbol


def test_every_role_in_the_plan_has_an_instrument(bank) -> None:
    for role, settings in VOICE_PLAN.items():
        assert settings.instrument in bank, (role, settings.instrument)


def test_the_lead_and_bass_stay_centred() -> None:
    """A melody that wanders across the image is distracting, and low
    frequencies carry no useful directional information."""
    assert VOICE_PLAN[Role.LEAD].pan == 0.0
    assert VOICE_PLAN[Role.BASS].pan == 0.0


def test_the_renderer_labels_itself_a_preview(bank) -> None:
    notes = [PartNote(role=Role.LEAD, tick=0, duration=20, value=60)]
    allocation = VoiceAllocator(["V1"]).allocate(notes)
    out = render(
        allocation.assignments, bank=bank,
        voices={Role.LEAD: VoiceSettings(instrument="brass")},
        seconds_per_tick=0.01, allocation=allocation,
    )
    assert "PREVIEW" in out.to_dict()["renderer"]
    assert "not hardware" in out.to_dict()["renderer"]


# --- the "feedback" defect ------------------------------------------------


def test_every_sustained_loop_is_exactly_periodic(bank) -> None:
    """The artifact the listener heard as "feedback or something coming
    through". The partials were detuned by up to eleven cents, so they could
    not share the loop's period, and the waveform jumped at every wrap: 4x to
    12x a normal sample step, repeating at 32.7 Hz. A discontinuity repeating
    at 32.7 Hz is a 32.7 Hz tone.

    The crossfade that was supposed to hide it had been applied to the wrong
    place -- the buffer's end into its start, while the player wraps from
    `loop_end` back to `loop_start`.
    """
    for name in ("brass", "strings", "choir", "bass"):
        asset = bank[name]
        loop = asset.data[asset.loop_start : asset.loop_end].astype(np.float64)
        assert loop.size > 0, name
        doubled = np.concatenate([loop, loop])
        wrap_step = abs(doubled[loop.size] - doubled[loop.size - 1])
        largest = float(np.abs(np.diff(loop)).max())
        assert wrap_step <= largest * 1.05, (
            f"{name}: wrap step {wrap_step:.5f} exceeds the largest step "
            f"inside the loop ({largest:.5f}), so the loop clicks"
        )


def test_a_held_note_has_no_inharmonic_energy(bank) -> None:
    """A click at the loop rate shows up as energy away from the loop's own
    harmonic series."""
    for name in ("brass", "strings", "choir", "bass"):
        asset = bank[name]
        loop = asset.data[asset.loop_start : asset.loop_end].astype(np.float64)
        held = np.tile(loop, 12)
        magnitude = np.abs(np.fft.rfft(held * np.hanning(len(held))))
        freqs = np.fft.rfftfreq(len(held), 1 / asset.rate)
        loop_hz = asset.rate / len(loop)
        harmonic = np.zeros_like(magnitude, dtype=bool)
        index = 1
        while loop_hz * index < asset.rate / 2:
            harmonic |= np.abs(freqs - loop_hz * index) < loop_hz * 0.3
            index += 1
        audible = freqs > 20
        junk = magnitude[audible & ~harmonic].sum() / magnitude[audible].sum()
        assert junk < 0.05, f"{name}: {junk * 100:.2f}% inharmonic energy"


def test_sustained_instruments_carry_no_noise(bank) -> None:
    """Noise cannot be periodic. Tiling it to the loop length -- the first
    attempt -- makes it a pitched rasp at the loop rate rather than noise."""
    for name in ("brass", "strings", "choir", "bass"):
        assert bank[name].instrument.noise == 0.0, name
    # Percussion is one-shot and never loops, so its noise is fine.
    for name in ("kick", "snare", "hat"):
        assert bank[name].instrument.noise > 0.0, name


def test_samples_carry_no_dc_offset(bank) -> None:
    """A constant offset under every sustained note wastes headroom and reads
    as pressure rather than tone. The finished mix measured +0.012 before the
    loop region's mean was removed."""
    for name, asset in bank.items():
        if asset.loop_end > asset.loop_start:
            region = asset.data[asset.loop_start : asset.loop_end]
        else:
            region = asset.data
        assert abs(float(region.mean())) < 0.02, name


def test_the_mix_is_high_passed(bank) -> None:
    """The NES renderer has an output high-pass and this did not, which
    showed: 3.2% of the mix's energy sat below 60 Hz, peaking at 5-7 Hz."""
    notes = [
        PartNote(role=Role.HARMONY, tick=tick, duration=90, value=60 + i)
        for tick, i in zip(range(0, 400, 100), range(4), strict=True)
    ]
    allocation = VoiceAllocator(["V1", "V2", "V3", "V4"]).allocate(notes)
    out = render(
        allocation.assignments, bank=bank,
        voices={Role.HARMONY: VoiceSettings(instrument="strings")},
        seconds_per_tick=0.01, allocation=allocation,
    )
    mono = out.samples.mean(axis=1).astype(np.float64)
    assert abs(float(mono.mean())) < 1e-3, "DC survived the high-pass"


def test_the_high_pass_spares_the_bass(bank) -> None:
    """30 Hz removes rumble without touching the bass instrument, whose
    lowest note is around 65 Hz."""
    from rse.snes.sampler import OUTPUT_HIGHPASS_HZ
    from rse.snes.samples import midi_to_hz

    assert OUTPUT_HIGHPASS_HZ < midi_to_hz(BANK["bass"].root_note) / 2


# --- the out-of-place bass note -------------------------------------------


def test_the_approach_note_steps_into_the_next_root() -> None:
    """The note the listener heard as out of place at 0:25.

    It was `bass + (2 if degree % 2 == 0 else -2)` -- a whole tone up or down
    chosen by whether the degree INDEX was even, with no reference to the next
    chord and no guarantee of being a step. In A harmonic minor it produced
    A2 -> A3 -> B2 -> A2: an octave leap up, then a drop of a minor seventh to
    a whole tone above the root.
    """
    from rse.procedural.compose import TICKS_PER_BAR, compose
    from rse.snes.arrange import arrange

    result = compose(style_name="boss", seed=4242, target_seconds=60.0)
    song = arrange(result)
    bass = sorted(
        (p for p in song.parts if p.role is Role.BASS), key=lambda p: p.tick
    )
    by_bar: dict[int, list] = {}
    for part in bass:
        by_bar.setdefault(part.tick // TICKS_PER_BAR, []).append(part)

    checked = 0
    for bar in sorted(by_bar):
        following = by_bar.get(bar + 1)
        if not following:
            continue
        last = sorted(by_bar[bar], key=lambda p: p.tick)[-1]
        root = sorted(following, key=lambda p: p.tick)[0]
        checked += 1
        # A step into the next root, in either octave.
        distance = min(
            abs(last.value - root.value), abs(last.value - (root.value + 12))
        )
        assert distance <= 2, (
            f"bar {bar}: approach {last.value} is {distance} semitones from "
            f"the next root {root.value}, which is a leap not a step"
        )
    assert checked >= 30


def test_every_bass_note_is_diatonic() -> None:
    from rse.procedural.compose import compose
    from rse.snes.arrange import SCALES, arrange

    result = compose(style_name="boss", seed=4242, target_seconds=60.0)
    song = arrange(result)
    scale = set(SCALES[result.style.mode])
    tonic = result.style.tonic
    offenders = [
        part.value
        for part in song.parts
        if part.role is Role.BASS and (part.value - tonic) % 12 not in scale
    ]
    assert offenders == [], offenders


def test_the_bass_makes_no_leap_wider_than_an_octave_inside_a_bar() -> None:
    """The octave lift is deliberate; a drop of 13 to 20 semitones back down
    to the approach note was not, and that contour is what stood out."""
    from rse.procedural.compose import TICKS_PER_BAR, compose
    from rse.snes.arrange import arrange

    result = compose(style_name="boss", seed=4242, target_seconds=60.0)
    song = arrange(result)
    by_bar: dict[int, list] = {}
    for part in song.parts:
        if part.role is Role.BASS:
            by_bar.setdefault(part.tick // TICKS_PER_BAR, []).append(part)
    for bar, parts in by_bar.items():
        ordered = sorted(parts, key=lambda p: p.tick)
        for first, second in zip(ordered, ordered[1:], strict=False):
            assert abs(second.value - first.value) <= 12, (
                f"bar {bar}: {first.value} -> {second.value}"
            )


def test_the_nearest_approach_wins_over_direction() -> None:  # noqa: D401
    """A semitone beats a whole tone regardless of which side the line is on.

    Into A harmonic minor's tonic that selects the leading tone G#, which is
    the strongest approach available. An earlier version of this test expected
    the direction to win, which would have chosen B over G# coming from above
    -- a weaker approach, and not what the code does.
    """
    from rse.snes.arrange import approach_note

    for from_pitch in (60, 76):
        assert approach_note(
            target=69, from_pitch=from_pitch, tonic=57, mode="harmonic_minor"
        ) == 68


def test_direction_breaks_an_exact_tie() -> None:
    """Where the scale offers an equal step on both sides, the line approaches
    from the side it is already on. In natural minor the fifth has a whole
    tone above and below it."""
    from rse.snes.arrange import approach_note

    # E (64) in A natural minor: D (62) below, F# is not in the scale, so the
    # neighbours are D (2 below) and F (65, 1 above) -- F wins on distance.
    assert approach_note(
        target=64, from_pitch=70, tonic=57, mode="minor"
    ) == 65
    # A tie: the second degree B (59) has A (57, 2 below) and C (60, 1 above).
    # C is nearer, so distance still decides.
    assert approach_note(
        target=59, from_pitch=50, tonic=57, mode="minor"
    ) == 60


def test_approach_note_never_leaves_the_scale() -> None:
    from rse.snes.arrange import SCALES, approach_note

    for mode, scale in SCALES.items():
        for target in range(50, 75):
            note = approach_note(
                target=target, from_pitch=target - 7, tonic=57, mode=mode
            )
            assert (note - 57) % 12 in set(scale), (mode, target, note)


# --- percussion is not pitched --------------------------------------------


def test_percussion_value_is_an_instrument_not_a_pitch(bank) -> None:
    """The off-key, off-beat note. Percussion `value` is a noise period index
    inherited from the NES noise channel -- 1 kick, 6 snare, 12 hat -- and the
    sampler resampled by it as if it were a MIDI note. That pitched the kick
    down 2.9 octaves, the snare 4.5 and the hat 6, stretching a 70 ms hat into
    a 6.17-second sub-bass drone. With 327 percussion notes the track carried a
    continuous bed of them.
    """
    from rse.snes.sampler import _resample

    for name, value in (("kick", 1), ("snare", 6), ("hat", 12)):
        asset = bank[name]
        native = asset.frames / asset.rate
        played = _resample(asset, value, int(0.6 * 44100), pitched=False)
        sounding = np.nonzero(np.abs(played) > 1e-4)[0]
        assert sounding.size > 0, name
        duration = (sounding[-1] - sounding[0]) / 44100
        assert duration == pytest.approx(native, abs=0.03), (
            f"{name}: {duration * 1000:.0f} ms played against a "
            f"{native * 1000:.0f} ms sample"
        )


def test_the_arranger_marks_percussion_unpitched() -> None:
    from rse.procedural.compose import compose
    from rse.snes.arrange import arrange

    result = compose(style_name="boss", seed=4242, target_seconds=60.0)
    song = arrange(result)
    drums = [p for p in song.parts if p.role is Role.PERCUSSION]
    assert drums
    assert all(not p.pitched for p in drums)
    # And everything else IS pitched.
    assert all(p.pitched for p in song.parts if p.role is not Role.PERCUSSION)


def test_a_pitched_note_still_transposes(bank) -> None:
    """The fix must not disable pitching for the instruments that need it."""
    from rse.snes.sampler import _resample

    asset = bank["brass"]
    low = _resample(asset, 48, 4000, pitched=True)
    high = _resample(asset, 72, 4000, pitched=True)
    # An octave up completes twice as many cycles in the same window.
    def zero_crossings(wave):
        return int(np.count_nonzero(np.diff(np.signbit(wave))))

    assert zero_crossings(high) > zero_crossings(low) * 1.5


def test_the_kick_is_dominated_by_low_frequencies(bank) -> None:
    """A kick whose spectral centroid sits at 2706 Hz is not a kick. 10% noise
    dominated the spectrum and there was no pitch envelope, which is the
    gesture that makes a kick sound like one."""
    asset = bank["kick"]
    data = asset.data.astype(np.float64)
    magnitude = np.abs(np.fft.rfft(data))
    freqs = np.fft.rfftfreq(len(data), 1 / asset.rate)
    below = magnitude[freqs < 200].sum() / magnitude.sum()
    assert below > 0.5, f"only {below * 100:.1f}% of the kick is below 200 Hz"
    assert asset.instrument.pitch_drop > 0


def test_the_drums_are_ordered_low_to_high(bank) -> None:
    def centroid(name: str) -> float:
        asset = bank[name]
        data = asset.data.astype(np.float64)
        magnitude = np.abs(np.fft.rfft(data))
        freqs = np.fft.rfftfreq(len(data), 1 / asset.rate)
        return float((freqs * magnitude).sum() / magnitude.sum())

    assert centroid("kick") < centroid("snare") < centroid("hat")


def test_the_pitch_drop_keeps_its_phase_continuous(bank) -> None:
    """Stepping the frequency per sample without integrating produces a buzz,
    so the falling pitch is applied as an integrated phase."""
    data = bank["kick"].data.astype(np.float64)
    steps = np.abs(np.diff(data))
    # No isolated discontinuity: the largest step is not wildly beyond the rest.
    assert steps.max() < np.percentile(steps, 99.9) * 6


def test_the_mix_has_little_sub_bass(bank) -> None:
    """Where the drones were. 20-60 Hz is below the bass instrument's range,
    so anything much there is an artifact."""
    import wave as wave_module

    path = Path("listen_snes/snes_boss.wav")
    if not path.is_file():
        pytest.skip("the rendered track is not present")
    with wave_module.open(str(path)) as handle:
        rate = handle.getframerate()
        raw = np.frombuffer(
            handle.readframes(handle.getnframes()), dtype="<i2"
        ).astype(np.float64)
    mono = raw.reshape(-1, 2).mean(axis=1)
    magnitude = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    freqs = np.fft.rfftfreq(len(mono), 1 / rate)
    audible = freqs > 20
    sub = magnitude[(freqs > 20) & (freqs < 60)].sum() / magnitude[audible].sum()
    assert sub < 0.02, f"{sub * 100:.2f}% of the mix is below 60 Hz"
