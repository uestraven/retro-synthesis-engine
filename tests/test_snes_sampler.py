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


def test_the_instruments_are_spectrally_distinct(bank) -> None:
    """A brass sample must not be a string sample with a different name."""
    import itertools

    spectra = {}
    for name in ("brass", "strings", "choir", "bass"):
        asset = bank[name]
        region = asset.data[asset.loop_start : asset.loop_end]
        magnitude = np.abs(np.fft.rfft(region * np.hanning(len(region))))
        spectra[name] = magnitude / (magnitude.max() or 1)
    for first, second in itertools.combinations(spectra, 2):
        length = min(len(spectra[first]), len(spectra[second]))
        difference = float(
            np.abs(spectra[first][:length] - spectra[second][:length]).mean()
        )
        assert difference > 0.002, f"{first} and {second} are too similar"


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


def test_the_manifest_declares_provenance_and_refuses_to_claim_brr(bank) -> None:
    report = manifest(bank)
    assert "Original synthesised prototype" in report["provenance"]
    assert "No commercial or" in report["provenance"]
    assert "is BRR-encoded" in report["caveat"]
    assert "no ARAM compliance is claimed" in report["caveat"]
    assert report["brr_estimate_bytes"] < report["aram_total_bytes"]


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
