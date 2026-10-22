from __future__ import annotations

import pytest

from rse import paths
from rse.hardware.profile import HardwareProfile, ProfileError, available_profiles, load_profile
from rse.jsonio import write_json


def test_committed_nes_profile_describes_four_channels(nes_profile):
    assert nes_profile.profile_id == "nes"
    assert nes_profile.channel_ids == ("P1", "P2", "TR", "NO")
    assert all(ch.max_simultaneous_notes == 1 for ch in nes_profile.channels)


def test_noise_channel_is_not_pitched(nes_profile):
    noise = nes_profile.channel("NO")
    assert noise.pitched is False
    # Its limits are hardware period values, not MIDI pitches.
    assert (noise.value_min, noise.value_max) == (1, 16)
    assert noise.in_range(8)
    assert not noise.in_range(60)


def test_pitched_channels_carry_midi_pitch_limits(nes_profile):
    triangle = nes_profile.channel_by_role("triangle")
    assert triangle.pitched is True
    assert triangle.in_range(60)
    assert not triangle.in_range(120)


def test_track_names_match_case_insensitively(nes_profile):
    assert nes_profile.match_track_name("P1").channel_id == "P1"
    assert nes_profile.match_track_name(" pulse 2 ").channel_id == "P2"
    assert nes_profile.match_track_name("Triangle").channel_id == "TR"
    assert nes_profile.match_track_name("noise").channel_id == "NO"
    assert nes_profile.match_track_name("Lead Synth") is None
    assert nes_profile.match_track_name(None) is None


def test_nes_is_discoverable():
    assert "nes" in available_profiles(configs_dir=paths.HARDWARE_CONFIGS_DIR)


def test_unknown_channel_raises(nes_profile):
    with pytest.raises(KeyError):
        nes_profile.channel("FM1")


def test_future_schema_major_is_refused():
    with pytest.raises(ProfileError, match="unsupported hardware schema"):
        HardwareProfile.from_dict(
            {"schema_version": "rse-hardware/9.0", "profile_id": "x", "channels": []}
        )


def test_duplicate_channel_ids_are_refused():
    channel = {
        "channel_id": "P1",
        "role": "pulse1",
        "type": "pulse",
        "pitch_min": 33,
        "pitch_max": 108,
    }
    with pytest.raises(ProfileError, match="duplicate channel_id"):
        HardwareProfile.from_dict(
            {
                "schema_version": "rse-hardware/0.1.0",
                "profile_id": "x",
                "channels": [channel, dict(channel)],
            }
        )


def test_missing_profile_file_reports_the_path(tmp_path):
    with pytest.raises(ProfileError, match="no hardware profile at"):
        load_profile("genesis", configs_dir=tmp_path)


def test_profile_without_channels_is_refused(tmp_path):
    write_json(
        tmp_path / "empty.json",
        {"schema_version": "rse-hardware/0.1.0", "profile_id": "empty", "channels": []},
    )
    with pytest.raises(ProfileError, match="declares no channels"):
        load_profile("empty", configs_dir=tmp_path)
