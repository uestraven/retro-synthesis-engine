from __future__ import annotations

import pytest

from rse.data.midi_io import parse_midi
from rse.data.normalize import build_intake_record
from rse.events.convert import (
    EXACT,
    Tolerance,
    compare_compositions,
    composition_from_intake,
    composition_from_midi,
    composition_to_intake,
    round_trip_midi,
    write_composition_midi,
)
from rse.events.schema import Composition, Controller, EffectEvent, Noise, Note
from rse.generation.plan import make_plan
from rse.generation.procedural import generate


@pytest.fixture
def record(midi_path, source_record, nes_profile):
    return build_intake_record(parse_midi(midi_path, nes_profile), source_record, nes_profile)


@pytest.fixture
def composition(record):
    return composition_from_intake(record)


# -- intake -> canonical ------------------------------------------------------


def test_every_sounding_event_survives_conversion(record, composition):
    expected = sum(1 for e in record["events"] if e["type"] in ("note", "noise"))
    assert len(composition.sounding_events()) == expected


def test_noise_events_stay_noise(composition):
    noise = [e for e in composition.events if isinstance(e, Noise)]
    assert noise
    assert all(e.channel_id == "NO" for e in noise)
    assert not any(isinstance(e, Note) and e.channel_id == "NO" for e in composition.events)


def test_controllers_survive_conversion(record, composition):
    expected = sum(1 for e in record["events"] if e["type"] == "control")
    found = [e for e in composition.events if isinstance(e, Controller)]
    assert len(found) == expected
    assert expected > 0


def test_timing_and_loop_metadata_survive(record, composition):
    assert composition.ticks_per_beat == record["midi"]["ticks_per_beat"]
    assert composition.tempo_map[0].us_per_beat == record["tempo_map"][0]["us_per_beat"]
    assert composition.loop_start_tick == 0
    assert composition.loop_end_tick is not None


def test_provenance_is_carried_into_source_metadata(composition, source_record):
    assert composition.source_metadata["sha256"] == source_record.sha256
    assert composition.source_metadata["composition_id"] == source_record.source_id


def test_pitch_bend_becomes_an_effect_not_a_controller(nes_profile):
    """A register-level effect must not be flattened into controller data."""
    record = {
        "hardware_profile": "nes",
        "midi": {"ticks_per_beat": 480},
        "events": [{"ch": "P1", "type": "pitchwheel", "tick": 0, "pitch": -2048}],
    }
    composition = composition_from_intake(record)
    effects = [e for e in composition.events if isinstance(e, EffectEvent)]
    assert len(effects) == 1
    assert effects[0].effect == "pitch_bend"
    assert dict(effects[0].params)["pitch"] == -2048


def test_events_without_a_channel_are_dropped(nes_profile):
    record = {
        "hardware_profile": "nes",
        "midi": {"ticks_per_beat": 480},
        "events": [{"ch": None, "type": "note", "tick": 0, "pitch": 60, "dur_ticks": 10}],
    }
    assert composition_from_intake(record).events == []


# -- canonical -> intake ------------------------------------------------------


def test_projection_back_to_intake_preserves_notes(composition, nes_profile):
    projected = composition_to_intake(composition, profile=nes_profile)
    again = composition_from_intake(projected)
    assert again.sounding_events() == composition.sounding_events()


def test_projection_uses_profile_channel_order(composition, nes_profile):
    projected = composition_to_intake(composition, profile=nes_profile)
    assert [c["channel_id"] for c in projected["channels"]] == ["P1", "P2", "TR", "NO"]


# -- round trips --------------------------------------------------------------


def test_json_round_trip_is_exact(composition):
    restored = Composition.from_dict(composition.to_dict())
    report = compare_compositions(composition, restored)
    assert report.ok, report.to_dict()
    assert report.n_matched == report.n_original


def test_midi_round_trip_is_lossless_for_a_dataset_file(composition, nes_profile):
    """MVP criterion 2, at exact tolerance rather than merely within one."""
    _, report = round_trip_midi(composition, nes_profile, tolerance=EXACT)
    assert report.ok, report.to_dict()
    assert report.n_matched == report.n_original > 0


def test_midi_round_trip_is_lossless_for_generated_music(nes_profile):
    generated = generate(nes_profile, make_plan(seed=71, n_bars=8, energy="high"))
    composition = composition_from_intake(generated.record)
    _, report = round_trip_midi(composition, nes_profile, tolerance=EXACT)
    assert report.ok, report.to_dict()


def test_round_trip_preserves_tempo_and_loop(composition, nes_profile):
    restored, report = round_trip_midi(composition, nes_profile)
    assert restored.ticks_per_beat == composition.ticks_per_beat
    assert restored.loop_end_tick == composition.loop_end_tick
    assert "tempo map did not survive" not in report.notes


def test_midi_from_canonical_is_readable(composition, nes_profile, tmp_path):
    path = write_composition_midi(tmp_path / "c.mid", composition, profile=nes_profile)
    reparsed = composition_from_midi(path, nes_profile)
    assert reparsed.sounding_events() == composition.sounding_events()


# -- comparison and tolerance -------------------------------------------------


def base() -> Composition:
    return Composition(
        hardware_profile="nes",
        ticks_per_beat=480,
        events=[Note(channel_id="P1", tick=0, pitch=60, duration=240, velocity=100)],
    )


def test_a_missing_event_is_reported():
    empty = Composition(hardware_profile="nes", ticks_per_beat=480)
    report = compare_compositions(base(), empty)
    assert not report.ok
    assert len(report.missing) == 1
    assert report.missing[0]["type"] == "note"


def test_an_extra_event_is_reported():
    report = compare_compositions(Composition(hardware_profile="nes", ticks_per_beat=480), base())
    assert not report.ok
    assert len(report.added) == 1


def test_timing_drift_beyond_tolerance_is_caught():
    shifted = Composition(
        hardware_profile="nes",
        ticks_per_beat=480,
        events=[Note(channel_id="P1", tick=5, pitch=60, duration=240, velocity=100)],
    )
    exact = compare_compositions(base(), shifted)
    assert not exact.ok
    assert exact.missing  # too far to match at all under exact tolerance

    # A tolerance is a budget: a 5-tick shift inside an 8-tick budget is
    # acceptable, so it is counted but must not fail the report.
    loose = compare_compositions(base(), shifted, tolerance=Tolerance(timing_ticks=8))
    assert loose.n_matched == 1
    assert loose.drifted == []
    assert loose.n_within_tolerance == 1
    assert loose.ok

    # Narrower than the shift: the event is now too far away to be identified
    # as the same one, so it reads as loss rather than drift.
    tight = compare_compositions(base(), shifted, tolerance=Tolerance(timing_ticks=2))
    assert not tight.ok
    assert tight.missing and tight.added
    assert tight.n_matched == 0


def test_velocity_drift_is_reported_as_drift_not_loss():
    quieter = Composition(
        hardware_profile="nes",
        ticks_per_beat=480,
        events=[Note(channel_id="P1", tick=0, pitch=60, duration=240, velocity=64)],
    )
    report = compare_compositions(base(), quieter)
    assert report.n_matched == 1
    assert report.drifted[0]["drift"]["velocity"] == -36


def test_a_changed_pitch_is_loss_not_drift():
    """Pitch is identity: a different pitch is a different note."""
    transposed = Composition(
        hardware_profile="nes",
        ticks_per_beat=480,
        events=[Note(channel_id="P1", tick=0, pitch=67, duration=240, velocity=100)],
    )
    report = compare_compositions(base(), transposed)
    assert report.missing and report.added
    assert report.n_matched == 0


def test_a_changed_channel_is_loss():
    moved = Composition(
        hardware_profile="nes",
        ticks_per_beat=480,
        events=[Note(channel_id="P2", tick=0, pitch=60, duration=240, velocity=100)],
    )
    assert compare_compositions(base(), moved).n_matched == 0


def test_resolution_and_loop_changes_are_noted():
    changed = Composition(
        hardware_profile="nes",
        ticks_per_beat=96,
        events=list(base().events),
        loop_end_tick=100,
    )
    report = compare_compositions(base(), changed)
    assert any("resolution changed" in note for note in report.notes)
    assert any("loop boundaries" in note for note in report.notes)


def test_tolerance_serializes_for_the_report():
    payload = compare_compositions(base(), base(), tolerance=Tolerance(timing_ticks=2)).to_dict()
    assert payload["tolerance"]["timing_ticks"] == 2
    assert payload["ok"] is True
