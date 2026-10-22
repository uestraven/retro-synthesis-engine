from __future__ import annotations

import pytest

from rse import INTAKE_SCHEMA_VERSION
from rse.data.midi_io import parse_midi
from rse.data.normalize import build_intake_record, quantize_beat


@pytest.fixture
def record(midi_path, source_record, nes_profile):
    comp = parse_midi(midi_path, nes_profile)
    return build_intake_record(comp, source_record, nes_profile)


def test_record_carries_schema_profile_and_provenance(record, source_record):
    assert record["schema_version"] == INTAKE_SCHEMA_VERSION
    assert record["hardware_profile"] == "nes"
    assert record["composition_id"] == source_record.source_id
    assert record["source"]["sha256"] == source_record.sha256
    assert record["source"]["use_class"] == "commercially_cleared"


def test_every_event_keeps_its_original_tick_alongside_derived_positions(record):
    assert record["events"]
    for event in record["events"]:
        assert isinstance(event["tick"], int)
        assert event["beat"] == pytest.approx(event["tick"] / 480, abs=1e-6)
        assert event["bar"] >= 0
        assert event["sec"] >= 0


def test_pitched_and_noise_events_use_different_semantics(record):
    pitched = [e for e in record["events"] if e["type"] == "note"]
    noise = [e for e in record["events"] if e["type"] == "noise"]
    assert pitched and noise
    assert all("pitch" in e and "value" not in e for e in pitched)
    # The noise channel's number is a period index, so it must never be called
    # a pitch — nothing downstream may transpose it.
    assert all("value" in e and "pitch" not in e for e in noise)
    assert {e["ch"] for e in noise} == {"NO"}


def test_controller_events_reach_the_record(record):
    controls = [e for e in record["events"] if e["type"] == "control"]
    assert controls
    assert all("control" in e and "value" in e for e in controls)
    assert record["measurements"]["n_controls"] == len(
        [e for e in record["events"] if e["type"] not in ("note", "noise")]
    )


def test_channel_summaries_cover_the_whole_profile(record):
    assert [ch["channel_id"] for ch in record["channels"]] == ["P1", "P2", "TR", "NO"]
    for channel in record["channels"]:
        assert channel["n_notes"] > 0
        assert channel["max_simultaneous"] == 1
        assert channel["overlap_count"] == 0
        assert channel["out_of_range_values"] == []
        assert channel["assignment_method"] == "track_name"


def test_measurements_describe_the_composition(record):
    m = record["measurements"]
    assert m["n_notes"] > 0
    assert m["n_bars"] == 8
    assert m["channels_used"] == 4
    assert m["hardware_compliant"] is True
    assert m["constant_tempo"] is True and m["constant_meter"] is True
    assert m["dropped_notes"] == 0 and m["dropped_controls"] == 0


def test_loop_markers_are_extracted(record):
    assert [m["text"] for m in record["loop_markers"]] == ["loop_start", "loop_end"]


def test_quantization_is_additive_not_destructive(midi_path, source_record, nes_profile):
    comp = parse_midi(midi_path, nes_profile)
    plain = build_intake_record(comp, source_record, nes_profile)
    quantized = build_intake_record(comp, source_record, nes_profile, quantize_grid=0.25)

    assert plain["quantize_grid"] is None
    assert quantized["quantize_grid"] == 0.25
    assert all("q_beat" not in e for e in plain["events"])
    for original, snapped in zip(plain["events"], quantized["events"], strict=True):
        # Original timing is untouched; the grid position sits beside it.
        assert snapped["tick"] == original["tick"]
        assert snapped["beat"] == original["beat"]
        assert snapped["q_beat"] == pytest.approx(round(original["beat"] / 0.25) * 0.25)


def test_quantized_durations_never_collapse_to_zero(midi_path, source_record, nes_profile):
    comp = parse_midi(midi_path, nes_profile)
    record = build_intake_record(comp, source_record, nes_profile, quantize_grid=1.0)
    notes = [e for e in record["events"] if e["type"] in ("note", "noise")]
    assert notes
    assert all(e["q_dur_beats"] >= 1.0 for e in notes)


def test_quantize_beat_snaps_to_the_nearest_grid_point():
    assert quantize_beat(0.26, 0.25) == pytest.approx(0.25)
    assert quantize_beat(0.39, 0.25) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="grid must be positive"):
        quantize_beat(1.0, 0)


def test_out_of_range_notes_are_measured_not_removed(tmp_path, source_record, nes_profile):
    import mido

    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name="TR", time=0))
    track.append(mido.Message("note_on", note=12, velocity=100, time=0))  # below pitch_min 21
    track.append(mido.Message("note_off", note=12, velocity=0, time=240))
    midi.tracks.append(track)
    path = tmp_path / "low.mid"
    midi.save(str(path))

    comp = parse_midi(path, nes_profile)
    record = build_intake_record(comp, source_record, nes_profile)
    triangle = next(ch for ch in record["channels"] if ch["channel_id"] == "TR")
    assert triangle["out_of_range_values"] == [12]
    assert record["measurements"]["hardware_compliant"] is False
    # The event is still present — Phase 1 measures, Phase 2 enforces.
    assert any(e["type"] == "note" and e["pitch"] == 12 for e in record["events"])
