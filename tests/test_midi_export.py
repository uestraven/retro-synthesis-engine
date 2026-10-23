from __future__ import annotations

import mido
import pytest

from rse.data.midi_io import parse_midi
from rse.data.normalize import build_intake_record
from rse.data.provenance import describe_file
from rse.exporters.midi import record_to_midi, write_midi
from rse.generation.plan import make_plan
from rse.generation.procedural import generate


@pytest.fixture
def record(midi_path, source_record, nes_profile):
    return build_intake_record(parse_midi(midi_path, nes_profile), source_record, nes_profile)


def test_export_names_a_track_per_channel(record):
    midi = record_to_midi(record)
    names = [m.name for t in midi.tracks for m in t if m.type == "track_name"]
    assert names == ["meta", "P1", "P2", "TR", "NO"]


def test_export_preserves_resolution_and_tempo(record):
    midi = record_to_midi(record)
    assert midi.ticks_per_beat == record["midi"]["ticks_per_beat"]
    tempos = [m.tempo for t in midi.tracks for m in t if m.type == "set_tempo"]
    assert tempos == [record["tempo_map"][0]["us_per_beat"]]


def test_export_round_trips_every_note(record, nes_profile, tmp_path):
    """Export then re-import must preserve note timing, pitch and channel."""
    path = write_midi(tmp_path / "out.mid", record)
    reparsed = build_intake_record(
        parse_midi(path, nes_profile),
        describe_file(path, root=path.parent, dataset_id="rt", dataset_version="rt/1"),
        nes_profile,
    )

    def signature(rec):
        return sorted(
            (e["ch"], e["tick"], e["dur_ticks"], e.get("pitch", e.get("value")))
            for e in rec["events"]
            if e["type"] in ("note", "noise")
        )

    assert signature(reparsed) == signature(record)


def test_round_trip_preserves_loop_markers(record, nes_profile, tmp_path):
    path = write_midi(tmp_path / "loops.mid", record)
    reparsed = build_intake_record(
        parse_midi(path, nes_profile),
        describe_file(path, root=path.parent, dataset_id="rt", dataset_version="rt/1"),
        nes_profile,
    )
    assert [m["text"] for m in reparsed["loop_markers"]] == [
        m["text"] for m in record["loop_markers"]
    ]


def test_round_trip_preserves_controller_data(record, nes_profile, tmp_path):
    path = write_midi(tmp_path / "cc.mid", record)
    reparsed = build_intake_record(
        parse_midi(path, nes_profile),
        describe_file(path, root=path.parent, dataset_id="rt", dataset_version="rt/1"),
        nes_profile,
    )
    original = sorted(
        (e["ch"], e["tick"], e["control"], e["value"])
        for e in record["events"]
        if e["type"] == "control"
    )
    assert original
    assert (
        sorted(
            (e["ch"], e["tick"], e["control"], e["value"])
            for e in reparsed["events"]
            if e["type"] == "control"
        )
        == original
    )


def test_a_generated_composition_round_trips(nes_profile, tmp_path):
    composition = generate(nes_profile, make_plan(seed=61, n_bars=8, energy="high"))
    path = write_midi(tmp_path / "gen.mid", composition.record)
    reparsed = build_intake_record(
        parse_midi(path, nes_profile),
        describe_file(path, root=path.parent, dataset_id="g", dataset_version="g/1"),
        nes_profile,
    )
    assert reparsed["measurements"]["n_notes"] == composition.record["measurements"]["n_notes"]
    assert reparsed["measurements"]["hardware_compliant"] is True


def test_noise_events_survive_as_their_hardware_values(nes_profile, tmp_path):
    composition = generate(nes_profile, make_plan(seed=62, n_bars=4))
    path = write_midi(tmp_path / "noise.mid", composition.record)
    reparsed = build_intake_record(
        parse_midi(path, nes_profile),
        describe_file(path, root=path.parent, dataset_id="g", dataset_version="g/1"),
        nes_profile,
    )
    values = {e["value"] for e in reparsed["events"] if e["type"] == "noise"}
    assert values
    assert values <= {e["value"] for e in composition.record["events"] if e["type"] == "noise"}


def test_export_handles_a_record_without_channel_summaries(tmp_path):
    bare = {
        "midi": {"ticks_per_beat": 480},
        "events": [
            {"ch": "P1", "type": "note", "tick": 0, "dur_ticks": 240, "pitch": 60, "velocity": 90}
        ],
    }
    midi = record_to_midi(bare)
    notes = [m for t in midi.tracks for m in t if m.type == "note_on"]
    assert len(notes) == 1


def test_export_defaults_the_tempo_when_none_is_recorded():
    midi = record_to_midi({"midi": {"ticks_per_beat": 96}, "events": []})
    tempos = [m.tempo for t in midi.tracks for m in t if m.type == "set_tempo"]
    assert tempos == [500_000]


def test_exported_file_is_valid_midi(record, tmp_path):
    path = write_midi(tmp_path / "valid.mid", record)
    assert mido.MidiFile(str(path)).ticks_per_beat == 480
