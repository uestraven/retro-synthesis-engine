from __future__ import annotations

import mido
import pytest

from rse.data.midi_io import MidiParseError, parse_midi
from rse.data.synthetic import EXPRESSION_CC, write_nes_style_midi


def write(path, tracks, *, ticks_per_beat=480):
    midi = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    for track in tracks:
        midi.tracks.append(track)
    midi.save(str(path))
    return path


def track(name, messages):
    t = mido.MidiTrack()
    t.append(mido.MetaMessage("track_name", name=name, time=0))
    t.extend(messages)
    return t


def test_parses_a_four_channel_file(midi_path, nes_profile):
    comp = parse_midi(midi_path, nes_profile)
    assert comp.ticks_per_beat == 480
    assert {n.channel_id for n in comp.notes} == {"P1", "P2", "TR", "NO"}
    assert all(n.dur_ticks > 0 for n in comp.notes)
    assert comp.end_tick > 0


def test_controller_data_is_preserved(midi_path, nes_profile):
    comp = parse_midi(midi_path, nes_profile)
    expression = [
        c
        for c in comp.controls
        if c.type == "control_change" and c.data["control"] == EXPRESSION_CC
    ]
    assert expression, "expression controllers must survive parsing"
    assert {c.channel_id for c in expression} <= {"P1", "P2", "NO"}


def test_note_on_with_zero_velocity_ends_the_note(tmp_path, nes_profile):
    path = write(
        tmp_path / "a.mid",
        [
            track(
                "P1",
                [
                    mido.Message("note_on", note=60, velocity=100, time=0),
                    mido.Message("note_on", note=60, velocity=0, time=240),
                ],
            )
        ],
    )
    comp = parse_midi(path, nes_profile)
    assert len(comp.notes) == 1
    assert comp.notes[0].dur_ticks == 240
    assert comp.warnings == [] or all("note_off without" not in w for w in comp.warnings)


def test_repeated_note_on_closes_the_first_note(tmp_path, nes_profile):
    path = write(
        tmp_path / "b.mid",
        [
            track(
                "P1",
                [
                    mido.Message("note_on", note=60, velocity=100, time=0),
                    mido.Message("note_on", note=60, velocity=100, time=120),
                    mido.Message("note_off", note=60, velocity=0, time=120),
                ],
            )
        ],
    )
    comp = parse_midi(path, nes_profile)
    assert [n.dur_ticks for n in comp.notes] == [120, 120]


def test_unterminated_note_is_truncated_and_reported(tmp_path, nes_profile):
    path = write(
        tmp_path / "c.mid",
        [
            track(
                "P1",
                [
                    mido.Message("note_on", note=60, velocity=100, time=0),
                    mido.Message("control_change", control=11, value=64, time=480),
                ],
            )
        ],
    )
    comp = parse_midi(path, nes_profile)
    assert len(comp.notes) == 1
    assert comp.notes[0].dur_ticks == 480
    assert any("note left open" in w for w in comp.warnings)


def test_orphan_note_off_is_reported_without_inventing_a_note(tmp_path, nes_profile):
    path = write(
        tmp_path / "d.mid",
        [track("P1", [mido.Message("note_off", note=60, velocity=0, time=240)])],
    )
    comp = parse_midi(path, nes_profile)
    assert comp.notes == []
    assert any("note_off without note_on" in w for w in comp.warnings)


def test_zero_length_note_is_dropped_and_reported(tmp_path, nes_profile):
    path = write(
        tmp_path / "e.mid",
        [
            track(
                "P1",
                [
                    mido.Message("note_on", note=60, velocity=100, time=0),
                    mido.Message("note_off", note=60, velocity=0, time=0),
                ],
            )
        ],
    )
    comp = parse_midi(path, nes_profile)
    assert comp.notes == []
    assert any("zero-length note" in w for w in comp.warnings)


def test_tempo_and_meta_are_collected(tmp_path, nes_profile):
    path = write_nes_style_midi(tmp_path / "f.mid", seed=1, bars=4, tempo_bpm=150)
    comp = parse_midi(path, nes_profile)
    assert comp.time.tempo_at(0).bpm == pytest.approx(150, abs=0.5)
    assert any("loop" in e["text"] for e in comp.text_events)


def test_a_non_midi_file_raises_a_parse_error(tmp_path, nes_profile):
    bad = tmp_path / "not.mid"
    bad.write_bytes(b"this is not a midi file")
    with pytest.raises(MidiParseError):
        parse_midi(bad, nes_profile)
