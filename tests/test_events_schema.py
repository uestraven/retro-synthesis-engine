from __future__ import annotations

import pytest

from rse.data.timing import TempoChange, TimeSignatureChange
from rse.events.schema import (
    RSE_EF_SCHEMA_VERSION,
    Composition,
    Controller,
    EffectEvent,
    Noise,
    Note,
    Program,
    SchemaError,
    event_from_dict,
    event_sort_key,
)


def sample() -> Composition:
    return Composition(
        hardware_profile="nes",
        ticks_per_beat=480,
        events=[
            Note(channel_id="P1", tick=0, pitch=60, duration=240, velocity=100),
            Note(channel_id="P1", tick=240, pitch=64, duration=240, velocity=90),
            Noise(channel_id="NO", tick=0, value=7, duration=60, velocity=110),
            Controller(channel_id="P1", tick=0, controller=11, value=80),
            Program(channel_id="P2", tick=0, program=80),
            EffectEvent(channel_id="P1", tick=120, effect="duty", params=(("value", 2),)),
        ],
        tempo_map=[TempoChange(0, 500_000)],
        time_signatures=[TimeSignatureChange(0, 4, 4)],
        loop_start_tick=0,
        loop_end_tick=960,
    )


def test_noise_and_note_keep_separate_semantics():
    note = Note(channel_id="P1", tick=0, pitch=60, duration=10)
    noise = Noise(channel_id="NO", tick=0, value=7, duration=10)
    # The field that does not apply is absent, not null, so pitch-based code
    # cannot silently operate on a noise event.
    assert "pitch" in note.to_dict() and "value" not in note.to_dict()
    assert "value" in noise.to_dict() and "pitch" not in noise.to_dict()


def test_event_types_report_themselves():
    assert Note(channel_id="P1", tick=0).event_type == "note"
    assert Noise(channel_id="NO", tick=0).event_type == "noise"
    assert Controller(channel_id="P1", tick=0).event_type == "controller"
    assert Program(channel_id="P1", tick=0).event_type == "program"
    assert EffectEvent(channel_id="P1", tick=0).event_type == "effect"


def test_end_tick_accounts_for_duration():
    assert Note(channel_id="P1", tick=100, duration=240).end_tick == 340
    # Instantaneous events end where they start.
    assert Controller(channel_id="P1", tick=100).end_tick == 100


def test_composition_rejects_a_nonpositive_resolution():
    with pytest.raises(SchemaError, match="ticks_per_beat"):
        Composition(hardware_profile="nes", ticks_per_beat=0)


def test_queries():
    composition = sample()
    assert composition.end_tick == 480
    assert composition.channel_ids == ["NO", "P1", "P2"]
    assert len(composition.sounding_events()) == 3
    assert len(composition.sounding_events("P1")) == 2
    assert len(composition.events_on("P1")) == 4


def test_derived_timing_is_computed_not_stored():
    composition = sample()
    assert composition.time.beat(480) == pytest.approx(1.0)
    assert composition.time.seconds(960) == pytest.approx(1.0)
    # Nothing in the serialized form carries a beat or second value.
    for event in composition.to_dict()["events"]:
        assert "beat" not in event and "sec" not in event


def test_events_sort_deterministically():
    composition = sample()
    first = [event_sort_key(e) for e in composition.sorted_events()]
    composition.events.reverse()
    assert [event_sort_key(e) for e in composition.sorted_events()] == first


def test_normalize_orders_in_place():
    composition = sample()
    composition.events.reverse()
    composition.normalize()
    assert composition.events == composition.sorted_events()


def test_round_trips_through_its_dict():
    original = sample()
    restored = Composition.from_dict(original.to_dict())
    assert restored.to_dict() == original.to_dict()
    assert restored.events == original.sorted_events()
    assert restored.loop_end_tick == 960
    assert restored.ticks_per_beat == 480


def test_effects_survive_serialization():
    original = EffectEvent(
        channel_id="P1", tick=10, effect="slide", params=(("cents", -50), ("ticks", 20))
    )
    assert event_from_dict(original.to_dict()) == original


def test_instrument_id_is_optional_but_preserved():
    with_id = Note(channel_id="P1", tick=0, pitch=60, duration=10, instrument_id="lead")
    assert event_from_dict(with_id.to_dict()).instrument_id == "lead"
    without = Note(channel_id="P1", tick=0, pitch=60, duration=10)
    assert "instrument_id" not in without.to_dict()


def test_a_future_major_schema_is_refused():
    payload = sample().to_dict()
    payload["schema_version"] = "rse-ef/9.0.0"
    with pytest.raises(SchemaError, match="unsupported RSE-EF schema"):
        Composition.from_dict(payload)


def test_an_unknown_event_type_is_refused():
    with pytest.raises(SchemaError, match="unknown event type"):
        event_from_dict({"type": "sysex", "ch": "P1", "tick": 0})


def test_the_schema_version_is_stamped_on_new_compositions():
    assert sample().to_dict()["schema_version"] == RSE_EF_SCHEMA_VERSION
