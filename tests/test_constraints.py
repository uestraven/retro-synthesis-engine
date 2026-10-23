from __future__ import annotations

import pytest

from rse.events.convert import composition_from_intake
from rse.events.schema import Composition, Controller, Noise, Note
from rse.generation.plan import make_plan
from rse.generation.procedural import generate
from rse.hardware.constraints import (
    CLAMPED,
    REJECTED,
    SHORTENED,
    TRANSPOSED,
    ChannelAllocator,
    ConstraintEngine,
    channel_priority,
)


def composition(*events, loop_end=None) -> Composition:
    return Composition(
        hardware_profile="nes",
        ticks_per_beat=480,
        events=list(events),
        loop_end_tick=loop_end,
    )


def note(tick, pitch=60, duration=240, channel="P1", velocity=100):
    return Note(channel_id=channel, tick=tick, pitch=pitch, duration=duration, velocity=velocity)


# -- allocator ----------------------------------------------------------------


def test_allocator_starts_empty(nes_profile):
    allocator = ChannelAllocator(nes_profile)
    assert allocator.spans("P1") == []
    assert allocator.can_place("P1", 0, 240)


def test_allocator_tracks_claims(nes_profile):
    allocator = ChannelAllocator(nes_profile)
    assert allocator.place("P1", 0, 240)
    assert allocator.spans("P1") == [(0, 240)]
    assert not allocator.can_place("P1", 100, 240)


def test_back_to_back_notes_are_allowed(nes_profile):
    """Half-open intervals: a note ending where the next starts is monophonic."""
    allocator = ChannelAllocator(nes_profile)
    allocator.place("P1", 0, 240)
    assert allocator.can_place("P1", 240, 240)
    assert allocator.place("P1", 240, 240)


def test_channels_are_independent(nes_profile):
    allocator = ChannelAllocator(nes_profile)
    allocator.place("P1", 0, 480)
    assert allocator.can_place("P2", 0, 480)
    assert allocator.can_place("NO", 0, 480)


def test_zero_length_claims_are_refused(nes_profile):
    allocator = ChannelAllocator(nes_profile)
    assert not allocator.can_place("P1", 0, 0)
    assert not allocator.place("P1", 0, -5)


def test_overlapping_reports_the_blocking_spans(nes_profile):
    allocator = ChannelAllocator(nes_profile)
    allocator.place("P1", 0, 240)
    allocator.place("P1", 480, 240)
    assert allocator.overlapping("P1", 100, 100) == [(0, 240)]
    assert allocator.overlapping("P1", 240, 240) == []
    assert allocator.overlapping("P1", 0, 1000) == [(0, 240), (480, 720)]


def test_room_until_supports_shortening_instead_of_dropping(nes_profile):
    allocator = ChannelAllocator(nes_profile)
    allocator.place("P1", 480, 240)
    assert allocator.room_until("P1", 0) == 480
    assert allocator.room_until("P1", 720) is None


def test_unknown_channel_is_an_error(nes_profile):
    with pytest.raises(KeyError):
        ChannelAllocator(nes_profile).overlapping("FM1", 0, 10)


# -- validation (read-only) ---------------------------------------------------


def test_clean_music_has_no_violations(nes_profile):
    engine = ConstraintEngine(nes_profile)
    assert engine.validate(composition(note(0), note(240, 64))) == []


def test_polyphony_is_detected(nes_profile):
    engine = ConstraintEngine(nes_profile)
    violations = engine.validate(composition(note(0, duration=480), note(240, 64)))
    assert [v.kind for v in violations] == ["polyphony"]


def test_out_of_range_pitch_is_detected(nes_profile):
    engine = ConstraintEngine(nes_profile)
    violations = engine.validate(composition(note(0, pitch=12)))
    assert [v.kind for v in violations] == ["out_of_range"]
    assert "12" in violations[0].detail


def test_out_of_range_noise_value_is_detected(nes_profile):
    engine = ConstraintEngine(nes_profile)
    event = Noise(channel_id="NO", tick=0, value=60, duration=60)
    assert [v.kind for v in engine.validate(composition(event))] == ["out_of_range"]


def test_zero_duration_is_detected(nes_profile):
    engine = ConstraintEngine(nes_profile)
    assert [v.kind for v in engine.validate(composition(note(0, duration=0)))] == ["zero_duration"]


def test_unknown_channel_is_detected(nes_profile):
    engine = ConstraintEngine(nes_profile)
    violations = engine.validate(composition(note(0, channel="FM1")))
    assert [v.kind for v in violations] == ["unknown_channel"]


def test_notes_past_the_loop_end_are_detected(nes_profile):
    engine = ConstraintEngine(nes_profile)
    violations = engine.validate(composition(note(0, duration=2000), loop_end=960))
    assert "past_loop_end" in [v.kind for v in violations]


def test_a_profile_mismatch_is_detected(nes_profile):
    engine = ConstraintEngine(nes_profile)
    foreign = Composition(hardware_profile="genesis", ticks_per_beat=480)
    assert [v.kind for v in engine.validate(foreign)] == ["profile_mismatch"]


def test_validation_changes_nothing(nes_profile):
    engine = ConstraintEngine(nes_profile)
    original = composition(note(0, pitch=12), note(100, 64))
    before = original.to_dict()
    engine.validate(original)
    assert original.to_dict() == before


# -- repair -------------------------------------------------------------------


def test_clean_music_passes_through_untouched(nes_profile):
    engine = ConstraintEngine(nes_profile)
    original = composition(note(0), note(240, 64))
    repaired, report = engine.apply(original)
    assert report.ok
    assert report.modifications == []
    assert repaired.sounding_events() == original.sounding_events()


def test_an_out_of_range_pitch_is_transposed_rather_than_dropped(nes_profile):
    engine = ConstraintEngine(nes_profile)
    repaired, report = engine.apply(composition(note(0, pitch=12)))
    assert [m.kind for m in report.modifications] == [TRANSPOSED]
    kept = repaired.sounding_events()
    assert len(kept) == 1
    # 12 is below P1's floor of 33; octaves up land on 36.
    assert kept[0].pitch == 36
    assert nes_profile.channel("P1").in_range(kept[0].pitch)


def test_a_pitch_too_high_is_transposed_down(nes_profile):
    engine = ConstraintEngine(nes_profile)
    repaired, report = engine.apply(composition(note(0, pitch=125)))
    assert [m.kind for m in report.modifications] == [TRANSPOSED]
    assert repaired.sounding_events()[0].pitch == 101


def test_a_noise_value_is_clamped_because_it_has_no_octave(nes_profile):
    engine = ConstraintEngine(nes_profile)
    event = Noise(channel_id="NO", tick=0, value=60, duration=60)
    repaired, report = engine.apply(composition(event))
    assert [m.kind for m in report.modifications] == [CLAMPED]
    assert repaired.sounding_events()[0].value == 16


def test_overlapping_notes_are_shortened_so_both_survive(nes_profile):
    """Proposal 3.2 prefers rescheduling to discarding."""
    engine = ConstraintEngine(nes_profile)
    repaired, report = engine.apply(composition(note(0, duration=480), note(240, 64)))
    assert [m.kind for m in report.modifications] == [SHORTENED]
    kept = sorted(repaired.sounding_events(), key=lambda e: e.tick)
    assert len(kept) == 2
    assert kept[0].duration == 240  # truncated to where the next note begins
    assert kept[1].tick == 240
    assert engine.validate(repaired) == []


def test_a_note_with_nowhere_to_go_is_rejected(nes_profile):
    """Two notes at the same instant on a monophonic channel: one must lose."""
    engine = ConstraintEngine(nes_profile)
    repaired, report = engine.apply(composition(note(0, 60), note(0, 64)))
    assert [m.kind for m in report.modifications] == [REJECTED]
    assert len(repaired.sounding_events()) == 1
    assert report.n_rejected == 1


def test_notes_past_the_loop_end_are_trimmed_to_it(nes_profile):
    engine = ConstraintEngine(nes_profile)
    repaired, report = engine.apply(composition(note(0, duration=2000), loop_end=960))
    assert [m.kind for m in report.modifications] == [SHORTENED]
    assert repaired.sounding_events()[0].end_tick == 960


def test_a_note_starting_after_the_loop_end_is_rejected(nes_profile):
    engine = ConstraintEngine(nes_profile)
    repaired, report = engine.apply(composition(note(1000, duration=240), loop_end=960))
    assert [m.kind for m in report.modifications] == [REJECTED]
    assert repaired.sounding_events() == []


def test_events_on_an_unknown_channel_are_rejected(nes_profile):
    engine = ConstraintEngine(nes_profile)
    repaired, report = engine.apply(composition(note(0, channel="FM1")))
    assert report.n_rejected == 1
    assert repaired.events == []


def test_controllers_are_never_contended(nes_profile):
    """Only notes occupy a channel; expression data always passes through."""
    engine = ConstraintEngine(nes_profile)
    controllers = [Controller(channel_id="P1", tick=t, controller=11, value=t) for t in range(5)]
    repaired, report = engine.apply(composition(*controllers))
    assert report.ok
    assert len(repaired.events) == 5


def test_repair_is_idempotent(nes_profile):
    engine = ConstraintEngine(nes_profile)
    broken = composition(note(0, pitch=12, duration=480), note(240, 64), note(240, 67))
    once, first_report = engine.apply(broken)
    twice, second_report = engine.apply(once)
    assert not first_report.ok
    assert second_report.ok, second_report.to_dict()
    assert twice.sounding_events() == once.sounding_events()


def test_repair_output_always_validates_clean(nes_profile):
    engine = ConstraintEngine(nes_profile)
    broken = composition(
        note(0, pitch=200, duration=1000),
        note(100, pitch=64),
        note(100, pitch=67),
        Noise(channel_id="NO", tick=0, value=99, duration=60),
        note(0, channel="FM1"),
        loop_end=480,
    )
    repaired, report = engine.apply(broken)
    assert engine.validate(repaired) == [], report.to_dict()


def test_the_report_counts_what_it_did(nes_profile):
    engine = ConstraintEngine(nes_profile)
    _, report = engine.apply(
        composition(note(0, pitch=12), note(480, 64, duration=480), note(600, 67))
    )
    payload = report.to_dict()
    assert payload["profile_id"] == "nes"
    assert payload["n_events_before"] == 3
    assert payload["n_modifications"] == len(report.modifications)
    assert sum(payload["counts"].values()) == payload["n_modifications"]
    assert payload["ok"] is False


def test_min_duration_rejects_slivers(nes_profile):
    engine = ConstraintEngine(nes_profile, min_duration=100)
    _, report = engine.apply(composition(note(0, duration=10)))
    assert [m.kind for m in report.modifications] == [REJECTED]
    assert "below the minimum" in report.modifications[0].detail


# -- against real output ------------------------------------------------------


def test_the_generator_produces_nothing_the_engine_has_to_fix(nes_profile):
    """The strongest check available: generated music needs no repair."""
    engine = ConstraintEngine(nes_profile)
    for mood in ("heroic", "dark", "mysterious", "melancholic"):
        for energy in ("low", "medium", "high"):
            generated = generate(
                nes_profile, make_plan(seed=13, mood=mood, energy=energy, n_bars=8)
            )
            composed = composition_from_intake(generated.record)
            assert engine.validate(composed) == [], f"{mood}/{energy}"
            _, report = engine.apply(composed)
            assert report.ok, f"{mood}/{energy}: {report.counts()}"


def test_a_dataset_file_needs_no_repair(midi_path, nes_profile, source_record):
    from rse.data.midi_io import parse_midi
    from rse.data.normalize import build_intake_record

    record = build_intake_record(parse_midi(midi_path, nes_profile), source_record, nes_profile)
    engine = ConstraintEngine(nes_profile)
    _, report = engine.apply(composition_from_intake(record))
    assert report.ok, report.to_dict()


def test_channel_priority_puts_melody_above_percussion(nes_profile):
    priorities = {spec.role: channel_priority(spec) for spec in nes_profile.channels}
    assert priorities["pulse1"] > priorities["triangle"] > priorities["pulse2"]
    assert priorities["pulse2"] > priorities["noise"]
