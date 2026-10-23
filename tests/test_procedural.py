from __future__ import annotations

import pytest

from rse.generation.plan import ENERGIES, MOODS, make_plan
from rse.generation.procedural import REGISTERS, generate


@pytest.fixture
def composition(nes_profile):
    return generate(nes_profile, make_plan(seed=99, mood="heroic", energy="medium", n_bars=8))


def test_all_four_channels_are_written(composition):
    channels = {n.channel_id for n in composition.notes}
    assert channels == {"P1", "P2", "TR", "NO"}


def test_output_satisfies_the_hardware_profile(composition):
    """The generator must not emit anything the console could not play."""
    measurements = composition.record["measurements"]
    assert measurements["hardware_compliant"] is True
    for channel in composition.record["channels"]:
        assert channel["max_simultaneous"] == 1, channel["channel_id"]
        assert channel["overlap_count"] == 0, channel["channel_id"]
        assert channel["out_of_range_values"] == [], channel["channel_id"]


@pytest.mark.parametrize("mood", MOODS)
@pytest.mark.parametrize("energy", ENERGIES)
def test_every_mood_and_energy_stays_compliant(nes_profile, mood, energy):
    result = generate(nes_profile, make_plan(seed=5, mood=mood, energy=energy, n_bars=8))
    assert result.record["measurements"]["hardware_compliant"] is True
    for channel in result.record["channels"]:
        assert channel["max_simultaneous"] == 1


def test_percussion_never_doubles_up(nes_profile):
    """High energy adds hats and a pickup snare; they must not collide."""
    result = generate(nes_profile, make_plan(seed=7, energy="high", n_bars=8))
    drums = sorted((n.start_tick, n.dur_ticks) for n in result.notes if n.channel_id == "NO")
    for (start, duration), (next_start, _) in zip(drums, drums[1:], strict=False):
        assert start + duration <= next_start, "two drums sounding at once"


def test_generation_is_reproducible(nes_profile):
    first = generate(nes_profile, make_plan(seed=2024, mood="dark", n_bars=8))
    second = generate(nes_profile, make_plan(seed=2024, mood="dark", n_bars=8))
    assert first.notes == second.notes
    assert first.record["events"] == second.record["events"]


def test_different_seeds_produce_different_music(nes_profile):
    a = generate(nes_profile, make_plan(seed=1, mood="dark", n_bars=8))
    b = generate(nes_profile, make_plan(seed=2, mood="dark", n_bars=8))
    assert a.notes != b.notes


def test_notes_stay_inside_their_register(composition):
    for note in composition.notes:
        if note.channel_id == "NO":
            assert 1 <= note.value <= 16
            continue
        low, high = REGISTERS[note.channel_id]
        assert low <= note.value <= high, note


def test_nothing_crosses_the_loop_boundary(composition):
    limit = composition.plan.total_ticks
    for note in composition.notes:
        assert note.start_tick + note.dur_ticks <= limit


def test_notes_are_articulated_rather_than_held_full_length(composition):
    """Gaps between notes are what stop chiptune sounding like an organ."""
    melody = sorted((n.start_tick, n.dur_ticks) for n in composition.notes if n.channel_id == "P1")
    gaps = [
        next_start - (start + duration)
        for (start, duration), (next_start, _) in zip(melody, melody[1:], strict=False)
    ]
    assert gaps and all(gap >= 0 for gap in gaps)
    assert sum(1 for gap in gaps if gap > 0) > len(gaps) // 2


def test_velocity_varies_with_metrical_position(composition):
    velocities = {n.velocity for n in composition.notes}
    # Flat velocity is what made the first renders sound lifeless.
    assert len(velocities) >= 3


def test_downbeats_are_louder_than_subdivisions(composition):
    ticks_per_bar = composition.plan.ticks_per_bar
    downbeats = [n.velocity for n in composition.notes if n.start_tick % ticks_per_bar == 0]
    assert max(downbeats) >= 112


def test_the_melody_follows_the_harmony(nes_profile):
    """Each bar's melody should be anchored to that bar's chord."""
    plan = make_plan(seed=31, mood="heroic", energy="low", n_bars=8)
    result = generate(nes_profile, plan)
    key = plan.key
    matches = 0
    bars = 0
    for bar in range(plan.n_bars):
        chord_root = key.degree_to_pitch(plan.chord_at_bar(bar), 4) % 12
        in_bar = [
            n
            for n in result.notes
            if n.channel_id == "P1"
            and bar * plan.ticks_per_bar <= n.start_tick < (bar + 1) * plan.ticks_per_bar
        ]
        if not in_bar:
            continue
        bars += 1
        matches += in_bar[0].value % 12 == chord_root
    assert bars and matches / bars > 0.7


def test_the_bass_plays_chord_roots(nes_profile):
    plan = make_plan(seed=33, energy="low", n_bars=8)
    result = generate(nes_profile, plan)
    hits = total = 0
    for bar in range(plan.n_bars):
        root = plan.key.degree_to_pitch(plan.chord_at_bar(bar), 2) % 12
        for note in result.notes:
            if note.channel_id != "TR":
                continue
            if not (bar * plan.ticks_per_bar <= note.start_tick < (bar + 1) * plan.ticks_per_bar):
                continue
            total += 1
            hits += note.value % 12 == root
    assert total and hits / total > 0.6


def test_the_last_bar_resolves_to_the_tonic(nes_profile):
    plan = make_plan(seed=44, mood="dark", n_bars=8)
    result = generate(nes_profile, plan)
    final_bar_start = (plan.n_bars - 1) * plan.ticks_per_bar
    final = [
        n for n in result.notes if n.channel_id in ("P1", "TR") and n.start_tick >= final_bar_start
    ]
    assert any(n.value % 12 == plan.key.tonic for n in final)


def test_repeated_phrases_reuse_their_rhythm(nes_profile):
    """An AABA form must reuse material, or the form means nothing."""
    plan = make_plan(seed=55, form="AABA", n_bars=16, energy="medium")
    result = generate(nes_profile, plan)

    def rhythm_of(bar):
        return tuple(
            n.start_tick - bar * plan.ticks_per_bar
            for n in sorted(result.notes, key=lambda n: n.start_tick)
            if n.channel_id == "P1"
            and bar * plan.ticks_per_bar <= n.start_tick < (bar + 1) * plan.ticks_per_bar
        )

    first_a = plan.sections[0]
    second_a = plan.sections[1]
    assert rhythm_of(first_a.start_bar) == rhythm_of(second_a.start_bar)


def test_higher_energy_means_more_notes(nes_profile):
    counts = [
        len(generate(nes_profile, make_plan(seed=8, energy=energy, n_bars=8)).notes)
        for energy in ("low", "medium", "high")
    ]
    assert counts == sorted(counts)
    assert counts[2] > counts[0] * 2


def test_bar_count_is_respected(nes_profile):
    for n_bars in (4, 8, 16):
        plan = make_plan(seed=12, n_bars=n_bars)
        result = generate(nes_profile, plan)
        assert result.record["measurements"]["n_bars"] == n_bars


def test_loop_markers_are_emitted_for_looping_pieces(nes_profile):
    looped = generate(nes_profile, make_plan(seed=13, n_bars=8, loop=True))
    assert [m["text"] for m in looped.record["loop_markers"]] == ["loop_start", "loop_end"]
    one_shot = generate(nes_profile, make_plan(seed=13, n_bars=8, loop=False))
    assert one_shot.record["loop_markers"] == []


def test_the_record_carries_the_plan_for_reproducibility(composition):
    generation = composition.record["generation"]
    assert generation["generator"].startswith("procedural/")
    assert generation["plan"]["seed"] == 99
    assert composition.record["source"]["use_class"] == "commercially_cleared"
    assert composition.composition_id.startswith("generated:")


def test_generated_midi_has_one_named_track_per_channel(composition, nes_profile):
    names = [
        message.name
        for track in composition.midi.tracks
        for message in track
        if message.type == "track_name"
    ]
    assert names == ["meta", "P1", "P2", "TR", "NO"]


def test_plan_and_kwargs_are_mutually_exclusive(nes_profile):
    with pytest.raises(TypeError, match="either a plan or plan keyword arguments"):
        generate(nes_profile, make_plan(seed=1), seed=2)


def test_generator_refuses_a_profile_it_does_not_write_for(nes_profile):
    from dataclasses import replace

    with pytest.raises(ValueError, match="writes for the 'nes' profile"):
        generate(replace(nes_profile, profile_id="snes"), make_plan(seed=1))


def test_generate_can_plan_for_itself(nes_profile):
    result = generate(nes_profile, seed=77, mood="melancholic", n_bars=4)
    assert result.plan.seed == 77
    assert result.plan.mood == "melancholic"
