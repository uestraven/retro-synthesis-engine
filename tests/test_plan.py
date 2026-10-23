from __future__ import annotations

import pytest

from rse.generation.plan import ENERGIES, FORMS, LENGTHS, MOODS, PlanSummary, make_plan


def test_a_seed_fully_determines_the_plan():
    first = make_plan(seed=42)
    second = make_plan(seed=42)
    assert first == second
    assert first.to_dict() == second.to_dict()


def test_different_seeds_give_different_plans():
    plans = {make_plan(seed=s).to_dict()["key"]["tonic"] for s in range(20)}
    assert len(plans) > 1


def test_an_omitted_seed_is_recorded_so_the_result_stays_reproducible():
    plan = make_plan()
    assert isinstance(plan.seed, int)
    assert make_plan(seed=plan.seed, mood=plan.mood) == plan


def test_explicit_parameters_are_honoured():
    plan = make_plan(
        seed=1, mood="dark", energy="high", tempo_bpm=165, n_bars=32, form="ABAC", tonic=3
    )
    assert plan.mood == "dark"
    assert plan.energy == "high"
    assert plan.tempo_bpm == 165
    assert plan.n_bars == 32
    assert plan.form == "ABAC"
    assert plan.key.tonic == 3


def test_mood_selects_a_fitting_scale():
    assert make_plan(seed=1, mood="heroic").key.scale == "major"
    assert make_plan(seed=1, mood="dark").key.scale == "minor"


def test_scale_can_override_the_mood_default():
    plan = make_plan(seed=1, mood="heroic", scale="minor_pentatonic")
    assert plan.key.scale == "minor_pentatonic"


def test_energy_drives_tempo():
    slow = [make_plan(seed=s, energy="low").tempo_bpm for s in range(8)]
    fast = [make_plan(seed=s, energy="high").tempo_bpm for s in range(8)]
    assert max(slow) < min(fast)


@pytest.mark.parametrize("bad", ["upbeat", "", "HEROIC"])
def test_unknown_mood_is_refused(bad):
    with pytest.raises(ValueError, match="unknown mood"):
        make_plan(seed=1, mood=bad)


def test_unknown_energy_and_scale_are_refused():
    with pytest.raises(ValueError, match="unknown energy"):
        make_plan(seed=1, energy="blistering")
    with pytest.raises(ValueError, match="unknown scale"):
        make_plan(seed=1, scale="phrygian_dominant")


def test_unsupported_length_is_refused():
    with pytest.raises(ValueError, match="n_bars must be one of"):
        make_plan(seed=1, n_bars=7)


@pytest.mark.parametrize("n_bars", LENGTHS)
@pytest.mark.parametrize("form", FORMS)
def test_sections_tile_the_bars_exactly(form, n_bars):
    plan = make_plan(seed=3, form=form, n_bars=n_bars)
    assert sum(s.n_bars for s in plan.sections) == n_bars
    # Contiguous, no gaps and no overlaps.
    bar = 0
    for section in plan.sections:
        assert section.start_bar == bar
        bar += section.n_bars
    assert bar == n_bars


@pytest.mark.parametrize("n_bars", LENGTHS)
def test_every_bar_resolves_to_a_section_and_a_chord(n_bars):
    plan = make_plan(seed=4, n_bars=n_bars)
    for bar in range(n_bars):
        assert plan.section_at(bar) is not None
        assert 0 <= plan.chord_at_bar(bar) <= 6


def test_repeated_form_letters_share_a_progression():
    """What makes AABA hear as a return rather than four unrelated ideas."""
    plan = make_plan(seed=5, form="AABA", n_bars=16)
    by_letter: dict[str, set[tuple[int, ...]]] = {}
    for section in plan.sections:
        by_letter.setdefault(section.label, set()).add(section.chord_degrees)
    assert all(len(progressions) == 1 for progressions in by_letter.values())
    assert len(by_letter["A"]) == 1


def test_section_lengths_differ_by_at_most_one_bar():
    plan = make_plan(seed=6, form="AABA", n_bars=8)
    lengths = [s.n_bars for s in plan.sections]
    assert max(lengths) - min(lengths) <= 1


def test_tick_arithmetic():
    plan = make_plan(seed=7, n_bars=8)
    assert plan.ticks_per_bar == 480 * 4
    assert plan.total_ticks == 480 * 4 * 8


def test_plan_round_trips_through_its_dict():
    plan = make_plan(seed=8, mood="mysterious", n_bars=16)
    payload = plan.to_dict()
    assert payload["seed"] == 8
    assert payload["mood"] == "mysterious"
    assert len(payload["sections"]) == len(plan.sections)
    assert payload["key"]["scale"] == plan.key.scale


def test_summary_names_the_key_and_lists_sections():
    plan = make_plan(seed=9, tonic=0, scale="major", form="AABA", n_bars=16)
    lines = PlanSummary(plan).lines()
    assert "C major" in lines[0]
    assert len(lines) == 1 + len(plan.sections)


@pytest.mark.parametrize("mood", MOODS)
@pytest.mark.parametrize("energy", ENERGIES)
def test_every_mood_and_energy_combination_plans(mood, energy):
    plan = make_plan(seed=11, mood=mood, energy=energy)
    assert plan.sections
    assert plan.tempo_bpm > 0
