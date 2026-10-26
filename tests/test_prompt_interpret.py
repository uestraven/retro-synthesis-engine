"""The interpreter must not overclaim.

The failure mode that matters is not a missed keyword -- it is reporting a
requirement as satisfied when nothing in the pipeline delivers it. These tests
pin the three-way split, and in particular that "memorable" always lands in
`unsupported`.
"""

from __future__ import annotations

import pytest

from rse.prompt.interpret import MODES, interpret

ACCEPTANCE = (
    "Compose a 60-second NES-era JRPG battle theme. Dark and ominous but "
    "energetic, with a memorable opening melody, driving percussion, a "
    "contrasting middle section and a return to the main theme. Make it "
    "seamlessly loopable."
)


def statuses(plan, text: str) -> list[str]:
    return [r.status for r in plan.requirements if r.text == text]


def test_acceptance_prompt_parses() -> None:
    plan = interpret(ACCEPTANCE)
    assert plan.target_seconds == 60.0
    assert plan.key_mode == "minor"
    assert plan.genre == "soundtrack"
    assert plan.beats_per_bar == 4
    assert plan.percussion is True
    assert plan.tempo_bpm > 140, "an energetic battle theme should not be slow"


def test_memorable_is_reported_unsupported() -> None:
    """The one claim this system must never make."""
    plan = interpret(ACCEPTANCE)
    assert statuses(plan, "memorable") == ["unsupported"]
    report = plan.report()
    assert any(r["text"] == "memorable" for r in report["unsupported"])
    assert not any(r["text"] == "memorable" for r in report["satisfied"])


def test_structure_is_imposed_not_satisfied() -> None:
    """A returning theme is pasted by the assembler, not chosen by the model."""
    plan = interpret(ACCEPTANCE)
    report = plan.report()
    imposed = {r["text"] for r in report["imposed"]}
    assert {"contrasting", "return", "loopable"} <= imposed
    satisfied = {r["text"] for r in report["satisfied"]}
    assert not ({"contrasting", "return", "loopable"} & satisfied)
    assert set(plan.structure) == {"contrast_section", "theme_return", "loop"}


def test_mood_is_approximated_never_satisfied() -> None:
    """Mapping a word to a mode is a reading, not an understanding."""
    plan = interpret(ACCEPTANCE)
    assert statuses(plan, "dark") == ["approximated"]
    assert all(r.status != "satisfied" for r in plan.requirements if r.text == "dark")


def test_naming_a_melody_does_not_remove_the_band() -> None:
    """Instrument words are additive.

    "A memorable opening melody" once cut the plan to a single track, which
    would have produced a one-voice arrangement from a prompt that wanted a
    band.
    """
    plan = interpret(ACCEPTANCE)
    assert plan.programs == [80, 81, 38]
    assert plan.plan_tokens() == [
        "<TRACK_80>",
        "<TRACK_81>",
        "<TRACK_38>",
        "<DRUM_TRACK>",
    ]


def test_explicit_bpm_beats_an_energy_word() -> None:
    plan = interpret("a fast battle theme at 96 bpm")
    assert plan.tempo_bpm == 96
    assert statuses(plan, "96 bpm") == ["satisfied"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("90 seconds", 90.0),
        ("2 minutes", 120.0),
        ("a minute and a half", 90.0),
        ("a minute", 60.0),
    ],
)
def test_duration_forms(text: str, expected: float) -> None:
    assert interpret(f"a town theme, {text} long").target_seconds == expected


def test_duration_is_satisfied_because_it_is_arithmetic() -> None:
    """Bars follow from tempo and meter, so duration is not a guess."""
    plan = interpret("a 60 second theme at 120 bpm")
    assert plan.seconds_per_bar == pytest.approx(2.0)
    assert plan.bars_for_target(section_bars=8) == 32
    assert plan.bars_for_target(section_bars=8) * plan.seconds_per_bar == 64.0


def test_bars_round_to_whole_sections() -> None:
    """Form matters more than hitting the second exactly."""
    plan = interpret("a 60 second theme at 170 bpm")
    bars = plan.bars_for_target(section_bars=8)
    assert bars % 8 == 0
    assert abs(bars * plan.seconds_per_bar - 60.0) < plan.seconds_per_bar * 8


def test_waltz_sets_three_four() -> None:
    plan = interpret("a slow waltz for a ballroom scene")
    assert plan.beats_per_bar == 3


def test_every_mode_emitted_is_in_the_composers_vocabulary() -> None:
    """A mode the header cannot express would fail at generation time."""
    for word in ("dark", "sinister", "hopeful", "heroic", "peaceful", "mysterious"):
        assert interpret(f"a {word} theme").key_mode in MODES


def test_unrecognised_prompt_still_produces_a_usable_plan() -> None:
    """No keywords is not an error; it is the documented default."""
    plan = interpret("something for the credits")
    assert plan.tempo_bpm > 0
    assert plan.programs
    assert plan.key_mode in MODES
    assert plan.bars_for_target(section_bars=8) > 0


def test_requirements_are_not_duplicated() -> None:
    plan = interpret("dark dark dark and ominous, loop it, loopable, looping")
    texts = [(r.text, r.status) for r in plan.requirements]
    assert len(texts) == len(set(texts))


def test_report_buckets_are_exhaustive() -> None:
    """Every requirement lands in exactly one bucket."""
    plan = interpret(ACCEPTANCE)
    report = plan.report()
    total = sum(len(v) for v in report.values())
    assert total == len(plan.requirements)
    assert set(report) == {"satisfied", "approximated", "imposed", "unsupported"}
