"""Phrase structure, and the measured defect it addresses.

The v1 melody had no phrase structure at all: section A of every style used
ONE distinct attack pattern across eight bars, rested 0% of the time, and
landed on the tonic at a bar end once or twice in eight by accident. The motif
was stated, re-anchored on a random chord tone, and restated every bar.

These tests pin the plan, the cadences, and -- most importantly -- that the v1
path is untouched, because v1 is the version the listener approved and
phrasing is still an unproven idea.
"""

from __future__ import annotations

import pytest

from rse.procedural.compose import (
    ROWS_PER_BAR,
    STYLES,
    TICKS_PER_BAR,
    TICKS_PER_ROW,
    compose,
)
from rse.procedural.phrase import PHRASE_BARS, ROLES, describe, plan_phrases

# --- the plan -------------------------------------------------------------


def test_an_eight_bar_section_is_two_four_bar_phrases() -> None:
    roles = plan_phrases(8)
    assert len(roles) == 8
    assert [r.phrase for r in roles] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert [r.role for r in roles] == [
        "state", "vary", "climb", "cadence",
        "state", "vary", "climb", "cadence",
    ]


def test_only_the_last_phrase_closes() -> None:
    """What makes a period one sentence rather than two halves: the first
    phrase asks, the second answers."""
    roles = plan_phrases(8)
    cadences = [r for r in roles if r.is_cadence]
    assert [r.cadence for r in cadences] == ["open", "closed"]


def test_every_phrase_ends_in_a_cadence_with_a_rest() -> None:
    for n_bars in (4, 8, 12, 16):
        roles = plan_phrases(n_bars)
        for phrase in {r.phrase for r in roles}:
            bars = [r for r in roles if r.phrase == phrase]
            assert bars[-1].is_cadence, f"{n_bars} bars, phrase {phrase}"
            assert bars[-1].rest_rows > 0, "a cadence must leave a breath"
            # Only the last bar of a phrase rests.
            assert all(r.rest_rows == 0 for r in bars[:-1])


def test_every_bar_gets_exactly_one_role() -> None:
    for n_bars in range(1, 25):
        roles = plan_phrases(n_bars)
        assert len(roles) == n_bars, n_bars
        assert [r.bar for r in roles] == list(range(n_bars))
        assert all(r.role in ROLES for r in roles)


def test_a_short_section_still_arrives() -> None:
    """A two-bar section gets a statement and a cadence, not a truncated
    four-bar plan with the arrival cut off."""
    roles = plan_phrases(2)
    assert [r.role for r in roles] == ["state", "cadence"]
    assert roles[-1].cadence == "closed"
    assert plan_phrases(1)[0].is_cadence


def test_an_odd_bar_count_lengthens_the_last_phrase() -> None:
    """The remainder joins the final phrase rather than forming a stub whose
    cadence would be clipped."""
    roles = plan_phrases(10)
    assert len({r.phrase for r in roles}) == 2
    assert sum(1 for r in roles if r.phrase == 0) == PHRASE_BARS
    assert sum(1 for r in roles if r.phrase == 1) == 6
    assert roles[-1].is_cadence and roles[-1].cadence == "closed"


def test_no_bars_is_no_plan() -> None:
    assert plan_phrases(0) == []


def test_only_the_climb_lifts() -> None:
    roles = plan_phrases(8)
    assert all(r.lift > 0 for r in roles if r.role == "climb")
    assert all(r.lift == 0 for r in roles if r.role != "climb")


def test_describe_names_the_cadences() -> None:
    text = describe(plan_phrases(8))
    assert "cadence(open)" in text
    assert "cadence(closed)" in text


# --- v1 must be untouched -------------------------------------------------


@pytest.mark.parametrize("style", sorted(STYLES))
def test_phrasing_off_is_byte_identical_to_v1(style: str) -> None:
    """The control arm. `phrasing=None` must consume the same draws and
    produce the same events, or the baseline would move under an opt-in."""
    plain = compose(style_name=style, seed=800, target_seconds=60.0)
    explicit = compose(
        style_name=style, seed=800, target_seconds=60.0, phrasing=None
    )
    signature = lambda r: [  # noqa: E731 - local, single use
        (e.channel_id, e.tick, getattr(e, "pitch", getattr(e, "value", 0)),
         e.duration, e.velocity)
        for e in r.composition.sounding_events()
    ]
    assert signature(plain) == signature(explicit)


def test_an_unknown_phrasing_is_refused() -> None:
    with pytest.raises(KeyError):
        compose(style_name="boss", seed=1, phrasing="sonata")


# --- what phrasing actually changes ---------------------------------------


def section_a_melody(result):
    bars = result.sections[0].bars
    return sorted(
        (
            e
            for e in result.composition.sounding_events()
            if e.channel_id == "P1" and e.tick < bars * TICKS_PER_BAR
        ),
        key=lambda e: e.tick,
    ), bars


def attack_patterns(notes) -> set[tuple[int, ...]]:
    per_bar: dict[int, list] = {}
    for note in notes:
        per_bar.setdefault(note.tick // TICKS_PER_BAR, []).append(note)
    return {
        tuple(
            (n.tick % TICKS_PER_BAR) // TICKS_PER_ROW
            for n in sorted(bar, key=lambda n: n.tick)
        )
        for bar in per_bar.values()
    }


def rest_rows(notes, bars) -> int:
    covered: set[int] = set()
    for note in notes:
        start = note.tick // TICKS_PER_ROW
        covered.update(range(start, start + note.duration // TICKS_PER_ROW))
    total = bars * ROWS_PER_BAR
    return total - len(covered & set(range(total)))


@pytest.mark.parametrize("style", sorted(STYLES))
def test_v1_has_one_attack_pattern_and_no_rest(style: str) -> None:
    """The defect, pinned. If this ever fails, v1 changed."""
    notes, bars = section_a_melody(
        compose(style_name=style, seed=800, target_seconds=60.0)
    )
    assert len(attack_patterns(notes)) == 1, style
    assert rest_rows(notes, bars) == 0, style


@pytest.mark.parametrize("style", sorted(STYLES))
def test_phrasing_adds_a_breath(style: str) -> None:
    notes, bars = section_a_melody(
        compose(
            style_name=style, seed=800, target_seconds=60.0, phrasing="period"
        )
    )
    assert rest_rows(notes, bars) > 0, style


@pytest.mark.parametrize("style", sorted(STYLES))
def test_phrasing_adds_rhythmic_variety(style: str) -> None:
    """More than one attack pattern per section. Sparse styles reach this by
    displacement rather than augmentation, since a one-note bar has nothing to
    augment -- without that they kept their single pattern."""
    notes, _bars = section_a_melody(
        compose(
            style_name=style, seed=800, target_seconds=60.0, phrasing="period"
        )
    )
    assert len(attack_patterns(notes)) >= 2, style


@pytest.mark.parametrize("style", sorted(STYLES))
@pytest.mark.parametrize("seed", [800, 801, 802, 803])
def test_cadences_land_where_the_plan_says(style: str, seed: int) -> None:
    """A closed cadence on the tonic, an open one anywhere but. This is the
    difference between a line that arrives and one that stops."""
    result = compose(
        style_name=style, seed=seed, target_seconds=60.0, phrasing="period"
    )
    notes, bars = section_a_melody(result)
    per_bar: dict[int, list] = {}
    for note in notes:
        per_bar.setdefault(note.tick // TICKS_PER_BAR, []).append(note)
    tonic_pc = STYLES[style].tonic % 12
    checked = 0
    for role in plan_phrases(bars):
        if not role.is_cadence or role.bar not in per_bar:
            continue
        last = sorted(per_bar[role.bar], key=lambda n: n.tick)[-1]
        checked += 1
        if role.cadence == "closed":
            assert last.pitch % 12 == tonic_pc, f"{style}/{seed} bar {role.bar}"
        else:
            assert last.pitch % 12 != tonic_pc, f"{style}/{seed} bar {role.bar}"
    assert checked >= 2


@pytest.mark.parametrize("style", sorted(STYLES))
def test_phrasing_keeps_the_hardware_valid(style: str) -> None:
    from rse.hardware.constraints import ConstraintEngine
    from rse.hardware.profile import load_profile

    result = compose(
        style_name=style, seed=800, target_seconds=60.0, phrasing="period"
    )
    engine = ConstraintEngine(load_profile("nes"))
    assert engine.validate(result.composition) == []


# --- the isolated melody generator ----------------------------------------


def test_isolating_the_melody_rng_leaves_the_backing_alone() -> None:
    """Why this exists: the melody draws once per note, the phrased melody has
    fewer notes, so the drums and every later motif shifted. The first build
    of the paired listening test had different backing in 12 of 12 pairs and
    would have measured phrasing plus a different arrangement.
    """
    def channel(result, channel_id):
        return sorted(
            (e.tick, getattr(e, "pitch", getattr(e, "value", 0)), e.duration)
            for e in result.composition.sounding_events()
            if e.channel_id == channel_id
        )

    for style in sorted(STYLES):
        plain = compose(
            style_name=style, seed=800, target_seconds=60.0,
            phrasing=None, isolate_melody_rng=True,
        )
        phrased = compose(
            style_name=style, seed=800, target_seconds=60.0,
            phrasing="period", isolate_melody_rng=True,
        )
        for channel_id in ("P2", "TR", "NO"):
            assert channel(plain, channel_id) == channel(phrased, channel_id), (
                f"{style}: {channel_id} differs between arms"
            )
        assert channel(plain, "P1") != channel(phrased, "P1"), style


def test_isolation_is_off_by_default() -> None:
    """It changes v1's output, so it cannot be the default while v1 is the
    approved baseline."""
    signature = lambda r: [  # noqa: E731 - local, single use
        (e.channel_id, e.tick, getattr(e, "pitch", getattr(e, "value", 0)))
        for e in r.composition.sounding_events()
    ]
    default = compose(style_name="boss", seed=800, target_seconds=60.0)
    isolated = compose(
        style_name="boss", seed=800, target_seconds=60.0, isolate_melody_rng=True
    )
    assert signature(default) != signature(isolated)


def test_isolation_is_deterministic() -> None:
    signature = lambda r: [  # noqa: E731 - local, single use
        (e.channel_id, e.tick, getattr(e, "pitch", getattr(e, "value", 0)))
        for e in r.composition.sounding_events()
    ]
    first = compose(
        style_name="boss", seed=5, target_seconds=60.0,
        phrasing="period", isolate_melody_rng=True,
    )
    second = compose(
        style_name="boss", seed=5, target_seconds=60.0,
        phrasing="period", isolate_melody_rng=True,
    )
    assert signature(first) == signature(second)
